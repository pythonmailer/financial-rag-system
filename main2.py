import os
import time
import json
import tempfile
import hashlib
import numpy as np
import asyncio
import contextvars
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from database import SessionLocal, CacheEntry, FeedbackEntry

load_dotenv()

TESTING = os.getenv("TESTING", "False") == "True"
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = "financial_documents"

request_queue = None
MAX_BATCH_SIZE = 32
MAX_CONCURRENT_LLM_CALLS = 10
llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

# ==========================================
# CONDITIONAL IMPORTS & TRACING
# ==========================================
if not TESTING:
    import mlflow
    import mlflow.openai
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from openai import AsyncOpenAI
    from tenacity import retry, wait_exponential, stop_after_attempt

    trace = mlflow.trace
else:
    SentenceTransformer = None
    CrossEncoder = None
    QDRANT_URL = None
    models = None
    AsyncOpenAI = None
    retry = lambda *a, **k: (lambda f: f)
    wait_exponential = None
    stop_after_attempt = None
    trace = lambda name=None, **kwargs: (lambda f: f)

# ==========================================
# LAZY LOADERS
# ==========================================
@lru_cache()
def get_embedder():
    if TESTING: return None
    device = "cuda" if USE_GPU else "cpu"
    return SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

@lru_cache()
def get_reranker():
    if TESTING: return None
    device = "cuda" if USE_GPU else "cpu"
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

@lru_cache()
def get_qdrant():
    if TESTING: return None
    return QdrantClient(url=QDRANT_URL)

# ==========================================
# LLM CLIENTS
# ==========================================
if not TESTING:
    primary_client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    )

    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_client = (
        AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gemini_key,
        ) if gemini_key else None
    )

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_client = (
        AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
        if openrouter_key else None
    )
else:
    primary_client = None
    gemini_client = None
    openrouter_client = None

# ==========================================
# APP INIT
# ==========================================
app = FastAPI(title="Financial RAG API - Stream-Batched")

if not TESTING:
    FastAPIInstrumentor.instrument_app(app)

# ==========================================
# DB DEP
# ==========================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# SCHEMAS
# ==========================================
class QueryRequest(BaseModel):
    query: str
    ticker: str
    document_type: str | None = None
    top_k: int = 5

class FeedbackRequest(BaseModel):
    query_hash: str
    rating: int

class EmbedRequest(BaseModel):
    texts: list[str]

# ==========================================
# HEALTH
# ==========================================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    if TESTING:
        return {"status": "ready"}
    try:
        get_qdrant().get_collections()
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}

@app.get("/queue_status")
def queue_status():
    return {"queue_size": request_queue.qsize() if request_queue else 0}

# ==========================================
# EMBED
# ==========================================
@app.post("/embed")
def embed(req: EmbedRequest):
    if TESTING:
        return {"embeddings": [[0.0] * 384 for _ in req.texts]}
    vectors = get_embedder().encode(req.texts)
    return {"embeddings": vectors.tolist()}

# ==========================================
# RETRIEVAL
# ==========================================
@trace(name="Qdrant_Vector_Search")
def retrieve_from_qdrant(query_vector, ticker, document_type=None, limit=15):
    if TESTING:
        return type("obj", (object,), {"points": []})

    must = [
        models.FieldCondition(
            key="ticker",
            match=models.MatchValue(value=ticker.upper()),
        )
    ]

    if document_type:
        must.append(
            models.FieldCondition(
                key="document_type",
                match=models.MatchValue(value=document_type.upper()),
            )
        )

    try:
        return get_qdrant().query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            query_filter=models.Filter(must=must),
        )
    except Exception:
        return type("obj", (object,), {"points": []})

# ==========================================
# RERANK
# ==========================================
@trace(name="CrossEncoder_Rerank")
def rerank_documents(query, texts, top_k):
    if TESTING or not texts:
        idx = list(range(min(top_k, len(texts))))
        scores = np.zeros(len(texts))
        return idx, scores

    scores = get_reranker().predict([[query, t] for t in texts])
    idx = np.argsort(scores)[::-1][:top_k]
    return idx, scores

# ==========================================
# EMBED BATCH
# ==========================================
@trace(name="Batch_Embed_Queries")
def embed_query_batch(queries):
    if TESTING:
        return [[0.0] * 384 for _ in queries]
    return get_embedder().encode(queries).tolist()

# ==========================================
# CACHE SAVE
# ==========================================
def save_to_cache(q_hash, user_query, llm_response, ticker, provider):
    db = SessionLocal()
    try:
        db.add(
            CacheEntry(
                query_hash=q_hash,
                user_query=user_query,
                llm_response=llm_response,
                ticker=ticker.upper(),
                provider=provider,
            )
        )
        db.commit()
    except:
        db.rollback()
    finally:
        db.close()

# ==========================================
# LLM CALL
# ==========================================
if not TESTING:
    @retry(wait=wait_exponential(min=2, max=6), stop=stop_after_attempt(3))
    async def safe_llm_call(client, model_name, messages):
        return await asyncio.wait_for(
            client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
            ),
            timeout=12,
        )

# ==========================================
# GENERATION
# ==========================================
async def generate_answer(system_prompt, user_query):
    if TESTING:
        return "Mock financial analysis response.", "MockProvider"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    if groq_breaker.is_healthy:
        try:
            resp = await safe_llm_call(primary_client, "llama-3.1-8b-instant", messages)
            return resp.choices[0].message.content, "Groq"
        except:
            groq_breaker.trip()

    if gemini_client and gemini_breaker.is_healthy:
        try:
            resp = await safe_llm_call(gemini_client, "gemini-2.5-flash", messages)
            return resp.choices[0].message.content, "Gemini"
        except:
            gemini_breaker.trip()

    if openrouter_client and openrouter_breaker.is_healthy:
        try:
            resp = await safe_llm_call(
                openrouter_client,
                "meta-llama/llama-3.1-8b-instruct",
                messages,
            )
            return resp.choices[0].message.content, "OpenRouter"
        except:
            openrouter_breaker.trip()

    return "⚠️ All providers unavailable.", "System Degraded"

# ==========================================
# PROCESS (WORKER)
# ==========================================
@trace(name="Full_RAG_Pipeline")
async def process_independently(i, fut, req, q_hash, run_id, batch_vectors):
    if not TESTING and run_id:
        mlflow.start_run(run_id=run_id)

    async with llm_semaphore:
        try:
            search = await asyncio.to_thread(
                retrieve_from_qdrant, batch_vectors[i], req.ticker, req.document_type
            )

            texts = [hit.payload.get("text", "") for hit in search.points if hit.payload]

            if not texts:
                combined = "No relevant financial data found."
                sources = []
            else:
                idx, scores = await asyncio.to_thread(
                    rerank_documents, req.query, texts, req.top_k
                )
                combined = "\n\n".join([texts[j] for j in idx])
                sources = [
                    {
                        "score": float(scores[j]),
                        "text": texts[j],
                        "document_type": "SEC Filing",
                    }
                    for j in idx
                ]

            if not TESTING:
                mlflow.set_tag("ticker", req.ticker)
                mlflow.set_tag("top_k", req.top_k)
                mlflow.log_metric("retrieved_docs", len(texts))
                mlflow.log_metric("reranked_docs", len(sources))

            answer, provider = await generate_answer(
                f"You are a Wall Street analyst. Use ONLY this context:\n{combined}",
                req.query,
            )

            asyncio.create_task(
                asyncio.to_thread(
                    save_to_cache, q_hash, req.query, answer, req.ticker, provider
                )
            )

            fut.set_result(
                {
                    "query_hash": q_hash,
                    "query": req.query,
                    "answer": answer,
                    "sources": sources,
                    "cached": False,
                    "provider": provider,
                }
            )

        except Exception as e:
            fut.set_exception(e)

# ==========================================
# BATCH ENGINE
# ==========================================
async def batch_processor():
    while True:
        batch = []

        fut, req, q_hash, run_id = await request_queue.get()
        batch.append((fut, req, q_hash, run_id))

        await asyncio.sleep(0.05)

        while not request_queue.empty() and len(batch) < MAX_BATCH_SIZE:
            batch.append(request_queue.get_nowait())

        queries = [item[1].query for item in batch]
        vectors = await asyncio.to_thread(embed_query_batch, queries)

        ctx = contextvars.copy_context()

        for i, (fut, req, q_hash, run_id) in enumerate(batch):
            ctx.run(
                asyncio.create_task,
                process_independently(i, fut, req, q_hash, run_id, vectors),
            )

# ==========================================
# LIFESPAN
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global request_queue
    request_queue = asyncio.Queue()

    if not TESTING:
        try:
            mlflow.set_tracking_uri("http://mlflow:5001")
            mlflow.set_experiment("Financial-RAG")
            mlflow.openai.autolog(log_traces=True)
            print("✅ MLflow initialized successfully.")
        except Exception as e:
            print(f"⚠️ MLflow initialization failed: {e}")

    task = asyncio.create_task(batch_processor())
    yield
    task.cancel()

app.router.lifespan_context = lifespan

# ==========================================
# CACHE + FEEDBACK
# ==========================================
@app.delete("/cache/clear/{ticker}")
def clear_cache(ticker: str, db: Session = Depends(get_db)):
    count = db.query(CacheEntry).filter(CacheEntry.ticker == ticker.upper()).delete()
    db.commit()
    return {"cleared_entries": count}

@app.post("/feedback")
def feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    db.add(FeedbackEntry(query_hash=req.query_hash, rating=req.rating))
    db.commit()
    return {"status": "ok"}

# ==========================================
# ASK IMPLEMENTATION
# ==========================================
async def _ask_impl(req: QueryRequest, db: Session):
    q_hash = hashlib.sha256(f"{req.ticker}_{req.query.lower()}".encode()).hexdigest()

    cached = db.query(CacheEntry).filter(CacheEntry.query_hash == q_hash).first()
    if cached:
        return {
            "query_hash": q_hash,
            "query": req.query,
            "answer": cached.llm_response,
            "sources": [
                {"score": 1.0, "text": "Semantic Cache", "document_type": "Cache"}
            ],
            "cached": True,
            "provider": "Cache",
        }

    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    run_id = mlflow.active_run().info.run_id if (not TESTING and mlflow.active_run()) else None

    await request_queue.put((fut, req, q_hash, run_id))

    try:
        return await asyncio.wait_for(fut, timeout=30)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")

# ==========================================
# ASK ENDPOINT
# ==========================================
@app.post("/ask")
async def ask(req: QueryRequest, db: Session = Depends(get_db)):
    if not TESTING:
        with mlflow.start_run(run_name="rag_request", nested=True):
            with mlflow.start_span(name="ASK_Request"):
                return await _ask_impl(req, db)
    else:
        return await _ask_impl(req, db)