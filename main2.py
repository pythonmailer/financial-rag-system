import os
import time
import json
import tempfile
import hashlib
import numpy as np
import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.orm import Session

import mlflow
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import AsyncOpenAI
from database import SessionLocal, CacheEntry, FeedbackEntry
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

load_dotenv()

# ==========================================
# CONFIG
# ==========================================
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = "financial_documents"

request_queue = None
MAX_BATCH_SIZE = 32
MAX_CONCURRENT_LLM_CALLS = 50
llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

# ==========================================
# LAZY MODELS
# ==========================================
@lru_cache()
def get_embedder():
    device = "cuda" if USE_GPU else "cpu"
    return SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

@lru_cache()
def get_reranker():
    device = "cuda" if USE_GPU else "cpu"
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

@lru_cache()
def get_qdrant():
    return QdrantClient(url=QDRANT_URL)

# ==========================================
# MLflow
# ==========================================
mlflow.set_tracking_uri("http://mlflow:5001")
mlflow.set_experiment("Financial-RAG")
mlflow.openai.autolog()

# ==========================================
# DATABASE DEPENDENCY
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
# HEALTH ENDPOINTS
# ==========================================
app = FastAPI(title="Financial RAG API - Stream-Batched")
FastAPIInstrumentor.instrument_app(app)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    try:
        get_qdrant().get_collections()
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}

@app.get("/queue_status")
def queue_status():
    return {"queue_size": request_queue.qsize() if request_queue else 0}

# ==========================================
# EMBED ENDPOINT (used by ingestor)
# ==========================================
@app.post("/embed")
def embed(req: EmbedRequest):
    embedder = get_embedder()
    vectors = embedder.encode(req.texts)
    return {"embeddings": vectors.tolist()}

# ==========================================
# CIRCUIT BREAKER
# ==========================================
class CircuitBreaker:
    def __init__(self, service_name="groq"):
        self.file_path = os.path.join(tempfile.gettempdir(), f"{service_name}_cb_state.json")

    def _write_state(self, state):
        tmp = self.file_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(state, f)
            os.replace(tmp, self.file_path)
        except:
            pass

    @property
    def is_healthy(self):
        if not os.path.exists(self.file_path):
            return True
        try:
            with open(self.file_path) as f:
                state = json.load(f)
            if not state.get("healthy", True):
                if time.time() > state.get("disabled_until", 0):
                    self.set_healthy(True)
                    return True
                return False
            return True
        except:
            return True

    def trip(self, cooldown=60):
        self._write_state({"healthy": False, "disabled_until": time.time() + cooldown})

    def set_healthy(self, healthy=True):
        self._write_state({"healthy": healthy, "disabled_until": 0})

groq_breaker = CircuitBreaker("groq")
gemini_breaker = CircuitBreaker("gemini")
openrouter_breaker = CircuitBreaker("openrouter")

# ==========================================
# LLM CLIENTS
# ==========================================
primary_client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

gemini_key = os.getenv("GEMINI_API_KEY")
gemini_client = (
    AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=gemini_key,
    )
    if gemini_key
    else None
)

openrouter_key = os.getenv("OPENROUTER_API_KEY")
openrouter_client = (
    AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
    if openrouter_key
    else None
)

# ==========================================
# RETRIEVAL + RERANK
# ==========================================
def retrieve_from_qdrant(query_vector, ticker, document_type=None, limit=15):
    if TESTING or not qdrant:
        return type("obj", (object,), {"points": []})

    try:
        must_conditions = [
            models.FieldCondition(
                key="ticker",
                match=models.MatchValue(value=ticker.upper())
            )
        ]

        if document_type:
            must_conditions.append(
                models.FieldCondition(
                    key="document_type",
                    match=models.MatchValue(value=document_type.upper())
                )
            )

        return qdrant.query_points(
            collection_name="financial_documents",
            query=query_vector,
            limit=limit,
            query_filter=models.Filter(must=must_conditions)
        )
    except Exception:
        return type("obj", (object,), {"points": []})

def rerank_documents(query, texts, top_k):
    reranker = get_reranker()
    scores = reranker.predict([[query, t] for t in texts])
    idx = np.argsort(scores)[::-1][:top_k]
    return idx, scores

def embed_query_batch(queries):
    embedder = get_embedder()
    return embedder.encode(queries).tolist()

# ==========================================
# CACHE HELPERS
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
    except Exception:
        db.rollback()
    finally:
        db.close()

# ==========================================
# SAFE LLM CALL
# ==========================================
@retry(wait=wait_exponential(min=2, max=6), stop=stop_after_attempt(3))
async def safe_llm_call(client, model_name, messages, max_tokens=None):
    kwargs = {"model": model_name, "messages": messages, "temperature": 0.2}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return await asyncio.wait_for(client.chat.completions.create(**kwargs), timeout=12)

# ==========================================
# SIMPLE ROUTER (kept lightweight)
# ==========================================
async def route_query(query: str):
    if len(query) < 120:
        return "SIMPLE", "Groq"
    return "COMPLEX", "Groq"

# ==========================================
# GENERATION WITH FALLBACK
# ==========================================
async def generate_answer(system_prompt, user_query):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    if groq_breaker.is_healthy:
        try:
            resp = await safe_llm_call(primary_client, "llama-3.1-8b-instant", messages)
            return resp.choices[0].message.content, "Groq"
        except Exception:
            groq_breaker.trip()

    if gemini_client and gemini_breaker.is_healthy:
        try:
            resp = await safe_llm_call(gemini_client, "gemini-2.5-flash", messages)
            return resp.choices[0].message.content, "Gemini"
        except Exception:
            gemini_breaker.trip()

    if openrouter_client and openrouter_breaker.is_healthy:
        try:
            resp = await safe_llm_call(
                openrouter_client, "meta-llama/llama-3.1-8b-instruct", messages
            )
            return resp.choices[0].message.content, "OpenRouter"
        except Exception:
            openrouter_breaker.trip()

    return "⚠️ All providers unavailable.", "System Degraded"

# ==========================================
# BATCH ENGINE
# ==========================================
async def process_independently(i, fut, req, batch_vectors):
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
                    {"score": float(scores[j]), "text": texts[j], "document_type": "SEC Filing"}
                    for j in idx
                ]

            answer, provider = await generate_answer(
                f"You are a Wall Street analyst. Use ONLY this context:\n{combined}",
                req.query,
            )

            q_hash = hashlib.sha256(
                f"{req.ticker}_{req.query.lower()}".encode()
            ).hexdigest()

            asyncio.create_task(
                asyncio.to_thread(save_to_cache, q_hash, req.query, answer, req.ticker, provider)
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

async def batch_processor():
    while True:
        batch = []
        fut, req = await request_queue.get()
        batch.append((fut, req))

        await asyncio.sleep(0.05)

        while not request_queue.empty() and len(batch) < MAX_BATCH_SIZE:
            batch.append(request_queue.get_nowait())

        queries = [item[1].query for item in batch]
        vectors = await asyncio.to_thread(embed_query_batch, queries)

        for i, (fut, req) in enumerate(batch):
            asyncio.create_task(process_independently(i, fut, req, vectors))

# ==========================================
# LIFESPAN
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global request_queue
    request_queue = asyncio.Queue()
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
# ASK ENDPOINT
# ==========================================
@app.post("/ask")
async def ask(req: QueryRequest, db: Session = Depends(get_db)):
    q_hash = hashlib.sha256(f"{req.ticker}_{req.query.lower()}".encode()).hexdigest()

    cached = db.query(CacheEntry).filter(CacheEntry.query_hash == q_hash).first()
    if cached:
        return {
            "query_hash": q_hash,
            "query": req.query,
            "answer": cached.llm_response,
            "sources": [{"score": 1.0, "text": "Semantic Cache", "document_type": "Cache"}],
            "cached": True,
            "provider": "Cache",
        }

    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    await request_queue.put((fut, req))

    try:
        return await asyncio.wait_for(fut, timeout=30)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")