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
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends
from mlflow.entities import SpanType
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.orm import Session

load_dotenv()

AWS_IP = "13.232.197.229"

QDRANT_URL = os.getenv("QDRANT_URL", f"http://{AWS_IP}:6333")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"http://{AWS_IP}:5001")
DB_URL = os.getenv("DATABASE_URL", f"postgresql://admin:adminpassword@{AWS_IP}:5432/financial_rag")

from database import SessionLocal, CacheEntry, FeedbackEntry

TESTING = os.getenv("TESTING", "False") == "True"
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
COLLECTION_NAME = "financial_documents"

request_queue = None
MAX_BATCH_SIZE = 32
MAX_CONCURRENT_LLM_CALLS = 25
llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

class EmbedRequest(BaseModel):
    texts: list[str]

class QueryRequest(BaseModel):
    query: str
    ticker: str
    document_type: str | None = None
    top_k: int = 5

# =========================
# HASH FUNCTION (SHARED)
# =========================
def make_query_hash(req: QueryRequest) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "ticker": req.ticker,
                "query": req.query.lower(),
                "top_k": req.top_k,
                "doc_type": req.document_type,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

# =========================
# CONDITIONAL IMPORTS
# =========================
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
    start_span = mlflow.start_span
else:
    import contextlib

    SentenceTransformer = CrossEncoder = models = AsyncOpenAI = None
    retry = lambda *a, **k: (lambda f: f)
    wait_exponential = stop_after_attempt = None
    trace = lambda name=None, **kwargs: (lambda f: f)

    @contextlib.contextmanager
    def start_span(*args, **kwargs):
        yield None

# =========================
# LAZY LOADERS
# =========================
@lru_cache()
def get_embedder():
    if TESTING:
        return None
    device = "cpu"
    if USE_GPU:
        import torch

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
    return SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

@lru_cache()
def get_reranker():
    if TESTING:
        return None
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

@lru_cache()
def get_qdrant():
    if TESTING:
        return None
    return QdrantClient(url=QDRANT_URL)

# =========================
# LLM CLIENT
# =========================
if not TESTING:
    primary_client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    )
else:
    primary_client = None

# =========================
# APP INIT
# =========================
app = FastAPI(title="Financial RAG API")
if not TESTING:
    FastAPIInstrumentor.instrument_app(app)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =========================
# CORE HELPERS
# =========================
def route_query(query: str) -> str:
    complex_keywords = ["compare", "analyze", "why", "impact", "trends", "growth", "risk"]
    return "COMPLEX" if len(query.split()) > 20 or any(
        kw in query.lower() for kw in complex_keywords
    ) else "SIMPLE"

def retrieve_from_qdrant(query_vector, ticker, document_type=None, limit=15):
    must = [
        models.FieldCondition(
            key="ticker", match=models.MatchValue(value=ticker.upper())
        )
    ]
    if document_type:
        must.append(
            models.FieldCondition(
                key="document_type",
                match=models.MatchValue(value=document_type.upper()),
            )
        )

    return get_qdrant().query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        query_filter=models.Filter(must=must),
    )

def rerank_documents(query, texts, top_k):
    scores = get_reranker().predict([[query, t] for t in texts])
    idx = np.argsort(scores)[::-1][:top_k]
    return idx, scores

def embed_query_batch(queries):
    return get_embedder().encode(queries).tolist()

# =========================
# LLM CALL
# =========================
async def generate_answer(system_prompt, user_query, complexity="SIMPLE"):
    with start_span(name="LLM_Generation", span_type=SpanType.LLM) as span:
        if TESTING:
            return "Mock answer.", "Mock"

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        model = (
            "llama-3.3-70b-versatile"
            if complexity == "COMPLEX"
            else "llama-3.1-8b-instant"
        )

        res = await safe_llm_call(primary_client, model, msgs)
        ans = res.choices[0].message.content

        if span:
            span.set_attribute("llm_model", model)
            span.set_outputs({"llm_answer": ans})

        return ans, f"Groq ({model})"

if not TESTING:
    @retry(wait=wait_exponential(min=2, max=6), stop=stop_after_attempt(3))
    async def safe_llm_call(client, model_name, messages):
        return await asyncio.wait_for(
            client.chat.completions.create(
                model=model_name, messages=messages, temperature=0.2
            ),
            timeout=12,
        )

# =========================
# WORKER PIPELINE
# =========================
async def process_independently(i, fut, req, q_hash, batch_vectors, req_arrival_time):
    async with llm_semaphore:
        db = SessionLocal()  # ✅ NEW DB SESSION PER TASK

        try:
            with mlflow.start_run(run_name=f"Request-{req.ticker}"):

                complexity = route_query(req.query)

                t_start = time.time()
                search = await asyncio.to_thread(
                    retrieve_from_qdrant,
                    batch_vectors[i],
                    req.ticker,
                    req.document_type,
                )
                q_latency = (time.time() - t_start) * 1000

                texts = [
                    hit.payload.get("text", "")
                    for hit in search.points
                    if hit.payload
                ]

                if not texts:
                    combined, sources = "No context found.", []
                    rerank_ms = 0
                else:
                    t_rerank = time.time()
                    idx, scores = await asyncio.to_thread(
                        rerank_documents, req.query, texts, req.top_k
                    )
                    rerank_ms = (time.time() - t_rerank) * 1000

                    combined = "\n\n".join([texts[j] for j in idx])
                    sources = [
                        {"score": float(scores[j]), "text": texts[j]} for j in idx
                    ]

                answer, provider = await generate_answer(
                    f"Analyst context:\n{combined}", req.query, complexity
                )

                # ✅ MLflow LOGGING
                mlflow.log_param("ticker", req.ticker)
                mlflow.log_param("query", req.query)
                mlflow.log_param("top_k", req.top_k)
                mlflow.log_param("doc_type", req.document_type)
                mlflow.log_param("provider", provider)

                mlflow.log_metric("qdrant_ms", q_latency)
                mlflow.log_metric("rerank_ms", rerank_ms)
                mlflow.log_metric(
                    "total_e2e_ms", (time.time() - req_arrival_time) * 1000
                )

                # ✅ CACHE WRITE
                cache_entry = CacheEntry(
                    query_hash=q_hash,
                    user_query=req.query,
                    llm_response=answer,
                    created_at=datetime.now(timezone.utc),
                    ticker=req.ticker,
                    provider=provider,
                )
                db.add(cache_entry)
                db.commit()

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
        finally:
            db.close()

# =========================
# BATCH ENGINE
# =========================
async def batch_processor():
    while True:
        batch = []
        fut, req, q_hash, ctx, req_arrival_time = await request_queue.get()
        batch.append((fut, req, q_hash, ctx, req_arrival_time))
        await asyncio.sleep(0.05)

        while not request_queue.empty() and len(batch) < MAX_BATCH_SIZE:
            batch.append(request_queue.get_nowait())

        queries = [item[1].query for item in batch]
        vectors = await asyncio.to_thread(embed_query_batch, queries)

        for i, (fut, req, q_hash, ctx, req_arrival_time) in enumerate(batch):
            ctx.run(
                asyncio.create_task,
                process_independently(
                    i, fut, req, q_hash, vectors, req_arrival_time
                ),
            )

# =========================
# LIFESPAN
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global request_queue
    request_queue = asyncio.Queue()

    if not TESTING:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("Financial-RAG")
        mlflow.openai.autolog(log_traces=True)
        print(f"🚀 MLflow Connected to: {MLFLOW_URI}")

    task = asyncio.create_task(batch_processor())
    yield
    task.cancel()

app.router.lifespan_context = lifespan

# =========================
# ENDPOINTS
# =========================
@app.post("/ask")
async def ask(req: QueryRequest, db: Session = Depends(get_db)):
    q_hash = make_query_hash(req)

    cached = (
        db.query(CacheEntry)
        .filter(CacheEntry.query_hash == q_hash)
        .first()
    )

    if cached:
        return {
            "query_hash": q_hash,
            "answer": cached.llm_response,
            "sources": [],
            "cached": True,
            "provider": "Cache",
        }

    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    ctx = contextvars.copy_context()

    await request_queue.put((fut, req, q_hash, ctx, time.time()))

    return await asyncio.wait_for(fut, timeout=90)

@app.get("/ready")
def ready():
    try:
        get_qdrant().get_collections()
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}

@app.post("/embed")
async def embed(req: EmbedRequest):
    return {"embeddings": await asyncio.to_thread(embed_query_batch, req.texts)}