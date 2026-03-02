import os
import time
import json
import tempfile
import hashlib
import numpy as np
import asyncio
from functools import lru_cache
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from database import SessionLocal, CacheEntry, FeedbackEntry

load_dotenv()

# ==========================================
# CONFIG
# ==========================================
TESTING = os.getenv("TESTING", "False") == "True"
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = "financial_documents"

# ==========================================
# CONDITIONAL IMPORTS & TRACING SETUP
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
    start_span = mlflow.start_span
else:
    import contextlib

    SentenceTransformer = None
    CrossEncoder = None
    QdrantClient = None
    models = None
    AsyncOpenAI = None
    retry = lambda *a, **k: (lambda f: f)
    trace = lambda name=None, **kwargs: (lambda f: f)

    @contextlib.contextmanager
    def start_span(*args, **kwargs):
        yield None

# ==========================================
# LIFESPAN & APP INIT
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not TESTING:
        try:
            mlflow.set_tracking_uri("http://mlflow:5001")
            mlflow.set_experiment("Financial-RAG-Sequential")
            mlflow.openai.autolog(log_traces=True)
            print("✅ MLflow initialized successfully.")
        except Exception as e:
            print(f"⚠️ MLflow initialization failed: {e}")
    yield

app = FastAPI(title="Financial RAG API - Sequential (Baseline)", lifespan=lifespan)

if not TESTING:
    FastAPIInstrumentor.instrument_app(app)

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
# DATABASE DEP
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
class FeedbackRequest(BaseModel):
    query_hash: str
    rating: int

class QueryRequest(BaseModel):
    query: str
    ticker: str
    document_type: str | None = None
    top_k: int = 5

class EmbedRequest(BaseModel):
    texts: list[str]

# ==========================================
# HEALTH + EMBED
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
    return {"mode": "sequential", "queue_size": 0}

@app.post("/embed")
def embed(req: EmbedRequest):
    if TESTING:
        return {"embeddings": [[0.0] * 384 for _ in req.texts]}
    vectors = get_embedder().encode(req.texts)
    return {"embeddings": vectors.tolist()}

# ==========================================
# CIRCUIT BREAKER
# ==========================================
class CircuitBreaker:
    def __init__(self, name="groq"):
        self.file_path = os.path.join(tempfile.gettempdir(), f"{name}_cb_state.json")

    def _write(self, state):
        tmp = self.file_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(state, f)
            os.replace(tmp, self.file_path)
        except:
            pass

    @property
    def is_healthy(self):
        if not os.path.exists(self.file_path): return True
        try:
            with open(self.file_path) as f: state = json.load(f)
            if not state.get("healthy", True):
                if time.time() > state.get("disabled_until", 0):
                    self.set_healthy(True)
                    return True
                return False
            return True
        except:
            return True

    def trip(self, cooldown=60):
        self._write({"healthy": False, "disabled_until": time.time() + cooldown})

    def set_healthy(self, healthy=True):
        self._write({"healthy": healthy, "disabled_until": 0})

groq_breaker = CircuitBreaker("groq")

# ==========================================
# LLM CLIENT
# ==========================================
if not TESTING:
    primary_client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    )

# ==========================================
# QUERY ROUTER
# ==========================================
def route_query(query: str) -> str:
    """Lightweight heuristic to classify queries as SIMPLE or COMPLEX."""
    complex_keywords = ["compare", "analyze", "why", "impact", "trends", "growth", "risk"]
    if len(query.split()) > 20 or any(kw in query.lower() for kw in complex_keywords):
        return "COMPLEX"
    return "SIMPLE"

# ==========================================
# RAG CORE (TRACED)
# ==========================================
def embed_query(query: str):
    if TESTING: return [0.0] * 384
    return get_embedder().encode(query).tolist()

def retrieve_from_qdrant(query_vector, ticker, document_type=None, limit=15):
    if TESTING: return type("obj", (object,), {"points": []})
    try:
        must = [
            models.FieldCondition(
                key="ticker",
                match=models.MatchValue(value=ticker.upper())
            )
        ]
        if document_type:
            must.append(
                models.FieldCondition(
                    key="document_type",
                    match=models.MatchValue(value=document_type.upper())
                )
            )

        return get_qdrant().query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            query_filter=models.Filter(must=must)
        )
    except Exception:
        return type("obj", (object,), {"points": []})

def rerank_documents(query, texts, top_k):
    if TESTING or not texts:
        return list(range(min(top_k, len(texts)))), np.zeros(len(texts))

    scores = get_reranker().predict([[query, t] for t in texts])
    idx = np.argsort(scores)[::-1][:top_k]
    return idx, scores

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
                model=model_name, messages=messages, temperature=0.2
            ),
            timeout=12,
        )

@trace(name="Generate_Answer")
async def generate_answer(system_prompt, user_query, complexity="SIMPLE"):
    if TESTING:
        return "Mock financial analysis response.", "MockProvider"

    # 🚨 DYNAMIC ROUTING: Use 8B for fast queries, 70B for deep analysis
    groq_model = "llama-3.3-70b-versatile" if complexity == "COMPLEX" else "llama-3.1-8b-instant"

    if groq_breaker.is_healthy:
        try:
            resp = await safe_llm_call(
                primary_client,
                groq_model,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
            )
            return resp.choices[0].message.content, f"Groq ({groq_model})"
        except:
            groq_breaker.trip()

    return "⚠️ LLM unavailable.", "System Degraded"

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
# ASK (SEQUENTIAL BASELINE WITH ROOT SPAN)
# ==========================================
@app.post("/ask")
async def ask(req: QueryRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    if not TESTING:
        with mlflow.start_span(name="ASK_Request"):
            return await _ask_impl(req, background_tasks, db)
    else:
        return await _ask_impl(req, background_tasks, db)

# ==========================================
# IMPLEMENTATION
# ==========================================
async def _ask_impl(req: QueryRequest, background_tasks: BackgroundTasks, db: Session):
    # 🚨 Capture arrival time for End-to-End latency
    req_arrival_time = time.time()
    
    q_hash = hashlib.sha256(
        f"{req.ticker}_{req.query.lower()}".encode()
    ).hexdigest()

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

    if not TESTING:
        mlflow.set_tag("ticker", req.ticker)
        mlflow.set_tag("top_k", req.top_k)

    # 1. ROUTER TIMING
    with start_span(name="Query_Router"):
        t_router = time.time()
        complexity = route_query(req.query)
        if not TESTING:
            mlflow.log_metric("router_latency_ms", (time.time() - t_router) * 1000)
            mlflow.set_tag("query_complexity", complexity)

    # 2. EMBED TIMING
    with start_span(name="Embed_Query"):
        t0 = time.time()
        query_vector = await asyncio.to_thread(embed_query, req.query)
        if not TESTING:
            mlflow.log_metric("embed_latency_ms", (time.time() - t0) * 1000)

    # 3. RETRIEVAL TIMING
    with start_span(name="Qdrant_Vector_Search"):
        t1 = time.time()
        search = await asyncio.to_thread(retrieve_from_qdrant, query_vector, req.ticker, req.document_type)
        if not TESTING:
            mlflow.log_metric("qdrant_latency_ms", (time.time() - t1) * 1000)

    texts = [hit.payload.get("text", "") for hit in search.points if hit.payload]

    # 4. RERANK TIMING
    with start_span(name="CrossEncoder_Rerank"):
        t2 = time.time()
        idx, scores = await asyncio.to_thread(rerank_documents, req.query, texts, req.top_k)
        if not TESTING:
            mlflow.log_metric("rerank_latency_ms", (time.time() - t2) * 1000)
            mlflow.log_metric("retrieved_docs", len(texts))
            mlflow.log_metric("reranked_docs", len(idx))

    context = "\n\n".join([texts[i] for i in idx]) if len(idx) else "No relevant context."

    # 5. LLM TIMING
    t3 = time.time()
    answer, provider = await generate_answer(
        f"You are a Wall Street analyst. Use ONLY this context:\n{context}",
        req.query,
        complexity  # Pass complexity to trigger the 8B vs 70B shift
    )
    if not TESTING:
        mlflow.log_metric("llm_latency_ms", (time.time() - t3) * 1000)
        mlflow.set_tag("provider", provider)
        
        # Total End-to-End Latency
        mlflow.log_metric("total_e2e_ms", (time.time() - req_arrival_time) * 1000)

    sources = [
        {"score": float(scores[i]), "text": texts[i], "document_type": "SEC Filing"}
        for i in idx
    ] if len(idx) else []

    if provider != "System Degraded":
        background_tasks.add_task(
            save_to_cache, q_hash, req.query, answer, req.ticker, provider
        )

    return {
        "query_hash": q_hash,
        "query": req.query,
        "answer": answer,
        "sources": sources,
        "cached": False,
        "provider": provider,
    }