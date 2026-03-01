import os
import time
import json
import tempfile
import hashlib
import numpy as np
import asyncio
from functools import lru_cache
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

app = FastAPI(title="Financial RAG API - Sequential (Baseline)")

# ==========================================
# CONDITIONAL IMPORTS (Skip heavy deps in CI)
# ==========================================
if not TESTING:
    import mlflow
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from openai import AsyncOpenAI
    from tenacity import retry, wait_exponential, stop_after_attempt

    mlflow.set_tracking_uri("http://13.232.197.229:5001")
    mlflow.set_experiment("Financial-RAG")
    mlflow.openai.autolog()

    FastAPIInstrumentor.instrument_app(app)
else:
    SentenceTransformer = None
    CrossEncoder = None
    QdrantClient = None
    models = None
    AsyncOpenAI = None
    retry = None

# ==========================================
# LAZY LOADERS
# ==========================================
@lru_cache()
def get_embedder():
    if TESTING:
        return None
    device = "cuda" if USE_GPU else "cpu"
    return SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

@lru_cache()
def get_reranker():
    if TESTING:
        return None
    device = "cuda" if USE_GPU else "cpu"
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

@lru_cache()
def get_qdrant():
    if TESTING:
        return None
    return QdrantClient(url=QDRANT_URL)

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
    return {
        "mode": "sequential",
        "queue_size": 0
    }

@app.post("/embed")
def embed(req: EmbedRequest):
    if TESTING:
        return {"embeddings": [[0.0] * 384 for _ in req.texts]}

    embedder = get_embedder()
    vectors = embedder.encode(req.texts)
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
        self._write({"healthy": False, "disabled_until": time.time() + cooldown})

    def set_healthy(self, healthy=True):
        self._write({"healthy": healthy, "disabled_until": 0})

groq_breaker = CircuitBreaker("groq")

# ==========================================
# LLM CLIENT (Production only)
# ==========================================
if not TESTING:
    primary_client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    )

# ==========================================
# RAG FUNCTIONS
# ==========================================
def embed_query(query: str):
    if TESTING:
        return [0.0] * 384
    return get_embedder().encode(query).tolist()

def retrieve_from_qdrant(query_vector, ticker, document_type=None, limit=15):
    if TESTING:
        return type("obj", (object,), {"points": []})

    try:
        qdrant = get_qdrant()

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

        return qdrant.query_points(
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

    reranker = get_reranker()
    scores = reranker.predict([[query, t] for t in texts])
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
    except Exception:
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

async def generate_answer(system_prompt, user_query):
    if TESTING:
        return "Mock financial analysis response.", "MockProvider"

    if groq_breaker.is_healthy:
        try:
            resp = await safe_llm_call(
                primary_client,
                "llama-3.1-8b-instant",
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
            )
            return resp.choices[0].message.content, "Groq"
        except Exception:
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
# ASK (SEQUENTIAL BASELINE)
# ==========================================
@app.post("/ask")
async def ask(req: QueryRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
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

    query_vector = embed_query(req.query)
    search = retrieve_from_qdrant(query_vector, req.ticker, req.document_type)

    texts = [hit.payload.get("text", "") for hit in search.points if hit.payload]

    idx, scores = rerank_documents(req.query, texts, req.top_k)

    context = "\n\n".join([texts[i] for i in idx]) if len(idx) else "No relevant context."

    answer, provider = await generate_answer(
        f"You are a Wall Street analyst. Use ONLY this context:\n{context}",
        req.query,
    )

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