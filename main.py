import os
import time
import json
import tempfile
import hashlib
import numpy as np
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
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
# 1. INFRASTRUCTURE & TESTING FLAGS
# ==========================================
app = FastAPI(title="Financial RAG API - Sequential (Baseline)")

# CI/CD Fix: Bypass heavy connections if TESTING=True
TESTING = os.getenv("TESTING", "False") == "True"

if not TESTING:
    mlflow.set_tracking_uri("http://mlflow:5001")
    mlflow.set_experiment("Financial-RAG")
    mlflow.openai.autolog()
    
    print("Loading AI Models (Embedder & Reranker)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    qdrant = QdrantClient(url="http://qdrant:6333")
    # Only instrument if not testing to avoid startup crashes
    FastAPIInstrumentor.instrument_app(app)
else:
    model = None
    reranker = None
    qdrant = None

# ==========================================
# 2. DATABASE DEPENDENCY
# ==========================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 3. PYDANTIC MODELS & PRICING
# ==========================================
class FeedbackRequest(BaseModel):
    query_hash: str
    rating: int

class QueryRequest(BaseModel):
    query: str
    ticker: str
    document_type: str = None
    top_k: int = 5

MODEL_PRICING = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
}

def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = MODEL_PRICING.get(model_name, {"input": 0.0, "output": 0.0})
    return round((prompt_tokens * rates["input"] / 1_000_000) + (completion_tokens * rates["output"] / 1_000_000), 6)

# ==========================================
# 4. CIRCUIT BREAKER & CLIENTS
# ==========================================
class CircuitBreaker:
    def __init__(self, service_name="groq"):
        self.file_path = os.path.join(tempfile.gettempdir(), f"{service_name}_cb_state.json")
    
    def _write_state(self, state: dict):
        tmp_path = self.file_path + ".tmp"
        try:
            with open(tmp_path, "w") as f: json.dump(state, f)
            os.replace(tmp_path, self.file_path)
        except Exception: pass

    @property
    def is_healthy(self):
        if not os.path.exists(self.file_path): return True
        try:
            with open(self.file_path, "r") as f: state = json.load(f)
            if not state.get("healthy", True):
                if time.time() > state.get("disabled_until", 0):
                    self.set_healthy(True)
                    return True
                return False
            return True
        except Exception: return True

    def trip(self, cooldown_seconds=60):
        self._write_state({"healthy": False, "disabled_until": time.time() + cooldown_seconds})

    def set_healthy(self, healthy=True):
        self._write_state({"healthy": healthy, "disabled_until": 0})

groq_breaker = CircuitBreaker("groq")

primary_client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
gemini_key = os.getenv("GEMINI_API_KEY")
gemini_client = AsyncOpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=gemini_key) if gemini_key else None
openrouter_key = os.getenv("OPENROUTER_API_KEY")
openrouter_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key) if openrouter_key else None

# ==========================================
# 5. CORE RAG FUNCTIONS
# ==========================================
def embed_query(query: str):
    return model.encode(query).tolist() if model else [0.0]*384

def retrieve_from_qdrant(query_vector, ticker, document_type=None, limit=15):
    if not qdrant: return type('obj', (object,), {'points': []})
    must = [models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.upper()))]
    if document_type:
        must.append(models.FieldCondition(key="document_type", match=models.MatchValue(value=document_type.upper())))
    return qdrant.query_points(collection_name="financial_documents", query=query_vector, limit=limit, query_filter=models.Filter(must=must))

def rerank_documents(query, retrieved_texts, top_k=5):
    if not reranker or not retrieved_texts: return [], np.array([])
    top_k = min(top_k, len(retrieved_texts))
    scores = reranker.predict([[query, text] for text in retrieved_texts])
    return np.argsort(scores)[::-1][:top_k], scores

def save_to_cache(q_hash, user_query, llm_response, ticker, provider):
    db_session = SessionLocal()
    try:
        new_cache = CacheEntry(query_hash=q_hash, user_query=user_query, llm_response=llm_response, ticker=ticker.upper(), provider=provider)
        db_session.add(new_cache)
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        print(f"Cache failed: {e}")
    finally: db_session.close()

# ==========================================
# 6. RESILIENCE & GENERATION
# ==========================================
@retry(wait=wait_exponential(multiplier=1, min=2, max=6), stop=stop_after_attempt(3), retry=retry_if_exception_type((asyncio.TimeoutError, Exception)))
async def safe_llm_call(client, model_name, messages, max_tokens=None):
    kwargs = {"model": model_name, "messages": messages, "temperature": 0.2}
    if max_tokens: kwargs["max_tokens"] = max_tokens
    return await asyncio.wait_for(client.chat.completions.create(**kwargs), timeout=8)

async def route_query(query: str) -> tuple[str, str]:
    system_prompt = "You are a financial router. Output exactly SIMPLE or COMPLEX."
    if groq_breaker.is_healthy:
        try:
            resp = await safe_llm_call(primary_client, "llama-3.1-8b-instant", [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}], max_tokens=5)
            return ("COMPLEX" if "COMPLEX" in resp.choices[0].message.content.upper() else "SIMPLE", "Groq")
        except Exception: groq_breaker.trip(60)
    return ("SIMPLE", "System Degraded")

async def generate_with_fallback(system_prompt, user_query, complexity: str, router_provider: str):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]
    m_name = "llama-3.3-70b-versatile" if complexity == "COMPLEX" else "llama-3.1-8b-instant"
    
    try:
        completion = await safe_llm_call(primary_client, m_name, messages)
        usage = completion.usage
        return completion.choices[0].message.content, "Groq", m_name, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
    except Exception:
        return "⚠️ Service Notice: Providers unavailable.", "System Degraded", "None", 0, 0, 0

# ==========================================
# 7. API ENDPOINTS (Optimized)
# ==========================================
@app.delete("/cache/clear/{ticker}")
def clear_semantic_cache(ticker: str, db: Session = Depends(get_db)):
    deleted_count = db.query(CacheEntry).filter(CacheEntry.ticker == ticker.upper()).delete()
    db.commit()
    return {"status": "success", "cleared_entries": deleted_count, "ticker": ticker.upper()}

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    new_feedback = FeedbackEntry(query_hash=request.query_hash, rating=request.rating)
    db.add(new_feedback)
    db.commit()
    return {"status": "success", "recorded_rating": request.rating}

@app.post("/ask")
async def ask_financial_question(request: QueryRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    hash_input = f"{request.ticker.upper()}_{request.query.strip().lower()}"
    if request.document_type: hash_input += f"_{request.document_type.upper()}"
    query_hash = hashlib.sha256(hash_input.encode()).hexdigest()

    cached = db.query(CacheEntry).filter(CacheEntry.query_hash == query_hash).first()
    if cached:
        return {"query_hash": query_hash, "answer": cached.llm_response, "cached": True, "provider": "Cache"}

    # Sequential execution (Safe for baseline comparison)
    with mlflow.start_span(name=f"Sequential_{request.ticker}") if not TESTING else tempfile.TemporaryDirectory() as span:
        query_vector = embed_query(request.query)
        search_res = retrieve_from_qdrant(query_vector, request.ticker, request.document_type)
        
        texts = [hit.payload.get("text", "") for hit in search_res.points]
        top_idx, _ = rerank_documents(request.query, texts, request.top_k)
        context = "".join([f"- {texts[i]}\n" for i in top_idx])

        complexity, r_provider = await route_query(request.query)
        answer, prov, model_used, p_tok, c_tok, t_tok = await generate_with_fallback(f"Use context:\n{context}", request.query, complexity, r_provider)

        if prov != "System Degraded":
            background_tasks.add_task(save_to_cache, query_hash, request.query, answer, request.ticker, prov)

        return {"query_hash": query_hash, "answer": answer, "cached": False, "provider": prov}