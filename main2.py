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
MAX_CONCURRENT_LLM_CALLS = 25
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
    start_span = mlflow.start_span
else:
    import contextlib

    SentenceTransformer = None
    CrossEncoder = None
    models = None
    AsyncOpenAI = None
    retry = lambda *a, **k: (lambda f: f)
    wait_exponential = None
    stop_after_attempt = None
    trace = lambda name=None, **kwargs: (lambda f: f)

    @contextlib.contextmanager
    def start_span(*args, **kwargs):
        yield None

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
# CIRCUIT BREAKERS
# ==========================================
class CircuitBreaker:
    def __init__(self, service_name="groq"):
        self.file_path = os.path.join(tempfile.gettempdir(), f"{service_name}_cb_state.json")

    def _write_state(self, state):
        try:
            with open(self.file_path, "w") as f:
                json.dump(state, f)
        except: pass

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
        except: return True

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
if not TESTING:
    primary_client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    )
    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_client = AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=gemini_key,
    ) if gemini_key else None

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=openrouter_key
    ) if openrouter_key else None
else:
    primary_client = gemini_client = openrouter_client = None

# ==========================================
# APP INIT
# ==========================================
app = FastAPI(title="Financial RAG API - Stream-Batched")
if not TESTING: FastAPIInstrumentor.instrument_app(app)

# ==========================================
# SCHEMAS & DEPENDENCIES
# ==========================================
def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

class QueryRequest(BaseModel):
    query: str; ticker: str; document_type: str | None = None; top_k: int = 5
class FeedbackRequest(BaseModel):
    query_hash: str; rating: int
class EmbedRequest(BaseModel):
    texts: list[str]

# ==========================================
# HEALTH
# ==========================================
@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/ready")
def ready():
    if TESTING: return {"status": "ready"}
    try:
        get_qdrant().get_collections()
        return {"status": "ready"}
    except Exception as e: return {"status": "not_ready", "error": str(e)}

@app.get("/queue_status")
def queue_status():
    return {"queue_size": request_queue.qsize() if request_queue else 0}

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
# RETRIEVAL & RERANK
# ==========================================
def retrieve_from_qdrant(query_vector, ticker, document_type=None, limit=15):
    if TESTING: return type("obj", (object,), {"points": []})
    must = [models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.upper()))]
    if document_type: must.append(models.FieldCondition(key="document_type", match=models.MatchValue(value=document_type.upper())))
    try:
        return get_qdrant().query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=limit, query_filter=models.Filter(must=must))
    except: return type("obj", (object,), {"points": []})

def rerank_documents(query, texts, top_k):
    if TESTING or not texts: return list(range(min(top_k, len(texts)))), np.zeros(len(texts))
    scores = get_reranker().predict([[query, t] for t in texts])
    idx = np.argsort(scores)[::-1][:top_k]
    return idx, scores

def embed_query_batch(queries):
    if TESTING: return [[0.0] * 384 for _ in queries]
    return get_embedder().encode(queries).tolist()

def save_to_cache(q_hash, user_query, llm_response, ticker, provider):
    db = SessionLocal()
    try:
        db.add(CacheEntry(query_hash=q_hash, user_query=user_query, llm_response=llm_response, ticker=ticker.upper(), provider=provider))
        db.commit()
    except: db.rollback()
    finally: db.close()

# ==========================================
# LLM CALL
# ==========================================
if not TESTING:
    @retry(wait=wait_exponential(min=2, max=6), stop=stop_after_attempt(3))
    async def safe_llm_call(client, model_name, messages):
        return await asyncio.wait_for(client.chat.completions.create(model=model_name, messages=messages, temperature=0.2), timeout=12)

@trace(name="Generate_Answer")
async def generate_answer(system_prompt, user_query, complexity="SIMPLE"):
    if TESTING: return "Mock answer.", "Mock"
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]
    
    # DYNAMIC ROUTING: Use 8B for fast queries, 70B for deep analysis
    groq_model = "llama-3.3-70b-versatile" if complexity == "COMPLEX" else "llama-3.1-8b-instant"

    if groq_breaker.is_healthy:
        try:
            res = await safe_llm_call(primary_client, groq_model, msgs)
            return res.choices[0].message.content, f"Groq ({groq_model})"
        except: groq_breaker.trip()
    if gemini_client and gemini_breaker.is_healthy:
        try:
            res = await safe_llm_call(gemini_client, "gemini-2.5-flash", msgs)
            return res.choices[0].message.content, "Gemini"
        except: gemini_breaker.trip()
    if openrouter_client and openrouter_breaker.is_healthy:
        try:
            res = await safe_llm_call(openrouter_client, "meta-llama/llama-3.1-8b-instruct", msgs)
            return res.choices[0].message.content, "OpenRouter"
        except: openrouter_breaker.trip()
    return "⚠️ All providers unavailable.", "System Degraded"

# ==========================================
# PROCESS (WORKER)
# ==========================================
@trace(name="Full_RAG_Pipeline")
async def process_independently(i, fut, req, q_hash, batch_vectors, req_arrival_time, embed_ms):
    # Calculate Time-in-Queue
    queue_wait_ms = (time.time() - req_arrival_time) * 1000

    async with llm_semaphore:
        try:
            # 1. ROUTER TIMING (Creates a Trace Waterfall Block)
            t0 = time.time()
            with start_span(name="Query_Router"):
                complexity = route_query(req.query)
            router_ms = (time.time() - t0) * 1000

            # 2. RETRIEVAL TIMING (Creates a Trace Waterfall Block)
            t1 = time.time()
            with start_span(name="Qdrant_Vector_Search"):
                search = await asyncio.to_thread(retrieve_from_qdrant, batch_vectors[i], req.ticker, req.document_type)
            qdrant_ms = (time.time() - t1) * 1000

            texts = [hit.payload.get("text", "") for hit in search.points if hit.payload]
            
            rerank_ms = 0
            if not texts:
                combined, sources = "No context found.", []
            else:
                # 3. RERANK TIMING (Creates a Trace Waterfall Block)
                t2 = time.time()
                with start_span(name="CrossEncoder_Rerank"):
                    idx, scores = await asyncio.to_thread(rerank_documents, req.query, texts, req.top_k)
                rerank_ms = (time.time() - t2) * 1000
                
                combined = "\n\n".join([texts[j] for j in idx])
                sources = [{"score": float(scores[j]), "text": texts[j]} for j in idx]

            # 4. LLM GENERATION TIMING
            t3 = time.time()
            answer, provider = await generate_answer(f"Analyst context:\n{combined}", req.query, complexity)
            llm_ms = (time.time() - t3) * 1000

            # 5. LOG EXPLICIT METRICS TO MLFLOW (Builds the Line Charts)
            if not TESTING:
                mlflow.set_tag("ticker", req.ticker)
                mlflow.set_tag("query_complexity", complexity)
                mlflow.log_metric("retrieved_docs", len(texts))
                
                # The complete lifecycle!
                mlflow.log_metric("queue_wait_ms", queue_wait_ms)
                mlflow.log_metric("embed_latency_ms", embed_ms)
                mlflow.log_metric("router_latency_ms", router_ms)
                mlflow.log_metric("qdrant_latency_ms", qdrant_ms)
                if texts: mlflow.log_metric("rerank_latency_ms", rerank_ms)
                mlflow.log_metric("llm_latency_ms", llm_ms)
                
                # Total End-to-End time
                mlflow.log_metric("total_e2e_ms", (time.time() - req_arrival_time) * 1000)

            asyncio.create_task(asyncio.to_thread(save_to_cache, q_hash, req.query, answer, req.ticker, provider))

            fut.set_result({"query_hash": q_hash, "query": req.query, "answer": answer, "sources": sources, "cached": False, "provider": provider})
        except Exception as e:
            fut.set_exception(e)

# ==========================================
# BATCH ENGINE (Context Restorer)
# ==========================================
async def batch_processor():
    while True:
        batch = []
        fut, req, q_hash, ctx, req_arrival_time = await request_queue.get()
        batch.append((fut, req, q_hash, ctx, req_arrival_time))
        await asyncio.sleep(0.05)

        while not request_queue.empty() and len(batch) < MAX_BATCH_SIZE:
            batch.append(request_queue.get_nowait())

        queries = [item[1].query for item in batch]
        
        t_embed = time.time()
        with start_span(name="Batch_Embed_Queries"):
            vectors = await asyncio.to_thread(embed_query_batch, queries)
        embed_ms = (time.time() - t_embed) * 1000 / len(batch) 

        for i, (fut, req, q_hash, ctx, req_arrival_time) in enumerate(batch):
            # Preserving the trace context inside the background queue
            ctx.run(
                asyncio.create_task,
                process_independently(i, fut, req, q_hash, vectors, req_arrival_time, embed_ms),
            )

# ==========================================
# LIFESPAN
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global request_queue
    request_queue = asyncio.Queue()
    if not TESTING:
        mlflow.set_tracking_uri("http://mlflow:5001")
        mlflow.set_experiment("Financial-RAG")
        mlflow.openai.autolog(log_traces=True)
    task = asyncio.create_task(batch_processor())
    yield
    task.cancel()

app.router.lifespan_context = lifespan

# ==========================================
# ENDPOINTS
# ==========================================
@app.delete("/cache/clear/{ticker}")
def clear_cache(ticker: str, db: Session = Depends(get_db)):
    count = db.query(CacheEntry).filter(CacheEntry.ticker == ticker.upper()).delete()
    db.commit()
    return {"cleared_entries": count}

async def _ask_impl(req: QueryRequest, db: Session):
    q_hash = hashlib.sha256(f"{req.ticker}_{req.query.lower()}".encode()).hexdigest()
    cached = db.query(CacheEntry).filter(CacheEntry.query_hash == q_hash).first()
    if cached:
        return {"query_hash": q_hash, "answer": cached.llm_response, "sources": [], "cached": True, "provider": "Cache"}

    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    
    ctx = contextvars.copy_context()
    req_arrival_time = time.time() 
    
    await request_queue.put((fut, req, q_hash, ctx, req_arrival_time))

    try:
        return await asyncio.wait_for(fut, timeout=90)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Queue timeout")

@app.post("/ask")
async def ask(req: QueryRequest, db: Session = Depends(get_db)):
    if not TESTING:
        with mlflow.start_run(run_name="rag_request", nested=True):
            # Creates the root trace in the waterfall
            with mlflow.start_span(name="ASK_Request"):
                return await _ask_impl(req, db)
    return await _ask_impl(req, db)