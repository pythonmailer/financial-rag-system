import os
import time
import json
import tempfile
import hashlib
import numpy as np
import asyncio
import contextvars
import socket
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Depends
from mlflow.entities import SpanType
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.orm import Session

# Load local .env if it exists
load_dotenv()

# ==========================================
# 🤖 SMART AUTO-DETECTION
# ==========================================
def resolve_host(service_name, default="localhost"):
    try:
        socket.gethostbyname(service_name)
        return service_name
    except socket.gaierror:
        return default

def get_service_url(service_name, port, protocol="http"):
    host = resolve_host(service_name, "localhost")
    return f"{protocol}://{host}:{port}"

DB_HOST = resolve_host("postgres", "localhost")
QDRANT_URL = os.getenv("QDRANT_URL", get_service_url("qdrant", 6333))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", get_service_url("mlflow", 5001))

from database import SessionLocal, CacheEntry, FeedbackEntry

# ==========================================
# CONFIG
# ==========================================
TESTING = os.getenv("TESTING", "False") == "True"
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
COLLECTION_NAME = "financial_documents"

request_queue = None
MAX_BATCH_SIZE = 32
MAX_CONCURRENT_LLM_CALLS = 25
llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

class EmbedRequest(BaseModel):
    texts: list[str]

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
    SentenceTransformer = CrossEncoder = models = AsyncOpenAI = None
    retry = lambda *a, **k: (lambda f: f)
    wait_exponential = stop_after_attempt = None
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
    device = "cpu"
    if USE_GPU:
        import torch
        if torch.backends.mps.is_available(): device = "mps"
        elif torch.cuda.is_available(): device = "cuda"
    return SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

@lru_cache()
def get_reranker():
    if TESTING: return None
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
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
            with open(self.file_path, "w") as f: json.dump(state, f)
        except: pass

    @property
    def is_healthy(self):
        if not os.path.exists(self.file_path): return True
        try:
            with open(self.file_path) as f: state = json.load(f)
            return state.get("healthy", True) or time.time() > state.get("disabled_until", 0)
        except: return True

    def trip(self, cooldown=60):
        self._write_state({"healthy": False, "disabled_until": time.time() + cooldown})

    def set_healthy(self, healthy=True):
        self._write_state({"healthy": healthy, "disabled_until": 0})

groq_breaker = CircuitBreaker("groq")

# ==========================================
# LLM CLIENTS
# ==========================================
if not TESTING:
    primary_client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
else:
    primary_client = None

# ==========================================
# APP INIT
# ==========================================
app = FastAPI(title="Financial RAG API")
if not TESTING: FastAPIInstrumentor.instrument_app(app)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

class QueryRequest(BaseModel):
    query: str; ticker: str; document_type: str | None = None; top_k: int = 5

# ==========================================
# CORE LOGIC
# ==========================================
def route_query(query: str) -> str:
    complex_keywords = ["compare", "analyze", "why", "impact", "trends", "growth", "risk"]
    return "COMPLEX" if len(query.split()) > 20 or any(kw in query.lower() for kw in complex_keywords) else "SIMPLE"

def retrieve_from_qdrant(query_vector, ticker, document_type=None, limit=15):
    must = [models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.upper()))]
    if document_type: must.append(models.FieldCondition(key="document_type", match=models.MatchValue(value=document_type.upper())))
    return get_qdrant().query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=limit, query_filter=models.Filter(must=must))

def rerank_documents(query, texts, top_k):
    scores = get_reranker().predict([[query, t] for t in texts])
    idx = np.argsort(scores)[::-1][:top_k]
    return idx, scores

def embed_query_batch(queries):
    return get_embedder().encode(queries).tolist()

async def generate_answer(system_prompt, user_query, complexity="SIMPLE"):
    # Span Type: LLM
    with start_span(name="LLM_Generation", span_type=SpanType.LLM) as span:
        if TESTING: return "Mock answer.", "Mock"
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]
        groq_model = "llama-3.3-70b-versatile" if complexity == "COMPLEX" else "llama-3.1-8b-instant"

        if groq_breaker.is_healthy:
            try:
                res = await safe_llm_call(primary_client, groq_model, msgs)
                ans = res.choices[0].message.content
                if span: 
                    span.set_attribute("llm_model", groq_model)
                    span.set_outputs({"llm_answer": ans})
                return ans, f"Groq ({groq_model})"
            except: groq_breaker.trip()
        return "⚠️ LLM Service Degraded.", "System Offline"

if not TESTING:
    @retry(wait=wait_exponential(min=2, max=6), stop=stop_after_attempt(3))
    async def safe_llm_call(client, model_name, messages):
        return await asyncio.wait_for(client.chat.completions.create(model=model_name, messages=messages, temperature=0.2), timeout=12)

# ==========================================
# PROCESS (CHRONOLOGICAL STEP-BY-STEP TRACING)
# ==========================================
async def process_independently(i, fut, req, q_hash, batch_vectors, req_arrival_time):
    # Root Parent Span for the entire RAG lifecycle
    with start_span(name=f"RAG-Workflow-{req.ticker}") as root_span:
        async with llm_semaphore:
            try:
                if root_span: 
                    root_span.set_inputs({"user_query": req.query, "ticker": req.ticker})

                # --- STEP 1: ROUTER ---
                with start_span(name="1_Query_Routing", span_type=SpanType.TOOL) as span:
                    complexity = route_query(req.query)
                    span.set_attribute("complexity", complexity)
                    span.set_outputs({"routing_result": complexity})

                # --- STEP 2: RETRIEVAL (RETRIEVER type) ---
                with start_span(name="2_Vector_Retrieval", span_type=SpanType.RETRIEVER) as span:
                    t_start = time.time()
                    search = await asyncio.to_thread(retrieve_from_qdrant, batch_vectors[i], req.ticker, req.document_type)
                    q_latency = (time.time() - t_start) * 1000
                    
                    texts = [hit.payload.get("text", "") for hit in search.points if hit.payload]
                    span.set_attribute("qdrant_latency_ms", q_latency)
                    span.set_attribute("chunks_found", len(texts))
                    span.set_outputs({"retrieved_count": len(texts)})

                if not texts:
                    combined, sources = "No context found.", []
                    rerank_ms = 0
                else:
                    # --- STEP 3: RERANKING (TOOL type) ---
                    with start_span(name="3_Reranking", span_type=SpanType.TOOL) as span:
                        t_rerank = time.time()
                        idx, scores = await asyncio.to_thread(rerank_documents, req.query, texts, req.top_k)
                        rerank_ms = (time.time() - t_rerank) * 1000
                        
                        combined = "\n\n".join([texts[j] for j in idx])
                        sources = [{"score": float(scores[j]), "text": texts[j]} for j in idx]
                        
                        span.set_attribute("rerank_latency_ms", rerank_ms)
                        span.set_outputs({"reranked_top_k": len(idx)})

                # --- STEP 4: LLM GENERATION ---
                answer, provider = await generate_answer(f"Analyst context:\n{combined}", req.query, complexity)

                # Set Final Output on Root Span
                if root_span:
                    root_span.set_outputs({
                        "final_answer": answer,
                        "provider": provider
                    })

                # Log performance metrics for charts
                if not TESTING:
                    mlflow.log_metric("qdrant_ms", q_latency)
                    mlflow.log_metric("rerank_ms", rerank_ms)
                    mlflow.log_metric("total_e2e_ms", (time.time() - req_arrival_time) * 1000)

                fut.set_result({"query_hash": q_hash, "query": req.query, "answer": answer, "sources": sources, "cached": False, "provider": provider})
            
            except Exception as e:
                if root_span:
                    root_span.set_attribute("error", True)
                    root_span.set_attribute("error_msg", str(e))
                fut.set_exception(e)

# ==========================================
# BATCH ENGINE
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
        with start_span(name="Batch_Embedding", span_type=SpanType.TOOL) as span:
            if span: span.set_attribute("batch_size", len(queries))
            vectors = await asyncio.to_thread(embed_query_batch, queries)

        for i, (fut, req, q_hash, ctx, req_arrival_time) in enumerate(batch):
            ctx.run(asyncio.create_task, process_independently(i, fut, req, q_hash, vectors, req_arrival_time))

# ==========================================
# LIFESPAN & ENDPOINTS
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global request_queue
    request_queue = asyncio.Queue()
    if not TESTING:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment("Financial-RAG")
            mlflow.openai.autolog(log_traces=True)
            print(f"✅ MLflow connected: {MLFLOW_URI}")
        except: print("⚠️ MLflow Offline")
    task = asyncio.create_task(batch_processor())
    yield
    task.cancel()

app.router.lifespan_context = lifespan

@app.post("/ask")
async def ask(req: QueryRequest, db: Session = Depends(get_db)):
    q_hash = hashlib.sha256(f"{req.ticker}_{req.query.lower()}".encode()).hexdigest()
    cached = db.query(CacheEntry).filter(CacheEntry.query_hash == q_hash).first()
    if cached: return {"query_hash": q_hash, "answer": cached.llm_response, "sources": [], "cached": True, "provider": "Cache"}

    loop, fut, ctx = asyncio.get_running_loop(), asyncio.get_running_loop().create_future(), contextvars.copy_context()
    await request_queue.put((fut, req, q_hash, ctx, time.time()))

    if not TESTING:
        # Start a Run in MLflow to group the following traces
        with mlflow.start_run(run_name=f"Request-{req.ticker}", nested=True):
            return await asyncio.wait_for(fut, timeout=90)
    return await asyncio.wait_for(fut, timeout=90)

@app.get("/ready")
def ready():
    try:
        get_qdrant().get_collections()
        return {"status": "ready"}
    except: return {"status": "not_ready"}

@app.post("/embed")
async def embed(req: EmbedRequest):
    return {"embeddings": await asyncio.to_thread(embed_query_batch, req.texts)}