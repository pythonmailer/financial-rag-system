# main2.py -- updated MLflow-safe version for async workers

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

# Load local .env if it exists
load_dotenv()

# ==========================================
# 🌐 HARDCODED INFRASTRUCTURE CONFIG
# ==========================================
AWS_IP = "13.232.197.229"
QDRANT_URL = os.getenv("QDRANT_URL", f"http://{AWS_IP}:6333")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"http://{AWS_IP}:5001")
DB_URL = os.getenv("DATABASE_URL", f"postgresql://admin:adminpassword@{AWS_IP}:5432/financial_rag")

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
    from mlflow.tracking import MlflowClient
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
# CORE LOGIC (same as you had)
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
# GLOBALS FOR MLflow (populated at lifespan)
# ==========================================
MLFLOW_EXPERIMENT_ID = None
MLFLOW_CLIENT = None

# ==========================================
# PROCESS PIPELINE (modified to accept run_id)
# ==========================================
async def process_independently(i, fut, req, q_hash, batch_vectors, req_arrival_time, run_id=None):
    """
    NOTE: run_id is the Mlflow run id created in the ask endpoint and passed into the queue.
    We use MlflowClient.log_metric/log_param with explicit run_id so logging works across async tasks.
    """
    client = None
    if not TESTING:
        client = MLFLOW_CLIENT  # MlflowClient instance created at lifespan
    with start_span(name=f"RAG-Workflow-{req.ticker}") as root_span:
        async with llm_semaphore:
            try:
                if root_span:
                    root_span.set_inputs({"user_query": req.query, "ticker": req.ticker})

                # --- STEP 1: ROUTER ---
                with start_span(name="1_Query_Routing", span_type=SpanType.TOOL) as span:
                    complexity = route_query(req.query)
                    span.set_outputs({"routing_result": complexity})

                # --- STEP 2: RETRIEVAL ---
                with start_span(name="2_Vector_Retrieval", span_type=SpanType.RETRIEVER) as span:
                    t_start = time.time()
                    search = await asyncio.to_thread(retrieve_from_qdrant, batch_vectors[i], req.ticker, req.document_type)
                    q_latency = (time.time() - t_start) * 1000
                    
                    texts = [hit.payload.get("text", "") for hit in search.points if hit.payload]
                    span.set_attribute("qdrant_latency_ms", q_latency)
                    span.set_outputs({"retrieved_count": len(texts)})

                if not texts:
                    combined, sources = "No context found.", []
                    rerank_ms = 0
                else:
                    # --- STEP 3: RERANKING ---
                    with start_span(name="3_Reranking", span_type=SpanType.TOOL) as span:
                        t_rerank = time.time()
                        idx, scores = await asyncio.to_thread(rerank_documents, req.query, texts, req.top_k)
                        rerank_ms = (time.time() - t_rerank) * 1000
                        
                        combined = "\n\n".join([texts[j] for j in idx])
                        sources = [{"score": float(scores[j]), "text": texts[j]} for j in idx]
                        span.set_outputs({"reranked_top_k": len(idx)})

                # --- STEP 4: LLM GENERATION ---
                answer, provider = await generate_answer(f"Analyst context:\n{combined}", req.query, complexity)

                if root_span:
                    root_span.set_outputs({"final_answer": answer, "provider": provider})

                # --- LOG METRICS TO MLflow USING MlflowClient (explicit run_id) ---
                if not TESTING and client and run_id:
                    try:
                        ts_ms = int(time.time() * 1000)
                        # log a few metrics
                        client.log_metric(run_id, "qdrant_ms", float(q_latency or 0.0), ts_ms, 0)
                        client.log_metric(run_id, "rerank_ms", float(rerank_ms or 0.0), ts_ms, 0)
                        client.log_metric(run_id, "total_e2e_ms", float((time.time() - req_arrival_time) * 1000), ts_ms, 0)
                        # add some params
                        client.log_param(run_id, "ticker", req.ticker)
                        client.log_param(run_id, "provider", provider)
                    except Exception as e:
                        # Keep worker robust if mlflow temporarily fails
                        print("⚠️ MLflow log error:", e)

                # --- PERSIST CACHE TO DB (if desired) ---
                try:
                    db = SessionLocal()
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
                    db.close()
                except Exception as e:
                    print("⚠️ Cache DB write failed:", e)

                fut.set_result({"query_hash": q_hash, "query": req.query, "answer": answer, "sources": sources, "cached": False, "provider": provider})
            
            except Exception as e:
                if root_span:
                    root_span.set_attribute("error", True)
                    root_span.set_attribute("error_msg", str(e))
                # If MLflow run exists, mark it failed
                if not TESTING and MLFLOW_CLIENT and run_id:
                    try:
                        MLFLOW_CLIENT.set_terminated(run_id, status="FAILED")
                    except Exception:
                        pass
                fut.set_exception(e)
            else:
                # mark run finished
                if not TESTING and MLFLOW_CLIENT and run_id:
                    try:
                        MLFLOW_CLIENT.set_terminated(run_id, status="FINISHED")
                    except Exception as e:
                        print("⚠️ MLflow finalize error:", e)

# ==========================================
# BATCH ENGINE (unchanged other than tuple carrying run_id)
# ==========================================
async def batch_processor():
    while True:
        batch = []
        # enqueue items are tuples: (fut, req, q_hash, ctx, req_arrival_time, run_id)
        fut, req, q_hash, ctx, req_arrival_time, run_id = await request_queue.get()
        batch.append((fut, req, q_hash, ctx, req_arrival_time, run_id))
        await asyncio.sleep(0.05)
        while not request_queue.empty() and len(batch) < MAX_BATCH_SIZE:
            # grab more if available
            item = request_queue.get_nowait()
            batch.append(item)

        queries = [item[1].query for item in batch]
        with start_span(name="Batch_Embedding", span_type=SpanType.TOOL) as span:
            vectors = await asyncio.to_thread(embed_query_batch, queries)

        for i, (fut, req, q_hash, ctx, req_arrival_time, run_id) in enumerate(batch):
            # schedule worker in the captured context
            ctx.run(asyncio.create_task, process_independently(i, fut, req, q_hash, vectors, req_arrival_time, run_id))

# ==========================================
# LIFESPAN & ENDPOINTS
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global request_queue, MLFLOW_CLIENT, MLFLOW_EXPERIMENT_ID
    request_queue = asyncio.Queue()
    if not TESTING:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            # ensure experiment exists and use its ID
            client = MlflowClient(tracking_uri=MLFLOW_URI)
            MLFLOW_CLIENT = client

            exp = client.get_experiment_by_name("Financial-RAG")
            if exp is None:
                MLFLOW_EXPERIMENT_ID = client.create_experiment("Financial-RAG")
            else:
                MLFLOW_EXPERIMENT_ID = exp.experiment_id

            # optional: autolog hooks
            try:
                mlflow.openai.autolog(log_traces=True)
            except Exception:
                # ignore if not available
                pass

            print(f"🚀 MLflow Connected to: {MLFLOW_URI} (experiment_id={MLFLOW_EXPERIMENT_ID})")

        except Exception as e:
            print(f"⚠️ MLflow Connection Failed: {e}")
            MLFLOW_CLIENT = None
            MLFLOW_EXPERIMENT_ID = None

    # start batch worker
    task = asyncio.create_task(batch_processor())
    yield
    task.cancel()

app.router.lifespan_context = lifespan

@app.post("/ask")
async def ask(req: QueryRequest, db: Session = Depends(get_db)):
    q_hash = hashlib.sha256(f"{req.ticker}_{req.query.lower()}".encode()).hexdigest()
    cached = (
        db.query(CacheEntry)
        .filter(CacheEntry.query_hash == q_hash, CacheEntry.ticker == req.ticker)
        .first()
    )
    if cached:
        return {"query_hash": q_hash, "answer": cached.llm_response, "sources": [], "cached": True, "provider": cached.provider or "Cache"}

    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    ctx = contextvars.copy_context()

    # Create MLflow run **explicitly** via MlflowClient so we can pass run_id across tasks
    run_id = None
    if not TESTING and MLFLOW_CLIENT and MLFLOW_EXPERIMENT_ID:
        try:
            run = MLFLOW_CLIENT.create_run(experiment_id=MLFLOW_EXPERIMENT_ID, run_name=f"Request-{req.ticker}")
            run_id = run.info.run_id
            # optional: initial tags/params
            MLFLOW_CLIENT.log_param(run_id, "ticker", req.ticker)
            MLFLOW_CLIENT.log_param(run_id, "query_hash", q_hash)
        except Exception as e:
            print("⚠️ MLflow create_run failed:", e)
            run_id = None

    # enqueue for processing; now includes run_id
    await request_queue.put((fut, req, q_hash, ctx, time.time(), run_id))

    # do not use mlflow.start_run here (it won't propagate to worker tasks). Return future result.
    return await asyncio.wait_for(fut, timeout=90)

@app.get("/ready")
def ready():
    try:
        # Check Qdrant Connection
        get_qdrant().get_collections()
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}

@app.post("/embed")
async def embed(req: EmbedRequest):
    return {"embeddings": await asyncio.to_thread(embed_query_batch, req.texts)}