import os
import time
import json
import tempfile
import hashlib
import numpy as np
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

import mlflow
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import AsyncOpenAI
from database import SessionLocal, CacheEntry

load_dotenv()

app = FastAPI(title="Financial RAG API - Stream-Batched")

# ==========================================
# 1. DYNAMIC BATCHING SETUP
# ==========================================
request_queue = asyncio.Queue()
MAX_BATCH_SIZE = 32

# ==========================================
# 2. INFRASTRUCTURE & MODELS
# ==========================================
mlflow.set_tracking_uri("http://mlflow:5001")
mlflow.set_experiment("Financial-RAG")
mlflow.openai.autolog()
FastAPIInstrumentor.instrument_app(app)

print("Loading AI Models (Embedder & Reranker)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
qdrant = QdrantClient(url="http://qdrant:6333")

# ==========================================
# CIRCUIT BREAKER STATE
# ==========================================
class CircuitBreaker:
    def __init__(self, service_name="groq"):
        self.file_path = os.path.join(tempfile.gettempdir(), f"{service_name}_cb_state.json")

    def _write_state(self, state: dict):
        tmp_path = self.file_path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(state, f)
            os.replace(tmp_path, self.file_path)
        except Exception:
            pass

    @property
    def is_healthy(self):
        if not os.path.exists(self.file_path):
            return True
        try:
            with open(self.file_path, "r") as f:
                state = json.load(f)
            if not state.get("healthy", True):
                if time.time() > state.get("disabled_until", 0):
                    self.set_healthy(True)
                    return True
                return False
            return True
        except Exception:
            return True

    def trip(self, cooldown_seconds=60):
        state = {"healthy": False, "disabled_until": time.time() + cooldown_seconds}
        self._write_state(state)

    def set_healthy(self, healthy=True):
        state = {"healthy": healthy, "disabled_until": 0}
        self._write_state(state)

groq_breaker = CircuitBreaker("groq")

# ==========================================
# API CLIENTS
# ==========================================
primary_client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

gemini_key = os.getenv("GEMINI_API_KEY")
gemini_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=gemini_key
) if gemini_key else None

openrouter_key = os.getenv("OPENROUTER_API_KEY")
openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key
) if openrouter_key else None

# Updated to match ingest.py
class QueryRequest(BaseModel):
    query: str
    ticker: str
    document_type: str = None  # matches payload={"ticker": ticker, "document_type": f_type, "text": chunk}
    top_k: int = 5


# ==========================================
# CORE RAG FUNCTIONS
# ==========================================
@mlflow.trace(span_type="BATCH_EMBEDDING")
def embed_query_batch(queries: list[str]):
    return model.encode(queries).tolist()

def retrieve_from_qdrant(query_vector, ticker, document_type=None, limit=15):
    must_conditions = [
        models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.upper()))
    ]
    
    # Filter by SEC filing type if requested
    if document_type:
        must_conditions.append(
            models.FieldCondition(key="document_type", match=models.MatchValue(value=document_type.upper()))
        )

    return qdrant.query_points(
        collection_name="financial_documents",
        query=query_vector,
        limit=limit,
        query_filter=models.Filter(must=must_conditions)
    )

def rerank_documents(query, retrieved_texts, top_k=5):
    if top_k <= 0 or not retrieved_texts:
        return [], np.array([])
        
    top_k = min(top_k, len(retrieved_texts))
    sentence_combinations = [[query, text] for text in retrieved_texts]
    scores = reranker.predict(sentence_combinations)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores

def save_to_cache(q_hash, user_query, llm_response, ticker):
    db_session = SessionLocal()
    try:
        new_cache = CacheEntry(
            query_hash=q_hash, 
            user_query=user_query, 
            llm_response=llm_response,
            ticker=ticker.upper()
        )
        db_session.add(new_cache)
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        print(f"Failed to save cache: {e}")
    finally:
        db_session.close()


# ==========================================
# ROUTER & GENERATION
# ==========================================
@mlflow.trace(span_type="ROUTER")
async def route_query(query: str) -> tuple[str, str]:
    system_prompt = "You are a financial router. Output exactly SIMPLE or COMPLEX."

    async def try_route(client, model_name):
        return await asyncio.wait_for(
            client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                temperature=0.0, max_tokens=5,
            ),
            timeout=4,
        )

    if groq_breaker.is_healthy:
        try:
            resp = await try_route(primary_client, "llama-3.1-8b-instant")
            classification = resp.choices[0].message.content.strip().upper()
            return ("COMPLEX" if "COMPLEX" in classification else "SIMPLE", "Groq")
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            if "401" in str(e) or "authentication" in str(e).lower():
                groq_breaker.trip(60)

    if gemini_client:
        try:
            resp = await try_route(gemini_client, "gemini-2.5-flash")
            classification = resp.choices[0].message.content.strip().upper()
            return ("COMPLEX" if "COMPLEX" in classification else "SIMPLE", "Gemini")
        except Exception:
            pass

    return ("SIMPLE", "System Degraded")


async def generate_with_fallback(system_prompt, user_query, complexity: str, router_provider: str):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]

    groq_model = "llama-3.3-70b-versatile" if complexity == "COMPLEX" else "llama-3.1-8b-instant"
    gemini_model = "gemini-2.5-pro" if complexity == "COMPLEX" else "gemini-2.5-flash"
    openrouter_model = "meta-llama/llama-3.1-70b-instruct" if complexity == "COMPLEX" else "meta-llama/llama-3.1-8b-instruct"

    if groq_breaker.is_healthy and router_provider == "Groq":
        try:
            completion = await asyncio.wait_for(
                primary_client.chat.completions.create(model=groq_model, messages=messages, temperature=0.2), timeout=8
            )
            return completion.choices[0].message.content, f"Groq ({groq_model})"
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            if "401" in str(e) or "authentication" in str(e).lower():
                groq_breaker.trip(60)

    if gemini_client:
        try:
            completion = await asyncio.wait_for(
                gemini_client.chat.completions.create(model=gemini_model, messages=messages, temperature=0.2), timeout=8
            )
            return completion.choices[0].message.content, f"Gemini ({gemini_model})"
        except Exception:
            pass

    if openrouter_client:
        try:
            completion = await asyncio.wait_for(
                openrouter_client.chat.completions.create(model=openrouter_model, messages=messages, temperature=0.2), timeout=8
            )
            return completion.choices[0].message.content, f"OpenRouter ({openrouter_model})"
        except Exception:
            pass

    return "⚠️ Service Notice: All providers unavailable.", "System Degraded"


# ==========================================
# BATCH ENGINE
# ==========================================
async def batch_processor():
    while True:
        batch = []
        fut, req, bg_tasks = await request_queue.get()
        batch.append((fut, req, bg_tasks))

        sleep_time = 0.05 if request_queue.qsize() > 0 else 0.5
        await asyncio.sleep(sleep_time)

        while not request_queue.empty() and len(batch) < MAX_BATCH_SIZE:
            batch.append(request_queue.get_nowait())

        queries = [item[1].query for item in batch]
        loop = asyncio.get_running_loop()

        with mlflow.start_span(name="Global_Batch_Embedding", span_type="SHARED_COMPUTE") as batch_span:
            batch_start = loop.time()
            batch_vectors = await asyncio.to_thread(embed_query_batch, queries)
            batch_latency = loop.time() - batch_start
            batch_span.set_inputs({"num_queries": len(batch)})
            batch_span.set_attribute("latency_ms", round(batch_latency * 1000, 2))

        async def process_independently(i, fut, req, bg_tasks):
            with mlflow.start_span(name=f"Request_{i+1}_{req.ticker}", span_type="USER_QUERY") as user_span:
                try:
                    user_span.set_inputs({"query": req.query, "ticker": req.ticker, "document_type": req.document_type})
                    user_span.set_attribute("batch_size", len(batch))
                    user_span.set_attribute("cache_hit", False)

                    # Retrieval
                    with mlflow.start_span(name="Retrieval_and_Rerank", span_type="RAG_PIPELINE") as rag_span:
                        search_response = await asyncio.to_thread(
                            retrieve_from_qdrant, batch_vectors[i], req.ticker, req.document_type
                        )

                        # Extract text AND document_type matching ingest.py
                        retrieved_data = [
                            {
                                "text": hit.payload.get("text", ""),
                                "document_type": hit.payload.get("document_type", "Unknown Source")
                            }
                            for hit in search_response.points if hit.payload and hit.payload.get("text")
                        ]

                        if not retrieved_data:
                            combined_text = "No relevant financial data found in context.\nIf context is empty, say you do not have enough data."
                            retrieved_chunks = []
                        else:
                            # Only rerank the text strings
                            texts_only = [d["text"] for d in retrieved_data]
                            top_indices, rerank_scores = await asyncio.to_thread(
                                rerank_documents, req.query, texts_only, req.top_k
                            )
                            
                            # Inject document_type into the prompt so LLM can cite its sources
                            combined_text = "".join(
                                [f"- [Source: {retrieved_data[idx]['document_type']}] {retrieved_data[idx]['text']}\n\n" 
                                 for idx in top_indices]
                            )
                            
                            retrieved_chunks = [
                                {
                                    "score": float(rerank_scores[idx]), 
                                    "text": retrieved_data[idx]["text"],
                                    "document_type": retrieved_data[idx]["document_type"]
                                }
                                for idx in top_indices
                            ]

                        rag_span.set_outputs({"num_chunks": len(retrieved_chunks)})

                    # Routing
                    with mlflow.start_span(name="Semantic_Router", span_type="CLASSIFICATION") as route_span:
                        complexity, router_provider = await route_query(req.query)
                        route_span.set_outputs({"decision": complexity, "routed_via": router_provider})

                    # Generation
                    with mlflow.start_span(name="LLM_Generation", span_type="INFERENCE") as gen_span:
                        final_answer, provider = await generate_with_fallback(
                            f"You are a Wall Street analyst. Use ONLY the context provided to answer. If quoting numbers, mention the Source document.\n\nContext:\n{combined_text}",
                            req.query, complexity, router_provider
                        )
                        gen_span.set_outputs({"answer": final_answer})
                        gen_span.set_attribute("provider", provider)

                    user_span.set_outputs({"final_answer": final_answer})

                    # Calculate deterministic Cache Hash (Including document_type)
                    hash_input = f"{req.ticker.upper()}_{req.query.strip().lower()}"
                    if req.document_type:
                        hash_input += f"_{req.document_type.upper()}"
                    q_hash = hashlib.sha256(hash_input.encode()).hexdigest()

                    if provider != "System Degraded":
                        bg_tasks.add_task(save_to_cache, q_hash, req.query, final_answer, req.ticker)

                    fut.set_result(
                        {"query": req.query, "answer": final_answer, "sources": retrieved_chunks, "cached": False, "provider": provider}
                    )

                except Exception as e:
                    user_span.set_outputs({"error": str(e)})
                    fut.set_exception(e)

        tasks = [process_independently(i, fut, req, bg_tasks) for i, (fut, req, bg_tasks) in enumerate(batch)]
        await asyncio.gather(*tasks)


# ==========================================
# API ENDPOINTS
# ==========================================
@app.on_event("startup")
async def startup_event():
    if not os.getenv("GEMINI_API_KEY"):
        print("\n🚨 WARNING: GEMINI_API_KEY is missing from environment!\n")
    asyncio.create_task(batch_processor())

@app.delete("/cache/clear")
async def clear_semantic_cache():
    """Wipes the PostgreSQL cache when new documents are ingested."""
    db = SessionLocal()
    try:
        # Deletes all rows in the CacheEntry table
        deleted_count = db.query(CacheEntry).delete()
        db.commit()
        return {"status": "success", "cleared_entries": deleted_count}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.post("/ask")
async def ask_financial_question(request: QueryRequest, background_tasks: BackgroundTasks):
    db = SessionLocal()
    try:
        # Include document_type in the hash so a search for 10-K vs 10-Q are cached separately
        hash_input = f"{request.ticker.upper()}_{request.query.strip().lower()}"
        if request.document_type:
            hash_input += f"_{request.document_type.upper()}"
            
        query_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        cached_result = db.query(CacheEntry).filter(CacheEntry.query_hash == query_hash).first()

        if cached_result:
            with mlflow.start_span(name="Cache_Hit", span_type="CACHE") as cache_span:
                cache_span.set_inputs({"query": request.query, "ticker": request.ticker, "document_type": request.document_type})
                cache_span.set_outputs({"provider": "Cache"})
                cache_span.set_attribute("cache_hit", True)

            return {
                "query": request.query,
                "answer": cached_result.llm_response,
                "sources": [{"score": 1.0, "text": "Semantic Cache", "document_type": request.document_type or "All"}],
                "cached": True,
                "provider": "Cache",
            }

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await request_queue.put((future, request, background_tasks))
        return await future

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()