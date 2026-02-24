import os
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

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_processor())

# ==========================================
# 2. INFRASTRUCTURE & MODELS (DOCKER UPDATED)
# ==========================================
# Use service names 'mlflow' and 'qdrant' instead of localhost
mlflow.set_tracking_uri("http://mlflow:5001") 
mlflow.set_experiment("Financial-RAG")
mlflow.openai.autolog()
FastAPIInstrumentor.instrument_app(app)

print("Loading AI Models (Embedder & Reranker)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
qdrant = QdrantClient(url="http://qdrant:6333") # Use service name 'qdrant'

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

class QueryRequest(BaseModel):
    query: str
    ticker: str
    top_k: int = 5

# ... [Keep embed_query_batch, retrieve_from_qdrant, rerank_documents, save_to_cache] ...

# ==========================================
# 4. GENERATION & BATCH ENGINE
# ==========================================
# [Your logic remains correct here, just ensure route_query and generate_with_fallback are async]

async def batch_processor():
    while True:
        batch = []
        fut, req, bg_tasks = await request_queue.get()
        batch.append((fut, req, bg_tasks))
        await asyncio.sleep(0.5) 
        
        while not request_queue.empty():
            batch.append(request_queue.get_nowait())

        print(f"\n📦 [BATCH WINDOW CLOSED] Processing {len(batch)} requests...")
        queries = [item[1].query for item in batch]

        # Wrap the shared batch embedding in its own trace span
        with mlflow.start_span(name="Global_Batch_Embedding", span_type="SHARED_COMPUTE") as batch_span:
            batch_vectors = await asyncio.to_thread(embed_query_batch, queries)
            mlflow.set_tag("batch_size", len(batch))

        async def process_independently(i, fut, req, bg_tasks):
            with mlflow.start_span(name=f"Request_{i+1}_{req.ticker}", span_type="USER_QUERY") as user_span:
                try:
                    user_span.set_inputs({"query": req.query, "ticker": req.ticker})
                    mlflow.set_tag("ticker", req.ticker)
                    
                    with mlflow.start_span(name="Retrieval_and_Rerank", span_type="RAG_PIPELINE") as rag_span:
                        search_response = await asyncio.to_thread(retrieve_from_qdrant, batch_vectors[i], req.ticker)
                        retrieved_texts = [hit.payload["text"] for hit in search_response.points]
                        top_indices, rerank_scores = await asyncio.to_thread(rerank_documents, req.query, retrieved_texts, req.top_k)

                        combined_text = "".join([f"- {retrieved_texts[idx]}\n\n" for idx in top_indices])
                        retrieved_chunks = [{"score": float(rerank_scores[idx]), "text": retrieved_texts[idx]} for idx in top_indices]
                        rag_span.set_outputs({"num_chunks": len(retrieved_chunks)})

                    with mlflow.start_span(name="Semantic_Router", span_type="CLASSIFICATION") as route_span:
                        route_decision = await route_query(req.query)
                        target_model = "llama-3.3-70b-versatile" if route_decision == "COMPLEX" else "llama-3.1-8b-instant"
                        mlflow.set_tag("route_decision", route_decision)
                        route_span.set_outputs({"decision": route_decision})

                    with mlflow.start_span(name="LLM_Generation", span_type="INFERENCE") as gen_span:
                        final_answer, provider = await generate_with_fallback(
                            f"You are a Wall Street analyst. Use ONLY the context:\n\n{combined_text}",
                            req.query,
                            target_model
                        )
                        gen_span.set_outputs({"answer": final_answer, "provider": provider})

                    user_span.set_outputs({"final_answer": final_answer})

                    # Finalize & Return
                    q_hash = hashlib.sha256(f"{req.ticker.upper()}_{req.query.strip().lower()}".encode()).hexdigest()
                    bg_tasks.add_task(save_to_cache, q_hash, req.query, final_answer)
                    
                    fut.set_result({
                        "query": req.query, "answer": final_answer,
                        "sources": retrieved_chunks, "cached": False, "provider": provider
                    })
                except Exception as e:
                    user_span.set_outputs({"error": str(e)})
                    fut.set_exception(e)

        tasks = [process_independently(i, fut, req, bg_tasks) for i, (fut, req, bg_tasks) in enumerate(batch)]
        await asyncio.gather(*tasks)

# ==========================================
# 5. THE API ENDPOINT
# ==========================================
@app.post("/ask")
async def ask_financial_question(request: QueryRequest, background_tasks: BackgroundTasks):
    db = SessionLocal()
    try:
        query_hash = hashlib.sha256(f"{request.ticker.upper()}_{request.query.strip().lower()}".encode()).hexdigest()
        cached_result = db.query(CacheEntry).filter(CacheEntry.query_hash == query_hash).first()
        
        if cached_result:
            return {
                "query": request.query, "answer": cached_result.llm_response,
                "sources": [{"score": 1.0, "text": "Retrieved from Postgres Semantic Cache"}],
                "cached": True, "provider": "Cache"
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