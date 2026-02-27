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
from database import SessionLocal, CacheEntry, FeedbackEntry
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

load_dotenv()

app = FastAPI(title="Financial RAG API - Sequential (Baseline)")

# ==========================================
# 1. INFRASTRUCTURE & MODELS
# ==========================================
mlflow.set_tracking_uri("http://mlflow:5001")
mlflow.set_experiment("Financial-RAG")
mlflow.openai.autolog()
FastAPIInstrumentor.instrument_app(app)

print("Loading AI Models (Embedder & Reranker)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
qdrant = QdrantClient(url="http://qdrant:6333")

class FeedbackRequest(BaseModel):
    query_hash: str
    rating: int

MODEL_PRICING = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "meta-llama/llama-3.1-8b-instruct": {"input": 0.05, "output": 0.05},
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.40, "output": 0.40},
}

def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = MODEL_PRICING.get(model_name, {"input": 0.0, "output": 0.0})
    cost = (prompt_tokens * rates["input"] / 1_000_000) + (completion_tokens * rates["output"] / 1_000_000)
    return round(cost, 6)

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

class QueryRequest(BaseModel):
    query: str
    ticker: str
    document_type: str = None
    top_k: int = 5


# ==========================================
# CORE RAG FUNCTIONS
# ==========================================
def embed_query(query: str):
    return model.encode(query).tolist()

def retrieve_from_qdrant(query_vector, ticker, document_type=None, limit=15):
    must_conditions = [
        models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.upper()))
    ]
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
# RESILIENCE LAYER (TENACITY RETRIES)
# ==========================================
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=6), 
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((asyncio.TimeoutError, Exception))
)
async def safe_llm_call(client, model_name, messages, max_tokens=None):
    kwargs = {"model": model_name, "messages": messages, "temperature": 0.2}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return await asyncio.wait_for(client.chat.completions.create(**kwargs), timeout=8)

# ==========================================
# ROUTER & GENERATION
# ==========================================
async def route_query(query: str) -> tuple[str, str]:
    system_prompt = "You are a financial router. Output exactly SIMPLE or COMPLEX."
    async def try_route(client, model_name):
        return await safe_llm_call(
            client, model_name, [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}], max_tokens=5
        )

    if groq_breaker.is_healthy:
        try:
            resp = await try_route(primary_client, "llama-3.1-8b-instant")
            classification = resp.choices[0].message.content.strip().upper()
            return ("COMPLEX" if "COMPLEX" in classification else "SIMPLE", "Groq")
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

    def extract_usage(completion):
        if hasattr(completion, "usage") and completion.usage:
            return completion.usage.prompt_tokens, completion.usage.completion_tokens, completion.usage.total_tokens
        return 0, 0, 0

    if groq_breaker.is_healthy and router_provider == "Groq":
        try:
            completion = await safe_llm_call(primary_client, groq_model, messages)
            p_tok, c_tok, t_tok = extract_usage(completion)
            return completion.choices[0].message.content, "Groq", groq_model, p_tok, c_tok, t_tok
        except Exception as e:
            if "401" in str(e) or "authentication" in str(e).lower():
                groq_breaker.trip(60)

    if gemini_client:
        try:
            completion = await safe_llm_call(gemini_client, gemini_model, messages)
            p_tok, c_tok, t_tok = extract_usage(completion)
            return completion.choices[0].message.content, "Gemini", gemini_model, p_tok, c_tok, t_tok
        except Exception:
            pass

    if openrouter_client:
        try:
            completion = await safe_llm_call(openrouter_client, openrouter_model, messages)
            p_tok, c_tok, t_tok = extract_usage(completion)
            return completion.choices[0].message.content, "OpenRouter", openrouter_model, p_tok, c_tok, t_tok
        except Exception:
            pass

    return "⚠️ Service Notice: All providers unavailable.", "System Degraded", "None", 0, 0, 0


# ==========================================
# API ENDPOINTS
# ==========================================
@app.delete("/cache/clear/{ticker}")
async def clear_semantic_cache(ticker: str):
    db = SessionLocal()
    try:
        deleted_count = db.query(CacheEntry).filter(CacheEntry.ticker == ticker.upper()).delete()
        db.commit()
        return {"status": "success", "cleared_entries": deleted_count, "ticker": ticker.upper()}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    db = SessionLocal()
    try:
        new_feedback = FeedbackEntry(query_hash=request.query_hash, rating=request.rating)
        db.add(new_feedback)
        db.commit()
        return {"status": "success", "recorded_rating": request.rating}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post("/ask")
async def ask_financial_question(request: QueryRequest, background_tasks: BackgroundTasks):
    db = SessionLocal()
    try:
        # 1. Deterministic Hashing
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
                "query_hash": query_hash,
                "query": request.query,
                "answer": cached_result.llm_response,
                "sources": [{"score": 1.0, "text": "Semantic Cache", "document_type": request.document_type or "All"}],
                "cached": True,
                "provider": "Cache",
            }

        # SEQUENTIAL EXECUTION START
        with mlflow.start_span(name=f"Sequential_Request_{request.ticker}", span_type="USER_QUERY") as user_span:
            user_span.set_inputs({"query": request.query, "ticker": request.ticker, "document_type": request.document_type})
            user_span.set_attribute("cache_hit", False)

            # 2. Sequential Retrieval
            with mlflow.start_span(name="Retrieval_and_Rerank", span_type="RAG_PIPELINE") as rag_span:
                query_vector = embed_query(request.query)
                search_response = retrieve_from_qdrant(query_vector, request.ticker, request.document_type)

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
                    texts_only = [d["text"] for d in retrieved_data]
                    top_indices, rerank_scores = rerank_documents(request.query, texts_only, request.top_k)
                    
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

            # 3. Sequential Routing
            with mlflow.start_span(name="Semantic_Router", span_type="CLASSIFICATION") as route_span:
                complexity, router_provider = await route_query(request.query)
                route_span.set_outputs({"decision": complexity, "routed_via": router_provider})

            # 4. Sequential Generation
            with mlflow.start_span(name="LLM_Generation", span_type="INFERENCE") as gen_span:
                final_answer, provider, model_used, p_tok, c_tok, t_tok = await generate_with_fallback(
                    f"You are a Wall Street analyst. Use ONLY the context provided to answer. If quoting numbers, mention the Source document.\n\nContext:\n{combined_text}",
                    request.query, complexity, router_provider
                )
                
                query_cost = calculate_cost(model_used, p_tok, c_tok)

                gen_span.set_outputs({"answer": final_answer})
                gen_span.set_attribute("provider", provider)
                gen_span.set_attribute("model", model_used)
                gen_span.set_attribute("prompt_tokens", p_tok)
                gen_span.set_attribute("completion_tokens", c_tok)
                gen_span.set_attribute("total_tokens", t_tok)
                gen_span.set_attribute("cost_usd", query_cost)
                user_span.set_attribute("total_cost_usd", query_cost)

            user_span.set_outputs({"final_answer": final_answer})

            # 5. Save and Return
            if provider != "System Degraded":
                background_tasks.add_task(save_to_cache, query_hash, request.query, final_answer, request.ticker)

            return {
                "query_hash": query_hash,
                "query": request.query,
                "answer": final_answer,
                "sources": retrieved_chunks,
                "cached": False,
                "provider": provider,
            }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()