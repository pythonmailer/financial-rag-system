import os
import hashlib
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

import mlflow
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from database import SessionLocal, CacheEntry

load_dotenv()

app = FastAPI(title="Financial RAG API")

# Setup MLflow
mlflow.set_tracking_uri("http://localhost:5001") 
mlflow.set_experiment("Financial-RAG")
mlflow.openai.autolog()
FastAPIInstrumentor.instrument_app(app)

print("Loading AI Models (Embedder & Reranker)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
qdrant = QdrantClient(url="http://localhost:6333")

# Setup Groq Client (Primary - Ultra Fast)
primary_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# Setup Gemini Client (Fallback 1 - High Intelligence)
gemini_key = os.getenv("GEMINI_API_KEY")
gemini_client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=gemini_key
) if gemini_key else None

# Setup OpenRouter Client (Fallback 2 - The Aggregator)
openrouter_key = os.getenv("OPENROUTER_API_KEY")
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key
) if openrouter_key else None

class QueryRequest(BaseModel):
    query: str
    ticker: str
    top_k: int = 5

@mlflow.trace(span_type="EMBEDDING")
def embed_query(query: str):
    """Converts the user question into a dense vector."""
    return model.encode(query).tolist()

@mlflow.trace(span_type="RETRIEVER")
def retrieve_from_qdrant(query_vector, ticker, limit=50):
    return qdrant.query_points(
        collection_name="financial_documents",
        query=query_vector,
        limit=limit,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.upper()))]
        )
    )

@mlflow.trace(span_type="RERANKER")
def rerank_documents(query, retrieved_texts, top_k=5):
    sentence_combinations = [[query, text] for text in retrieved_texts]
    scores = reranker.predict(sentence_combinations)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores

def save_to_cache(q_hash, user_q, llm_resp):
    """Saves the generated answer to Postgres in a background thread."""
    db_session = SessionLocal() 
    try:
        new_cache = CacheEntry(query_hash=q_hash, user_query=user_q, llm_response=llm_resp)
        db_session.add(new_cache)
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        print(f"Failed to save cache: {e}")
    finally:
        db_session.close()

@mlflow.trace(span_type="ROUTER")
def route_query(query: str) -> str:
    """
    Acts as a Traffic Cop. Uses the extremely fast 8B model to classify the query.
    Returns 'COMPLEX' for heavy reasoning, or 'SIMPLE' for basic extraction.
    """
    system_prompt = """You are a classification router for a financial system.
    Evaluate the user's question. 
    If it requires basic extraction (finding a specific number, name, or single fact), output exactly 'SIMPLE'.
    If it requires cross-referencing topics, deep synthesis, comparing risks, or advanced reasoning, output exactly 'COMPLEX'.
    Do not explain your reasoning. Output NOTHING ELSE besides SIMPLE or COMPLEX."""

    try:
        response = primary_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            max_tokens=5
        )
        classification = response.choices[0].message.content.strip().upper()
        return "COMPLEX" if "COMPLEX" in classification else "SIMPLE"
    except Exception as e:
        print(f"⚠️ Router failed, defaulting to SIMPLE: {e}")
        return "SIMPLE"

@app.post("/ask")
@mlflow.trace
async def ask_financial_question(request: QueryRequest, background_tasks: BackgroundTasks):
    db = SessionLocal()
    try:
        # --- 1. THE CACHE CHECK ---
        query_hash = hashlib.sha256(f"{request.ticker.upper()}_{request.query.strip().lower()}".encode()).hexdigest()
        cached_result = db.query(CacheEntry).filter(CacheEntry.query_hash == query_hash).first()
        
        if cached_result:
            return {
                "query": request.query,
                "answer": cached_result.llm_response,
                "sources": [{"score": 1.0, "text": "Retrieved from Postgres Semantic Cache"}],
                "cached": True,
                "provider": "Cache"
            }

        # --- 2. THE RAG PIPELINE ---
        query_vector = embed_query(request.query)
        search_response = retrieve_from_qdrant(query_vector, request.ticker)
        retrieved_texts = [hit.payload["text"] for hit in search_response.points]
        top_indices, rerank_scores = rerank_documents(request.query, retrieved_texts, request.top_k)

        retrieved_chunks = []
        combined_text = ""
        for idx in top_indices:
            text = retrieved_texts[idx]
            score = float(rerank_scores[idx])
            retrieved_chunks.append({"score": score, "text": text})
            combined_text += f"- {text}\n\n"

        # --- 3. DYNAMIC GENERATION WITH WATERFALL FALLBACK ---
        system_prompt = f"You are a Wall Street analyst. Use ONLY the context:\n\n{combined_text}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.query}
        ]

        # 3a. The Traffic Cop decides which model to use
        route_decision = route_query(request.query)
        
        if route_decision == "COMPLEX":
            print("🧠 Complex logic required. Routing to Heavy Model (70B)...")
            target_model = "llama-3.3-70b-versatile"
        else:
            print("⚡ Simple extraction. Routing to Fast Model (8B)...")
            target_model = "llama-3.1-8b-instant"

        # 3b. Waterfall Generation
        try:
            # Attempt 1: Groq (using the dynamically routed model)
            completion = primary_client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=0.2 
            )
            final_answer = completion.choices[0].message.content
            provider_used = f"Groq ({target_model})"

        except Exception as e_groq:
            print(f"⚠️ Groq failed: {e_groq}. Routing to Gemini...")
            
            if gemini_client:
                try:
                    # Attempt 2: Gemini (1.5 Flash)
                    completion = gemini_client.chat.completions.create(
                        model="gemini-1.5-flash", 
                        messages=messages,
                        temperature=0.2 
                    )
                    final_answer = completion.choices[0].message.content
                    provider_used = "Gemini"
                    
                except Exception as e_gemini:
                    print(f"⚠️ Gemini failed: {e_gemini}. Routing to OpenRouter...")
                    
                    if openrouter_client:
                        try:
                            # Attempt 3: OpenRouter (Free Llama 3 API)
                            completion = openrouter_client.chat.completions.create(
                                model="meta-llama/llama-3-8b-instruct:free",
                                messages=messages,
                                temperature=0.2 
                            )
                            final_answer = completion.choices[0].message.content
                            provider_used = "OpenRouter"
                            
                        except Exception as e_openrouter:
                            # Attempt 4: Ultimate Failure (Graceful Degradation)
                            print(f"🚨 All Cloud LLMs failed! Error: {e_openrouter}")
                            final_answer = "⚠️ **Service Notice:** Our AI generation engines are currently experiencing high traffic. However, we successfully retrieved the most relevant SEC documents for your query. Please review the sources below."
                            provider_used = "System Degraded (No AI)"
                            
                    else:
                        final_answer = "⚠️ **Service Notice:** All AI engines are down, and OpenRouter is not configured."
                        provider_used = "System Degraded"
            else:
                final_answer = "⚠️ **Service Notice:** Groq is down, and no fallback keys are configured."
                provider_used = "System Degraded"

        # --- 4. CACHE & RETURN ---
        background_tasks.add_task(save_to_cache, query_hash, request.query, final_answer)

        return {
            "query": request.query, 
            "answer": final_answer,
            "sources": retrieved_chunks,
            "cached": False,
            "provider": provider_used
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        db.close()