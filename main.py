import os
import hashlib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

import mlflow
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI, RateLimitError, APIConnectionError, InternalServerError
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

# Setup Groq Client (Primary)
primary_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# Optional Fallback Client (Gracefully disabled if you don't have a key)
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    fallback_client = OpenAI(api_key=openai_key)
else:
    print("⚠️ No OpenAI API Key found. Fallback routing will be disabled.")
    fallback_client = None

class QueryRequest(BaseModel):
    query: str
    ticker: str
    top_k: int = 5

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

@app.post("/ask")
@mlflow.trace
async def ask_financial_question(request: QueryRequest):
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
        query_vector = model.encode(request.query).tolist()
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

        # --- 3. GENERATION WITH GROQ ---
        system_prompt = f"You are a Wall Street analyst. Use ONLY the context:\n\n{combined_text}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.query}
        ]

        try:
            print("🟢 Using Groq...")
            completion = primary_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.2 
            )
            final_answer = completion.choices[0].message.content
            provider_used = "Groq"
        except (RateLimitError, APIConnectionError, InternalServerError) as e:
            if fallback_client:
                print("⚠️ Groq Failed. Switching to OpenAI...")
                completion = fallback_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.2 
                )
                final_answer = completion.choices[0].message.content
                provider_used = "OpenAI"
            else:
                raise e # Fail gracefully if Groq goes down and no fallback exists

        # --- 4. CACHE & RETURN ---
        new_cache = CacheEntry(query_hash=query_hash, user_query=request.query, llm_response=final_answer)
        db.add(new_cache)
        db.commit()

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