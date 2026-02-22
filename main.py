import os
import hashlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from database import SessionLocal, CacheEntry

load_dotenv()

app = FastAPI(title="Financial RAG API")

mlflow.openai.autolog()

# Enable auto-tracing for FastAPI incoming requests
FastAPIInstrumentor.instrument_app(app)

# Set your tracking server (Local by default, or point to DagsHub)
mlflow.set_tracking_uri("http://localhost:5001") 
mlflow.set_experiment("Financial-RAG")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Loading Cross-Encoder Reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
qdrant = QdrantClient(url="http://localhost:6333")
llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

class QueryRequest(BaseModel):
    query: str
    ticker: str
    top_k: int = 5

@app.post("/ask")
@mlflow.trace
async def ask_financial_question(request: QueryRequest):
    db = SessionLocal() # Open a connection to Postgres
    try:
        # ==========================================
        # 1. THE CACHE CHECK
        # ==========================================
        # Create a unique mathematical fingerprint of the ticker + question
        query_hash = hashlib.sha256(f"{request.ticker.upper()}_{request.query.strip().lower()}".encode()).hexdigest()
        
        # Look for this exact fingerprint in Postgres
        cached_result = db.query(CacheEntry).filter(CacheEntry.query_hash == query_hash).first()
        
        if cached_result:
            print("🚀 CACHE HIT! Skipping AI generation.")
            return {
                "query": request.query,
                "answer": cached_result.llm_response,
                "sources": [{"score": 1.0, "text": "Retrieved from Postgres Semantic Cache"}],
                "cached": True
            }

        # ==========================================
        # 2. THE RAG PIPELINE (If Cache Miss)
        # ==========================================
        print("🐌 CACHE MISS! Running full AI pipeline.")
        query_vector = model.encode(request.query).tolist()

        search_response = qdrant.query_points(
            collection_name="financial_documents",
            query=query_vector,
            limit=50, # CHANGED: Grab a huge net of 50 possible paragraphs
            query_filter=models.Filter(
                must=[models.FieldCondition(key="ticker", match=models.MatchValue(value=request.ticker.upper()))]
            )
        )
        
        # Extract just the text from those 50 hits
        retrieved_texts = [hit.payload["text"] for hit in search_response.points]

        # ==========================================
        # Stage 2: The Sniper (Rerank to Top 5)
        # ==========================================
        # Pair the user's query with every single one of the 50 paragraphs
        sentence_combinations = [[request.query, paragraph] for paragraph in retrieved_texts]
        
        # The Cross-Encoder scores all 50 pairs simultaneously
        rerank_scores = reranker.predict(sentence_combinations)
        
        # Sort the scores from highest to lowest and grab the indices of the Top 5
        top_5_indices = np.argsort(rerank_scores)[::-1][:request.top_k]

        # ==========================================
        # Format the surviving Top 5 for the LLM
        # ==========================================
        retrieved_chunks = []
        combined_text = ""
        for idx in top_5_indices:
            best_text = retrieved_texts[idx]
            best_score = float(rerank_scores[idx]) # Convert numpy float to standard Python float
            
            retrieved_chunks.append({"score": best_score, "text": best_text})
            combined_text += f"- {best_text}\n\n"

        system_prompt = f"""You are an expert Wall Street financial analyst. 
        Answer the user's question using ONLY the provided context from SEC filings below.
        If the context does not contain the answer, say "I cannot answer this based on the available documents."
        
        CONTEXT:
        {combined_text}
        """

        completion = llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.query}
            ],
            temperature=0.2 
        )
        
        final_answer = completion.choices[0].message.content

        # ==========================================
        # 3. SAVE TO CACHE FOR NEXT TIME
        # ==========================================
        new_cache = CacheEntry(
            query_hash=query_hash,
            user_query=request.query,
            llm_response=final_answer
        )
        db.add(new_cache)
        db.commit()

        return {
            "query": request.query, 
            "answer": final_answer,
            "sources": retrieved_chunks,
            "cached": False
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close() # Always close the connection safely