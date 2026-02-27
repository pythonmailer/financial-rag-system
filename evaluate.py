import time
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ==========================================
# 1. INITIALIZATION
# ==========================================
print("Loading Embedding Model and Qdrant...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# We use localhost because we run this script outside of Docker
qdrant = QdrantClient(url="http://localhost:6333") 

# ==========================================
# 2. THE GOLDEN DATASET
# ==========================================
# This is our ground-truth testing set. 
# We define the exact query, the target company, and the specific keywords 
# we absolutely MUST find in the database chunks for the answer to be considered "accurate".
EVAL_DATASET = [
    {
        "query": "What are the primary risk factors for Apple?", 
        "ticker": "AAPL",
        "expected_keywords": ["risk", "macroeconomic", "supply chain", "tariffs", "competition"]
    },
    {
        "query": "What was Apple's total net sales for the year?", 
        "ticker": "AAPL",
        "expected_keywords": ["net sales", "revenue", "billion", "total"]
    },
    {
        "query": "How is Apple utilizing Artificial Intelligence?", 
        "ticker": "AAPL",
        "expected_keywords": ["machine learning", "ai", "neural engine", "generative"]
    },
    {
        "query": "Who are Apple's main competitors?", 
        "ticker": "AAPL",
        "expected_keywords": ["competitors", "competition", "android", "samsung", "google"]
    }
]

# ==========================================
# 3. THE EVALUATION ENGINE
# ==========================================
def run_evaluation(k=5):
    print(f"\n📊 Running Retrieval Evaluation (Top {k} Chunks)...")
    print("-" * 50)
    
    hits = 0
    total_latency = 0

    for item in EVAL_DATASET:
        start_time = time.time()
        
        # 1. Embed the test query
        query_vector = model.encode(item["query"]).tolist()
        
        # 2. Query Qdrant for the top K results
        search_result = qdrant.query_points(
            collection_name="financial_documents",
            query=query_vector,
            limit=k,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="ticker", match=models.MatchValue(value=item["ticker"]))]
            )
        )
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        total_latency += latency
        
        # 3. Extract the text from all retrieved chunks
        retrieved_texts = [hit.payload.get("text", "").lower() for hit in search_result.points if hit.payload]
        combined_text = " ".join(retrieved_texts)
        
        # 4. Score the Retrieval (Hit or Miss)
        # If at least ONE expected keyword is found in the retrieved chunks, we score a Hit.
        # In a real enterprise system, you might require 2+ keywords or use an LLM as a judge.
        is_hit = any(keyword.lower() in combined_text for keyword in item["expected_keywords"])
        
        if is_hit:
            hits += 1
            print(f"✅ [HIT]  ({latency:.1f}ms) | {item['query']}")
        else:
            print(f"❌ [MISS] ({latency:.1f}ms) | {item['query']}")
            print(f"   ↳ Missing Keywords: {item['expected_keywords']}")

    # ==========================================
# 4. CALCULATE FINAL METRICS
# ==========================================
    hit_rate = (hits / len(EVAL_DATASET)) * 100
    avg_latency = total_latency / len(EVAL_DATASET)

    print("\n" + "="*50)
    print(f"🏆 FINAL EVALUATION METRICS (Hit@{k})")
    print("="*50)
    print(f"Total Test Queries: {len(EVAL_DATASET)}")
    print(f"Successful Hits:    {hits}")
    print(f"Hit Rate Accuracy:  {hit_rate:.1f}%")
    print(f"Avg Search Latency: {avg_latency:.1f} ms")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Test strict retrieval (Can it find the answer in just 3 chunks?)
    run_evaluation(k=3)
    
    # Test standard retrieval (Can it find the answer in 5 chunks?)
    run_evaluation(k=5)