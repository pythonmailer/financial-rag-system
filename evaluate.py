import time
import math
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ==========================================
# 🌐 HARDCODED INFRASTRUCTURE CONFIG
# ==========================================
# Using your provided AWS Elastic IP directly
AWS_IP = "13.232.197.229"

# Prefer the env var from docker-compose, fallback to the AWS IP
QDRANT_URL = os.getenv("QDRANT_URL", f"http://{AWS_IP}:6333")

# ==========================================
# 1. INITIALIZATION
# ==========================================
print(f"🚀 Initializing Evaluation Suite on: {QDRANT_URL}")
# Note: Using the same model as your embedder for consistency
model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url=QDRANT_URL)

# ==========================================
# 2. APPLE-SPECIFIC GOLDEN DATASET
# ==========================================
EVAL_DATASET = [
    {
        "query": "What are Apple's primary risk factors regarding the supply chain?", 
        "ticker": "AAPL",
        "expected_keywords": ["supply chain", "components", "manufacturing", "china", "disruption"]
    },
    {
        "query": "What was the total net sales for iPhone in the recent fiscal year?", 
        "ticker": "AAPL",
        "expected_keywords": ["iphone", "net sales", "billion", "revenue"]
    },
    {
        "query": "How much did Apple spend on Research and Development (R&D)?", 
        "ticker": "AAPL",
        "expected_keywords": ["research and development", "R&D", "innovation", "expense"]
    },
    {
        "query": "What is Apple's strategy for Artificial Intelligence and Machine Learning?", 
        "ticker": "AAPL",
        "expected_keywords": ["neural engine", "machine learning", "ai", "intelligence", "generative"]
    },
    {
        "query": "Discuss Apple's service sector revenue growth.", 
        "ticker": "AAPL",
        "expected_keywords": ["services", "subscription", "app store", "icloud", "growth"]
    }
]

# ==========================================
# 3. THE EVALUATION ENGINE (MRR + HIT RATE)
# ==========================================
def run_evaluation(k=5):
    print(f"\n📊 Evaluating Retrieval Performance (Top-{k})...")
    print("-" * 60)
    
    hits = 0
    reciprocal_ranks = []
    total_latency = 0

    # Ensure collection exists before evaluating
    try:
        if not qdrant.collection_exists("financial_documents"):
            print("❌ ERROR: 'financial_documents' collection not found in Qdrant!")
            return
    except Exception as e:
        print(f"❌ CONNECTION ERROR: Could not reach Qdrant at {QDRANT_URL}: {e}")
        return

    for item in EVAL_DATASET:
        start_time = time.time()
        
        # 1. Vectorize query
        query_vector = model.encode(item["query"]).tolist()
        
        # 2. Search Qdrant with Metadata Filtering
        search_result = qdrant.query_points(
            collection_name="financial_documents",
            query=query_vector,
            limit=k,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="ticker", match=models.MatchValue(value=item['ticker']))]
            )
        )
        
        latency = (time.time() - start_time) * 1000
        total_latency += latency
        
        # 3. Calculate Rank and MRR
        found_at_rank = 0
        for rank, hit in enumerate(search_result.points, start=1):
            combined_text = hit.payload.get("text", "").lower()
            if any(kw.lower() in combined_text for kw in item["expected_keywords"]):
                found_at_rank = rank
                break
        
        # 4. Scoring Logic
        if found_at_rank > 0:
            hits += 1
            rr = 1.0 / found_at_rank
            reciprocal_ranks.append(rr)
            print(f"✅ [HIT]  Rank: {found_at_rank} | {latency:4.1f}ms | {item['query'][:50]}...")
        else:
            reciprocal_ranks.append(0.0)
            print(f"❌ [MISS] Rank: N/A | {latency:4.1f}ms | {item['query'][:50]}...")

    # ==========================================
    # 4. CALCULATE FINAL METRICS
    # ==========================================
    avg_mrr = sum(reciprocal_ranks) / len(EVAL_DATASET)
    hit_rate = (hits / len(EVAL_DATASET)) * 100
    avg_latency = total_latency / len(EVAL_DATASET)

    print("\n" + "="*60)
    print(f"🏆 EVALUATION RESULTS (k={k})")
    print("="*60)
    print(f"Accuracy (Hit@{k}):   {hit_rate:.1f}%")
    print(f"Precision (MRR):     {avg_mrr:.3f}  (1.0 is Perfect)")
    print(f"Avg Search Latency:  {avg_latency:.1f} ms")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_evaluation(k=5)