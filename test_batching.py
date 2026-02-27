import asyncio
import httpx
import time

# ==========================================
# ⚙️ TEST CONFIGURATION
# ==========================================
# Toggle this to test your different endpoints!
# 8000 = Sequential Baseline (main.py)
# 8001 = Batched Engine (main2.py)
TARGET_PORT = 8000  

BASE_URL = f"http://localhost:{TARGET_PORT}"
API_URL = f"{BASE_URL}/ask"
CACHE_CLEAR_URL = f"{BASE_URL}/cache/clear/AAPL"

QUERIES = [
    "What are the primary macroeconomic risks mentioned?",
    "Who is the current CEO and CFO?",
    "What was the total revenue for the last fiscal year?",
    "Are there any ongoing legal proceedings or lawsuits?",
    "What is the company's strategy for artificial intelligence?",
    "How much was spent on Research and Development (R&D)?",
    "What are the international supply chain bottlenecks?",
    "How does inflation impact their profit margins?",
    "What is the total amount of outstanding debt?",
    "Did they announce any stock buybacks or dividends?"
]

async def clear_cache(client):
    print(f"🧹 Clearing cache on Port {TARGET_PORT}...")
    try:
        response = await client.delete(CACHE_CLEAR_URL)
        if response.status_code == 200:
            print("✨ Cache cleared successfully!\n")
    except Exception as e:
        print(f"⚠️ Could not clear cache: {e}\n")

async def fetch_answer(client, query_text, index):
    payload = {"query": query_text, "ticker": "AAPL", "top_k": 5}
    start_time = time.time()
    try:
        response = await client.post(API_URL, json=payload, timeout=60.0)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            provider = data.get('provider', 'Unknown')
            if data.get("cached"):
                print(f"⚠️ Request {index+1} Hit Cache! ({latency:.2f}s)")
            else:
                print(f"✅ Request {index+1} | {latency:.2f}s | Engine: {provider}")
        else:
            print(f"❌ Request {index+1} Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Request {index+1} Error: {e}")

async def main():
    mode = "SEQUENTIAL" if TARGET_PORT == 8000 else "BATCHED"
    print(f"🚀 Starting {mode} load test against Port {TARGET_PORT}...")
    
    async with httpx.AsyncClient() as client:
        await clear_cache(client)
        
        start_time = time.time()
        tasks = [fetch_answer(client, query, i) for i, query in enumerate(QUERIES)]
        await asyncio.gather(*tasks)
        
    total_time = time.time() - start_time
    print(f"\n🎉 {mode} ENGINE: Processed {len(QUERIES)} concurrent requests in {total_time:.2f} seconds!")

if __name__ == "__main__":
    asyncio.run(main())