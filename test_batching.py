import asyncio
import httpx
import time

# The URL of your FastAPI endpoint
API_URL = "http://localhost:8000/ask"

# 10 distinct questions to test the embedder
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

async def fetch_answer(client, query_text, index):
    """Sends a single request to the FastAPI server."""
    payload = {
        "query": query_text,
        "ticker": "AAPL", # You can change this to MSFT, NVDA, etc.
        "top_k": 5
    }
    
    print(f"🚀 Sending Request {index+1}...")
    try:
        # We set a long timeout because 10 LLM generations might take a few seconds
        response = await client.post(API_URL, json=payload, timeout=60.0)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Request {index+1} Completed! Engine: {data.get('provider')}")
            return data
        else:
            print(f"❌ Request {index+1} Failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Request {index+1} Error: {e}")
        return None

async def main():
    print(f"Starting load test with {len(QUERIES)} concurrent requests...\n")
    start_time = time.time()
    
    # Open a single async HTTP session
    async with httpx.AsyncClient() as client:
        # Create a list of concurrent tasks
        tasks = [
            fetch_answer(client, query, i) 
            for i, query in enumerate(QUERIES)
        ]
        
        # Fire them all at the exact same time
        await asyncio.gather(*tasks)
        
    total_time = time.time() - start_time
    print(f"\n🎉 Finished processing {len(QUERIES)} requests in {total_time:.2f} seconds!")

if __name__ == "__main__":
    # Run the async event loop
    asyncio.run(main())