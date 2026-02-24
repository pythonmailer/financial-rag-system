import asyncio
import httpx
import time

API_URL = "http://localhost:8001/ask"

QUERIES = [
    "What specific cybersecurity risks or recent data breaches are disclosed?",
    "Who are the primary competitors identified in the market?",
    "What are the company's environmental sustainability or carbon-neutral goals?",
    "Were there any significant mergers, acquisitions, or divestitures?",
    "What is the total number of full-time employees and their human capital strategy?",
    "How does the company protect its intellectual property and patents?",
    "What were the total operating expenses excluding Research and Development?",
    "Which geographic region or country generated the most revenue?",
    "What major physical properties, factories, or real estate does the company lease or own?",
    "What is the management's forward-looking guidance or financial outlook for the next quarter?"
]

async def fetch_answer(client, query_text, index, global_start_time):
    payload = {
        "query": query_text,
        "ticker": "JHDO", 
        "top_k": 5
    }
    
    print(f"🚀 Sent Request {index+1}...")
    try:
        response = await client.post(API_URL, json=payload, timeout=60.0)
        elapsed_time = time.time() - global_start_time # Time since the script started
        
        if response.status_code == 200:
            data = response.json()
            provider = data.get('provider')
            print(f"✅ [{elapsed_time:.2f}s] Request {index+1} Completed! Engine: {provider}")
            return data
    except Exception as e:
        print(f"❌ Request {index+1} Error: {e}")
        return None

async def main():
    print(f"Starting load test with {len(QUERIES)} concurrent requests...\n")
    global_start_time = time.time()
    
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_answer(client, query, i, global_start_time) 
            for i, query in enumerate(QUERIES)
        ]
        await asyncio.gather(*tasks)
        
    total_time = time.time() - global_start_time
    print(f"\n🎉 All requests finished. Maximum total time: {total_time:.2f} seconds!")

if __name__ == "__main__":
    asyncio.run(main())