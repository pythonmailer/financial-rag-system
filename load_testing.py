import asyncio
import httpx
import time
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
TARGET_IP = "13.232.197.229"
TARGET_PORT = 8001  # Your FastAPI port
BASE_URL = f"http://{TARGET_IP}:{TARGET_PORT}"
API_URL = f"{BASE_URL}/ask"
CACHE_CLEAR_URL = f"{BASE_URL}/cache/clear/AAPL"

# ==========================================
# 2. THE STRESS-TEST DATASET (100 QUESTIONS)
# ==========================================
QUERIES = [
    # --- SECTION 1: Revenue & Growth ---
    "What was the total net sales for the last fiscal year?",
    "How much revenue came from the iPhone segment specifically?",
    "What is the year-over-year growth rate for Services?",
    "Which geographic region showed the highest sales growth?",
    "How did the Wearables, Home, and Accessories segment perform?",
    "What percentage of total revenue is derived from international sales?",
    "Detail the net sales for the iPad category.",
    "Describe the impact of foreign currency fluctuations on net sales.",
    "What were the Mac net sales compared to the previous year?",
    "What are the primary drivers of Service revenue growth?",

    # --- SECTION 2: Profitability & Margins ---
    "What was the gross margin percentage for the last quarter?",
    "How do product margins compare to service margins?",
    "What caused the fluctuations in operating margins this year?",
    "Detail the net income for the current reporting period.",
    "What is the basic and diluted earnings per share (EPS)?",
    "Describe the impact of component costs on gross margins.",
    "What are the main operating expenses affecting profitability?",
    "How does Apple manage its cost of sales?",
    "What is the effective tax rate reported in the 10-K?",
    "Were there any significant one-time gains or losses affecting net income?",

    # --- SECTION 3: Risk Factors & Macro ---
    "What are the primary macroeconomic risks mentioned?",
    "How does inflation impact Apple's profit margins?",
    "What risks are associated with global trade and tariffs?",
    "Discuss the risks related to the COVID-19 pandemic recovery.",
    "How does Apple describe its dependency on third-party manufacturing?",
    "What are the risks concerning intellectual property litigation?",
    "Detail the risks of operating in the Greater China region.",
    "How do interest rate changes affect Apple's financial position?",
    "What are the risks related to the highly competitive smartphone market?",
    "Discuss potential disruptions in the global supply chain.",

    # --- SECTION 4: R&D & Innovation ---
    "How much was spent on Research and Development (R&D)?",
    "What is the company's strategy for artificial intelligence?",
    "Does the report mention specific investments in silicon or chips?",
    "What are the key areas of focus for R&D spending?",
    "How does Apple view its investment in 'Generative AI'?",
    "What is the Neural Engine's role in future product strategy?",
    "Does the filing discuss R&D for autonomous systems or cars?",
    "How has R&D spending changed as a percentage of revenue?",
    "What investments are being made in augmented reality (AR)?",
    "Describe Apple's approach to software and services innovation.",

    # --- SECTION 5: Debt & Capital Structure ---
    "What is the total amount of outstanding debt?",
    "What is the breakdown between short-term and long-term debt?",
    "Did they announce any stock buybacks or dividends?",
    "How much cash and cash equivalents are currently held?",
    "What is the total value of marketable securities?",
    "Describe the company's share repurchase program details.",
    "What is the interest expense on the company's debt?",
    "How does Apple use commercial paper for financing?",
    "What are the credit ratings mentioned for Apple's debt?",
    "Detail the dividend payments made to shareholders this year.",

    # --- SECTION 6: Legal & Regulatory ---
    "Are there any ongoing legal proceedings or lawsuits?",
    "What is the status of antitrust investigations against Apple?",
    "Discuss any significant patent infringement cases.",
    "How does Apple address environmental regulations and compliance?",
    "What are the potential legal liabilities mentioned in the 10-K?",
    "Detail any tax-related disputes with international governments.",
    "How does Apple manage data privacy and security regulations?",
    "What are the risks associated with the App Store's legal challenges?",
    "Describe the impact of the Digital Markets Act (DMA) in Europe.",
    "Are there any updates on Epic Games vs. Apple litigation?",

    # --- SECTION 7: Supply Chain & Operations ---
    "What are the international supply chain bottlenecks?",
    "Where are the majority of Apple's products manufactured?",
    "How does Apple manage its relationship with Foxconn and other suppliers?",
    "What is the strategy for diversifying manufacturing outside of China?",
    "How does Apple mitigate the risk of component shortages?",
    "Detail the company's commitment to renewable energy in the supply chain.",
    "What is the 'Supplier Code of Conduct' mentioned in the report?",
    "How many retail stores does Apple currently operate?",
    "Describe the logistics and distribution strategy for new products.",
    "How does the company manage inventory levels for seasonal demand?",

    # --- SECTION 8: ESG & Sustainability ---
    "What is Apple's goal for carbon neutrality?",
    "How much recycled material is used in current products?",
    "What are the social responsibility goals mentioned for employees?",
    "Describe Apple's 'Racial Equity and Justice Initiative'.",
    "What steps are taken to ensure ethical mineral sourcing?",
    "How does Apple manage electronic waste and recycling programs?",
    "What are the diversity and inclusion metrics for the workforce?",
    "Detail the water conservation efforts in manufacturing.",
    "How does Apple report on its Scope 3 greenhouse gas emissions?",
    "What is the company's stance on human rights in the supply chain?",

    # --- SECTION 9: Management & Governance ---
    "Who is the current CEO and CFO?",
    "How is executive compensation structured (base vs. stock)?",
    "Who are the members of the Board of Directors?",
    "What is the role of the Audit and Finance Committee?",
    "Describe the succession planning mentioned for senior leadership.",
    "How much stock do the top executives currently hold?",
    "What are the primary voting rights for common shareholders?",
    "When is the next annual meeting of shareholders?",
    "Describe the company's corporate governance guidelines.",
    "Who is the lead independent director of the board?",

    # --- SECTION 10: Future Outlook ---
    "What are the key growth opportunities identified for next year?",
    "Does Apple provide specific revenue guidance for the next quarter?",
    "What are the expectations for the Services segment growth?",
    "How does Apple view the expansion of the 'Services' ecosystem?",
    "What is the strategy for increasing market share in emerging markets?",
    "Describe the expected impact of new product launches on revenue.",
    "How is Apple preparing for future global economic downturns?",
    "What are the long-term capital allocation priorities?",
    "Does the company anticipate increased regulatory scrutiny in AI?",
    "What is the concluding outlook on the global smartphone market?"
]

async def clear_cache(client):
    print(f"🧹 Clearing cache on Port {TARGET_PORT}...")
    try:
        response = await client.delete(CACHE_CLEAR_URL, timeout=10.0)
        if response.status_code == 200:
            print("✨ Cache cleared successfully!\n")
    except Exception as e:
        print(f"⚠️ Could not clear cache: {e}\n")

async def fetch_answer(client, query_text, index, semaphore):
    # Use semaphore to prevent overwhelming the local OS sockets
    async with semaphore:
        payload = {"query": query_text, "ticker": "AAPL", "top_k": 5}
        start_time = time.time()
        try:
            response = await client.post(API_URL, json=payload, timeout=90.0)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                provider = data.get('provider', 'Unknown')
                if data.get("cached"):
                    print(f"🕒 [{index+1}/100] CACHE HIT  | {latency:.2f}s")
                else:
                    print(f"🚀 [{index+1}/100] LIVE ({provider}) | {latency:.2f}s")
                return latency
            else:
                print(f"❌ [{index+1}/100] FAILED ({response.status_code})")
                return None
        except Exception as e:
            print(f"❌ [{index+1}/100] ERROR: {e}")
            return None

async def main():
    print(f"🔥 Starting CONCURRENT Load Test (100 Queries) against {TARGET_IP}...")
    
    # Limits concurrent requests to 10 at a time to avoid crashing the server/network
    # Increase this to 50+ if your EC2 instance is powerful (t3.large+)
    semaphore = asyncio.Semaphore(10) 
    
    async with httpx.AsyncClient() as client:
        await clear_cache(client)
        
        start_time = time.time()
        tasks = [fetch_answer(client, query, i, semaphore) for i, query in enumerate(QUERIES)]
        latencies = await asyncio.gather(*tasks)
        
    total_time = time.time() - start_time
    valid_latencies = [l for l in latencies if l is not None]
    
    print("\n" + "="*50)
    print(f"📊 LOAD TEST RESULTS")
    print("="*50)
    print(f"Total Queries:     {len(QUERIES)}")
    print(f"Successful:        {len(valid_latencies)}")
    print(f"Total Duration:    {total_time:.2f} seconds")
    if valid_latencies:
        print(f"Avg Latency:       {sum(valid_latencies)/len(valid_latencies):.2f}s")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())