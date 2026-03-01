import os
import uuid
import hashlib
import requests
import time
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams

# ==========================================
# 1. CONFIGURATION & ENVIRONMENT
# ==========================================
# Use internal Docker DNS if running in a container, otherwise localhost
QDRANT_HOST = os.getenv("QDRANT_URL", "http://localhost:6333")
BACKEND_HOST = os.getenv("BACKEND_URL", "http://localhost:8001")
COLLECTION_NAME = "financial_documents"

def run_ingestion(ticker="AAPL", filing_types=["10-K", "10-Q"]):
    print(f"🚀 Initializing Ingestion Pipeline for: {ticker}")
    
    # Initialize Local AI Model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Connect to Qdrant
    qdrant = QdrantClient(url=QDRANT_HOST)

    # Ensure collection exists before upserting
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"✅ Created new Qdrant collection: {COLLECTION_NAME}")

    # ==========================================
    # 2. DOWNLOAD SEC FILINGS
    # ==========================================
    # Using a professional identity for the SEC downloader
    dl = Downloader("FinancialRAGProject", "chiragg948@gmail.com", "./sec_data")
    
    for f_type in filing_types:
        print(f"📡 Fetching latest {f_type} for {ticker}...")
        dl.get(f_type, ticker, limit=1, download_details=True)

        download_dir = f"./sec_data/sec-edgar-filings/{ticker}/{f_type}"
        
        # Walk through downloaded files
        for root, dirs, files in os.walk(download_dir):
            for file in files:
                if file.endswith(".html") or file == "primary_document.html":
                    html_path = os.path.join(root, file)
                    print(f"📄 Processing: {html_path}")
                    
                    with open(html_path, "r", encoding="utf-8") as f:
                        soup = BeautifulSoup(f.read(), "html.parser")
                    
                    # Clean the HTML to extract raw financial text
                    clean_text = soup.get_text(separator="\n", strip=True)

                    # ==========================================
                    # 3. SEMANTIC CHUNKING
                    # ==========================================
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    chunks = text_splitter.split_text(clean_text)

                    points = []
                    print(f"🧠 Generating embeddings for {len(chunks)} chunks...")
                    
                    for chunk in chunks:
                        # DETERMINISTIC ID: Prevents duplicates if the script runs twice
                        chunk_hash = hashlib.md5(f"{ticker}_{f_type}_{chunk}".encode()).hexdigest()
                        
                        vector = model.encode(chunk).tolist()
                        
                        points.append(PointStruct(
                            id=chunk_hash, 
                            vector=vector, 
                            payload={
                                "ticker": ticker.upper(), 
                                "document_type": f_type.upper(), 
                                "text": chunk,
                                "ingested_at": time.time()
                            }
                        ))

                    # ==========================================
                    # 4. UPSERT TO QDRANT
                    # ==========================================
                    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                    print(f"✅ Ingested {len(chunks)} chunks into Vector DB.")
                    
    # ==========================================
    # 5. CACHE INVALIDATION HOOK
    # ==========================================
    print(f"🧹 Triggering cache invalidation for {ticker}...")
    try:
        # We target the specific ticker so we don't wipe data for other stocks
        cache_clear_url = f"{BACKEND_HOST}/cache/clear/{ticker}"
        response = requests.delete(cache_clear_url, timeout=5)
        
        if response.status_code == 200:
            count = response.json().get('cleared_entries', 0)
            print(f"✨ Success: Deleted {count} stale cache entries for {ticker}.")
        else:
            print(f"⚠️ Cache clear returned status: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Could not reach Backend API to clear cache. This is normal if backend is offline. Error: {e}")

if __name__ == "__main__":
    run_ingestion()