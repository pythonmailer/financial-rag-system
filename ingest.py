import os
import uuid
import requests
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

def run_ingestion(ticker="AAPL", filing_types=["10-K"]):
    print(f"--- Starting Ingestion for {ticker} ---")
    
    # 1. Initialize Connection (Uses Docker service name 'qdrant')
    model = SentenceTransformer("all-MiniLM-L6-v2")
    qdrant = QdrantClient(url="http://qdrant:6333") 

    # 2. Download latest filings
    # Downloader skips files if they already exist locally
    dl = Downloader("MyRAGProject", "your.email@example.com", "./sec_data")
    
    for f_type in filing_types:
        print(f"Checking for new {f_type} filings...")
        dl.get(f_type, ticker, limit=1, download_details=True)

        # 3. Locate and Process the HTML File
        download_dir = f"./sec_data/sec-edgar-filings/{ticker}/{f_type}"
        for root, dirs, files in os.walk(download_dir):
            for file in files:
                if file.endswith(".html") or file == "primary_document.html":
                    html_path = os.path.join(root, file)
                    
                    with open(html_path, "r", encoding="utf-8") as f:
                        soup = BeautifulSoup(f.read(), "html.parser")
                    
                    clean_text = soup.get_text(separator="\n", strip=True)

                    # 4. Split and Embed
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=200
                    )
                    chunks = text_splitter.split_text(clean_text)

                    points = []
                    for chunk in chunks:
                        vector = model.encode(chunk).tolist()
                        points.append(PointStruct(
                            id=str(uuid.uuid4()), 
                            vector=vector, 
                            payload={"ticker": ticker, "document_type": f_type, "text": chunk}
                        ))

                    # 5. Push to Vector DB
                    qdrant.upsert(collection_name="financial_documents", points=points)
                    print(f"✅ Successfully ingested {len(chunks)} chunks for {ticker} {f_type}")
                    
    # 6. NEW: Clear the API Cache so the next user gets the fresh data
    print(f"🧹 Wiping stale semantic cache for {ticker}...")
    try:
        # Pass the ticker into the URL
        response = requests.delete(f"http://localhost:8001/cache/clear/{ticker}") 
        if response.status_code == 200:
            print(f"✨ Cache cleared! Deleted {response.json()['cleared_entries']} old entries for {ticker}.")
    except Exception as e:
        print(f"⚠️ Could not reach API to clear cache. Error: {e}")

if __name__ == "__main__":
    run_ingestion()