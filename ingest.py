import os
import uuid
import hashlib
import requests
import time
from datetime import datetime
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams

# ==========================================
# CONFIG
# ==========================================
QDRANT_HOST = os.getenv("QDRANT_URL", "http://localhost:6333")
BACKEND_HOST = os.getenv("BACKEND_URL", "http://localhost:8001")
COLLECTION_NAME = "financial_documents"

EMBED_URL = f"{BACKEND_HOST}/embed"
READY_URL = f"{BACKEND_HOST}/ready"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 64  # batch embedding size

# ==========================================
# WAIT FOR BACKEND
# ==========================================
def wait_for_backend():
    print("⏳ Waiting for backend readiness...")
    for _ in range(60):
        try:
            r = requests.get(READY_URL, timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ready":
                print("✅ Backend is ready")
                return
        except:
            pass
        time.sleep(2)
    raise RuntimeError("❌ Backend not ready after waiting")

# ==========================================
# EMBED VIA BACKEND (BATCHED)
# ==========================================
def embed_chunks(chunks):
    embeddings = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        r = requests.post(EMBED_URL, json={"texts": batch}, timeout=120)
        r.raise_for_status()
        embeddings.extend(r.json()["embeddings"])

    return embeddings

# ==========================================
# TEXT CHUNKING
# ==========================================
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)

# ==========================================
# QDRANT SETUP
# ==========================================
def ensure_collection(qdrant):
    if not qdrant.collection_exists(COLLECTION_NAME):
        print("🧱 Creating Qdrant collection...")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,  # must match embedding model
                distance=Distance.COSINE,
            ),
        )
        print("✅ Collection created")

# ==========================================
# PROCESS HTML → TEXT
# ==========================================
def extract_text_from_html(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text(separator="\n", strip=True)

# ==========================================
# INGESTION PIPELINE
# ==========================================
def run_ingestion(ticker="AAPL", filing_types=("10-K", "10-Q"), limit=1):
    print(f"🚀 Starting ingestion for {ticker}")

    wait_for_backend()

    qdrant = QdrantClient(url=QDRANT_HOST)
    ensure_collection(qdrant)

    dl = Downloader("FinancialRAGProject", "your_email@example.com", "./sec_data")

    total_chunks = 0

    for f_type in filing_types:
        print(f"📡 Downloading {f_type} for {ticker}")
        dl.get(f_type, ticker, limit=limit, download_details=True)

        base_path = f"./sec_data/sec-edgar-filings/{ticker}/{f_type}"

        for root, _, files in os.walk(base_path):
            for file in files:
                if file.endswith(".html") or file == "primary_document.html":
                    html_path = os.path.join(root, file)
                    print(f"📄 Processing {html_path}")

                    text = extract_text_from_html(html_path)
                    chunks = chunk_text(text)

                    print(f"🧠 Chunked into {len(chunks)} segments")

                    embeddings = embed_chunks(chunks)

                    points = []

                    for chunk, vector in zip(chunks, embeddings):
                        chunk_hash = hashlib.md5(
                            f"{ticker}_{f_type}_{chunk}".encode()
                        ).hexdigest()

                        points.append(
                            PointStruct(
                                id=chunk_hash,
                                vector=vector,
                                payload={
                                    "ticker": ticker.upper(),
                                    "document_type": f_type.upper(),
                                    "text": chunk,
                                    "source_file": file,
                                    "ingested_at": datetime.utcnow().isoformat(),
                                },
                            )
                        )

                    qdrant.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points,
                    )

                    total_chunks += len(points)
                    print(f"✅ Upserted {len(points)} vectors")

    print(f"🎉 Ingestion complete: {total_chunks} total chunks")

    # ======================================
    # CACHE INVALIDATION
    # ======================================
    try:
        cache_url = f"{BACKEND_HOST}/cache/clear/{ticker}"
        r = requests.delete(cache_url, timeout=10)

        if r.status_code == 200:
            cleared = r.json().get("cleared_entries", 0)
            print(f"🧹 Cleared {cleared} cache entries for {ticker}")
        else:
            print(f"⚠️ Cache clear returned {r.status_code}")
    except Exception as e:
        print(f"⚠️ Cache clear skipped: {e}")

# ==========================================
# ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    run_ingestion()