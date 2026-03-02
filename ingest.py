import os
import uuid
import hashlib
import requests
import time
import socket
import math
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams

# ==========================================
# 🤖 SMART AUTO-DETECTION
# ==========================================
def resolve_host(service_name, default="localhost"):
    try:
        socket.gethostbyname(service_name)
        return service_name
    except socket.gaierror:
        return default

# Detect hosts
QDRANT_HOST_NAME = resolve_host("qdrant", "localhost")
BACKEND_HOST_NAME = resolve_host("backend", "localhost")

# ==========================================
# CONFIG
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL", f"http://{QDRANT_HOST_NAME}:6333")
BACKEND_URL = os.getenv("BACKEND_URL", f"http://{BACKEND_HOST_NAME}:8001")
COLLECTION_NAME = "financial_documents"

EMBED_URL = f"{BACKEND_URL}/embed"
READY_URL = f"{BACKEND_URL}/ready"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 64  # batch embedding size

print(f"🛠️ Configured Ingestor: Backend @ {BACKEND_URL} | Qdrant @ {QDRANT_URL}")

# ==========================================
# WAIT FOR BACKEND
# ==========================================
def wait_for_backend():
    print("⏳ Waiting for backend readiness...")
    for i in range(30):
        try:
            r = requests.get(READY_URL, timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ready":
                print("✅ Backend is ready")
                return
        except:
            pass
        print(f"   (Retrying... {i+1}/30)")
        time.sleep(2)
    raise RuntimeError(f"❌ Backend at {BACKEND_URL} not ready")

# ==========================================
# EMBED VIA BACKEND (BATCHED)
# ==========================================
def embed_chunks(chunks):
    embeddings = []
    print(f"🧠 Embedding {len(chunks)} chunks...")
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        try:
            r = requests.post(EMBED_URL, json={"texts": batch}, timeout=120)
            r.raise_for_status()
            embeddings.extend(r.json()["embeddings"])
        except Exception as e:
            print(f"❌ Embedding failed: {e}")
            raise

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
        print(f"🧱 Creating Qdrant collection: {COLLECTION_NAME}")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,  # must match BGE embedding model
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

    qdrant = QdrantClient(url=QDRANT_URL)
    ensure_collection(qdrant)

    # Use a generic name and your real email as required by SEC
    dl = Downloader("FinancialRAGProject", "chiragg948@gmail.com", "./sec_data")

    total_chunks = 0

    for f_type in filing_types:
        print(f"📡 Downloading {f_type} for {ticker}")
        dl.get(f_type, ticker, limit=limit, download_details=True)

        base_path = f"./sec_data/sec-edgar-filings/{ticker}/{f_type}"

        for root, _, files in os.walk(base_path):
            for file in files:
                # SEC often names the main file 'primary_document.html'
                if file.endswith(".html") or file == "primary_document.html":
                    html_path = os.path.join(root, file)
                    print(f"📄 Processing {html_path}")

                    text = extract_text_from_html(html_path)
                    chunks = chunk_text(text)

                    print(f"✂️ Created {len(chunks)} segments")

                    embeddings = embed_chunks(chunks)

                    points = []

                    for chunk, vector in zip(chunks, embeddings):
                        # Unique ID based on content to prevent duplicates
                        chunk_hash = hashlib.md5(
                            f"{ticker}_{f_type}_{chunk[:100]}".encode()
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
                                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                                },
                            )
                        )

                    # Bulk upsert to Qdrant
                    qdrant.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points,
                    )

                    total_chunks += len(points)
                    print(f"✅ Upserted {len(points)} vectors to Qdrant")

    print(f"🎉 Ingestion complete: {total_chunks} total chunks indexed")

    # ======================================
    # CACHE INVALIDATION
    # ======================================
    try:
        cache_url = f"{BACKEND_URL}/cache/clear/{ticker}"
        requests.delete(cache_url, timeout=10)
        print(f"🧹 Semantic cache cleared for {ticker}")
    except:
        print("⚠️ Cache clear skipped")

if __name__ == "__main__":
    run_ingestion()