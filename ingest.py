import os
import hashlib
import requests
import time
from datetime import datetime, timezone

from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams

# ==========================================
# 🌐 INFRA CONFIG
# ==========================================
AWS_IP = "13.232.197.229"

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
BACKEND_URL = os.getenv("BACKEND_URL", f"http://{AWS_IP}:8001")

COLLECTION_NAME = "financial_documents"
EMBED_URL = f"{BACKEND_URL}/embed"
READY_URL = f"{BACKEND_URL}/ready"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_BATCH = 64
UPSERT_BATCH = 256

print(f"🛠️ Ingestor Config → Backend: {BACKEND_URL} | Qdrant: {QDRANT_URL}")

# ==========================================
# WAIT FOR BACKEND
# ==========================================
def wait_for_backend():
    print(f"⏳ Waiting for backend at {READY_URL}")
    for i in range(30):
        try:
            r = requests.get(READY_URL, timeout=5)
            if r.status_code == 200:
                print("✅ Backend ready")
                return
        except Exception:
            pass
        print(f"   Retry {i+1}/30")
        time.sleep(3)
    raise RuntimeError("❌ Backend not ready")

# ==========================================
# EMBEDDING (BATCHED VIA BACKEND)
# ==========================================
def embed_chunks(chunks):
    embeddings = []
    print(f"🧠 Embedding {len(chunks)} chunks")

    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        try:
            r = requests.post(EMBED_URL, json={"texts": batch}, timeout=180)
            r.raise_for_status()
            embeddings.extend(r.json()["embeddings"])
        except Exception as e:
            print(f"❌ Embedding failed: {e}")
            raise

    return embeddings

# ==========================================
# TEXT CHUNKING
# ==========================================
def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)

    # 🔧 remove empty / whitespace chunks
    chunks = [c for c in chunks if c.strip()]
    return chunks

# ==========================================
# QDRANT SETUP
# ==========================================
def ensure_collection(qdrant: QdrantClient):
    if not qdrant.collection_exists(COLLECTION_NAME):
        print(f"🧱 Creating collection: {COLLECTION_NAME}")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,  # BGE-small-en-v1.5
                distance=Distance.COSINE,
            ),
        )
        print("✅ Collection created")

# ==========================================
# HTML → TEXT
# ==========================================
def extract_text_from_html(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text(separator="\n", strip=True)

# ==========================================
# INGESTION PIPELINE
# ==========================================
def run_ingestion(ticker="AAPL", filing_types=("10-K", "10-Q"), limit=1):
    print(f"🚀 Ingesting {ticker}")

    wait_for_backend()

    qdrant = QdrantClient(url=QDRANT_URL)
    ensure_collection(qdrant)

    dl = Downloader("FinancialRAGProject", "chiragg948@gmail.com", "./sec_data")

    total_chunks = 0

    for f_type in filing_types:
        print(f"📡 Downloading {f_type}")
        dl.get(f_type, ticker, limit=limit, download_details=True)

        base_path = f"./sec_data/sec-edgar-filings/{ticker}/{f_type}"

        if not os.path.exists(base_path):
            print(f"⚠️ No filings found for {f_type}")
            continue

        for root, _, files in os.walk(base_path):
            for file in files:
                if file.endswith(".html") or file == "primary_document.html":
                    html_path = os.path.join(root, file)
                    print(f"📄 Processing {html_path}")

                    text = extract_text_from_html(html_path)
                    chunks = chunk_text(text)

                    if not chunks:
                        print("⚠️ No valid chunks")
                        continue

                    print(f"✂️ {len(chunks)} chunks created")

                    embeddings = embed_chunks(chunks)

                    points = []
                    for chunk, vector in zip(chunks, embeddings):

                        # ✅ deterministic, collision-safe ID
                        chunk_hash = hashlib.md5(
                            f"{ticker}_{f_type}_{file}_{chunk}".encode()
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

                    # ✅ memory-safe batched upsert
                    for i in range(0, len(points), UPSERT_BATCH):
                        qdrant.upsert(
                            collection_name=COLLECTION_NAME,
                            points=points[i : i + UPSERT_BATCH],
                        )

                    total_chunks += len(points)
                    print(f"✅ Indexed {len(points)} vectors")

    print(f"🎉 Done → {total_chunks} total chunks indexed")

    # ======================================
    # CACHE INVALIDATION
    # ======================================
    try:
        cache_url = f"{BACKEND_URL}/cache/clear/{ticker}"
        r = requests.delete(cache_url, timeout=10)
        if r.status_code == 200:
            print(f"🧹 Cache cleared for {ticker}")
        else:
            print(f"⚠️ Cache clear returned {r.status_code}")
    except Exception as e:
        print(f"⚠️ Cache clear failed: {e}")

# ==========================================
# ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    run_ingestion()