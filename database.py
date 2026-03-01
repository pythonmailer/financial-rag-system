import os
import time
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from datetime import datetime, timezone

# --- CI/CD Logic: Detect Testing Environment ---
TESTING = os.getenv("TESTING", "False") == "True"

if TESTING:
    # Use an in-memory SQLite database for GitHub Actions tests
    POSTGRES_URL = "sqlite:///./test_database.db"
    # SQLite requires 'check_same_thread: False' for FastAPI
    engine = create_engine(POSTGRES_URL, connect_args={"check_same_thread": False})
else:
    # Production PostgreSQL connection
    POSTGRES_URL = "postgresql://admin:adminpassword@postgres:5432/financial_rag"
    engine = create_engine(POSTGRES_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ... (Keep your FeedbackEntry and CacheEntry classes exactly as they are) ...

# ==========================================
# INFRASTRUCTURE INITIALIZATION
# ==========================================

# 1. Initialize SQLAlchemy Tables
if TESTING:
    Base.metadata.create_all(bind=engine)
    print("✅ Successfully initialized local SQLite testing database.")
else:
    # Keep your original retry logic for production Postgres
    MAX_RETRIES = 5
    RETRY_DELAY = 3
    for attempt in range(MAX_RETRIES):
        try:
            Base.metadata.create_all(bind=engine)
            print("✅ Successfully connected to PostgreSQL and verified tables.")
            break
        except Exception:
            print(f"⏳ Waiting for PostgreSQL to be ready... (Attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY)
    else:
        print("❌ CRITICAL: Could not connect to PostgreSQL.")

# 2. Initialize Qdrant Collection (Bypass if Testing)
if not TESTING:
    for attempt in range(MAX_RETRIES):
        try:
            qdrant = QdrantClient(url="http://qdrant:6333")
            collection_name = "financial_documents"
            if not qdrant.collection_exists(collection_name):
                qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
                print(f"✅ Qdrant collection '{collection_name}' created.")
            break
        except Exception:
            print(f"⏳ Waiting for Qdrant... (Attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY)
else:
    print("⏭️ Skipping Qdrant initialization (TESTING=True).")