import os
import time
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# ==========================================
# 🌐 HARDCODED INFRASTRUCTURE CONFIG
# ==========================================
# Using your provided AWS Elastic IP directly
AWS_IP = "13.232.197.229"

# We prioritize environment variables (from .env or docker-compose)
# but fallback to your static AWS IP.
QDRANT_URL = os.getenv("QDRANT_URL", f"http://{AWS_IP}:6333")
DATABASE_URL = os.getenv("DATABASE_URL", f"postgresql://admin:adminpassword@{AWS_IP}:5432/financial_rag")

COLLECTION_NAME = "financial_documents"
VECTOR_SIZE = 384  # must match BGE embedding model

# ==========================================
# CONFIG
# ==========================================
TESTING = os.getenv("TESTING", "False") == "True"

# ==========================================
# SQLALCHEMY ENGINE
# ==========================================
if TESTING:
    engine = create_engine(
        "sqlite:///./test_database.db",
        connect_args={"check_same_thread": False},
    )
else:
    # pool_pre_ping=True is vital for AWS to handle dropped connections
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==========================================
# MODELS
# ==========================================
class FeedbackEntry(Base):
    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True, index=True)
    query_hash = Column(String, index=True)
    rating = Column(Integer)  # 1 = 👍, -1 = 👎
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class CacheEntry(Base):
    __tablename__ = "semantic_cache"

    id = Column(Integer, primary_key=True, index=True)
    query_hash = Column(String, unique=True, index=True)
    user_query = Column(Text)
    llm_response = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ticker = Column(String, index=True)
    provider = Column(String, nullable=True)

# ==========================================
# DB INIT
# ==========================================
def init_db():
    if TESTING:
        Base.metadata.create_all(bind=engine)
        print("✅ SQLite test database initialized.")
        return

    try:
        Base.metadata.create_all(bind=engine)
        print(f"✅ DB connection verified at: {DATABASE_URL.split('@')[-1]}")
    except Exception as e:
        print(f"⚠️ DB init skipped (likely using Alembic): {e}")

# ==========================================
# QDRANT INIT
# ==========================================
def init_qdrant():
    if TESTING:
        print("跑 Skipping Qdrant init (TESTING=True).")
        return

    retries = 5
    delay = 3

    for attempt in range(1, retries + 1):
        try:
            client = QdrantClient(url=QDRANT_URL)

            if not client.collection_exists(COLLECTION_NAME):
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=VECTOR_SIZE,
                        distance=Distance.COSINE,
                    ),
                )
                print(f"✅ Qdrant collection '{COLLECTION_NAME}' created at {QDRANT_URL}")
            else:
                print(f"✅ Qdrant collection '{COLLECTION_NAME}' exists at {QDRANT_URL}")

            return

        except Exception as e:
            print(
                f"⏳ Waiting for Qdrant at {QDRANT_URL}... ({attempt}/{retries})"
            )
            time.sleep(delay)

    print("❌ CRITICAL: Could not connect to Qdrant.")

# ==========================================
# SAFE STARTUP HOOKS
# ==========================================
if __name__ == "__main__":
    init_db()
    init_qdrant()