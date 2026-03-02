import os
import time
import socket
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# ==========================================
# 🤖 SMART AUTO-DETECTION LOGIC
# ==========================================
def resolve_host(service_name, default="localhost"):
    """
    Checks if we can see the Docker service name. 
    If not, we are likely running locally on a Mac.
    """
    try:
        socket.gethostbyname(service_name)
        return service_name
    except socket.gaierror:
        return default

# Detect correct hosts for the current environment
DB_HOST = resolve_host("postgres", "localhost")
QDRANT_HOST = resolve_host("qdrant", "localhost")

# ==========================================
# CONFIG
# ==========================================
TESTING = os.getenv("TESTING", "False") == "True"

# Use the auto-detected DB_HOST if DATABASE_URL is not explicitly set in .env
DEFAULT_DB_URL = f"postgresql://admin:adminpassword@{DB_HOST}:5432/financial_rag"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# Use the auto-detected QDRANT_HOST
QDRANT_URL = os.getenv("QDRANT_URL", f"http://{QDRANT_HOST}:6333")

COLLECTION_NAME = "financial_documents"
VECTOR_SIZE = 384  # must match BGE embedding model

# ==========================================
# SQLALCHEMY ENGINE
# ==========================================
if TESTING:
    engine = create_engine(
        os.getenv("DATABASE_URL", "sqlite:///./test_database.db"),
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
        print(f"✅ DB connection verified on host: {DB_HOST}")
    except Exception as e:
        print(f"⚠️ DB init skipped (likely using Alembic): {e}")

# ==========================================
# QDRANT INIT
# ==========================================
def init_qdrant():
    if TESTING:
        print("⏭️ Skipping Qdrant init (TESTING=True).")
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