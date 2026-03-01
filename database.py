import os
import time
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# ==========================================
# CONFIG
# ==========================================
TESTING = os.getenv("TESTING", "False") == "True"
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://admin:adminpassword@postgres:5432/financial_rag",
)
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
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
# DB INIT (ONLY FOR TESTING OR FIRST BOOT WITHOUT ALEMBIC)
# ==========================================
def init_db():
    """
    In production with Alembic, tables are created via migrations.
    This function is safe for:
    - TESTING=True
    - First boot fallback (if Alembic not used)
    """
    if TESTING:
        Base.metadata.create_all(bind=engine)
        print("✅ SQLite test database initialized.")
        return

    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Tables verified (fallback mode).")
    except Exception as e:
        print(f"⚠️ DB init skipped: {e}")

# ==========================================
# QDRANT INIT
# ==========================================
def init_qdrant():
    """
    Ensures Qdrant collection exists.
    Safe to call multiple times.
    """
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
                print(f"✅ Qdrant collection '{COLLECTION_NAME}' created.")
            else:
                print(f"✅ Qdrant collection '{COLLECTION_NAME}' exists.")

            return

        except Exception as e:
            print(
                f"⏳ Waiting for Qdrant... ({attempt}/{retries}) → {str(e)[:120]}"
            )
            time.sleep(delay)

    print("❌ CRITICAL: Could not connect to Qdrant after retries.")

# ==========================================
# SAFE STARTUP HOOKS
# ==========================================
if __name__ == "__main__":
    init_db()
    init_qdrant()