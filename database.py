import os
import time
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Index,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# ==========================================
# 🌐 INFRA CONFIG
# ==========================================
AWS_IP = "13.232.197.229"

QDRANT_URL = os.getenv("QDRANT_URL", f"http://{AWS_IP}:6333")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://admin:adminpassword@{AWS_IP}:5432/financial_rag",
)

COLLECTION_NAME = "financial_documents"
VECTOR_SIZE = 384

TESTING = os.getenv("TESTING", "False") == "True"
USE_ALEMBIC = os.getenv("USE_ALEMBIC", "False") == "True"

# ==========================================
# SQLALCHEMY ENGINE
# ==========================================
if TESTING:
    engine = create_engine(
        "sqlite:///./test_database.db",
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

    def __repr__(self):
        return f"<Feedback {self.query_hash} rating={self.rating}>"


class CacheEntry(Base):
    __tablename__ = "semantic_cache"

    id = Column(Integer, primary_key=True, index=True)
    query_hash = Column(String, unique=True, index=True)
    user_query = Column(Text)
    llm_response = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ticker = Column(String, index=True)
    provider = Column(String, nullable=True)

    __table_args__ = (
        Index("idx_ticker_query", "ticker", "query_hash"),
    )

    def __repr__(self):
        return f"<Cache {self.ticker}:{self.query_hash[:8]}>"

# ==========================================
# DB INIT
# ==========================================
def init_db():
    if TESTING:
        Base.metadata.create_all(bind=engine)
        print("✅ SQLite test DB initialized")
        return

    if USE_ALEMBIC:
        print("⚠️ Alembic enabled → skipping SQLAlchemy create_all")
        return

    try:
        Base.metadata.create_all(bind=engine)
        print(f"✅ DB tables ready at {DATABASE_URL.split('@')[-1]}")
    except Exception as e:
        print(f"❌ DB init failed: {e}")
        raise

# ==========================================
# QDRANT INIT
# ==========================================
def init_qdrant():
    if TESTING:
        print("🧪 Skipping Qdrant init (TESTING=True)")
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
                print(f"✅ Qdrant collection created at {QDRANT_URL}")
            else:
                print(f"✅ Qdrant collection exists at {QDRANT_URL}")

            return

        except Exception as e:
            print(
                f"⏳ Qdrant not ready ({attempt}/{retries}) → {str(e)}"
            )
            time.sleep(delay)

    raise RuntimeError("❌ Could not connect to Qdrant")

# ==========================================
# STARTUP HOOK
# ==========================================
if __name__ == "__main__":
    init_db()
    init_qdrant()