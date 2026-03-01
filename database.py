import os
import time
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from datetime import datetime, timezone

POSTGRES_URL = "postgresql://admin:adminpassword@postgres:5432/financial_rag"

engine = create_engine(POSTGRES_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class FeedbackEntry(Base):
    __tablename__ = "user_feedback"
    id = Column(Integer, primary_key=True, index=True)
    query_hash = Column(String, index=True)
    rating = Column(Integer)  # 1 for Thumbs Up, -1 for Thumbs Down
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

MAX_RETRIES = 5
RETRY_DELAY = 3

for attempt in range(MAX_RETRIES):
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Successfully connected to PostgreSQL and verified tables.")
        break
    except Exception as e:
        print(f"⏳ Waiting for PostgreSQL to be ready... (Attempt {attempt + 1}/{MAX_RETRIES})")
        time.sleep(RETRY_DELAY)
else:
    print("❌ CRITICAL: Could not connect to PostgreSQL after multiple attempts.")

for attempt in range(MAX_RETRIES):
    try:
        qdrant = QdrantClient(url="http://qdrant:6333")
        collection_name = "financial_documents"
        
        if not qdrant.collection_exists(collection_name):
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            print(f"✅ Qdrant collection '{collection_name}' created successfully.")
        else:
            print(f"✅ Connected to Qdrant. Collection '{collection_name}' already exists.")
        break
    except Exception as e:
        print(f"⏳ Waiting for Qdrant to be ready... (Attempt {attempt + 1}/{MAX_RETRIES})")
        time.sleep(RETRY_DELAY)
else:
    print("❌ CRITICAL: Could not connect to Qdrant after multiple attempts.")