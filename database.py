import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from datetime import datetime, timezone

# Updated to use service name 'postgres' for Docker networking
POSTGRES_URL = "postgresql://admin:adminpassword@postgres:5432/financial_rag"

engine = create_engine(POSTGRES_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class CacheEntry(Base):
    __tablename__ = "semantic_cache"
    id = Column(Integer, primary_key=True, index=True)
    query_hash = Column(String, unique=True, index=True) 
    user_query = Column(Text)
    llm_response = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"❌ PostgreSQL Connection Error: {e}")

try:
    # Updated to use service name 'qdrant'
    qdrant = QdrantClient(url="http://qdrant:6333")
    collection_name = "financial_documents"
    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
except Exception as e:
    print(f"❌ Qdrant Connection Error: {e}")