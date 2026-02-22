import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from datetime import datetime

# ==========================================
# 1. POSTGRESQL SETUP (Inside Docker)
# ==========================================
# This matches the credentials in your docker-compose.yml
POSTGRES_URL = "postgresql://admin:adminpassword@localhost:5432/financial_rag"

engine = create_engine(POSTGRES_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the Schema for our Semantic Cache
class CacheEntry(Base):
    __tablename__ = "semantic_cache"

    id = Column(Integer, primary_key=True, index=True)
    query_hash = Column(String, unique=True, index=True) 
    user_query = Column(Text)
    llm_response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create the tables in the Postgres container
try:
    Base.metadata.create_all(bind=engine)
    print("✅ PostgreSQL: 'semantic_cache' table verified/created successfully.")
except Exception as e:
    print(f"❌ PostgreSQL Connection Error: {e}")

# ==========================================
# 2. QDRANT SETUP (Inside Docker)
# ==========================================
# Connect to the Qdrant REST API exposed on port 6333
try:
    qdrant = QdrantClient(url="http://localhost:6333")
    collection_name = "financial_documents"

    # Check if collection exists, if not, create it
    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"✅ Qdrant: Created new collection '{collection_name}'.")
    else:
        print(f"✅ Qdrant: Collection '{collection_name}' already exists.")
except Exception as e:
    print(f"❌ Qdrant Connection Error: {e}")

print("🚀 Infrastructure initialization complete!")