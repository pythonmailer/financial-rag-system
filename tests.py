import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import your app and DB dependencies
from main import app, get_db
from database import Base

# 1. Setup a temporary, local SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_database.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the tables in the test database
Base.metadata.create_all(bind=engine)

# 2. Override the get_db dependency
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Tell FastAPI: "Whenever you see Depends(get_db), use override_get_db instead!"
app.dependency_overrides[get_db] = override_get_db

# 3. Initialize the Test Client
client = TestClient(app)

# ==========================================
# TEST CASES
# ==========================================

def test_submit_feedback():
    """Test if the feedback endpoint successfully writes to the DB"""
    payload = {"query_hash": "test_hash_123", "rating": 1}
    response = client.post("/feedback", json=payload)
    
    assert response.status_code == 200
    assert response.json() == {"status": "success", "recorded_rating": 1}

def test_clear_semantic_cache():
    """Test if the cache clearing endpoint works for a specific ticker"""
    response = client.delete("/cache/clear/AAPL")
    
    assert response.status_code == 200
    assert response.json()["ticker"] == "AAPL"
    assert response.json()["status"] == "success"

def test_ask_endpoint_validation_error():
    """Test if the API correctly blocks requests missing required fields"""
    # Missing 'ticker'
    payload = {"query": "What is the revenue?"}
    response = client.post("/ask", json=payload)
    
    # 422 Unprocessable Entity is FastAPI's default error for bad Pydantic schemas
    assert response.status_code == 422