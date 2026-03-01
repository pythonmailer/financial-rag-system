import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure testing mode before importing app
os.environ["TESTING"] = "True"
os.environ["DATABASE_URL"] = "sqlite:///./test_database.db"

from main import app, get_db  # noqa: E402
from database import Base, CacheEntry  # noqa: E402

# ==========================================
# TEST DATABASE SETUP (SQLite)
# ==========================================
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_database.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

# ==========================================
# DEPENDENCY OVERRIDE
# ==========================================
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

# ==========================================
# FIXTURES
# ==========================================
@pytest.fixture(scope="function", autouse=True)
def clean_db():
    """Clean DB before each test for isolation."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield

# ==========================================
# TESTS
# ==========================================

def test_submit_feedback():
    payload = {"query_hash": "test_hash_123", "rating": 1}
    response = client.post("/feedback", json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_clear_semantic_cache_empty():
    response = client.delete("/cache/clear/AAPL")

    assert response.status_code == 200
    assert response.json()["cleared_entries"] == 0


def test_ask_validation_error():
    payload = {"query": "What is revenue?"}  # Missing ticker
    response = client.post("/ask", json=payload)

    assert response.status_code == 422


def test_cache_write_and_hit():
    """
    First request should write to cache.
    Second request should return cached response.
    """

    payload = {
        "query": "Test revenue question",
        "ticker": "AAPL",
        "top_k": 1
    }

    # First call → not cached
    response1 = client.post("/ask", json=payload)
    assert response1.status_code == 200
    assert response1.json()["cached"] is False

    # Verify cache row exists
    db = TestingSessionLocal()
    cached_rows = db.query(CacheEntry).all()
    db.close()
    assert len(cached_rows) == 1

    # Second call → cached
    response2 = client.post("/ask", json=payload)
    assert response2.status_code == 200
    assert response2.json()["cached"] is True
    assert response2.json()["provider"] == "Cache"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_queue_status_endpoint():
    response = client.get("/queue_status")
    assert response.status_code == 200
    assert response.json()["mode"] == "sequential"