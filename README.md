# 📈 Financial RAG System — High-Performance SEC Analyst

An institutional-grade **Retrieval-Augmented Generation (RAG)** platform for ingesting, indexing, and analyzing SEC filings with ultra-low latency and high concurrency. Optimized for **stream batching + async multiplexing**, reducing latency from ~30s (sequential RAG) to **~2.7s for 10 concurrent queries**.

---

# 🚀 Key Capabilities

### ⚡ Stream-Batched Inference Engine

Traditional RAG pipelines process queries sequentially. This system introduces a **decoupled batch + async execution model**:

* **Dynamic Batching (Max 32):** Shared embedding computation across requests within a 50ms window to optimize GPU/CPU utilization.
* **Independent Async Execution:** Fast queries are never blocked by slow LLM calls using `asyncio.Queue` and `contextvars`.
* **Thread Offloading:** CPU-heavy embedding & reranking run in separate threads to keep the FastAPI event loop responsive.
* **Instant Cache Hits:** SHA-256 hashed queries bypass the batch queue for sub-10ms response times.

### 🧠 Two-Stage Retrieval & Source Citation

1. **Bi-Encoder (`BAAI/bge-small-en-v1.5`):** State-of-the-art semantic search in **Qdrant**.
2. **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`):** Reranks top 15 candidates down to the `top_k` most relevant context chunks.
3. **Document Lineage:** Every response includes precise document source citations (Ticker, Score, and Text Chunks) for auditability.

### 🔁 Enterprise Resilience & LLM Routing

A lightweight heuristic router classifies queries as **SIMPLE** or **COMPLEX** to optimize cost vs. quality:

* **SIMPLE:** Routed to `llama-3.1-8b-instant`.
* **COMPLEX:** Routed to `llama-3.3-70b-versatile`.
* **Atomic Circuit Breaker:** Protects against Groq rate-limits. If a failure is detected, the system trips a cooldown period to protect system stability.

---

# ⚡ Exact-Match Cache & MLflow Observability

### PostgreSQL Semantic Cache

Queries are cryptographically hashed: `hash(ticker + normalized_query)`.

* **Targeted Invalidation:** Specific endpoints allow clearing the cache for a single ticker when new data is ingested.
* **Ground Truth Dataset:** The system logs every query and response to PostgreSQL, creating a dataset for future fine-tuning.

### GenAI Observability (MLflow 2.14+)

The system features a **customized GenAI Trace waterfall**, allowing you to inspect:

* **Retrieval Quality:** View exactly which chunks were pulled from Qdrant in the "Documents" tab.
* **Model Transparency:** Track which provider (Groq, Gemini, etc.) answered each query.
* **Latency Spikes:** Precise millisecond tracking for Embedding, Routing, Retrieval, Reranking, and LLM Generation.

---

# 🧪 Performance Metrics

| Metric | Performance |
| --- | --- |
| **Accuracy (Hit@5)** | **100.0%** |
| **Precision (MRR)** | **0.767** |
| **Concurrent Latency (10 queries)** | **~2.7s** |
| **Cached Query Latency** | **~7ms** |

---

# 🧱 Tech Stack

| Layer | Technology |
| --- | --- |
| **Backend** | FastAPI (async), Uvicorn, SQLAlchemy |
| **Vector DB** | Qdrant |
| **Relational DB** | PostgreSQL |
| **Models** | BGE-Small (Embedding), MS-Marco (Reranking), Llama 3.3/3.1 (LLM) |
| **Observability** | **MLflow (Tracing & Metrics)** |
| **Infra** | Docker Compose, AWS EC2 |

---

# ⚙️ Quick Start

### 1. Configure Environment

Create a `.env` file:

```env
GROQ_API_KEY=your_key
DATABASE_URL=postgresql://admin:adminpassword@postgres:5432/financial_rag
MLFLOW_TRACKING_URI=http://mlflow:5001

```

### 2. Launch Stack

```bash
docker-compose up --build -d

```

---

# 👨‍💻 Author

**Chirag Gupta**
AI Systems • LLM Infrastructure • High-Performance RAG