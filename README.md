# 📈 Financial RAG System — High-Performance SEC Analyst

An institutional-grade **Retrieval-Augmented Generation (RAG)** platform for ingesting, indexing, and analyzing SEC filings (**10-K, 10-Q, 8-K**) with ultra-low latency and high concurrency.

The system is optimized for **stream batching + async multiplexing**, reducing latency from ~30s (sequential RAG) to **~2.7s for 10 concurrent queries**, while maintaining high answer quality through cross-encoder reranking, precise document source citation, and a verifiable evaluation pipeline.

---

# 🚀 Key Capabilities

### ⚡ Stream-Batched Inference Engine

Traditional RAG pipelines process queries sequentially. This system introduces a **decoupled batch + async execution model**:

* **Dynamic Batching (Max 32):** Shared embedding computation across requests within a 50–500ms window to prevent OOM errors and latency spikes during traffic bursts.
* **Independent Async Execution:** Fast queries are never blocked by slow LLM calls.
* **Thread Offloading:** CPU-heavy embedding & reranking run outside the FastAPI event loop.
* **Instant Cache Hits:** Bypass the batch queue entirely.

### 🧠 Two-Stage Retrieval & Source Citation

1. **Bi-Encoder (`all-MiniLM-L6-v2`):** High-speed semantic search in **Qdrant**.
2. **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`):** Reranks top candidates to deliver highly relevant financial context.
3. **Document Lineage:** Vectors are tagged with SEC filing types (`10-K`, `10-Q`). The API forces the LLM to explicitly cite its sources (e.g., *"According to the Q3 10-Q..."*).

### 🔁 Enterprise Resilience & Cost-Aware Routing

A lightweight LLM router classifies queries as **SIMPLE** (Llama-3.1-8B) or **COMPLEX** (Llama-3.3-70B). To guarantee **99.9% uptime**, the system employs a two-tier resilience layer:

1. **Exponential Backoff (`tenacity`):** Automatically retries transient network errors.
2. **Atomic Circuit Breaker:** If the primary provider (Groq) rate-limits or fails, an atomic breaker triggers a cooldown. Traffic instantly falls back to:
* ➡️ **Gemini 2.0 Flash/Pro** (Primary Fallback)
* ➡️ **OpenRouter Llama-3-8B** (Secondary Fallback)



---

# ⚡ Exact-Match Cache & Implicit RLHF Loop (PostgreSQL)

Queries are cryptographically hashed for instantaneous retrieval: `hash(ticker + normalized_query + document_type)`.

* **Targeted Invalidation:** When the background worker ingests a new filing (e.g., AAPL), it triggers a REST webhook to surgically wipe *only* that company's stale cache.
* **Preference Dataset (RLHF):** The UI includes a Thumbs Up/Down feedback mechanism tied directly to the `query_hash`, silently building a ground-truth dataset for future model fine-tuning.

---

# 🧪 Performance & Evaluation

Includes a dedicated `evaluate.py` testing suite to mathematically verify retrieval accuracy.

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
| **Backend** | FastAPI (async), Uvicorn, Pydantic |
| **Frontend** | Streamlit |
| **Vector DB** | Qdrant |
| **Relational DB** | PostgreSQL (Caching & Feedback) |
| **LLMs** | Groq Llama-3, Gemini 2.0 Flash, OpenRouter |
| **Observability** | MLflow, OpenTelemetry |
| **Infra** | Docker, Docker Compose, AWS EC2 |

---

# ⚙️ Quick Start

### 1. Configure Environment

Create a `.env` file:

```env
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key
OPENROUTER_API_KEY=your_key
HF_TOKEN=your_key

```

### 2. Launch Stack

```bash
docker-compose up --build -d

```

### 3. Access Services

* **Analyst UI:** `http://localhost:8501`
* **API Docs:** `http://localhost:8001/docs`
* **MLflow UI:** `http://localhost:5001`

---

# 👨‍💻 Author

**Chirag Gupta**
AI Systems • LLM Infrastructure • High-Performance RAG