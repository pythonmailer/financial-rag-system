# 📈 Financial RAG System — High-Performance SEC Analyst

An institutional-grade **Retrieval-Augmented Generation (RAG)** platform designed to ingest, process, and analyze SEC financial filings (**10-K, 10-Q, 8-K**) with ultra-low latency.

This system has been re-engineered for **high-concurrency**, moving from a ~30s response time to **~2.7s for 10 simultaneous requests** through advanced asynchronous multiplexing and stream-batching.

---

# 🚀 Key Innovations & Capabilities

### ⚡ Stream-Batched Architecture (Performance Leader)

Unlike standard RAG pipelines that process requests sequentially, this system uses a **decoupled processing engine**:

* **Dynamic Batching**: Groups incoming queries into 500ms windows to perform heavy vector embedding math once for the entire group, saving significant CPU resources.
* **Independent Task Racing**: Once embeddings are ready, individual requests "sprint" independently through retrieval and LLM generation. Fast models (8B) are never held hostage by slow models (70B).
* **Thread Offloading**: Heavy CPU math (Reranking/Embedding) is offloaded to background threads, keeping the FastAPI event loop 100% responsive for instant cache hits.

### 🤖 Automated Midnight Ingestion

* **Autonomous Worker**: A dedicated scheduler monitors the SEC EDGAR feed every midnight.
* **Smart Updates**: Automatically detects, downloads, and vectorizes new AAPL filings only if they haven't been processed yet, ensuring your analyst always has the latest data.

### 🧠 Two-Stage Semantic Retrieval

1. **Bi-Encoder (`all-MiniLM-L6-v2`)**: Performs high-speed semantic search across the Qdrant vector database.
2. **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)**: Reranks the top results to ensure the LLM receives only the most contextually relevant financial data, drastically reducing hallucinations.

### 🔁 Multi-Model Failover & Routing

* **Semantic Router**: A "Traffic Cop" Llama-3 8B model classifies queries as **SIMPLE** (extraction) or **COMPLEX** (reasoning) to route them to the most efficient engine.
* **Waterfall Fallback**: If the primary engine (Groq) is rate-limited, the system automatically fails over to **Gemini 1.5 Flash** and then **OpenRouter**.

### ⚡ Semantic Caching

* **PostgreSQL Backend**: Stores exact and semantic matches of previous queries.
* **Latency Win**: Reduces response time for repeated queries from **~2.7s to ~7ms**.

---

# 📊 Observability

Integrated **MLflow + OpenTelemetry** provides a professional-grade audit trail:

* **Execution Waterfalls**: Visualize exactly how long was spent in retrieval vs. reranking vs. generation.
* **Trace Context**: Full visibility into the prompt context fed to the LLM and the raw metadata returned.

---

# 🧱 Tech Stack

| Component | Technology |
| --- | --- |
| **Backend** | FastAPI (Asynchronous), Uvicorn, Pydantic |
| **Frontend** | Streamlit |
| **Vector DB** | Qdrant |
| **Cache DB** | PostgreSQL |
| **LLMs** | Llama-3 (Groq), Gemini 1.5 Flash, OpenRouter |
| **Embedding** | sentence-transformers, Cross-Encoders |
| **Automation** | Python `schedule`, sec-edgar-downloader |
| **Observability** | MLflow, OpenTelemetry |
| **Infrastructure** | Docker, Docker Compose, AWS EC2/S3 |

---

# ⚙️ Setup & Deployment

### 1️⃣ Configure Environment

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key
OPENROUTER_API_KEY=your_key

```

### 2️⃣ Launch the Entire Stack

The system is fully containerized. One command starts the databases, the observability UI, the backend, the frontend, and the automated scheduler:

```bash
docker-compose up --build -d

```

### 3️⃣ Access Services

* **Analyst UI**: `http://localhost:8501`
* **API Docs**: `http://localhost:8001/docs`
* **Observability (MLflow)**: `http://localhost:5001`

---

**Author: Chirag Gupta**
