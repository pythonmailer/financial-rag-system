# 📈 Financial RAG System — High-Performance SEC Analyst

An institutional-grade **Retrieval-Augmented Generation (RAG)** platform for ingesting, indexing, and analyzing SEC filings (**10-K, 10-Q, 8-K**) with ultra-low latency and high concurrency.

The system is optimized for **stream batching + async multiplexing**, reducing latency from ~30 s (sequential RAG) to **~2.7 s for 10 concurrent queries**, while maintaining high answer quality through cross-encoder reranking.

---

# 🚀 Key Capabilities

## ⚡ Stream-Batched Inference Engine

Traditional RAG pipelines process queries sequentially.
This system introduces a **decoupled batch + async execution model**:

* **Dynamic batching (500 ms window)** → shared embedding computation across requests
* **Independent async execution** → fast queries are never blocked by slow LLM calls
* **Thread offloading** → CPU-heavy embedding & reranking run outside the FastAPI event loop
* **Instant cache hits** → bypass the batch queue entirely

📊 Result: ~10× throughput improvement under concurrent load

---

## 🧠 Two-Stage Retrieval (Accuracy First)

1. **Bi-Encoder — `all-MiniLM-L6-v2`**
   High-speed semantic search in **Qdrant**

2. **Cross-Encoder — `ms-marco-MiniLM-L-6-v2`**
   Reranks top candidates to deliver highly relevant financial context

🎯 Outcome: Lower hallucination rate and tighter grounding

---

## 🔁 Cost-Aware Model Routing + Failover

A lightweight LLM router classifies queries as:

* **SIMPLE** → routed to **Llama-3.1-8B (Groq)** for low latency
* **COMPLEX** → routed to **Llama-3.3-70B (Groq)** for deep reasoning

If Groq is unavailable:

➡️ **Gemini 1.5 Flash** → fallback
➡️ **OpenRouter Llama-3-8B** → final fallback

This ensures **high availability + cost efficiency**.

---

## ⚡ Exact-Match Query Cache (PostgreSQL)

Queries are hashed as:

```
hash(ticker + normalized_query)
```

Repeated queries return instantly from cache:

⏱️ ~2.7 s → ~7 ms latency

(Cache is exact-match; semantic cache planned.)

---

## 🤖 Autonomous Filing Ingestion

An asynchronous background worker:

* Monitors the SEC EDGAR feed
* Detects new filings for configured tickers
* Downloads, chunks, embeds, and upserts into Qdrant

Only **new documents** are processed → no duplicate vectorization.

---

# 📊 Observability (MLflow + OpenTelemetry)

Every user query generates a **single MLflow trace** with nested spans:

```
USER_QUERY
 ├── Retrieval_and_Rerank
 ├── Semantic_Router
 └── LLM_Generation
```

### Logged Metrics & Tags

* `batch_size`
* `shared_embedding_latency_ms`
* router decision (SIMPLE / COMPLEX)
* provider used (Groq / Gemini / OpenRouter)
* retrieved chunk count
* final answer payload

This enables:

📈 Latency vs batch size analysis
📉 Model performance comparison
🧪 Prompt and context inspection

---

# 🧱 Tech Stack

| Layer         | Technology                                                      |
| ------------- | --------------------------------------------------------------- |
| Backend       | FastAPI (async), Uvicorn, Pydantic                              |
| Frontend      | Streamlit                                                       |
| Vector DB     | Qdrant                                                          |
| Cache DB      | PostgreSQL                                                      |
| Embeddings    | sentence-transformers                                           |
| Reranking     | Cross-Encoder (MS MARCO)                                        |
| LLMs          | Groq Llama-3.1-8B / Llama-3.3-70B, Gemini 1.5 Flash, OpenRouter |
| Observability | MLflow, OpenTelemetry                                           |
| Automation    | Async background worker, sec-edgar-downloader                   |
| Infra         | Docker, Docker Compose                                          |

---

# ⚙️ Local Deployment

## 1️⃣ Configure Environment

Create `.env`:

```env
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key
OPENROUTER_API_KEY=your_key
```

All fallbacks are optional — the system runs with Groq only.

---

## 2️⃣ Launch the Full Stack

```bash
docker compose up --build
```

This starts:

* FastAPI backend
* Streamlit frontend
* Qdrant vector database
* PostgreSQL cache
* MLflow observability server
* Async ingestion worker

---

## 3️⃣ Access Services

| Service    | URL                                                      |
| ---------- | -------------------------------------------------------- |
| Analyst UI | [http://localhost:8501](http://localhost:8501)           |
| API Docs   | [http://localhost:8001/docs](http://localhost:8001/docs) |
| MLflow UI  | [http://localhost:5001](http://localhost:5001)           |

---

# 📈 Performance Benchmarks

| Scenario                      | Latency                   |
| ----------------------------- | ------------------------- |
| Sequential RAG (baseline)     | ~30 s                     |
| Single query (batched engine) | ~2.4–2.8 s                |
| 10 concurrent queries         | ~2.7 s (shared embedding) |
| Cached query                  | ~7 ms                     |

---

# 🏗️ Architecture Highlights

* Decoupled batch engine with async task racing
* Cost-aware model routing
* Two-stage retrieval with cross-encoder reranking
* Exact-match query cache
* MLflow-based per-request tracing
* Autonomous SEC ingestion worker
* Fully containerized, horizontally scalable design

---

# 🔮 Roadmap

* Hybrid BM25 + vector retrieval
* Semantic (embedding) cache layer
* Token usage + cost logging per provider
* Recall@K evaluation pipeline
* GPU embedding worker pool
* Multi-ticker ingestion orchestration

---

# 👨‍💻 Author

**Chirag Gupta**
AI Systems • LLM Infrastructure • High-Performance RAG