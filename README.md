Here is your **edited, fixed, and properly formatted README**.
I removed duplication, fixed broken markdown, corrected code blocks, and improved structure while keeping your content intact.

````markdown
# 📈 Financial RAG System — Enterprise-Grade SEC Analyst

An enterprise-grade **Retrieval-Augmented Generation (RAG)** platform that autonomously ingests, processes, and analyzes SEC financial filings (**10-K, 10-Q, 8-K, S-1**).

Deployed on **AWS** with a fully decoupled, production-ready architecture featuring:

- Automated batch ingestion  
- Two-stage semantic retrieval  
- High-availability LLM routing with failover  
- Semantic caching for ultra-low latency  
- End-to-end observability and tracing  

Designed for **institutional-grade financial research workflows**.

---

# 🚀 Core Capabilities

## 📄 Multi-Filing SEC Intelligence
Automatically ingests and processes:

- Annual Reports (**10-K**)  
- Quarterly Reports (**10-Q**)  
- Material Events (**8-K**)  
- IPO Prospectuses (**S-1**)  

Enabling comprehensive, cross-document financial analysis.

---

## ⚡ Automated Batch Ingestion
- Scheduled **EventBridge/Cron jobs** monitor the SEC EDGAR feed  
- New filings are:
  1. Downloaded to an **S3 data lake**
  2. Parsed and cleaned
  3. Chunked and vectorized
  4. Indexed into **Qdrant**

Fully asynchronous and fault-tolerant.

---

## 🧠 Two-Stage Retrieval (Hallucination Control)
1. **Bi-Encoder — SentenceTransformers (`all-MiniLM-L6-v2`)** → High-recall vector search  
2. **Cross-Encoder — MS MARCO (`ms-marco-MiniLM-L-6-v2`)** → Precision reranking of top-K results  

Significantly improves factual grounding and reduces hallucinations.

---

## 🔁 High-Availability LLM Routing
Intelligent API router with automatic failover:

- Primary: **Llama-3 via Groq**  
- Fallback: **OpenAI API**  
- Optional: **Local Ollama**  

Ensures **zero downtime** during rate limits or provider outages.

---

## ⚡ Semantic Caching Layer
- SHA-256 exact-match + semantic similarity detection  
- Backed by **PostgreSQL**  
- Reduces latency from **~7.0s → ~7ms** for repeated queries  
- Cuts LLM cost and token usage dramatically

---

## 📊 Observability & Tracing
Integrated **MLflow + OpenTelemetry**:

- Token usage tracking  
- Latency breakdown (retrieval → rerank → generation)  
- Execution waterfalls  
- Model performance metrics  

Production-grade monitoring for LLM systems.

---

# ☁️ Cloud Architecture (AWS)

| Layer       | Service            | Purpose                                      |
|------------|--------------------|----------------------------------------------|
| Compute    | EC2                | FastAPI backend, Streamlit frontend, Qdrant |
| Storage    | S3                 | Immutable SEC data lake                     |
| Scheduling | EventBridge / Cron | Automated ingestion pipeline                |
| Containers | Docker + Compose   | Reproducible deployment                     |

---

# 🧱 Tech Stack

### Backend
- FastAPI  
- Uvicorn  
- Pydantic  

### Frontend
- Streamlit  

### Databases
- Qdrant (Vector DB)  
- PostgreSQL (Semantic Cache)  

### AI / ML
- sentence-transformers  
- Cross-Encoders (MS MARCO)  
- Groq API (Llama-3)  
- OpenAI API (fallback)  

### Data Ingestion
- sec-edgar-downloader  
- BeautifulSoup4  
- LangChain text splitters  

### Observability
- MLflow  
- OpenTelemetry  

### Infrastructure
- AWS EC2  
- AWS S3  
- Docker / Docker Compose  

---

# ⚙️ Setup & Deployment

## 1️⃣ Clone Repository

```bash
git clone https://github.com/pythonmailer/financial-rag-system.git
cd financial-rag-system
pip install -r requirements.txt
````

---

## 2️⃣ Configure Environment Variables

Create a `.env` file in the project root:

```env
# AWS
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=your_sec_data_lake

# LLM Routing
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_fallback_openai_key
```

---

## 3️⃣ Launch the Cloud Infrastructure

```bash
docker-compose up -d
```

---

## 4️⃣ Start the Application Services

### Terminal 1 — Tracing

```bash
mlflow ui --port 5001
```

### Terminal 2 — Backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Terminal 3 — Frontend

```bash
streamlit run frontend.py --server.port 8501
```

---

# 👤 Author

**Chirag Gupta**

```

If you want, next I can add:

- GitHub badges  
- Architecture diagram section  
- API endpoints documentation  
- Benchmarks (latency, cost savings, recall@k)  
- Screenshots/GIF demo section
```
