# 📈 Financial RAG System: Enterprise-Grade SEC Analyst
An enterprise-grade Retrieval-Augmented Generation (RAG) system designed to autonomously ingest, process, and analyze SEC financial filings (10-K, 10-Q, 8-K, and S-1). Hosted on AWS, this project implements a fully decoupled architecture featuring automated batch processing, two-stage retrieval, semantic caching, and highly available LLM routing.

# 🚀 Key Features
1. Multi-Document SEC Pipeline: Automatically ingests and processes Annual Reports (10-K), Quarterly Reports (10-Q), Material Events (8-K), and IPO Prospectuses (S-1) for comprehensive financial analysis.

2. Automated Batch Processing: Scheduled chron jobs continuously monitor the SEC EDGAR database for new filings, automatically downloading raw data to an S3 data lake and triggering asynchronous vectorization into Qdrant.

3. High-Availability LLM Routing: Implements an intelligent API routing and fallback mechanism. If the primary generation model (e.g., Llama-3 via Groq) experiences rate limits or downtime, the system gracefully falls back to a secondary provider (e.g., OpenAI or local Ollama) ensuring zero downtime.

4. Semantic Caching Layer: Bypasses expensive LLM generation for repeated or semantically similar queries using an exact-match SHA-256 hashing function backed by PostgreSQL. Drops response latency from ~7.0 seconds to ~0.007 seconds.

5. Two-Stage Retrieval: Eliminates LLM hallucinations by using a SentenceTransformer (all-MiniLM-L6-v2) for broad vector search, followed by a Cross-Encoder (ms-marco-MiniLM-L-6-v2) for strict semantic reranking of the top 50 documents.

6. Observability & Tracing: Integrated MLflow and OpenTelemetry automatically trace LLM token usage, backend latency, and execution waterfalls.

# ☁️ Cloud Architecture (AWS)

1. Compute (Amazon EC2): Hosts the decoupled FastAPI backend, Streamlit frontend, and the open-source Qdrant vector database.

2. Data Lake (Amazon S3): Serves as the immutable storage layer for all raw, scraped SEC HTML documents and system logs.

3. Event Scheduling (EventBridge / Cron): Triggers the automated ingestion pipeline to run at scheduled intervals, keeping the vector database perfectly synced with Wall Street filings.

# 🛠️ Technology Stack
Backend & API: FastAPI, Uvicorn, Pydantic

Frontend: Streamlit

Databases: Qdrant (Vector DB), PostgreSQL (Semantic Cache)

AI & ML: HuggingFace sentence-transformers, Groq API, OpenAI API

Data Ingestion: sec-edgar-downloader, BeautifulSoup4, LangChain Text Splitters

Observability: MLflow, OpenTelemetry

Infrastructure: AWS EC2, AWS S3, Docker, Docker Compose


# ⚙️ Setup & Deployment
1. Clone the repository and install dependencies

Bash

git clone https://github.com/pythonmailer/financial-rag-system.git

cd financial-rag-system

pip install -r requirements.txt


2. Set up environment variables

Create a .env file in the root directory:

AWS Credentials

AWS_ACCESS_KEY_ID=your_access_key

AWS_SECRET_ACCESS_KEY=your_secret_key

S3_BUCKET_NAME=your_sec_data_lake

LLM Routing Keys

GROQ_API_KEY=your_groq_api_key

OPENAI_API_KEY=your_fallback_openai_key

3. Launch the Cloud Infrastructure

Bash

docker-compose up -d

4. Start the Application Services

Bash

Terminal 1 (Tracing)

mlflow ui --port 5001

Terminal 2 (Backend)

uvicorn main:app --host 0.0.0.0 --port 8000

Terminal 3 (Frontend)

streamlit run frontend.py --server.port 8501

# 👤 Author

Chirag Gupta