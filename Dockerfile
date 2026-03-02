FROM python:3.10-slim

# 1. Set environment variables early
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# 2. Group system dependencies and clean up in the same layer
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. Upgrade pip and install heavy ML dependencies first
# This layer is cached unless your requirements change
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 4. Copy ONLY requirements first to leverage Docker Cache
# If you only change your code, Docker will skip this slow install step
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir alembic schedule

# 5. Pre-download the Embedding Model (CRITICAL for AWS)
# This prevents the backend from timing out on the first request while trying to download the model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# 6. Copy the rest of the application code
COPY . .

# 7. Expose the port your FastAPI app runs on
EXPOSE 8001

# Command to run the application
CMD ["uvicorn", "main2:app", "--host", "0.0.0.0", "--port", "8001"]