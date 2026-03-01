FROM python:3.10-slim

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip
RUN pip install --upgrade pip

# 3. CRITICAL: Install Torch CPU-ONLY FIRST
# This ensures that subsequent requirement installs don't pull the 5GB GPU version
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining requirements
COPY requirements.txt .
# --no-cache-dir is essential to keep your 20GB disk from filling up
RUN pip install --no-cache-dir -r requirements.txt

# 5. Install small utilities
RUN pip install --no-cache-dir schedule

# 6. Copy code
COPY . .

ENV PYTHONUNBUFFERED=1