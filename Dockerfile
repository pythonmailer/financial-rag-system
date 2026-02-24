FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for C compilers and Postgres libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install schedule for the automated worker
RUN pip install schedule 

COPY . .

# Ensure Python logs are sent straight to the terminal
ENV PYTHONUNBUFFERED=1