# .dockerignore
# .git
# .gitignore
# __pycache__
# *.pyc
# *.pyo
# .env
# .env.*
# *.log
# .DS_Store
# tests/
# docs/
# README.md
# .idea/
# .vscode/
# *.egg-info/
# dist/
# build/

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY rag_system.py .
COPY .env* ./

# Create directories for vector storage (mounted at runtime)
RUN mkdir -p /mnt/vectordb

EXPOSE 8000

CMD ["python", "rag_system.py"]