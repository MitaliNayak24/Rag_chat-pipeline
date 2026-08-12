# ── RAG Pipeline — Dockerfile ─────────────────────────────────
# Optimised for Render.com deployment

FROM python:3.11-slim

# System deps needed by some LangChain/chroma packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create default ChromaDB directory
RUN mkdir -p /data/chroma

# Expose port (Render injects $PORT at runtime)
EXPOSE 8000

# Start the server
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
