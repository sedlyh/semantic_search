# API container: build after `chroma_data/` exists (run embed_listings locally or in CI).
# Example (from repo root machine_learning/):
#   docker build --platform linux/amd64 -t fl-search-api -f semantic_search/Dockerfile .
#   docker run -p 8000:8000 -e ALLOW_ORIGINS=http://localhost:3000 fl-search-api

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (avoids pulling ~2 GB of CUDA libraries).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY semantic_search/requirements.txt /app/semantic_search/requirements.txt
RUN pip install --no-cache-dir -r /app/semantic_search/requirements.txt

COPY semantic_search/constants.py \
     semantic_search/core.py \
     semantic_search/server.py \
     semantic_search/__init__.py \
     /app/semantic_search/

# Persisted index from embed_listings (must be present in build context).
COPY semantic_search/chroma_data /app/semantic_search/chroma_data

# Pre-download the HF model so the first request doesn't stall.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "semantic_search.server:app", "--host", "0.0.0.0", "--port", "8000"]
