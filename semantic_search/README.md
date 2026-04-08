# Semantic search (portfolio stack)

Florida MLS listing descriptions → embeddings (sentence-transformers) → ChromaDB → **FastAPI** or **Streamlit**. A **Next.js** app on Vercel calls the API over HTTPS.

## Why two deployments?

Vercel runs the Next.js front end. **Chroma, PyTorch, and the embedding model do not run on Vercel serverless** in this setup. The Python API runs on any host that supports long-lived processes and enough RAM (Railway, Render, Fly.io, your laptop, or the included Dockerfile).

```mermaid
flowchart LR
  Vercel[Vercel_Next.js]
  API[FastAPI_host]
  Vercel -->|POST_/search| API
```

## Local development

### 1. Python dependencies

From repo root `machine_learning/`:

```bash
python -m pip install -r semantic_search/requirements.txt
```

### 2. Build the vector index (once, or after changing data/model)

```bash
python semantic_search/embed_listings.py --limit 200   # quick test
# or full dataset:
python semantic_search/embed_listings.py
```

This writes `semantic_search/chroma_data/`.

### 3. Run the API

```bash
cd /path/to/machine_learning
ALLOW_ORIGINS=http://localhost:3000 uvicorn semantic_search.server:app --reload --host 0.0.0.0 --port 8000
```

- `GET http://127.0.0.1:8000/health` — liveness and vector-store status  
- `POST http://127.0.0.1:8000/search` — JSON body `{"query":"...","k":5}`

### 4. Run the Next.js app

```bash
cd semantic_search/web
cp .env.local.example .env.local
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The default API URL in code is `http://127.0.0.1:8000`.

### 5. Streamlit (optional, same backend logic)

```bash
streamlit run semantic_search/search_app.py
```

Shared search logic lives in `semantic_search/core.py`.

## Environment variables

- **FastAPI — `ALLOW_ORIGINS`:** comma-separated CORS origins, e.g. `http://localhost:3000,https://your-app.vercel.app`. Restart the API after changes.
- **Next.js (`.env.local`) — `NEXT_PUBLIC_SEARCH_API_URL`:** API base URL, no trailing slash, e.g. `https://api.example.com`.
- **Vercel project settings — `NEXT_PUBLIC_SEARCH_API_URL`:** same value, pointing at your deployed FastAPI HTTPS URL.

## Deploy Next.js to Vercel

1. Connect the Git repo to Vercel.  
2. Set **Root Directory** to `semantic_search/web`.  
3. Add environment variable `NEXT_PUBLIC_SEARCH_API_URL` = your public API base URL.  
4. Deploy.

## Deploy the API (examples)

- **Railway / Render / Fly:** Python service, start command  
  `uvicorn semantic_search.server:app --host 0.0.0.0 --port $PORT`  
  (set `PORT` if the platform provides it).  
- Copy or generate `chroma_data` on the server (run `embed_listings.py` in build or mount a volume).  
- Set `ALLOW_ORIGINS` to your Vercel URL(s).

### Docker

Build from **repo root** `machine_learning/` (so paths match the Dockerfile):

```bash
python semantic_search/embed_listings.py   # ensure chroma_data exists
docker build -t fl-search-api -f semantic_search/Dockerfile .
docker run -p 8000:8000 -e ALLOW_ORIGINS=http://localhost:3000 fl-search-api
```

The first real query may download Hugging Face model weights into the container unless you bake a cache layer.

## Project layout

- `constants.py` — paths, default model, collection name  
- `embed_listings.py` — CSV → embeddings → Chroma  
- `core.py` — shared `search_listings()`  
- `server.py` — FastAPI app  
- `search_app.py` — Streamlit UI  
- `web/` — Next.js (Vercel)  

## API shapes

**POST /search**

```json
{ "query": "pool renovated kitchen", "k": 5, "model_id": null }
```

**Response**

```json
{
  "results": [
    {
      "id": "row_0",
      "text": "...",
      "metadata": { "lastSoldPrice": 605000, "zip": "33446" },
      "distance": 0.12,
      "similarity": 0.88
    }
  ]
}
```
