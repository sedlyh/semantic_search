"""
FastAPI search API for the Next.js front end (Vercel) and other clients.

Run from repo root (machine_learning/):
    uvicorn semantic_search.server:app --reload --host 0.0.0.0 --port 8000

Environment:
    ALLOW_ORIGINS — comma-separated CORS origins (default: http://localhost:3000)
    PORT — listen port (App Runner, Render, etc.); use scripts/run_api.sh on App Runner
    CHROMA_HTTP_URL — optional HTTPS URL to a .zip of chroma_data/ if the index is not in the image
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_sem = Path(__file__).resolve().parent
if str(_sem) not in sys.path:
    sys.path.insert(0, str(_sem))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from constants import CHROMA_DIR, COLLECTION_NAME, DEFAULT_MODEL
from core import MAX_K, get_collection, search_listings, vector_store_ready

_DEFAULT_ORIGINS = "http://localhost:3000"


def _cors_origins() -> list[str]:
    raw = os.environ.get("ALLOW_ORIGINS", _DEFAULT_ORIGINS)
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(title="Florida MLS semantic search", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search")
    k: int = Field(default=5, ge=1, le=MAX_K)
    model_id: str | None = Field(default=None, description="Override embedding model")


class SearchHit(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None
    similarity: float | None


class SearchResponse(BaseModel):
    results: list[SearchHit]


@app.get("/health")
def health():
    """Liveness + whether the vector store is usable."""
    ready = vector_store_ready()
    if ready:
        try:
            get_collection()
            return {"status": "ok", "ready": True, "chroma_dir": str(CHROMA_DIR)}
        except Exception as e:
            return {
                "status": "degraded",
                "ready": False,
                "error": str(e),
                "chroma_dir": str(CHROMA_DIR),
            }
    return {
        "status": "degraded",
        "ready": False,
        "detail": "Run embed_listings.py to create chroma_data",
        "chroma_dir": str(CHROMA_DIR),
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not vector_store_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Vector store missing. Run embed_listings.py; expected {CHROMA_DIR}",
        )
    try:
        get_collection()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot open Chroma collection `{COLLECTION_NAME}`: {e}",
        ) from e

    mid = req.model_id or DEFAULT_MODEL
    hits = search_listings(req.query, k=req.k, model_id=mid)
    return SearchResponse(results=[SearchHit(**h) for h in hits])
