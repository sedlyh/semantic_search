"""
Shared search logic for Streamlit, FastAPI, and tests.

Both ingest (embed_listings.py) and this module MUST use the same model
with normalize_embeddings=True. If they drift, cosine distances are meaningless
because the vectors live in different spaces.
"""

from __future__ import annotations

from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from constants import CHROMA_DIR, COLLECTION_NAME, DEFAULT_MODEL

MAX_K = 20

_collection = None


def vector_store_ready() -> bool:
    """True when chroma_data/ exists (ingest has been run at least once)."""
    return CHROMA_DIR.is_dir()


def get_collection():
    """Lazy-open the Chroma persistent client and return the collection."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


@lru_cache(maxsize=4)
def _get_model(model_id: str) -> SentenceTransformer:
    """Load (and cache) the embedding model so repeat queries skip download."""
    return SentenceTransformer(model_id)


def search_listings(
    query: str,
    k: int = 5,
    model_id: str | None = None,
) -> list[dict]:
    """
    Encode *query* into the same vector space as stored listings, then ask
    Chroma for the k nearest neighbors.

    Returns JSON-serializable dicts:
        id, text, metadata, distance, similarity  (similarity = 1 - distance)
    """
    q = query.strip()
    if not q:
        return []

    mid = model_id or DEFAULT_MODEL
    k = max(1, min(int(k), MAX_K))

    # Encode the query with the SAME normalization used at ingest time.
    model = _get_model(mid)
    q_emb = model.encode(
        [q],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    raw = get_collection().query(
        query_embeddings=q_emb.tolist(),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    ids = raw["ids"][0]
    docs = raw["documents"][0]
    metas = raw["metadatas"][0]
    dists = raw["distances"][0]

    out: list[dict] = []
    for doc_id, text, meta, dist in zip(ids, docs, metas, dists, strict=True):
        d = float(dist) if dist is not None else None
        sim = (1.0 - d) if d is not None else None
        safe_meta = {str(k): v for k, v in (meta or {}).items() if v is not None}
        out.append(
            {
                "id": doc_id,
                "text": text or "",
                "metadata": safe_meta,
                "distance": d,
                "similarity": sim,
            }
        )
    return out
