"""
semantic_search/constants.py
------------------------------
Single source of truth for paths and model name.

Why this file exists:
  Ingest (embed_listings.py) and search (core.py / server.py) must use the SAME
  embedding model and the SAME Chroma collection. If they drift, query vectors
  live in a different space than stored vectors — results are meaningless.

You edit values here once; both scripts import from here.
"""

from pathlib import Path

# --- Repository layout ---------------------------------------------------------
# This file lives in semantic_search/; repo root is one level up.
_SEMANTIC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SEMANTIC_DIR.parent

# CSV created for your Florida price project; column "sanitized_text" holds MLS text.
CSV_PATH = _REPO_ROOT / "Florida Real Estate Sold Properties.csv"

# Chroma persists SQLite + index files under this folder (safe to delete to rebuild).
CHROMA_DIR = _SEMANTIC_DIR / "chroma_data"

# Name of the collection inside Chroma (one DB can hold many collections).
COLLECTION_NAME = "florida_listings"

# --- Embedding model -----------------------------------------------------------
# sentence-transformers loads this from Hugging Face (downloads on first use).
# all-MiniLM-L6-v2: small, fast, 384-dimensional vectors — good default for learning.
# Must match at ingest AND query time.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
