"""
semantic_search/embed_listings.py
---------------------------------
Offline job: read listing descriptions from the CSV, embed each one, store
vectors + text + metadata in ChromaDB.

Run (from repo root, machine_learning/):
    python semantic_search/embed_listings.py

Quick test (few rows only):
    python semantic_search/embed_listings.py --limit 100

After this succeeds, run the Streamlit search app.
"""

from __future__ import annotations

# --- Imports -------------------------------------------------------------------
# argparse: CLI flags (--limit) without extra dependencies.
# sys / Path: so we can import constants when this file is run as a script.
import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from constants import CHROMA_DIR, COLLECTION_NAME, CSV_PATH, DEFAULT_MODEL

# How many texts to send through the model at once (memory vs speed tradeoff).
BATCH_SIZE = 64


# --- Helper: Chroma metadata must be JSON-serializable primitives ----------------
def _scalar_meta(val) -> str | int | float | bool:
    """
    Chroma rejects numpy types and NaN in metadata.
    Convert each field to str / int / float / bool / empty string.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return float(val) if isinstance(val, float) else int(val)
    return str(val)


# --- Main pipeline -------------------------------------------------------------
def main() -> None:
    # 1) Parse CLI: optional --model override, --limit for fast experiments.
    parser = argparse.ArgumentParser(description="Embed MLS texts into ChromaDB.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Must stay in sync with search_app.py / constants.py",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Embed only the first N rows (omit for full ~10k dataset).",
    )
    args = parser.parse_args()

    # 2) Fail fast if CSV path is wrong (typo, wrong working directory).
    if not CSV_PATH.is_file():
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    # 3) Load rows. utf-8-sig strips a BOM if present; low_memory=False avoids dtype warnings.
    print(f"Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", low_memory=False)

    if "sanitized_text" not in df.columns:
        raise SystemExit("Expected column 'sanitized_text' in CSV.")

    # 4) Keep only rows with usable text (empty descriptions cannot be embedded meaningfully).
    df = df[df["sanitized_text"].notna() & (df["sanitized_text"].astype(str).str.strip() != "")]
    df = df.reset_index(drop=True)

    if args.limit is not None:
        df = df.head(args.limit)

    n = len(df)
    print(f"Rows to embed: {n}")

    # 5) Load the sentence-transformer model (downloads weights on first run).
    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    # 6) Encode all descriptions into a matrix of shape (n, embedding_dim).
    #    normalize_embeddings=True → unit vectors, so cosine distance matches "angle" intuition.
    texts = df["sanitized_text"].astype(str).tolist()
    print("Encoding (this may take a few minutes on full data)…")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # 7) Open or create persistent Chroma storage on disk under CHROMA_DIR.
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # 8) If we re-run, replace the collection so we don't duplicate IDs.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # 9) cosine space: distance relates to angle between vectors (standard for normalized embeddings).
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 10) Build parallel lists: ids, vectors, raw text, metadata for UI / debugging.
    ids = [f"row_{i}" for i in range(n)]
    metadatas = []
    for _, row in df.iterrows():
        metadatas.append(
            {
                "lastSoldPrice": _scalar_meta(row.get("lastSoldPrice")),
                "listPrice": _scalar_meta(row.get("listPrice")),
                "sqft": _scalar_meta(row.get("sqft")),
                "beds": _scalar_meta(row.get("beds")),
                "baths": _scalar_meta(row.get("baths")),
                "zip": _scalar_meta(row.get("zip")),
                "type": _scalar_meta(row.get("type")),
            }
        )

    # 11) Add in chunks so one giant payload doesn't choke the client.
    chunk = 500
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Indexed {end}/{n}")

    print(f"Done. Vector store: {CHROMA_DIR}")
    print("Next: streamlit run semantic_search/search_app.py")


if __name__ == "__main__":
    main()
