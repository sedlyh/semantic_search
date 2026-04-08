"""
semantic_search/search_app.py
-----------------------------
Streamlit UI: user types a natural-language query → embed with the SAME model
as in embed_listings.py → ask Chroma for nearest listing vectors → show text
and metadata.

Run (from repo root):
    streamlit run semantic_search/search_app.py

Prerequisite: embed_listings.py has been run at least once so chroma_data/ exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import streamlit as st

from constants import CHROMA_DIR, COLLECTION_NAME, DEFAULT_MODEL
from core import get_collection, search_listings, vector_store_ready


def main() -> None:
    st.set_page_config(page_title="Florida listings — semantic search", layout="wide")
    st.title("Semantic search over MLS descriptions")
    st.markdown(
        "Search by **meaning** (not exact keywords). "
        "Each listing was embedded with the same model used here."
    )

    if not vector_store_ready():
        st.error(
            f"No data at `{CHROMA_DIR}`. "
            "Run `python semantic_search/embed_listings.py` first."
        )
        return

    try:
        get_collection()
    except Exception as e:
        st.error(f"Could not open collection `{COLLECTION_NAME}`: {e}")
        return

    model_id = st.sidebar.text_input(
        "Embedding model",
        value=DEFAULT_MODEL,
        help="Change only if you re-embedded with a different --model.",
    )
    k = st.sidebar.slider("How many results", 1, 20, 5)

    query = st.text_input(
        "Your search",
        placeholder="e.g. gated community golf course renovated kitchen",
    )

    if st.button("Search") and query.strip():
        hits = search_listings(query, k=k, model_id=model_id)

        st.subheader("Results")
        for rank, hit in enumerate(hits, start=1):
            doc_id = hit["id"]
            text = hit["text"]
            meta = hit.get("metadata") or {}
            dist = hit.get("distance")
            sim = hit.get("similarity")

            price = meta.get("lastSoldPrice", "")
            zip_code = meta.get("zip", "")
            prop_type = meta.get("type", "")

            header = f"**{rank}.** `{doc_id}`"
            if sim is not None:
                header += f" — similarity ≈ **{sim:.3f}** (distance {float(dist):.4f})"
            st.markdown(header)

            cols = st.columns(4)
            cols[0].markdown(f"Sold: **${price}**" if str(price) else "Sold: —")
            cols[1].markdown(f"ZIP: **{zip_code}**" if zip_code else "ZIP: —")
            cols[2].markdown(f"Type: **{prop_type}**" if prop_type else "Type: —")
            cols[3].markdown(f"Sqft: **{meta.get('sqft', '')}**")

            snippet = text[:1200] + ("…" if len(text) > 1200 else "")
            st.caption(snippet)
            st.divider()


if __name__ == "__main__":
    main()
