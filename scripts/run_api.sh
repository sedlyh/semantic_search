#!/usr/bin/env bash
# App Runner and other hosts: respects PORT (default 8000). Repo root must be cwd.
set -euo pipefail

export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"

# Optional: download packaged chroma_data if the repo does not include it (see README).
if [[ -n "${CHROMA_HTTP_URL:-}" ]]; then
  python semantic_search/download_chroma_if_needed.py
fi

exec python -m uvicorn semantic_search.server:app --host 0.0.0.0 --port "${PORT:-8000}"
