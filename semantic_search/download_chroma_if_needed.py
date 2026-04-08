"""
If CHROMA_HTTP_URL points to a .zip of the chroma_data/ folder and chroma_data is
missing or empty, download and extract it. Use for App Runner / CI where the index
is not committed (see README).
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

from constants import CHROMA_DIR


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract zip under *dest*, rejecting paths that escape *dest* (zip slip)."""
    dest = dest.resolve()
    for member in zf.infolist():
        if member.is_dir():
            continue
        parts = Path(member.filename).parts
        if ".." in parts or Path(member.filename).is_absolute():
            raise RuntimeError(f"Unsafe zip path: {member.filename!r}")
        target = (dest / member.filename).resolve()
        try:
            target.relative_to(dest)
        except ValueError as e:
            raise RuntimeError(f"Unsafe zip path: {member.filename!r}") from e
    zf.extractall(dest)


def _needs_download() -> bool:
    if not CHROMA_DIR.is_dir():
        return True
    return not any(CHROMA_DIR.iterdir())


def main() -> int:
    url = os.environ.get("CHROMA_HTTP_URL", "").strip()
    if not url:
        return 0
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise SystemExit("CHROMA_HTTP_URL must use https://")
    if not _needs_download():
        print(f"Chroma already present at {CHROMA_DIR}, skipping download.")
        return 0
    if not url.lower().endswith(".zip"):
        raise SystemExit("CHROMA_HTTP_URL must end with .zip (zip of chroma_data contents).")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zpath = tmp.name
    try:
        print("Downloading Chroma index from CHROMA_HTTP_URL …")
        urlretrieve(url, zpath)
        # Extract into parent so zip root folders land under chroma_data/
        parent = CHROMA_DIR.parent
        with zipfile.ZipFile(zpath, "r") as zf:
            _safe_extract_zip(zf, parent)
        print(f"Extracted index under {CHROMA_DIR}")
    finally:
        Path(zpath).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
