"""Shared paths and helper functions for Section B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List


STUDENT_ROOT = Path(__file__).resolve().parent
DATA_DIR = STUDENT_ROOT / "data"
ENTRIES_DIR = DATA_DIR / "Wikipedia Entries"
PUBLIC_QUERIES_PATH = DATA_DIR / "public_queries.json"
ARTIFACTS_DIR = STUDENT_ROOT / "artifacts"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_EVAL = 10


def normalize_page_id(value: Any) -> int:
    """
    Convert page_id from JSON to int.

    Official files may store page_id as int or numeric string.
    """
    if isinstance(value, int):
        return value

    if isinstance(value, str):
        value = value.strip()
        if value.isdigit():
            return int(value)

    raise ValueError(f"Invalid page_id: {value!r}")


def load_public_queries(path: Path | None = None) -> List[Dict[str, Any]]:
    """Load public queries and normalize relevant_page_ids to int."""
    path = path or PUBLIC_QUERIES_PATH
    rows = json.loads(path.read_text(encoding="utf-8"))

    for row in rows:
        row["relevant_page_ids"] = [
            normalize_page_id(pid) for pid in row["relevant_page_ids"]
        ]

    return rows


def iter_entries(entries_dir: Path | None = None) -> Iterator[Dict[str, Any]]:
    """
    Yield one JSON record per Wikipedia entry.

    Expected official format:
    {
        "page_id": "1000",
        "title": "...",
        "content": "..."
    }
    """
    root = entries_dir or ENTRIES_DIR

    if not root.is_dir():
        raise FileNotFoundError(
            f"Corpus directory not found: {root}. "
            "Expected: data/Wikipedia Entries/"
        )

    for path in sorted(root.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))

        # If page_id is missing, use filename stem as fallback.
        record["page_id"] = normalize_page_id(record.get("page_id", path.stem))

        yield record


def entry_text(record: Dict[str, Any]) -> str:
    """
    Convert one page record to the text that will be embedded.

    We include the title because many queries mention information
    strongly connected to the page title.
    """
    title = str(record.get("title", "")).strip()
    content = str(record.get("content", "")).strip()

    if title and content:
        return f"{title}\n\n{content}"

    if title:
        return title

    return content


def ensure_artifacts_dir() -> Path:
    """Create artifacts/ if needed and return its path."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR