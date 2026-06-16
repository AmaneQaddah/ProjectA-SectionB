"""Shared I/O and corpus-loading helpers used across pipeline modules."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def load_json(path: str) -> Any:
    """Load and return the contents of a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    """Serialise obj to a JSON file, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CORPUS_DIR = os.path.join(DATA_DIR, "Wikipedia_Entries")
QUERIES_PATH = os.path.join(DATA_DIR, "public_queries.json")


def load_corpus(corpus_dir: str = CORPUS_DIR) -> List[Dict[str, Any]]:
    """Load all Wikipedia entry JSON files from corpus_dir."""
    documents: List[Dict[str, Any]] = []
    if not os.path.isdir(corpus_dir):
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    for fname in sorted(os.listdir(corpus_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(corpus_dir, fname)
        doc = load_json(fpath)
        if "id" not in doc:
            doc["id"] = os.path.splitext(fname)[0]
        documents.append(doc)

    return documents


def load_queries(queries_path: str = QUERIES_PATH) -> List[Dict[str, Any]]:
    """Load the public queries list from the JSON file."""
    return load_json(queries_path)


def flatten(list_of_lists: List[List[Any]]) -> List[Any]:
    """Flatten one level of nesting."""
    return [item for sublist in list_of_lists for item in sublist]


def deduplicate(items: List[Any]) -> List[Any]:
    """Return list with duplicates removed, preserving first-seen order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
