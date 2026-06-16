"""
index.py – Builds and persists a FAISS vector index over chunk embeddings.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(ARTIFACTS_DIR, "chunks.pkl")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_index(chunks: List[Dict[str, Any]], embeddings: np.ndarray):
    """
    Create a FAISS flat inner-product index from pre-computed embeddings.

    Because embeddings are L2-normalised, inner product == cosine similarity.

    Args:
        chunks: list of chunk dicts (same order as embeddings).
        embeddings: float32 array of shape (N, dim).

    Returns:
        (index, chunks) tuple – the FAISS index and the chunk list.
    """
    import faiss  # type: ignore

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, chunks


# ---------------------------------------------------------------------------
# Persist / load
# ---------------------------------------------------------------------------


def save_index(
    index,
    chunks: List[Dict[str, Any]],
    index_path: str = INDEX_PATH,
    chunks_path: str = CHUNKS_PATH,
) -> None:
    """Save FAISS index and chunk metadata to disk."""
    import faiss  # type: ignore

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)


def load_index(index_path: str = INDEX_PATH, chunks_path: str = CHUNKS_PATH):
    """
    Load a previously saved FAISS index and chunk list.

    Returns:
        (index, chunks) tuple.

    Raises:
        FileNotFoundError if artifacts are missing.
    """
    import faiss  # type: ignore

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Index not found at {index_path}. "
            "Run scripts/build_index.py to build the index first."
        )
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"Chunk metadata not found at {chunks_path}. "
            "Run scripts/build_index.py to build the index first."
        )

    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_index(
    index, chunks: List[Dict[str, Any]], query_embeddings: np.ndarray, top_k: int = 10
) -> List[List[Dict[str, Any]]]:
    """
    Query the FAISS index and return ranked chunk results per query.

    Args:
        index: loaded FAISS index.
        chunks: list of chunk dicts aligned with the index.
        query_embeddings: float32 array of shape (Q, dim).
        top_k: number of results to return per query.

    Returns:
        List of length Q, where each element is a list of up to top_k chunk dicts
        sorted by descending relevance, each augmented with a 'score' key.
    """
    scores, indices = index.search(query_embeddings, top_k)
    results = []
    for row_scores, row_indices in zip(scores, indices):
        ranked = []
        for score, idx in zip(row_scores, row_indices):
            if idx == -1:
                continue
            chunk = dict(chunks[idx])
            chunk["score"] = float(score)
            ranked.append(chunk)
        results.append(ranked)
    return results
