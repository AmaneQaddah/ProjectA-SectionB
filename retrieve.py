"""Query-time retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from embed import embed_queries
from index import load_index
from utils import K_EVAL


_cached_vectors: np.ndarray | None = None
_cached_page_ids: List[int] | None = None


def _load_cached_index(
    artifacts_dir: Optional[Path] = None,
) -> tuple[np.ndarray, List[int]]:
    """
    Load the index once and cache it in memory.

    This helps because run() may search multiple queries in one call.
    """
    global _cached_vectors, _cached_page_ids

    if _cached_vectors is None or _cached_page_ids is None:
        _cached_vectors, _cached_page_ids = load_index(artifacts_dir)

    return _cached_vectors, _cached_page_ids


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """
    Return ranked page_id lists for each query.

    Steps:
    1. Load precomputed corpus vectors from artifacts/.
    2. Embed the query batch.
    3. Compute dot-product similarity.
    4. Sort pages by score.
    5. Return top_k page_id values.
    """
    corpus_vectors, page_ids = _load_cached_index(artifacts_dir)

    if not queries:
        return []

    query_vectors = embed_queries(queries)

    if query_vectors.size == 0:
        return [[] for _ in queries]

    # Since embeddings are normalized, dot product = cosine similarity.
    scores = query_vectors @ corpus_vectors.T

    all_ranked: List[List[int]] = []

    for row in scores:
        order = np.argsort(-row)

        seen: set[int] = set()
        ranked_ids: List[int] = []

        for idx in order:
            page_id = int(page_ids[int(idx)])

            if page_id in seen:
                continue

            seen.add(page_id)
            ranked_ids.append(page_id)

            if len(ranked_ids) >= top_k:
                break

        all_ranked.append(ranked_ids)

    return all_ranked