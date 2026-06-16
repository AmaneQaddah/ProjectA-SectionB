"""
retrieve.py – High-level retrieval interface: query → ranked doc IDs.
"""

from typing import Any, Dict, List

from embed import embed_queries
from index import load_index, search_index

_index = None
_chunks = None


def _ensure_loaded() -> None:
    """Load the FAISS index and chunk list into module-level caches once."""
    global _index, _chunks
    if _index is None or _chunks is None:
        _index, _chunks = load_index()


def retrieve(queries: List[str], top_k: int = 10) -> List[List[str]]:
    """Retrieve the top-k document IDs for each query, most relevant first."""
    _ensure_loaded()

    query_embeddings = embed_queries(queries)
    raw_results = search_index(_index, _chunks, query_embeddings, top_k=top_k * 5)

    ranked_doc_ids: List[List[str]] = []
    for result_chunks in raw_results:
        doc_best_score: Dict[str, float] = {}
        for chunk in result_chunks:
            doc_id = chunk["doc_id"]
            score = chunk["score"]
            if doc_id not in doc_best_score or score > doc_best_score[doc_id]:
                doc_best_score[doc_id] = score

        ranked = sorted(doc_best_score, key=doc_best_score.__getitem__, reverse=True)
        ranked_doc_ids.append(ranked[:top_k])

    return ranked_doc_ids


def retrieve_with_scores(
    queries: List[str], top_k: int = 10
) -> List[List[Dict[str, Any]]]:
    """Like retrieve(), but return result dicts including similarity scores."""
    _ensure_loaded()

    query_embeddings = embed_queries(queries)
    raw_results = search_index(_index, _chunks, query_embeddings, top_k=top_k * 5)

    all_results: List[List[Dict[str, Any]]] = []
    for result_chunks in raw_results:
        doc_best: Dict[str, Dict[str, Any]] = {}
        for chunk in result_chunks:
            doc_id = chunk["doc_id"]
            if doc_id not in doc_best or chunk["score"] > doc_best[doc_id]["score"]:
                doc_best[doc_id] = {
                    "doc_id": doc_id,
                    "title": chunk.get("title", ""),
                    "score": chunk["score"],
                }

        ranked = sorted(doc_best.values(), key=lambda x: x["score"], reverse=True)
        all_results.append(ranked[:top_k])

    return all_results
