from __future__ import annotations

import math
from typing import Dict, List, Set


def _dcg(relevances: List[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0

    relevances = [1 if doc_id in relevant else 0 for doc_id in retrieved[:k]]
    ideal = sorted(
        [1] * min(len(relevant), k) + [0] * max(0, k - len(relevant)), reverse=True
    )

    actual_dcg = _dcg(relevances, k)
    ideal_dcg = _dcg(ideal, k)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg



def evaluate(
    results: Dict[str, List[str]],
    labels: Dict[str, List[str]],
    k: int = 10,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for qid, relevant_docs in labels.items():
        retrieved = results.get(qid, [])
        scores[qid] = ndcg_at_k(retrieved, set(relevant_docs), k=k)

    scores["mean_ndcg"] = sum(v for k2, v in scores.items() if k2 != "mean_ndcg") / max(
        1, len(labels)
    )
    return scores
