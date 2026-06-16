import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval import evaluate
from main import run
from utils import load_queries


def main() -> None:
    print("Loading public queries …")
    query_records = load_queries()

    queries = [r["query"] for r in query_records]
    query_ids = [r["query_id"] for r in query_records]
    labels = {r["query_id"]: r["relevant_docs"] for r in query_records}

    print(f"Running retrieval on {len(queries)} queries …")
    raw_results = run(queries)

    results = {qid: docs for qid, docs in zip(query_ids, raw_results)}

    print("Computing NDCG@10 …")
    scores = evaluate(results, labels, k=10)

    mean_score = scores.pop("mean_ndcg")
    for qid, score in sorted(scores.items()):
        print(f"  {qid}: {score:.4f}")

    print(f"\nMean NDCG@10: {mean_score:.4f}")


if __name__ == "__main__":
    main()
