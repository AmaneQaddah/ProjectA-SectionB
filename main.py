"""
main.py – Entry point called by the autograder.

The autograder will call:

    from main import run
    results = run(queries)

where `queries` is a list of query strings.
`run` must return a list of lists of document IDs (strings), one per query,
ranked from most to least relevant (up to 10 results each).
"""

from typing import List

from retrieve import retrieve


def run(queries: List[str]) -> List[List[str]]:
    """
    Retrieve the top-10 Wikipedia document IDs for each query.

    Args:
        queries: list of natural-language query strings.

    Returns:
        List of length len(queries). Each element is an ordered list of
        document ID strings, most relevant first, with at most 10 entries.
    """
    return retrieve(queries, top_k=10)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_queries = ["What is the capital of France?", "History of the Roman Empire"]
    results = run(test_queries)
    for query, docs in zip(test_queries, results):
        print(f"Query : {query}")
        print(f"Top-10: {docs}\n")
