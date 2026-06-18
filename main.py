"""
Section B entry point.

The autograder calls run(queries) once with all evaluation queries.
The offline build script calls build_offline_index().
"""

from __future__ import annotations

from typing import List

from index import build_index
from retrieve import search_batch


def run(queries: List[str]) -> List[List[int]]:
    """
    Rank corpus pages for each query.

    Parameters
    ----------
    queries : list[str]
        Batch of query strings.

    Returns
    -------
    list[list[int]]
        One ranked list of page_id values per query.
        Only the first 10 IDs per list are scored.
    """
    return search_batch(queries)


def build_offline_index() -> None:
    """
    Build the corpus index offline and save artifacts/.

    This is used locally by:
        python scripts/build_index.py

    The autograder does not call this during grading.
    """
    build_index()


if __name__ == "__main__":
    build_offline_index()
    print("Index built under artifacts/. Run: python scripts/eval_public.py")