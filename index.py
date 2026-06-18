"""Offline index build and load."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, EMBEDDING_MODEL_NAME, ensure_artifacts_dir, iter_entries


INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME = "index_meta.json"


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Build the offline retrieval index.

    This function:
    1. Loads all Wikipedia entries.
    2. Converts them to chunks.
    3. Embeds all chunks.
    4. Saves vectors and metadata under artifacts/.
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()

    print("Loading corpus...")
    records = list(iter_entries(entries_dir))
    print(f"Loaded {len(records)} pages.")

    print("Creating chunks...")
    chunks: List[Chunk] = chunk_corpus(records)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding corpus chunks...")
    texts = [chunk.text for chunk in chunks]
    vectors = embed_texts(texts)
    print(f"Vector matrix shape: {vectors.shape}")

    page_ids = [chunk.page_id for chunk in chunks]
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    print("Saving artifacts...")
    np.save(out_dir / INDEX_VECTORS_NAME, vectors)

    meta = {
        "page_ids": page_ids,
        "chunk_ids": chunk_ids,
        "model": EMBEDDING_MODEL_NAME,
        "num_vectors": len(page_ids),
    }

    (out_dir / INDEX_META_NAME).write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {INDEX_VECTORS_NAME} and {INDEX_META_NAME} under artifacts/.")

    return vectors, page_ids


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Load precomputed vectors and metadata from artifacts/.

    This is what run() uses during evaluation.
    """
    root = artifacts_dir or ARTIFACTS_DIR

    vectors_path = root / INDEX_VECTORS_NAME
    meta_path = root / INDEX_META_NAME

    if not vectors_path.exists():
        raise FileNotFoundError(
            f"Missing artifact: {vectors_path}. "
            "Run python scripts/build_index.py first."
        )

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing artifact: {meta_path}. "
            "Run python scripts/build_index.py first."
        )

    vectors = np.load(vectors_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    page_ids = [int(pid) for pid in meta["page_ids"]]

    return vectors, page_ids