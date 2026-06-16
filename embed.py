"""Generate dense vector embeddings for text chunks and queries."""

from typing import Any, Dict, List

import numpy as np

_model = None

_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"


def _get_model():
    """Load and cache the sentence-transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(
    texts: List[str], batch_size: int = 64, show_progress: bool = False
) -> np.ndarray:
    """Encode strings into L2-normalised float32 dense vectors."""
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_chunks(
    chunks: List[Dict[str, Any]], batch_size: int = 64, show_progress: bool = False
) -> np.ndarray:
    """Embed a list of chunk dicts using their 'text' field."""
    texts = [c["text"] for c in chunks]
    return embed_texts(texts, batch_size=batch_size, show_progress=show_progress)


def embed_queries(queries: List[str], batch_size: int = 64) -> np.ndarray:
    """Encode query strings into L2-normalised dense vectors."""
    return embed_texts(queries, batch_size=batch_size, show_progress=False)
