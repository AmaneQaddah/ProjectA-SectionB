"""Split Wikipedia page documents into overlapping text chunks."""

from typing import Any, Dict, List


def chunk_document(
    doc: Dict[str, Any], chunk_size: int = 200, overlap: int = 50
) -> List[Dict[str, Any]]:
    """Split a single document into overlapping word-level chunks."""
    doc_id = doc.get("id", doc.get("title", "unknown"))
    title = doc.get("title", "")
    words = doc.get("text", "").split()
    title_prefix = f"{title}: " if title else ""

    chunks = []
    step = max(1, chunk_size - overlap)
    for i, start in enumerate(range(0, max(1, len(words)), step)):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(
            {
                "chunk_id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "title": title,
                "text": title_prefix + " ".join(chunk_words),
            }
        )
    return chunks


def chunk_corpus(
    documents: List[Dict[str, Any]], chunk_size: int = 200, overlap: int = 50
) -> List[Dict[str, Any]]:
    """Chunk every document in a corpus and return a flat list of chunks."""
    all_chunks: List[Dict[str, Any]] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
