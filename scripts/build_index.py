import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chunk import chunk_corpus

from embed import embed_chunks
from index import build_index, save_index
from utils import load_corpus


def main() -> None:
    t0 = time.time()

    # 1. Load corpus
    print("Loading corpus …")
    documents = load_corpus()
    print(f"  Loaded {len(documents)} documents.")

    # 2. Chunk
    print("Chunking documents …")
    chunks = chunk_corpus(documents, chunk_size=200, overlap=50)
    print(f"  Created {len(chunks)} chunks.")

    # 3. Embed
    print("Embedding chunks (this may take a while) …")
    embeddings = embed_chunks(chunks, show_progress=True)
    print(f"  Embedding shape: {embeddings.shape}")

    # 4. Build index
    print("Building FAISS index …")
    index, chunks = build_index(chunks, embeddings)
    print(f"  Index contains {index.ntotal} vectors.")

    # 5. Save
    print("Saving artifacts …")
    save_index(index, chunks)
    print("  Saved to artifacts/faiss.index and artifacts/chunks.pkl")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
