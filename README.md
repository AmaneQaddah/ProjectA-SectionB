# Wikipedia Retrieval Pipeline

Dense retrieval system over a Wikipedia corpus, evaluated with NDCG@10.

## Project Structure

```
.
├── main.py               # Autograder entry point — implements run(queries)
├── chunk.py              # Document chunking
├── embed.py              # Sentence-transformer embeddings
├── index.py              # FAISS index build, save, load, search
├── retrieve.py           # High-level retrieval interface
├── utils.py              # Shared I/O and corpus helpers
├── eval.py               # READ-ONLY — NDCG@10 evaluation utilities
├── requirements.txt
├── scripts/
│   ├── build_index.py    # READ-ONLY — offline index build
│   └── eval_public.py    # READ-ONLY — self-test on 50 public queries
├── data/
│   ├── public_queries.json       # 50 public queries with relevance labels
│   └── Wikipedia_Entries/        # One JSON file per Wikipedia page
└── artifacts/            # Precomputed index & embeddings (empty in handout)
    ├── faiss.index
    └── chunks.pkl
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add corpus data

Place the Wikipedia entry JSON files into `data/Wikipedia_Entries/`.
Each file must be a JSON object with at least:
- `"id"` or `"title"` – unique document identifier
- `"text"` – full article text

### 3. Build the index

```bash
python scripts/build_index.py
```

This populates `artifacts/faiss.index` and `artifacts/chunks.pkl`.

### 4. Self-test

```bash
python scripts/eval_public.py
```

Expected output: per-query NDCG@10 scores and a macro-average.

## Pipeline Overview

```
Wikipedia JSON files
       │
   chunk.py          Split each article into overlapping 200-word windows
       │
   embed.py          Encode with sentence-transformers (all-MiniLM-L6-v2)
       │
   index.py          Store in a FAISS flat inner-product index
       │
  retrieve.py        Embed query → search index → deduplicate by doc_id
       │
   main.run()        Return top-10 doc IDs per query
```


## Evaluation Metric

**NDCG@10** (Normalised Discounted Cumulative Gain at rank 10).
Implemented in `eval.py::ndcg_at_k`.
