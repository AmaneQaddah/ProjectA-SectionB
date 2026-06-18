# Project A — Section B: Wikipedia Retrieval Pipeline

## Team Members

* Amane Qaddah
* Partner Name

## Video Presentation

Video link: **PUT_VIDEO_LINK_HERE**

---

## Project Overview

This repository contains our implementation for **Project A — Section B**.

The goal of this section is to build an end-to-end retrieval pipeline over a collection of Wikipedia-style entries.
Given a batch of natural-language queries, the system returns, for each query, a ranked list of relevant `page_id` values.

The official evaluation measures the quality of the top-10 returned page IDs using **NDCG@10**.

---

## Pipeline Summary

Our retrieval pipeline contains four main stages:

1. **Chunk**
   Each Wikipedia entry is converted into a retrieval unit.
   In our final implementation, we use one retrieval unit per page as a stable page-level baseline.

2. **Embed**
   Each retrieval unit is embedded using:

   ```text
   sentence-transformers/all-MiniLM-L6-v2
   ```

   The embeddings are L2-normalized.

3. **Index**
   Corpus embeddings are built offline using:

   ```bash
   python scripts/build_index.py
   ```

   The resulting vectors and metadata are saved under the `artifacts/` directory.

4. **Retrieve**
   At query time, `run(queries)` embeds all input queries, computes dot-product similarity against the saved corpus vectors, sorts pages by score, and returns the top-10 `page_id` values for each query.

---

## Repository Structure

```text
.
├── artifacts/
│   ├── index_vectors.npy
│   └── index_meta.json
├── data/
│   ├── public_queries.json
│   └── Wikipedia Entries/
├── scripts/
│   ├── build_index.py
│   └── eval_public.py
├── chunk.py
├── embed.py
├── eval.py
├── index.py
├── main.py
├── retrieve.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## Data Format

The corpus is stored under:

```text
data/Wikipedia Entries/
```

Each Wikipedia-style entry is a JSON file with the following structure:

```json
{
  "page_id": "1000",
  "title": "...",
  "content": "..."
}
```

The public queries are stored in:

```text
data/public_queries.json
```

Each query contains:

```json
{
  "query_id": "q_public_001",
  "query": "...",
  "relevant_page_ids": ["20263"]
}
```

---

## Artifacts

The repository includes precomputed artifacts under:

```text
artifacts/
```

The required artifact files are:

```text
artifacts/index_vectors.npy
artifacts/index_meta.json
```

### Artifact Descriptions

* `index_vectors.npy`
  A NumPy matrix containing the precomputed L2-normalized embeddings of all corpus retrieval units.

* `index_meta.json`
  Metadata that maps each vector row to its corresponding `page_id` and `chunk_id`.

The autograder does **not** rebuild the index during grading.
It only calls:

```python
run(queries)
```

Therefore, the precomputed artifacts must be included in the repository.

---

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
numpy
sentence-transformers
faiss-cpu
```

---

## Build the Offline Index

To rebuild the index locally, run:

```bash
python scripts/build_index.py
```

This command loads the corpus, embeds all pages, and saves the required artifacts under:

```text
artifacts/
```

This step is performed offline and is not executed by the autograder.

---

## Run Public Evaluation

After the artifacts are available, run:

```bash
python scripts/eval_public.py
```

This evaluates the retrieval pipeline on the 50 public queries.

Our public evaluation result:

```text
Mean NDCG@10: PUT_YOUR_NDCG_SCORE_HERE
Runtime: PUT_YOUR_RUNTIME_HERE
```

---

## Required API

The autograder calls the following function from `main.py`:

```python
def run(queries: list[str]) -> list[list[int]]:
```

Input:

```text
queries: list of query strings
```

Output:

```text
A list of ranked page_id lists.
Each inner list contains the top-10 retrieved page IDs for one query.
```

Example output:

```python
[
    [20263, 9112, 25051],
    [42955, 1000, 3001]
]
```

---

## Design Decisions

### Page-Level Retrieval

We chose a page-level retrieval baseline where each page is represented as one retrieval unit.
This design is simple, stable, and directly aligned with the official page-level evaluation.

### Title + Content Representation

For each page, we embed both the title and the content.
Including the title helps match queries that refer to the main topic of a page.

### Normalized Embeddings

All embeddings are L2-normalized.
This allows dot-product similarity to behave like cosine similarity.

### Offline Artifacts

The corpus index is built offline and saved to disk.
This makes query-time evaluation faster and matches the grading requirement that the index should already exist when `run(queries)` is called.

---

## Possible Future Improvements

Possible improvements include:

* splitting long pages into overlapping chunks;
* aggregating chunk scores by `page_id`;
* using FAISS for faster approximate nearest-neighbor search;
* adding a reranking step over the top retrieved candidates;
* tuning chunk size and overlap based on public NDCG@10.

---

## How to Reproduce

From a fresh clone:

```bash
pip install -r requirements.txt
python scripts/eval_public.py
```

If the artifacts need to be rebuilt:

```bash
python scripts/build_index.py
python scripts/eval_public.py
```
