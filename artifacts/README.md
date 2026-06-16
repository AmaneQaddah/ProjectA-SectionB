# artifacts/

This directory stores precomputed pipeline artifacts:

| File            | Description                                        |
|-----------------|----------------------------------------------------|
| `faiss.index`   | FAISS flat inner-product index over chunk vectors  |
| `chunks.pkl`    | Pickled list of chunk dicts aligned with the index |

## How to generate

Run from the project root:

```bash
python scripts/build_index.py
```

This directory is **empty in the handout** and must be populated before running
`main.py` or `scripts/eval_public.py`. Include these files in your submission.
