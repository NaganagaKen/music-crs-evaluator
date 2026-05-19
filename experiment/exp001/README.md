# exp001: BM25/Cosine + LGBMRanker Baseline

This experiment trains a two-stage baseline for RecSys Challenge 2026 Music-CRS.

1. Retrieve `--candidate_count 300` tracks from `all_tracks` with BM25, TF-IDF cosine, or a hybrid score.
2. Train `lightgbm.LGBMRanker` on retrieved candidates from the train split.
3. Rerank the 300 candidates and write top-20 predictions with `predicted_response: ""`.

## Setup

```bash
uv pip install -r requirments.txt
```

## Devset Inference

```bash
.\.venv\Scripts\python.exe experiment/exp001/run_lgbm_ranker_baseline.py `
  --tid bm25_cosine_lgbmranker_devset `
  --eval_dataset devset
```

Evaluate with:

```bash
.\.venv\Scripts\python.exe evaluate_devset.py --eval_dataset devset --tid bm25_cosine_lgbmranker_devset
```

## Blind A Inference

```bash
.\.venv\Scripts\python.exe experiment/exp001/run_lgbm_ranker_baseline.py `
  --tid bm25_cosine_lgbmranker_blindset_A `
  --eval_dataset blindset_A
```

For a full 300-negative training group, set `--negative_sample_size -1`.
For all train queries, set `--max_train_queries 0`.
