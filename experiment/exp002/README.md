# exp002: CF-BPR Cosine Retrieval Recall

This experiment checks the upper-bound quality of the provided CF-BPR embeddings as a
retrieval stage for RecSys Challenge 2026 Music-CRS.

For each devset user, it computes cosine similarity between the user's `cf-bpr`
embedding and every track's `cf-bpr` embedding, retrieves the top 500 tracks, and
reports recall@20, 50, 100, 200, and 500.

## Run

```powershell
.\.venv\Scripts\python.exe experiment/exp002/run_cf_bpr_retrieval_recall.py `
  --tid exp002_cf_bpr_top500
```

## Outputs

- Retrieval JSON: `exp/retrieval/devset/exp002_cf_bpr_top500.json`
- Recall scores: `exp/scores/devset/exp002_cf_bpr_top500_recall.json`

The retrieval JSON stores top-500 candidates and is not intended as a final
submission file. Slice it to top 20 if you want to use it with the official
devset evaluator.

To calculate recall without writing the larger top-500 retrieval JSON:

```powershell
.\.venv\Scripts\python.exe experiment/exp002/run_cf_bpr_retrieval_recall.py `
  --tid exp002_cf_bpr_top500 `
  --no_save_retrieval
```

By default, users with missing or empty CF-BPR embeddings are assigned the mean
non-empty user CF-BPR vector so that all devset turns remain in the aggregate.
Use `--missing_user_strategy none` to score those users as zero recall instead.

## Re-evaluate Saved Retrieval

To recompute recall from a saved retrieval JSON while dropping empty candidate
lists from the evaluation denominator:

```powershell
.\.venv\Scripts\python.exe experiment/exp002/evaluate_retrieval_recall.py `
  --input_tid exp002_cf_bpr_top500_none
```

This writes:

```text
exp/scores/devset/exp002_cf_bpr_top500_none_drop_empty_eval_recall.json
```

Add `--keep_empty_predictions` if you want empty candidate lists to remain in
the denominator as zero recall.
