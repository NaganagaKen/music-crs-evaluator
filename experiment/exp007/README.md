# exp007: All-Track Metadata CountVectorizer + TruncatedSVD Retrieval

This experiment builds latent vectors for every track in
`TalkPlayData-Challenge-Track-Metadata/all_tracks`.

1. Each track is represented as a weighted metadata document containing
   `track_name`, `artist_name`, `album_name`, `tag_list`, and `release_date`.
2. `CountVectorizer` creates the sparse item-feature matrix.
3. `TruncatedSVD` transforms that matrix into dense item vectors.
4. For devset turn `t`, the user vector is the mean of item vectors for
   ground-truth tracks from earlier turns in the same session.
5. All metadata tracks are ranked by cosine similarity, excluding tracks
   already present in the session history.

Turn 1 has no history and receives an empty candidate list. It remains in the
recall denominator, matching exp004, exp005, and exp006.

## Run

```powershell
.\.venv\Scripts\python.exe experiment/exp007/run_metadata_svd_retrieval_recall.py `
  --tid exp007_metadata_svd_history_top500
```

The first run trains and saves CountVectorizer, TruncatedSVD, and the item
vectors. Later runs reuse that model unless `--force_retrain` is specified.

Use `--track_metadata_arrow_path` to load a cached all-tracks Arrow file
directly.

## Outputs

- Model: `exp/models/exp007/metadata_svd.pkl`
- Retrieval:
  `exp/retrieval/devset/exp007_metadata_svd_history_top500.json`
- Recall:
  `exp/scores/devset/exp007_metadata_svd_history_top500_recall.json`

Defaults are 128 SVD components, 50,000 maximum CountVectorizer features,
randomized SVD, and seven power iterations.
