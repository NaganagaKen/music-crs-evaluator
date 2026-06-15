# exp009: All-Track Metadata CountVectorizer + NMF Retrieval

This experiment follows exp007/exp008 while replacing `TruncatedSVD` with
non-negative matrix factorization (`NMF`).

1. Each track is represented as a weighted metadata document containing
   `track_name`, `artist_name`, `album_name`, `tag_list`, and `release_date`.
2. `CountVectorizer` creates the sparse non-negative item-feature matrix.
3. `NMF` transforms that matrix into dense non-negative item vectors.
4. For devset turn `t`, the user vector is the mean of item vectors for
   ground-truth tracks from earlier turns in the same session.
5. All tracks in `Track-Metadata/all_tracks` are ranked by cosine similarity,
   excluding tracks already present in the session history.

Turn 1 has no history and receives an empty candidate list. It remains in the
recall denominator, matching exp007 and exp008.

## Run

```powershell
.\.venv\Scripts\python.exe experiment/exp009/run_metadata_nmf_retrieval_recall.py `
  --tid exp009_metadata_nmf_history_top500
```

Use `--track_metadata_arrow_path` to load a cached all-tracks Arrow file
directly.

The first run trains and saves CountVectorizer, NMF, and normalized item
vectors. Later runs reuse that model only when the catalog and all model
settings match. Use `--force_retrain` to replace it.

## Outputs

- Model: `exp/models/exp009/metadata_nmf.pkl`
- Retrieval:
  `exp/retrieval/devset/exp009_metadata_nmf_history_top500.json`
- Recall:
  `exp/scores/devset/exp009_metadata_nmf_history_top500_recall.json`

Defaults are 128 NMF components, 50,000 maximum CountVectorizer features,
`nndsvda` initialization, coordinate-descent optimization, and 200 maximum
iterations.
