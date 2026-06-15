# exp006: All-Track Metadata CountVectorizer + LDA Retrieval

This experiment builds topic vectors for every track in
`TalkPlayData-Challenge-Track-Metadata/all_tracks`.

1. Each track is represented as a weighted metadata document containing
   `track_name`, `artist_name`, `album_name`, `tag_list`, and `release_date`.
2. `CountVectorizer` creates the sparse item-feature matrix.
3. `LatentDirichletAllocation` transforms that matrix into item-topic vectors.
4. For devset turn `t`, the query vector is the mean of topic vectors for
   ground-truth tracks from earlier turns in the same session.
5. All metadata tracks are ranked by cosine similarity, excluding tracks
   already present in the session history.

Turn 1 has no history and receives an empty candidate list. It remains in the
recall denominator, matching exp004 and exp005.

## Run

```powershell
.\.venv\Scripts\python.exe experiment/exp006/run_metadata_lda_retrieval_recall.py `
  --tid exp006_metadata_lda_history_top500
```

The first run trains and saves CountVectorizer, LDA, and the item vectors.
Later runs reuse that model unless `--force_retrain` is specified.

Use `--track_metadata_arrow_path` to load a cached all-tracks Arrow file
directly.

## Outputs

- Model: `exp/models/exp006/metadata_lda.pkl`
- Retrieval:
  `exp/retrieval/devset/exp006_metadata_lda_history_top500.json`
- Recall:
  `exp/scores/devset/exp006_metadata_lda_history_top500_recall.json`

Defaults are 128 topics, 50,000 maximum CountVectorizer features, online LDA,
and ten LDA iterations.
