# exp005: All-Track Metadata-Augmented Item2Vec

This experiment extends exp004 so every track in
`TalkPlayData-Challenge-Track-Metadata/all_tracks` receives an Item2Vec vector.

Track Metadata has no listening order, so the catalog is not treated as one
artificial sequence. The Word2Vec corpus combines:

- ordered ground-truth track sequences from the train conversations;
- `track_id <-> artist_id` metadata pairs;
- `track_id <-> album_id` metadata pairs; and
- `track_id <-> tag` metadata pairs.

The default retains the ten most frequent tags attached to each track. All
47,071 metadata tracks are used and remain retrieval candidates.

For devset turn `t`, the user vector is the mean of Item2Vec vectors for
ground-truth tracks from turns earlier than `t` in the same session. Candidates
are ranked by cosine similarity, excluding tracks already in that history.
Turn 1 has no history and therefore receives an empty candidate list.

## Run

```powershell
.\.venv\Scripts\python.exe experiment/exp005/run_metadata_item2vec_retrieval_recall.py `
  --tid exp005_metadata_item2vec_history_top500
```

Cached Arrow files can be supplied with `--train_arrow_path` and
`--track_metadata_arrow_path` when Hugging Face resolution is unavailable.

## Outputs

- Model: `exp/models/exp005/metadata_item2vec.model`
- Retrieval:
  `exp/retrieval/devset/exp005_metadata_item2vec_history_top500.json`
- Recall:
  `exp/scores/devset/exp005_metadata_item2vec_history_top500_recall.json`

The recall JSON includes macro, micro, and turn-wise
`recall@20/50/100/200/500`, along with catalog and vocabulary coverage.
