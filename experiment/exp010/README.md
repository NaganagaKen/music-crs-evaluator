# exp010: Rich Context Item2Vec

This experiment extends `exp005` by feeding more non-leaky catalog and
conversation context into the Item2Vec corpus.

The Word2Vec corpus combines:

- ordered ground-truth track sequences from train conversations;
- all-track metadata pairs from `Track-Metadata/all_tracks`;
- richer metadata tokens from track, artist, and album names;
- release year/decade, popularity bins, duration bins, and ISRC prefixes;
- train turn context pairs linking each train target track to user text,
  conversation goal, user profile, session date, and optionally user ID.

For devset turn `t`, the query vector is a weighted mean of:

- Item2Vec vectors for ground-truth tracks from turns earlier than `t`;
- context tokens from the same session up to the current user turn.

The dev/test `music`, `assistant`, and `thought` rows are not used as query
context. This keeps target-track and future-turn information out of the query
side. Turn 1 can still produce candidates from profile, goal, session date, and
current user text.

## Run

```powershell
.\.venv\Scripts\python.exe experiment/exp010/run_rich_item2vec_retrieval_recall.py `
  --tid exp010_rich_item2vec_context_top500
```

Cached Arrow files can be supplied when Hugging Face resolution is unavailable:

```powershell
.\.venv\Scripts\python.exe experiment/exp010/run_rich_item2vec_retrieval_recall.py `
  --tid exp010_rich_item2vec_context_top500 `
  --train_arrow_path <path-to-train.arrow> `
  --eval_arrow_path <path-to-test.arrow> `
  --track_metadata_arrow_path <path-to-track-metadata-all_tracks.arrow>
```

Useful ablations:

```powershell
# Disable warm-user user_id tokens.
.\.venv\Scripts\python.exe experiment/exp010/run_rich_item2vec_retrieval_recall.py `
  --tid exp010_rich_item2vec_no_user_id_top500 `
  --no-include_user_id_token `
  --model_path exp/models/exp010/rich_item2vec_no_user_id.model

# Make the query rely more heavily on text/profile/goal context.
.\.venv\Scripts\python.exe experiment/exp010/run_rich_item2vec_retrieval_recall.py `
  --tid exp010_rich_item2vec_context_w070_top500 `
  --context_weight 0.70 `
  --model_path exp/models/exp010/rich_item2vec.model
```

## Outputs

- Model: `exp/models/exp010/rich_item2vec.model`
- Model config: `exp/models/exp010/rich_item2vec.model.config.json`
- Retrieval:
  `exp/retrieval/devset/exp010_rich_item2vec_context_top500.json`
- Recall:
  `exp/scores/devset/exp010_rich_item2vec_context_top500_recall.json`

The recall JSON includes macro, micro, turn-wise recall, query-status counts,
catalog coverage, vocabulary coverage, context-token coverage, and all major
training corpus counts.
