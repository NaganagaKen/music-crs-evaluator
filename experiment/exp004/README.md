# exp004: Item2Vec Mean-History Cosine Retrieval

This experiment trains Item2Vec on the ordered ground-truth tracks from each
training session. For devset turn `t`, it:

1. uses only ground-truth tracks from turns earlier than `t` in the same session;
2. averages their Item2Vec vectors to create the user vector;
3. ranks Item2Vec items by cosine similarity to the user vector; and
4. excludes tracks already present in that session history.

The current turn's ground truth and future turns are never included in the query.
Turn 1 and turns whose full history is outside the Item2Vec vocabulary receive
an empty candidate list. These rows remain in the recall denominator.

## Run

```powershell
.\.venv\Scripts\python.exe experiment/exp004/run_item2vec_retrieval_recall.py `
  --tid exp004_item2vec_history_top500
```

The first run trains and saves the model. Later runs reuse the saved model unless
`--force_retrain` is specified.

If Hugging Face dataset resolution is unavailable, pass a cached train Arrow
file directly with `--train_arrow_path`.

## Outputs

- Model: `exp/models/exp004/item2vec.model`
- Retrieval JSON:
  `exp/retrieval/devset/exp004_item2vec_history_top500.json`
- Recall JSON:
  `exp/scores/devset/exp004_item2vec_history_top500_recall.json`

The score JSON contains:

- macro `recall@20`, `recall@50`, `recall@100`, `recall@200`, `recall@500`;
- micro recall;
- recall and availability counts for every turn number; and
- Item2Vec vocabulary and OOV metadata.

Use `--save_per_query` to additionally save session-turn details as CSV.
