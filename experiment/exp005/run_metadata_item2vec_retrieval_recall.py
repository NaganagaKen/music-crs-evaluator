from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm


CHALLENGE_DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"
TRACK_METADATA_DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
DEFAULT_K_VALUES = (20, 50, 100, 200, 500)


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def extract_music_sequence(conversations: list[dict[str, Any]]) -> list[str]:
    music_rows = [
        row
        for row in conversations
        if row.get("role") == "music" and row.get("turn_number") is not None
    ]
    music_rows.sort(key=lambda row: int(row["turn_number"]))
    return [
        value_to_text(row.get("content"))
        for row in music_rows
        if value_to_text(row.get("content"))
    ]


def load_training_sequences(
    dataset_name: str,
    split: str,
    arrow_path: str | None,
) -> list[list[str]]:
    dataset = (
        Dataset.from_file(arrow_path)
        if arrow_path is not None
        else load_dataset(dataset_name, split=split)
    )
    sequences = [
        extract_music_sequence(list(item["conversations"]))
        for item in tqdm(dataset, desc="Building Item2Vec corpus")
    ]
    return [sequence for sequence in sequences if sequence]


def attribute_token(prefix: str, value: Any) -> str:
    text = value_to_text(value).strip().casefold()
    return f"__{prefix}__:{text}" if text else ""


def load_metadata_sequences(
    dataset_name: str,
    split: str,
    arrow_path: str | None,
    max_tags_per_track: int,
) -> tuple[list[str], list[list[str]], dict[str, int]]:
    dataset = (
        Dataset.from_file(arrow_path)
        if arrow_path is not None
        else load_dataset(dataset_name, split=split)
    )
    tag_counts = Counter(
        attribute_token("tag", tag)
        for tags in dataset["tag_list"]
        for tag in (tags or [])
        if attribute_token("tag", tag)
    )

    track_ids: list[str] = []
    sequences: list[list[str]] = []
    relation_counts: Counter[str] = Counter()
    seen_track_ids: set[str] = set()

    for row in tqdm(dataset, desc="Building metadata corpus"):
        track_id = value_to_text(row["track_id"])
        if not track_id or track_id in seen_track_ids:
            continue
        seen_track_ids.add(track_id)
        track_ids.append(track_id)

        attribute_tokens: list[tuple[str, str]] = []
        attribute_tokens.extend(
            ("artist", token)
            for value in (row.get("artist_id") or [])
            if (token := attribute_token("artist", value))
        )
        attribute_tokens.extend(
            ("album", token)
            for value in (row.get("album_id") or [])
            if (token := attribute_token("album", value))
        )

        tag_tokens = {
            attribute_token("tag", value)
            for value in (row.get("tag_list") or [])
            if attribute_token("tag", value)
        }
        ranked_tags = sorted(
            tag_tokens,
            key=lambda token: (-tag_counts[token], token),
        )
        if max_tags_per_track >= 0:
            ranked_tags = ranked_tags[:max_tags_per_track]
        attribute_tokens.extend(("tag", token) for token in ranked_tags)

        if not attribute_tokens:
            sequences.append([track_id])
            relation_counts["track_only"] += 1
            continue

        for relation, token in attribute_tokens:
            sequences.append([track_id, token])
            relation_counts[relation] += 1

    stats = {
        "metadata_track_count": len(track_ids),
        "metadata_sequence_count": len(sequences),
        **{
            f"metadata_{relation}_relation_count": int(count)
            for relation, count in relation_counts.items()
        },
    }
    return track_ids, sequences, stats


def load_ground_truth(path: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or not records:
        raise ValueError("Ground truth must be a non-empty JSON list.")

    required = {"session_id", "user_id", "turn_number", "ground_truth_track_id"}
    missing = required - set(records[0])
    if missing:
        raise ValueError(f"Ground truth is missing fields: {sorted(missing)}")

    normalized = [
        {
            "session_id": value_to_text(record["session_id"]),
            "user_id": value_to_text(record["user_id"]),
            "turn_number": int(record["turn_number"]),
            "ground_truth_track_id": value_to_text(
                record["ground_truth_track_id"]
            ),
        }
        for record in records
    ]
    normalized.sort(key=lambda row: (row["session_id"], row["turn_number"]))
    return normalized


def load_or_train_item2vec(
    sequences: list[list[str]],
    model_path: Path,
    vector_size: int,
    window: int,
    min_count: int,
    negative: int,
    epochs: int,
    workers: int,
    seed: int,
    force_retrain: bool,
) -> Any:
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise ImportError(
            "gensim is required. Install dependencies from requirments.txt."
        ) from exc

    if model_path.exists() and not force_retrain:
        print(f"Loading Item2Vec model from {model_path}")
        return Word2Vec.load(str(model_path))

    print(f"Training Item2Vec on {len(sequences)} sequences...")
    model = Word2Vec(
        sentences=sequences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        negative=negative,
        epochs=epochs,
        workers=workers,
        seed=seed,
        sorted_vocab=1,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"Saved Item2Vec model to {model_path}")
    return model


def normalize_vector(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= eps:
        return None
    return (vector / norm).astype(np.float32)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    available_count = int(np.isfinite(scores).sum())
    k = min(k, available_count)
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if k == scores.shape[0]:
        return np.argsort(-scores, kind="stable")

    indices = np.argpartition(-scores, kth=k - 1)[:k]
    return indices[np.argsort(-scores[indices], kind="stable")]


def build_user_vector(
    history_track_ids: list[str],
    model: Any,
) -> tuple[np.ndarray | None, str, int]:
    if not history_track_ids:
        return None, "no_history", 0

    in_vocab_history = [
        track_id for track_id in history_track_ids if track_id in model.wv
    ]
    if not in_vocab_history:
        return None, "all_history_oov", 0

    history_matrix = np.asarray(
        [model.wv[track_id] for track_id in in_vocab_history],
        dtype=np.float32,
    )
    user_vector = normalize_vector(history_matrix.mean(axis=0))
    if user_vector is None:
        return None, "zero_norm_user_vector", len(in_vocab_history)
    return user_vector, "ok", len(in_vocab_history)


def build_retrieval(
    records: list[dict[str, Any]],
    model: Any,
    catalog_track_ids: list[str],
    top_k: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    missing_catalog_tracks = [
        track_id for track_id in catalog_track_ids if track_id not in model.wv
    ]
    if missing_catalog_tracks:
        raise ValueError(
            f"{len(missing_catalog_tracks)} catalog tracks are missing from "
            "the Item2Vec vocabulary."
        )

    track_ids = np.asarray(catalog_track_ids)
    normalized_track_matrix = np.asarray(
        [model.wv.get_vector(track_id, norm=True) for track_id in track_ids],
        dtype=np.float32,
    )
    track_index = {
        track_id: index for index, track_id in enumerate(track_ids.tolist())
    }

    status_counts: Counter[str] = Counter()
    history_by_session: dict[str, list[str]] = {}
    contexts: list[dict[str, Any]] = []

    for record in records:
        session_id = record["session_id"]
        history = history_by_session.setdefault(session_id, [])
        user_vector, status, in_vocab_history_count = build_user_vector(
            history_track_ids=history,
            model=model,
        )
        status_counts[status] += 1
        contexts.append(
            {
                "record": record,
                "history_track_ids": list(history),
                "user_vector": user_vector,
                "history_count": len(history),
                "in_vocab_history_count": in_vocab_history_count,
                "retrieval_status": status,
                "predicted_track_ids": [],
                "predicted_track_scores": [],
            }
        )
        history.append(record["ground_truth_track_id"])

    usable_indices = [
        index
        for index, context in enumerate(contexts)
        if context["user_vector"] is not None
    ]
    for start in tqdm(
        range(0, len(usable_indices), batch_size),
        desc="Retrieving",
    ):
        batch_indices = usable_indices[start : start + batch_size]
        query_matrix = np.stack(
            [contexts[index]["user_vector"] for index in batch_indices]
        )
        batch_scores = query_matrix @ normalized_track_matrix.T

        for row_index, context_index in enumerate(batch_indices):
            context = contexts[context_index]
            scores = batch_scores[row_index]
            for track_id in set(context["history_track_ids"]):
                index = track_index.get(track_id)
                if index is not None:
                    scores[index] = -np.inf

            indices = topk_indices(scores, top_k)
            context["predicted_track_ids"] = track_ids[indices].tolist()
            context["predicted_track_scores"] = (
                scores[indices].astype(float).tolist()
            )

    retrieval_entries: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    for context in contexts:
        record = context["record"]
        predictions = context["predicted_track_ids"]
        retrieval_entries.append(
            {
                "session_id": record["session_id"],
                "user_id": record["user_id"],
                "turn_number": record["turn_number"],
                "predicted_track_ids": predictions,
                "predicted_track_scores": context["predicted_track_scores"],
                "predicted_response": (
                    "metadata-augmented item2vec mean-history cosine retrieval"
                ),
            }
        )
        evaluation_rows.append(
            {
                **record,
                "history_count": context["history_count"],
                "in_vocab_history_count": context["in_vocab_history_count"],
                "retrieval_status": context["retrieval_status"],
                "candidate_count": len(predictions),
                "target_in_item2vec_vocab": record["ground_truth_track_id"]
                in model.wv,
                "predicted_track_ids": predictions,
            }
        )

    return retrieval_entries, evaluation_rows, dict(status_counts)


def compute_recall_scores(
    evaluation_rows: list[dict[str, Any]],
    k_values: list[int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for evaluation_row in evaluation_rows:
        predictions = evaluation_row.pop("predicted_track_ids")
        row = dict(evaluation_row)
        for k in k_values:
            row[f"recall@{k}"] = float(
                row["ground_truth_track_id"] in predictions[:k]
            )
        rows.append(row)

    per_query = pd.DataFrame(rows)
    metric_columns = [f"recall@{k}" for k in k_values]
    turn_wise_df = per_query.groupby("turn_number", sort=True)[
        metric_columns
    ].mean()
    macro = turn_wise_df.mean(axis=0).to_dict()
    micro = per_query[metric_columns].mean(axis=0).to_dict()

    turn_wise: dict[str, dict[str, Any]] = {}
    for turn_number, metrics in turn_wise_df.to_dict(orient="index").items():
        turn_rows = per_query[per_query["turn_number"] == turn_number]
        turn_wise[str(turn_number)] = {
            **{column: float(value) for column, value in metrics.items()},
            "turn_count": int(len(turn_rows)),
            "non_empty_prediction_count": int(
                (turn_rows["candidate_count"] > 0).sum()
            ),
            "target_in_item2vec_vocab_count": int(
                turn_rows["target_in_item2vec_vocab"].sum()
            ),
            "retrieval_status_counts": {
                str(key): int(value)
                for key, value in turn_rows["retrieval_status"]
                .value_counts()
                .to_dict()
                .items()
            },
        }

    scores: dict[str, Any] = {
        **{column: float(value) for column, value in macro.items()},
        "micro": {column: float(value) for column, value in micro.items()},
        "turn_wise": turn_wise,
    }
    return scores, per_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train metadata-augmented Item2Vec on all catalog tracks and "
            "retrieve by cosine similarity to mean prior-turn track vectors."
        )
    )
    parser.add_argument(
        "--tid",
        default="exp005_metadata_item2vec_history_top500",
    )
    parser.add_argument("--dataset_name", default=CHALLENGE_DATASET)
    parser.add_argument("--train_split", default="train")
    parser.add_argument(
        "--train_arrow_path",
        default=None,
        help="Optional local Arrow file; bypasses Hugging Face dataset loading.",
    )
    parser.add_argument(
        "--track_metadata_dataset_name",
        default=TRACK_METADATA_DATASET,
    )
    parser.add_argument("--track_metadata_split", default="all_tracks")
    parser.add_argument(
        "--track_metadata_arrow_path",
        default=None,
        help="Optional local all_tracks metadata Arrow file.",
    )
    parser.add_argument(
        "--max_tags_per_track",
        type=int,
        default=10,
        help="Most frequent metadata tags retained per track; -1 keeps all.",
    )
    parser.add_argument("--ground_truth_path", default="exp/ground_truth/devset.json")
    parser.add_argument("--eval_dataset", default="devset")
    parser.add_argument("--top_k", type=int, default=500)
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
    )
    parser.add_argument("--vector_size", type=int, default=128)
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--negative", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of user vectors scored against the item matrix at once.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Use 1 for deterministic training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_path",
        default="exp/models/exp005/metadata_item2vec.model",
    )
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--retrieval_dir", default="exp/retrieval")
    parser.add_argument("--score_dir", default="exp/scores")
    parser.add_argument("--save_per_query", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = sorted({k for k in args.k_values if 0 < k <= args.top_k})
    if not k_values:
        raise ValueError("At least one --k_values entry must be <= --top_k.")

    session_sequences = load_training_sequences(
        args.dataset_name,
        args.train_split,
        args.train_arrow_path,
    )
    catalog_track_ids, metadata_sequences, metadata_stats = (
        load_metadata_sequences(
            dataset_name=args.track_metadata_dataset_name,
            split=args.track_metadata_split,
            arrow_path=args.track_metadata_arrow_path,
            max_tags_per_track=args.max_tags_per_track,
        )
    )
    sequences = session_sequences + metadata_sequences
    training_track_counts = Counter(
        track_id for sequence in session_sequences for track_id in sequence
    )
    model = load_or_train_item2vec(
        sequences=sequences,
        model_path=Path(args.model_path),
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        negative=args.negative,
        epochs=args.epochs,
        workers=args.workers,
        seed=args.seed,
        force_retrain=args.force_retrain,
    )

    records = load_ground_truth(args.ground_truth_path)
    retrieval_entries, evaluation_rows, status_counts = build_retrieval(
        records=records,
        model=model,
        catalog_track_ids=catalog_track_ids,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )
    scores, per_query = compute_recall_scores(evaluation_rows, k_values)

    target_in_vocab_count = int(per_query["target_in_item2vec_vocab"].sum())
    scores["metadata"] = {
        "tid": args.tid,
        "eval_dataset": args.eval_dataset,
        "dataset_name": args.dataset_name,
        "train_split": args.train_split,
        "train_arrow_path": args.train_arrow_path,
        "track_metadata_dataset_name": args.track_metadata_dataset_name,
        "track_metadata_split": args.track_metadata_split,
        "track_metadata_arrow_path": args.track_metadata_arrow_path,
        "max_tags_per_track": args.max_tags_per_track,
        "ground_truth_path": args.ground_truth_path,
        "model_path": args.model_path,
        "model": "metadata_augmented_word2vec_skipgram",
        "metadata_relations": ["artist_id", "album_id", "tag_list"],
        "user_vector": "mean_of_prior_turn_item_vectors",
        "retrieval_metric": "cosine_similarity",
        "exclude_history_items": True,
        "empty_history_strategy": "empty_predictions",
        "top_k": args.top_k,
        "k_values": k_values,
        "vector_size": args.vector_size,
        "window": args.window,
        "min_count": args.min_count,
        "negative": args.negative,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "seed": args.seed,
        "training_session_count": int(len(session_sequences)),
        "training_interaction_count": int(
            sum(map(len, session_sequences))
        ),
        "training_unique_track_count": int(len(training_track_counts)),
        "total_training_sequence_count": int(len(sequences)),
        **metadata_stats,
        "candidate_track_count": int(len(catalog_track_ids)),
        "item2vec_vocab_size": int(len(model.wv)),
        "eval_turn_count": int(len(records)),
        "eval_session_count": int(len({row["session_id"] for row in records})),
        "eval_user_count": int(len({row["user_id"] for row in records})),
        "target_in_item2vec_vocab_count": target_in_vocab_count,
        "target_in_item2vec_vocab_rate": float(
            target_in_vocab_count / len(records)
        ),
        "retrieval_status_counts": {
            key: int(value) for key, value in status_counts.items()
        },
    }

    retrieval_path = (
        Path(args.retrieval_dir) / args.eval_dataset / f"{args.tid}.json"
    )
    retrieval_path.parent.mkdir(parents=True, exist_ok=True)
    with retrieval_path.open("w", encoding="utf-8") as f:
        json.dump(retrieval_entries, f, ensure_ascii=False, indent=2)

    score_path = (
        Path(args.score_dir)
        / args.eval_dataset
        / f"{args.tid}_recall.json"
    )
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open("w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    if args.save_per_query:
        per_query_path = (
            Path(args.score_dir)
            / args.eval_dataset
            / f"{args.tid}_per_query.csv"
        )
        per_query.to_csv(per_query_path, index=False)
        print(f"Saved per-query rows to {per_query_path}")

    print(f"Saved retrieval to {retrieval_path}")
    print(f"Saved recall scores to {score_path}")
    print("Macro recall:")
    for k in k_values:
        print(f"  recall@{k}: {scores[f'recall@{k}']:.6f}")
    print("Turn-wise recall:")
    for turn_number, turn_scores in scores["turn_wise"].items():
        values = " ".join(
            f"recall@{k}={turn_scores[f'recall@{k}']:.6f}"
            for k in k_values
        )
        print(f"  turn {turn_number}: {values}")


if __name__ == "__main__":
    main()
