from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


CHALLENGE_DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"
TRACK_EMBEDDINGS_DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
USER_EMBEDDINGS_DATASET = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings"

DEFAULT_K_VALUES = (20, 50, 100, 200, 500)
DEFAULT_USER_SPLITS = ("train", "test_warm", "test_cold")


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    valid = norms[:, 0] > eps
    normalized = matrix.astype(np.float32, copy=True)
    normalized[valid] /= norms[valid]
    normalized[~valid] = 0.0
    return normalized, valid


def load_track_embedding_matrix(
    dataset_name: str,
    split: str,
    embedding_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = load_dataset(dataset_name, split=split)
    raw_track_ids = [value_to_text(track_id) for track_id in dataset["track_id"]]
    raw_embeddings = dataset[embedding_column]
    lengths = [len(embedding) for embedding in raw_embeddings]
    non_empty_lengths = [length for length in lengths if length > 0]
    if not non_empty_lengths:
        raise ValueError(f"No non-empty {embedding_column} track embeddings found.")

    expected_dim = max(set(non_empty_lengths), key=non_empty_lengths.count)
    valid_shape = [length == expected_dim for length in lengths]
    if not all(valid_shape):
        dropped = len(valid_shape) - int(sum(valid_shape))
        print(
            f"Dropping {dropped} tracks whose {embedding_column} length "
            f"is not {expected_dim}."
        )

    track_ids = np.asarray(
        [
            track_id
            for track_id, is_valid in zip(raw_track_ids, valid_shape)
            if is_valid
        ]
    )
    embeddings = np.asarray(
        [
            embedding
            for embedding, is_valid in zip(raw_embeddings, valid_shape)
            if is_valid
        ],
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2-D track embeddings, got shape={embeddings.shape}")
    if len(track_ids) != embeddings.shape[0]:
        raise ValueError("Track ids and embeddings have different lengths.")

    embeddings, valid = normalize_rows(embeddings)
    if not valid.all():
        dropped = int((~valid).sum())
        print(f"Dropping {dropped} tracks with empty or zero-norm embeddings.")
        track_ids = track_ids[valid]
        embeddings = embeddings[valid]

    return track_ids, embeddings


def load_user_embedding_lookup(
    dataset_name: str,
    splits: list[str],
    embedding_column: str,
    expected_dim: int,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, str]]:
    lookup: dict[str, np.ndarray] = {}
    user_split: dict[str, str] = {}
    invalid_users: dict[str, str] = {}

    for split in splits:
        dataset = load_dataset(dataset_name, split=split)
        for user_id, embedding in zip(dataset["user_id"], dataset[embedding_column]):
            user_id = value_to_text(user_id)
            if len(embedding) == 0:
                invalid_users[user_id] = f"empty:{split}"
                user_split.setdefault(user_id, split)
                continue

            vector = np.asarray(embedding, dtype=np.float32)
            if vector.shape != (expected_dim,):
                invalid_users[user_id] = f"shape={vector.shape}:{split}"
                user_split.setdefault(user_id, split)
                continue

            lookup[user_id] = vector
            user_split[user_id] = split
            invalid_users.pop(user_id, None)

    return lookup, user_split, invalid_users


def first_music_content(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        if row.get("role") == "music":
            return value_to_text(row.get("content"))
    if len(rows) > 1:
        return value_to_text(rows[1].get("content"))
    return ""


def extract_eval_records(dataset_name: str, split: str) -> list[dict[str, Any]]:
    dataset = load_dataset(dataset_name, split=split)
    records: list[dict[str, Any]] = []

    for item in tqdm(dataset, desc="Extracting ground truth"):
        conversations = list(item["conversations"])
        turn_numbers = sorted(
            {
                int(row["turn_number"])
                for row in conversations
                if row.get("turn_number") is not None
            }
        )
        for turn_number in turn_numbers:
            turn_rows = [
                row
                for row in conversations
                if int(row.get("turn_number", 0) or 0) == turn_number
            ]
            ground_truth_track_id = first_music_content(turn_rows)
            if not ground_truth_track_id:
                continue
            records.append(
                {
                    "session_id": value_to_text(item["session_id"]),
                    "user_id": value_to_text(item["user_id"]),
                    "turn_number": turn_number,
                    "ground_truth_track_id": ground_truth_track_id,
                }
            )

    return records


def load_eval_records(
    dataset_name: str,
    split: str,
    ground_truth_path: str | None,
) -> list[dict[str, Any]]:
    if ground_truth_path is None:
        return extract_eval_records(dataset_name, split)

    with Path(ground_truth_path).open("r", encoding="utf-8") as f:
        records = json.load(f)

    required = {"session_id", "user_id", "turn_number", "ground_truth_track_id"}
    missing = required - set(records[0])
    if missing:
        raise ValueError(f"Ground truth file is missing required fields: {sorted(missing)}")

    return records


def mean_user_vector(user_lookup: dict[str, np.ndarray]) -> np.ndarray | None:
    if not user_lookup:
        return None
    matrix = np.stack(list(user_lookup.values())).astype(np.float32)
    mean_vector = matrix.mean(axis=0)
    norm = float(np.linalg.norm(mean_vector))
    if norm <= 1e-12:
        return None
    return mean_vector / norm


def build_user_query_matrix(
    user_ids: list[str],
    user_lookup: dict[str, np.ndarray],
    fallback_vector: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, str]]:
    vectors: list[np.ndarray] = []
    user_status: dict[str, str] = {}

    for user_id in user_ids:
        vector = user_lookup.get(user_id)
        if vector is None:
            if fallback_vector is None:
                user_status[user_id] = "missing"
                vectors.append(np.zeros(0, dtype=np.float32))
                continue
            user_status[user_id] = "fallback_mean_user"
            vectors.append(fallback_vector)
            continue

        user_status[user_id] = "ok"
        vectors.append(vector)

    valid_vectors = [vector for vector in vectors if vector.size > 0]
    if not valid_vectors:
        return np.empty((0, 0), dtype=np.float32), user_status

    query_matrix = np.stack(
        [
            vector if vector.size > 0 else np.zeros_like(valid_vectors[0])
            for vector in vectors
        ]
    ).astype(np.float32)
    query_matrix, valid = normalize_rows(query_matrix)

    for user_id, is_valid in zip(user_ids, valid):
        if not is_valid and user_status[user_id] == "ok":
            user_status[user_id] = "zero_norm"

    return query_matrix, user_status


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.shape[1]:
        return np.argsort(-scores, axis=1)

    candidate_indices = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    candidate_scores = np.take_along_axis(scores, candidate_indices, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    return np.take_along_axis(candidate_indices, order, axis=1)


def retrieve_topk_by_user(
    user_ids: list[str],
    query_matrix: np.ndarray,
    track_ids: np.ndarray,
    track_matrix: np.ndarray,
    max_k: int,
    batch_size: int,
    user_status: dict[str, str],
) -> dict[str, list[str]]:
    topk_by_user: dict[str, list[str]] = {}
    usable_indices = [
        idx
        for idx, user_id in enumerate(user_ids)
        if user_status[user_id] not in {"missing", "zero_norm"}
    ]

    if not usable_indices:
        return {user_id: [] for user_id in user_ids}

    for start in tqdm(range(0, len(usable_indices), batch_size), desc="Retrieving"):
        batch_indices = usable_indices[start : start + batch_size]
        batch = query_matrix[batch_indices]
        scores = batch @ track_matrix.T
        indices = topk_indices(scores, max_k)

        for row_position, user_index in enumerate(batch_indices):
            user_id = user_ids[user_index]
            topk_by_user[user_id] = track_ids[indices[row_position]].tolist()

    for user_id in user_ids:
        topk_by_user.setdefault(user_id, [])

    return topk_by_user


def build_retrieval_entries(
    records: list[dict[str, Any]],
    topk_by_user: dict[str, list[str]],
    max_k: int,
) -> list[dict[str, Any]]:
    entries = []
    for record in records:
        entries.append(
            {
                "session_id": record["session_id"],
                "user_id": record["user_id"],
                "turn_number": int(record["turn_number"]),
                "predicted_track_ids": topk_by_user.get(record["user_id"], [])[:max_k],
                "predicted_response": "",
            }
        )
    return entries


def compute_recall_scores(
    records: list[dict[str, Any]],
    topk_by_user: dict[str, list[str]],
    k_values: list[int],
    user_status: dict[str, str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    for record in records:
        user_id = record["user_id"]
        predictions = topk_by_user.get(user_id, [])
        ground_truth_track_id = record["ground_truth_track_id"]
        row = {
            "session_id": record["session_id"],
            "user_id": user_id,
            "turn_number": int(record["turn_number"]),
            "user_status": user_status.get(user_id, "unknown"),
            "ground_truth_track_id": ground_truth_track_id,
        }
        for k in k_values:
            row[f"recall@{k}"] = float(ground_truth_track_id in predictions[:k])
        rows.append(row)

    per_query = pd.DataFrame(rows)
    metric_columns = [f"recall@{k}" for k in k_values]
    turn_wise = per_query.groupby("turn_number", sort=True)[metric_columns].mean()
    macro = turn_wise.mean(axis=0).to_dict()
    micro = per_query[metric_columns].mean(axis=0).to_dict()

    scores: dict[str, Any] = {
        **{column: float(value) for column, value in macro.items()},
        "micro": {column: float(value) for column, value in micro.items()},
        "turn_wise": {
            str(turn_number): {
                column: float(value)
                for column, value in values.items()
            }
            for turn_number, values in turn_wise.to_dict(orient="index").items()
        },
    }
    return scores, per_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CF-BPR cosine top-500 retrieval recall."
    )
    parser.add_argument("--tid", default="exp002_cf_bpr_top500")
    parser.add_argument("--eval_dataset", default="devset")
    parser.add_argument("--eval_dataset_name", default=CHALLENGE_DATASET)
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--ground_truth_path", default=None)
    parser.add_argument("--track_embedding_dataset_name", default=TRACK_EMBEDDINGS_DATASET)
    parser.add_argument("--track_split", default="all_tracks")
    parser.add_argument("--user_embedding_dataset_name", default=USER_EMBEDDINGS_DATASET)
    parser.add_argument(
        "--user_splits",
        nargs="+",
        default=list(DEFAULT_USER_SPLITS),
        help="User embedding splits to load.",
    )
    parser.add_argument("--embedding_column", default="cf-bpr")
    parser.add_argument("--top_k", type=int, default=500)
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help="Recall cutoffs. Values larger than --top_k are ignored.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--missing_user_strategy",
        choices=["mean_user", "none"],
        default="mean_user",
        help="Fallback for missing or empty user CF-BPR embeddings.",
    )
    parser.add_argument("--retrieval_dir", default="exp/retrieval")
    parser.add_argument("--score_dir", default="exp/scores")
    parser.add_argument(
        "--no_save_retrieval",
        action="store_true",
        help="Compute recall without writing the top-k retrieval JSON.",
    )
    parser.add_argument(
        "--save_per_query",
        action="store_true",
        help="Also save per session-turn recall rows as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = sorted({k for k in args.k_values if k <= args.top_k})
    if not k_values:
        raise ValueError("At least one --k_values entry must be <= --top_k.")

    print("Loading track CF-BPR embeddings...")
    track_ids, track_matrix = load_track_embedding_matrix(
        args.track_embedding_dataset_name,
        args.track_split,
        args.embedding_column,
    )
    embedding_dim = track_matrix.shape[1]
    print(f"Loaded {len(track_ids)} tracks with dim={embedding_dim}.")

    print("Loading user CF-BPR embeddings...")
    user_lookup, user_split, invalid_users = load_user_embedding_lookup(
        args.user_embedding_dataset_name,
        args.user_splits,
        args.embedding_column,
        embedding_dim,
    )
    print(
        f"Loaded {len(user_lookup)} non-empty user embeddings "
        f"from splits={args.user_splits}."
    )

    print("Loading eval ground truth...")
    records = load_eval_records(
        args.eval_dataset_name,
        args.eval_split,
        args.ground_truth_path,
    )
    user_ids = sorted({record["user_id"] for record in records})
    print(f"Loaded {len(records)} eval turns across {len(user_ids)} users.")

    fallback_vector = None
    if args.missing_user_strategy == "mean_user":
        fallback_vector = mean_user_vector(user_lookup)
        if fallback_vector is None:
            print("Mean-user fallback is unavailable; missing users will score zero.")

    query_matrix, user_status = build_user_query_matrix(
        user_ids,
        user_lookup,
        fallback_vector,
    )
    status_counts = pd.Series(user_status).value_counts().to_dict()
    print(f"User retrieval status: {status_counts}")

    topk_by_user = retrieve_topk_by_user(
        user_ids,
        query_matrix,
        track_ids,
        track_matrix,
        args.top_k,
        args.batch_size,
        user_status,
    )

    scores, per_query = compute_recall_scores(
        records,
        topk_by_user,
        k_values,
        user_status,
    )
    track_id_set = set(track_ids.tolist())
    ground_truth_track_ids = {
        record["ground_truth_track_id"]
        for record in records
    }
    covered_eval_turns = sum(
        record["ground_truth_track_id"] in track_id_set
        for record in records
    )
    scores["metadata"] = {
        "tid": args.tid,
        "eval_dataset": args.eval_dataset,
        "eval_dataset_name": args.eval_dataset_name,
        "eval_split": args.eval_split,
        "track_embedding_dataset_name": args.track_embedding_dataset_name,
        "track_split": args.track_split,
        "user_embedding_dataset_name": args.user_embedding_dataset_name,
        "user_splits": args.user_splits,
        "embedding_column": args.embedding_column,
        "top_k": args.top_k,
        "k_values": k_values,
        "track_count": int(len(track_ids)),
        "eval_turn_count": int(len(records)),
        "eval_user_count": int(len(user_ids)),
        "unique_ground_truth_track_count": int(len(ground_truth_track_ids)),
        "covered_unique_ground_truth_track_count": int(
            len(ground_truth_track_ids & track_id_set)
        ),
        "covered_eval_turn_count": int(covered_eval_turns),
        "non_empty_user_embedding_count": int(len(user_lookup)),
        "invalid_user_embedding_count": int(len(invalid_users)),
        "eval_user_status_counts": {
            key: int(value)
            for key, value in status_counts.items()
        },
        "missing_user_strategy": args.missing_user_strategy,
    }

    retrieval_path = Path(args.retrieval_dir) / args.eval_dataset / f"{args.tid}.json"
    if not args.no_save_retrieval:
        retrieval_entries = build_retrieval_entries(records, topk_by_user, args.top_k)
        retrieval_path.parent.mkdir(parents=True, exist_ok=True)
        with retrieval_path.open("w", encoding="utf-8") as f:
            json.dump(retrieval_entries, f, ensure_ascii=False)

    score_path = Path(args.score_dir) / args.eval_dataset / f"{args.tid}_recall.json"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open("w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    if args.save_per_query:
        per_query_path = (
            Path(args.score_dir) / args.eval_dataset / f"{args.tid}_per_query.csv"
        )
        per_query.to_csv(per_query_path, index=False)
        print(f"Saved per-query rows to {per_query_path}")

    if not args.no_save_retrieval:
        print(f"Saved retrieval to {retrieval_path}")
    print(f"Saved recall scores to {score_path}")
    print("Macro recall:")
    for k in k_values:
        print(f"  recall@{k}: {scores[f'recall@{k}']:.6f}")


if __name__ == "__main__":
    main()
