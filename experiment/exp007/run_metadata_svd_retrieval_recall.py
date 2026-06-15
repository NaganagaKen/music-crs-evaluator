from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required. Install dependencies from requirments.txt."
    ) from exc


TRACK_METADATA_DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
DEFAULT_K_VALUES = (20, 50, 100, 200, 500)


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set, np.ndarray)):
        texts = [value_to_text(item) for item in value]
        return " ".join(text for text in texts if text)
    if isinstance(value, dict):
        texts = [value_to_text(item) for item in value.values()]
        return " ".join(text for text in texts if text)
    return str(value)


def build_track_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    weighted_columns = [
        ("track_name", 3),
        ("artist_name", 3),
        ("album_name", 2),
        ("tag_list", 1),
        ("release_date", 1),
    ]
    for column, weight in weighted_columns:
        text = value_to_text(row.get(column)).strip()
        if text:
            parts.extend([text] * weight)
    return " ".join(parts)


def load_track_metadata(
    dataset_name: str,
    split: str,
    arrow_path: str | None,
) -> tuple[list[str], list[str]]:
    dataset = (
        Dataset.from_file(arrow_path)
        if arrow_path is not None
        else load_dataset(dataset_name, split=split)
    )
    track_ids: list[str] = []
    track_texts: list[str] = []
    seen_track_ids: set[str] = set()
    for row in tqdm(dataset, desc="Building item documents"):
        track_id = value_to_text(row["track_id"])
        if not track_id or track_id in seen_track_ids:
            continue
        seen_track_ids.add(track_id)
        track_ids.append(track_id)
        track_texts.append(build_track_text(row))
    if not track_ids:
        raise ValueError("Track metadata produced no catalog tracks.")
    return track_ids, track_texts


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


def normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix, dtype=np.float32),
        where=norms > eps,
    ).astype(np.float32)


def load_or_train_item_vectors(
    track_ids: list[str],
    track_texts: list[str],
    model_path: Path,
    n_components: int,
    max_features: int | None,
    min_df: int,
    max_df: float,
    algorithm: str,
    n_iter: int,
    tol: float,
    seed: int,
    force_retrain: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if model_path.exists() and not force_retrain:
        print(f"Loading CountVectorizer + TruncatedSVD model from {model_path}")
        with model_path.open("rb") as f:
            bundle = pickle.load(f)
        if bundle.get("track_ids") != track_ids:
            raise ValueError(
                "Saved model track IDs do not match the current metadata. "
                "Use --force_retrain."
            )
        return np.asarray(bundle["item_vectors"], dtype=np.float32), {
            "vocabulary_size": int(len(bundle["vectorizer"].vocabulary_)),
            "empty_item_document_count": int(bundle["empty_item_document_count"]),
            "explained_variance_ratio_sum": float(
                bundle["svd"].explained_variance_ratio_.sum()
            ),
        }

    vectorizer = CountVectorizer(
        lowercase=True,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )
    print(f"Fitting CountVectorizer on {len(track_texts)} item documents...")
    item_matrix = vectorizer.fit_transform(track_texts)
    empty_item_document_count = int(
        np.count_nonzero(np.asarray(item_matrix.sum(axis=1)).ravel() == 0)
    )

    max_components = min(item_matrix.shape) - 1
    if n_components > max_components:
        raise ValueError(
            f"--n_components must be <= {max_components} for item matrix "
            f"shape {item_matrix.shape}."
        )

    svd = TruncatedSVD(
        n_components=n_components,
        algorithm=algorithm,
        n_iter=n_iter,
        random_state=seed,
        tol=tol,
    )
    print(
        f"Fitting TruncatedSVD on item matrix {item_matrix.shape} "
        f"with {n_components} components..."
    )
    item_vectors = normalize_rows(
        svd.fit_transform(item_matrix).astype(np.float32)
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(
            {
                "track_ids": track_ids,
                "vectorizer": vectorizer,
                "svd": svd,
                "item_vectors": item_vectors,
                "empty_item_document_count": empty_item_document_count,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Saved CountVectorizer + TruncatedSVD model to {model_path}")
    return item_vectors, {
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "empty_item_document_count": empty_item_document_count,
        "explained_variance_ratio_sum": float(
            svd.explained_variance_ratio_.sum()
        ),
    }


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
    item_vectors: np.ndarray,
    track_index: dict[str, int],
) -> tuple[np.ndarray | None, str, int]:
    if not history_track_ids:
        return None, "no_history", 0
    history_indices = [
        track_index[track_id]
        for track_id in history_track_ids
        if track_id in track_index
    ]
    if not history_indices:
        return None, "all_history_oov", 0
    user_vector = normalize_rows(
        item_vectors[np.asarray(history_indices)].mean(axis=0, keepdims=True)
    )[0]
    if not np.any(user_vector):
        return None, "zero_norm_user_vector", len(history_indices)
    return user_vector, "ok", len(history_indices)


def build_retrieval(
    records: list[dict[str, Any]],
    track_ids: list[str],
    item_vectors: np.ndarray,
    top_k: int,
    retrieval_batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    track_id_array = np.asarray(track_ids)
    track_index = {
        track_id: index for index, track_id in enumerate(track_ids)
    }
    status_counts: Counter[str] = Counter()
    history_by_session: dict[str, list[str]] = {}
    contexts: list[dict[str, Any]] = []

    for record in records:
        history = history_by_session.setdefault(record["session_id"], [])
        user_vector, status, in_catalog_history_count = build_user_vector(
            history, item_vectors, track_index
        )
        status_counts[status] += 1
        contexts.append(
            {
                "record": record,
                "history_track_ids": list(history),
                "user_vector": user_vector,
                "history_count": len(history),
                "in_catalog_history_count": in_catalog_history_count,
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
        range(0, len(usable_indices), retrieval_batch_size),
        desc="Retrieving",
    ):
        batch_indices = usable_indices[start : start + retrieval_batch_size]
        query_matrix = np.stack(
            [contexts[index]["user_vector"] for index in batch_indices]
        )
        batch_scores = query_matrix @ item_vectors.T
        for row_index, context_index in enumerate(batch_indices):
            context = contexts[context_index]
            scores = batch_scores[row_index]
            for track_id in set(context["history_track_ids"]):
                index = track_index.get(track_id)
                if index is not None:
                    scores[index] = -np.inf
            indices = topk_indices(scores, top_k)
            context["predicted_track_ids"] = track_id_array[indices].tolist()
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
                    "metadata CountVectorizer TruncatedSVD mean-history "
                    "cosine retrieval"
                ),
            }
        )
        evaluation_rows.append(
            {
                **record,
                "history_count": context["history_count"],
                "in_catalog_history_count": context[
                    "in_catalog_history_count"
                ],
                "retrieval_status": context["retrieval_status"],
                "candidate_count": len(predictions),
                "target_in_catalog": record["ground_truth_track_id"]
                in track_index,
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
            "target_in_catalog_count": int(
                turn_rows["target_in_catalog"].sum()
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
            "Build all-track metadata item vectors with CountVectorizer and "
            "TruncatedSVD, then retrieve by cosine similarity to mean history "
            "vectors."
        )
    )
    parser.add_argument(
        "--tid", default="exp007_metadata_svd_history_top500"
    )
    parser.add_argument(
        "--track_metadata_dataset_name", default=TRACK_METADATA_DATASET
    )
    parser.add_argument("--track_metadata_split", default="all_tracks")
    parser.add_argument("--track_metadata_arrow_path", default=None)
    parser.add_argument(
        "--ground_truth_path", default="exp/ground_truth/devset.json"
    )
    parser.add_argument("--eval_dataset", default="devset")
    parser.add_argument("--top_k", type=int, default=500)
    parser.add_argument(
        "--k_values", type=int, nargs="+", default=list(DEFAULT_K_VALUES)
    )
    parser.add_argument("--n_components", type=int, default=128)
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_df", type=float, default=0.95)
    parser.add_argument(
        "--svd_algorithm", choices=["randomized", "arpack"], default="randomized"
    )
    parser.add_argument("--svd_n_iter", type=int, default=7)
    parser.add_argument("--svd_tol", type=float, default=0.0)
    parser.add_argument("--retrieval_batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_path", default="exp/models/exp007/metadata_svd.pkl"
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

    track_ids, track_texts = load_track_metadata(
        args.track_metadata_dataset_name,
        args.track_metadata_split,
        args.track_metadata_arrow_path,
    )
    item_vectors, model_stats = load_or_train_item_vectors(
        track_ids=track_ids,
        track_texts=track_texts,
        model_path=Path(args.model_path),
        n_components=args.n_components,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        algorithm=args.svd_algorithm,
        n_iter=args.svd_n_iter,
        tol=args.svd_tol,
        seed=args.seed,
        force_retrain=args.force_retrain,
    )
    records = load_ground_truth(args.ground_truth_path)
    retrieval_entries, evaluation_rows, status_counts = build_retrieval(
        records=records,
        track_ids=track_ids,
        item_vectors=item_vectors,
        top_k=args.top_k,
        retrieval_batch_size=args.retrieval_batch_size,
    )
    scores, per_query = compute_recall_scores(evaluation_rows, k_values)

    target_in_catalog_count = int(per_query["target_in_catalog"].sum())
    scores["metadata"] = {
        "tid": args.tid,
        "eval_dataset": args.eval_dataset,
        "track_metadata_dataset_name": args.track_metadata_dataset_name,
        "track_metadata_split": args.track_metadata_split,
        "track_metadata_arrow_path": args.track_metadata_arrow_path,
        "ground_truth_path": args.ground_truth_path,
        "model_path": args.model_path,
        "model": "count_vectorizer_truncated_svd",
        "metadata_fields": [
            "track_name",
            "artist_name",
            "album_name",
            "tag_list",
            "release_date",
        ],
        "user_vector": "mean_of_prior_turn_item_svd_vectors",
        "retrieval_metric": "cosine_similarity",
        "exclude_history_items": True,
        "empty_history_strategy": "empty_predictions",
        "top_k": args.top_k,
        "k_values": k_values,
        "n_components": args.n_components,
        "max_features": args.max_features,
        "min_df": args.min_df,
        "max_df": args.max_df,
        "svd_algorithm": args.svd_algorithm,
        "svd_n_iter": args.svd_n_iter,
        "svd_tol": args.svd_tol,
        "retrieval_batch_size": args.retrieval_batch_size,
        "seed": args.seed,
        "candidate_track_count": int(len(track_ids)),
        **model_stats,
        "eval_turn_count": int(len(records)),
        "eval_session_count": int(len({row["session_id"] for row in records})),
        "eval_user_count": int(len({row["user_id"] for row in records})),
        "target_in_catalog_count": target_in_catalog_count,
        "target_in_catalog_rate": float(target_in_catalog_count / len(records)),
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
