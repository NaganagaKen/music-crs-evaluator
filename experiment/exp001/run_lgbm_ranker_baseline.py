from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy import sparse
from tqdm import tqdm

try:
    from lightgbm import LGBMRanker
except ImportError as exc:  # pragma: no cover - import guard for local setup
    raise SystemExit(
        "lightgbm is required. Install the baseline dependencies with "
        "`uv pip install -r requirments.txt`."
    ) from exc

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
except ImportError as exc:  # pragma: no cover - import guard for local setup
    raise SystemExit(
        "scikit-learn is required. Install the baseline dependencies with "
        "`uv pip install -r requirments.txt`."
    ) from exc


CHALLENGE_DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"
BLIND_A_DATASET = "talkpl-ai/TalkPlayData-Challenge-Blind-A"
TRACK_METADATA_DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['_-][a-z0-9]+)?")

FEATURE_NAMES = [
    "bm25_score",
    "cosine_score",
    "initial_score",
    "initial_rank_recip",
    "popularity",
    "duration_min",
    "release_year",
    "track_age_at_session",
    "turn_number",
    "history_track_count",
    "query_track_overlap",
    "query_track_jaccard",
    "artist_in_query",
    "album_in_query",
    "tag_overlap_query",
    "same_artist_as_history",
    "same_album_as_history",
    "tag_overlap_history",
]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set, np.ndarray)):
        texts = [value_to_text(v) for v in value]
        return " ".join(text for text in texts if text)
    if isinstance(value, dict):
        texts = [value_to_text(v) for v in value.values()]
        return " ".join(text for text in texts if text)
    return str(value)


def token_set(value: Any) -> set[str]:
    return set(tokenize(value_to_text(value)))


def parse_year(value: Any) -> float:
    match = re.search(r"\d{4}", value_to_text(value))
    if not match:
        return -1.0
    return float(match.group(0))


def first_content(rows: Iterable[dict[str, Any]], role: str) -> str:
    for row in rows:
        if row.get("role") == role:
            return value_to_text(row.get("content"))
    return ""


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    max_score = float(np.max(scores)) if scores.size else 0.0
    if max_score <= 0.0:
        return np.zeros_like(scores, dtype=np.float32)
    return scores / max_score


@dataclass
class TrackStore:
    df: pd.DataFrame
    track_ids: np.ndarray
    track_texts: list[str]
    popularity: np.ndarray
    duration_min: np.ndarray
    release_year: np.ndarray
    token_sets: list[set[str]]
    artist_token_sets: list[set[str]]
    album_token_sets: list[set[str]]
    tag_token_sets: list[set[str]]
    id_to_index: dict[str, int]
    popularity_order: np.ndarray

    @classmethod
    def from_huggingface(cls, dataset_name: str, split: str) -> "TrackStore":
        dataset = load_dataset(dataset_name, split=split)
        df = dataset.to_pandas()
        df = df.reset_index(drop=True)
        df["track_text"] = df.apply(build_track_text, axis=1)
        df["release_year_num"] = df["release_date"].map(parse_year)
        df["popularity_num"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
        df["duration_min_num"] = (
            pd.to_numeric(df["duration"], errors="coerce").fillna(0.0) / 60000.0
        )

        track_ids = df["track_id"].astype(str).to_numpy()
        popularity = df["popularity_num"].astype(np.float32).to_numpy()
        duration_min = df["duration_min_num"].astype(np.float32).to_numpy()
        release_year = df["release_year_num"].astype(np.float32).to_numpy()
        track_texts = df["track_text"].fillna("").astype(str).tolist()

        return cls(
            df=df,
            track_ids=track_ids,
            track_texts=track_texts,
            popularity=popularity,
            duration_min=duration_min,
            release_year=release_year,
            token_sets=[set(tokenize(text)) for text in track_texts],
            artist_token_sets=[token_set(v) for v in df["artist_name"].tolist()],
            album_token_sets=[token_set(v) for v in df["album_name"].tolist()],
            tag_token_sets=[token_set(v) for v in df["tag_list"].tolist()],
            id_to_index={track_id: idx for idx, track_id in enumerate(track_ids)},
            popularity_order=np.argsort(-popularity),
        )

    def metadata_text(self, track_id: str) -> str:
        idx = self.id_to_index.get(track_id)
        if idx is None:
            return track_id
        return self.track_texts[idx]


def build_track_text(row: pd.Series) -> str:
    parts: list[str] = []
    weighted_columns = [
        ("track_name", 3),
        ("artist_name", 3),
        ("album_name", 2),
        ("tag_list", 1),
        ("release_date", 1),
    ]
    for column, weight in weighted_columns:
        text = value_to_text(row.get(column))
        if text:
            parts.extend([text] * weight)
    return " ".join(parts)


@dataclass
class QueryExample:
    session_id: str
    user_id: str
    turn_number: int
    session_year: float
    query_text: str
    history_track_ids: list[str]
    positive_track_id: str | None = None


@dataclass
class RetrievalResult:
    indices: np.ndarray
    bm25_scores: np.ndarray
    cosine_scores: np.ndarray
    initial_scores: np.ndarray
    initial_ranks: np.ndarray
    all_bm25_scores: np.ndarray | None = None
    all_cosine_scores: np.ndarray | None = None
    all_initial_scores: np.ndarray | None = None

    def has_index(self, track_index: int) -> bool:
        return bool(np.any(self.indices == track_index))

    def add_index(self, track_index: int) -> "RetrievalResult":
        if self.has_index(track_index):
            return self
        if self.all_bm25_scores is None or self.all_cosine_scores is None or self.all_initial_scores is None:
            raise ValueError("Full score arrays are required to append a missing positive.")

        initial_score = float(self.all_initial_scores[track_index])
        if np.max(self.all_initial_scores) > 0.0:
            initial_rank = int(np.count_nonzero(self.all_initial_scores > initial_score) + 1)
        else:
            initial_rank = int(len(self.indices) + 1)

        return RetrievalResult(
            indices=np.append(self.indices, track_index),
            bm25_scores=np.append(self.bm25_scores, self.all_bm25_scores[track_index]),
            cosine_scores=np.append(self.cosine_scores, self.all_cosine_scores[track_index]),
            initial_scores=np.append(self.initial_scores, initial_score),
            initial_ranks=np.append(self.initial_ranks, initial_rank),
            all_bm25_scores=self.all_bm25_scores,
            all_cosine_scores=self.all_cosine_scores,
            all_initial_scores=self.all_initial_scores,
        )

    def take_positions(self, positions: np.ndarray) -> "RetrievalResult":
        return RetrievalResult(
            indices=self.indices[positions],
            bm25_scores=self.bm25_scores[positions],
            cosine_scores=self.cosine_scores[positions],
            initial_scores=self.initial_scores[positions],
            initial_ranks=self.initial_ranks[positions],
            all_bm25_scores=self.all_bm25_scores,
            all_cosine_scores=self.all_cosine_scores,
            all_initial_scores=self.all_initial_scores,
        )


class CandidateRetriever:
    def __init__(
        self,
        track_store: TrackStore,
        candidate_count: int,
        method: str,
        bm25_weight: float,
        cosine_weight: float,
        max_features: int | None,
        min_df: int,
        bm25_k1: float,
        bm25_b: float,
    ) -> None:
        self.track_store = track_store
        self.candidate_count = candidate_count
        self.method = method
        self.bm25_weight = bm25_weight
        self.cosine_weight = cosine_weight

        self.count_vectorizer = CountVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=False,
            max_features=max_features,
            min_df=min_df,
        )
        count_matrix = self.count_vectorizer.fit_transform(track_store.track_texts)
        self.bm25_matrix = self._build_bm25_matrix(count_matrix, bm25_k1, bm25_b)

        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=False,
            max_features=max_features,
            min_df=min_df,
            norm="l2",
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(track_store.track_texts).astype(
            np.float32
        )

    @staticmethod
    def _build_bm25_matrix(
        count_matrix: sparse.spmatrix,
        k1: float,
        b: float,
    ) -> sparse.csr_matrix:
        matrix = count_matrix.tocsr().astype(np.float32)
        n_docs = matrix.shape[0]
        doc_len = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
        avg_doc_len = float(doc_len.mean()) if doc_len.size else 1.0
        if avg_doc_len <= 0.0:
            avg_doc_len = 1.0

        df = np.diff(matrix.tocsc().indptr).astype(np.float32)
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        coo = matrix.tocoo(copy=True)
        denom = coo.data + k1 * (1.0 - b + b * doc_len[coo.row] / avg_doc_len)
        coo.data = idf[coo.col] * (coo.data * (k1 + 1.0) / denom)
        return coo.tocsr().astype(np.float32)

    def retrieve(self, query_text: str, keep_full_scores: bool = False) -> RetrievalResult:
        bm25_scores = self._bm25_scores(query_text)
        cosine_scores = self._cosine_scores(query_text)

        if self.method == "bm25":
            initial_scores = bm25_scores
        elif self.method == "cosine":
            initial_scores = cosine_scores
        else:
            initial_scores = (
                self.bm25_weight * normalize_scores(bm25_scores)
                + self.cosine_weight * normalize_scores(cosine_scores)
            ).astype(np.float32)

        top_indices = self._top_indices(initial_scores)
        initial_ranks = np.arange(1, len(top_indices) + 1, dtype=np.int32)

        return RetrievalResult(
            indices=top_indices,
            bm25_scores=bm25_scores[top_indices],
            cosine_scores=cosine_scores[top_indices],
            initial_scores=initial_scores[top_indices],
            initial_ranks=initial_ranks,
            all_bm25_scores=bm25_scores if keep_full_scores else None,
            all_cosine_scores=cosine_scores if keep_full_scores else None,
            all_initial_scores=initial_scores if keep_full_scores else None,
        )

    def _bm25_scores(self, query_text: str) -> np.ndarray:
        query_vec = self.count_vectorizer.transform([query_text]).astype(np.float32)
        if query_vec.nnz == 0:
            return np.zeros(len(self.track_store.track_ids), dtype=np.float32)
        query_vec.data = np.ones_like(query_vec.data, dtype=np.float32)
        return (self.bm25_matrix @ query_vec.T).toarray().ravel().astype(np.float32)

    def _cosine_scores(self, query_text: str) -> np.ndarray:
        query_vec = self.tfidf_vectorizer.transform([query_text])
        if query_vec.nnz == 0:
            return np.zeros(len(self.track_store.track_ids), dtype=np.float32)
        return (self.tfidf_matrix @ query_vec.T).toarray().ravel().astype(np.float32)

    def _top_indices(self, scores: np.ndarray) -> np.ndarray:
        n_tracks = len(scores)
        k = min(self.candidate_count, n_tracks)
        if k == 0:
            return np.array([], dtype=np.int64)

        if not np.any(scores > 0.0):
            return self.track_store.popularity_order[:k].astype(np.int64)

        if k == n_tracks:
            candidate_indices = np.arange(n_tracks, dtype=np.int64)
        else:
            candidate_indices = np.argpartition(scores, -k)[-k:].astype(np.int64)
        return candidate_indices[np.argsort(-scores[candidate_indices])]


class FeatureBuilder:
    def __init__(self, track_store: TrackStore) -> None:
        self.track_store = track_store

    def build(self, example: QueryExample, retrieval: RetrievalResult) -> list[list[float]]:
        query_tokens = set(tokenize(example.query_text))
        history_indices = [
            self.track_store.id_to_index[track_id]
            for track_id in example.history_track_ids
            if track_id in self.track_store.id_to_index
        ]
        history_artist_tokens = set().union(
            *(self.track_store.artist_token_sets[idx] for idx in history_indices)
        ) if history_indices else set()
        history_album_tokens = set().union(
            *(self.track_store.album_token_sets[idx] for idx in history_indices)
        ) if history_indices else set()
        history_tag_tokens = set().union(
            *(self.track_store.tag_token_sets[idx] for idx in history_indices)
        ) if history_indices else set()

        rows: list[list[float]] = []
        for pos, track_idx in enumerate(retrieval.indices):
            track_tokens = self.track_store.token_sets[track_idx]
            artist_tokens = self.track_store.artist_token_sets[track_idx]
            album_tokens = self.track_store.album_token_sets[track_idx]
            tag_tokens = self.track_store.tag_token_sets[track_idx]

            overlap = len(query_tokens & track_tokens)
            union_size = len(query_tokens | track_tokens)
            release_year = float(self.track_store.release_year[track_idx])
            track_age = (
                example.session_year - release_year
                if example.session_year > 0.0 and release_year > 0.0
                else -1.0
            )

            rows.append(
                [
                    float(retrieval.bm25_scores[pos]),
                    float(retrieval.cosine_scores[pos]),
                    float(retrieval.initial_scores[pos]),
                    1.0 / float(retrieval.initial_ranks[pos] + 1),
                    float(self.track_store.popularity[track_idx]),
                    float(self.track_store.duration_min[track_idx]),
                    release_year,
                    track_age,
                    float(example.turn_number),
                    float(len(history_indices)),
                    float(overlap),
                    float(overlap / union_size) if union_size else 0.0,
                    float(bool(query_tokens & artist_tokens)),
                    float(bool(query_tokens & album_tokens)),
                    float(len(query_tokens & tag_tokens)),
                    float(bool(history_artist_tokens & artist_tokens)),
                    float(bool(history_album_tokens & album_tokens)),
                    float(len(history_tag_tokens & tag_tokens)),
                ]
            )
        return rows


def build_query_text(
    item: dict[str, Any],
    conversations: list[dict[str, Any]],
    target_turn_number: int,
    current_user_query: str,
    track_store: TrackStore,
    include_user_profile: bool,
    include_goal_text: bool,
) -> str:
    parts = [current_user_query, current_user_query, current_user_query]

    for row in conversations:
        turn_number = int(row.get("turn_number", 0) or 0)
        if turn_number >= target_turn_number:
            continue
        role = row.get("role")
        content = value_to_text(row.get("content"))
        if role == "music":
            parts.append(track_store.metadata_text(content))
        elif content:
            parts.append(content)

    if include_user_profile:
        parts.append(value_to_text(item.get("user_profile")))
    if include_goal_text:
        parts.append(value_to_text(item.get("conversation_goal")))

    return " ".join(part for part in parts if part)


def iter_query_examples(
    dataset: Iterable[dict[str, Any]],
    track_store: TrackStore,
    prediction_mode: str,
    include_user_profile: bool,
    include_goal_text: bool,
    with_labels: bool,
) -> list[QueryExample]:
    examples: list[QueryExample] = []
    for item in dataset:
        conversations = list(item["conversations"])
        session_year = parse_year(item.get("session_date"))

        if prediction_mode == "last_turn":
            turn_numbers = [int(conversations[-1]["turn_number"])]
        else:
            turn_numbers = sorted(
                {
                    int(row["turn_number"])
                    for row in conversations
                    if row.get("role") == "user" and row.get("turn_number") is not None
                }
            )

        for turn_number in turn_numbers:
            current_rows = [
                row for row in conversations if int(row.get("turn_number", 0) or 0) == turn_number
            ]
            user_query = first_content(current_rows, "user")
            if not user_query:
                continue

            positive_track_id = first_content(current_rows, "music") if with_labels else None
            history_track_ids = [
                value_to_text(row.get("content"))
                for row in conversations
                if row.get("role") == "music"
                and int(row.get("turn_number", 0) or 0) < turn_number
            ]
            query_text = build_query_text(
                item=item,
                conversations=conversations,
                target_turn_number=turn_number,
                current_user_query=user_query,
                track_store=track_store,
                include_user_profile=include_user_profile,
                include_goal_text=include_goal_text,
            )
            examples.append(
                QueryExample(
                    session_id=value_to_text(item["session_id"]),
                    user_id=value_to_text(item["user_id"]),
                    turn_number=turn_number,
                    session_year=session_year,
                    query_text=query_text,
                    history_track_ids=history_track_ids,
                    positive_track_id=positive_track_id or None,
                )
            )
    return examples


def select_training_candidates(
    retrieval: RetrievalResult,
    positive_index: int,
    negative_sample_size: int | None,
    rng: np.random.Generator,
) -> RetrievalResult:
    retrieval = retrieval.add_index(positive_index)
    positive_positions = np.flatnonzero(retrieval.indices == positive_index)
    negative_positions = np.flatnonzero(retrieval.indices != positive_index)

    if negative_sample_size is None or negative_sample_size < 0:
        keep_positions = np.concatenate([positive_positions, negative_positions])
    elif len(negative_positions) <= negative_sample_size:
        keep_positions = np.concatenate([positive_positions, negative_positions])
    else:
        hard_count = min(len(negative_positions), max(1, int(negative_sample_size * 0.7)))
        hard_positions = negative_positions[:hard_count]
        remaining_positions = negative_positions[hard_count:]
        random_count = negative_sample_size - len(hard_positions)
        if random_count > 0 and len(remaining_positions) > 0:
            random_positions = rng.choice(
                remaining_positions,
                size=min(random_count, len(remaining_positions)),
                replace=False,
            )
            negative_keep = np.concatenate([hard_positions, random_positions])
        else:
            negative_keep = hard_positions
        keep_positions = np.concatenate([positive_positions, negative_keep])

    keep_positions = keep_positions[np.argsort(retrieval.initial_ranks[keep_positions])]
    return retrieval.take_positions(keep_positions)


def build_training_frame(
    examples: list[QueryExample],
    track_store: TrackStore,
    retriever: CandidateRetriever,
    feature_builder: FeatureBuilder,
    max_train_queries: int | None,
    negative_sample_size: int | None,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    if max_train_queries is not None and max_train_queries > 0 and len(examples) > max_train_queries:
        chosen = np.sort(rng.choice(len(examples), size=max_train_queries, replace=False))
        examples = [examples[idx] for idx in chosen]

    feature_rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    skipped = 0

    for example in tqdm(examples, desc="Building LGBMRanker train rows"):
        if not example.positive_track_id or example.positive_track_id not in track_store.id_to_index:
            skipped += 1
            continue
        positive_index = track_store.id_to_index[example.positive_track_id]
        retrieval = retriever.retrieve(example.query_text, keep_full_scores=True)
        retrieval = select_training_candidates(
            retrieval=retrieval,
            positive_index=positive_index,
            negative_sample_size=negative_sample_size,
            rng=rng,
        )
        rows = feature_builder.build(example, retrieval)
        feature_rows.extend(rows)
        labels.extend((retrieval.indices == positive_index).astype(int).tolist())
        groups.append(len(rows))

    if skipped:
        print(f"Skipped {skipped} training queries because the positive track was unavailable.")
    if not groups:
        raise RuntimeError("No training groups were built. Check dataset splits and track IDs.")

    return pd.DataFrame(feature_rows, columns=FEATURE_NAMES), np.asarray(labels), groups


def train_ranker(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    train_groups: list[int],
    args: argparse.Namespace,
) -> LGBMRanker:
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        importance_type="gain",
    )
    ranker.fit(train_x, train_y, group=train_groups)
    return ranker


def predict_examples(
    examples: list[QueryExample],
    track_store: TrackStore,
    retriever: CandidateRetriever,
    feature_builder: FeatureBuilder,
    ranker: LGBMRanker,
    top_k: int,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for example in tqdm(examples, desc="Predicting"):
        retrieval = retriever.retrieve(example.query_text, keep_full_scores=False)
        feature_rows = feature_builder.build(example, retrieval)
        features = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
        rerank_scores = ranker.predict(features)
        ordered_indices = retrieval.indices[np.argsort(-rerank_scores)]

        predicted_track_ids: list[str] = []
        seen: set[str] = set()
        for track_index in ordered_indices:
            track_id = str(track_store.track_ids[track_index])
            if track_id not in seen:
                predicted_track_ids.append(track_id)
                seen.add(track_id)
            if len(predicted_track_ids) >= top_k:
                break

        for track_index in track_store.popularity_order:
            if len(predicted_track_ids) >= top_k:
                break
            track_id = str(track_store.track_ids[track_index])
            if track_id not in seen:
                predicted_track_ids.append(track_id)
                seen.add(track_id)

        predictions.append(
            {
                "session_id": example.session_id,
                "user_id": example.user_id,
                "turn_number": example.turn_number,
                "predicted_track_ids": predicted_track_ids,
                "predicted_response": "",
            }
        )
    return predictions


def resolve_eval_dataset_name(eval_dataset: str, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    if eval_dataset.lower() in {"blindset_a", "blind_a", "blinda"}:
        return BLIND_A_DATASET
    return CHALLENGE_DATASET


def resolve_prediction_mode(eval_dataset: str, predict_mode: str) -> str:
    if predict_mode != "auto":
        return predict_mode
    if eval_dataset.lower() in {"devset", "dev", "development"}:
        return "all_turns"
    return "last_turn"


def save_feature_importance(ranker: LGBMRanker, output_path: Path) -> None:
    booster = ranker.booster_
    importance = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "importance_gain": booster.feature_importance(importance_type="gain"),
            "importance_split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage RecSys Challenge 2026 baseline: retrieve 300 tracks with "
            "BM25/TF-IDF cosine, then rerank with LightGBM LGBMRanker."
        )
    )
    parser.add_argument("--tid", default="bm25_cosine_lgbmranker_devset")
    parser.add_argument("--eval_dataset", default="devset")
    parser.add_argument("--train_dataset_name", default=CHALLENGE_DATASET)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_dataset_name", default=None)
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--track_metadata_dataset_name", default=TRACK_METADATA_DATASET)
    parser.add_argument("--track_split", default="all_tracks")
    parser.add_argument("--candidate_count", type=int, default=300)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--retrieval_method",
        choices=["bm25", "cosine", "hybrid"],
        default="hybrid",
    )
    parser.add_argument("--bm25_weight", type=float, default=0.65)
    parser.add_argument("--cosine_weight", type=float, default=0.35)
    parser.add_argument("--bm25_k1", type=float, default=1.5)
    parser.add_argument("--bm25_b", type=float, default=0.75)
    parser.add_argument("--max_features", type=int, default=200000)
    parser.add_argument("--min_df", type=int, default=1)
    parser.add_argument("--max_train_queries", type=int, default=30000)
    parser.add_argument(
        "--negative_sample_size",
        type=int,
        default=100,
        help="Number of negatives sampled from the 300 retrieved candidates per query. Use -1 for all.",
    )
    parser.add_argument(
        "--predict_mode",
        choices=["auto", "all_turns", "last_turn"],
        default="auto",
    )
    parser.add_argument(
        "--include_user_profile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include_goal_text",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--min_child_samples", type=int, default=30)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="exp/inference")
    parser.add_argument("--model_dir", default="exp/models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    negative_sample_size = None if args.negative_sample_size < 0 else args.negative_sample_size
    max_features = None if args.max_features <= 0 else args.max_features

    print("Loading track metadata...")
    track_store = TrackStore.from_huggingface(args.track_metadata_dataset_name, args.track_split)

    print("Building BM25 and TF-IDF cosine retrievers...")
    retriever = CandidateRetriever(
        track_store=track_store,
        candidate_count=args.candidate_count,
        method=args.retrieval_method,
        bm25_weight=args.bm25_weight,
        cosine_weight=args.cosine_weight,
        max_features=max_features,
        min_df=args.min_df,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
    )
    feature_builder = FeatureBuilder(track_store)

    print("Loading training conversations...")
    train_dataset = load_dataset(args.train_dataset_name, split=args.train_split)
    train_examples = iter_query_examples(
        dataset=train_dataset,
        track_store=track_store,
        prediction_mode="all_turns",
        include_user_profile=args.include_user_profile,
        include_goal_text=args.include_goal_text,
        with_labels=True,
    )
    print(f"Training query examples: {len(train_examples):,}")

    train_x, train_y, train_groups = build_training_frame(
        examples=train_examples,
        track_store=track_store,
        retriever=retriever,
        feature_builder=feature_builder,
        max_train_queries=args.max_train_queries,
        negative_sample_size=negative_sample_size,
        seed=args.seed,
    )
    print(
        f"LGBMRanker rows: {len(train_x):,}, groups: {len(train_groups):,}, "
        f"positive rows: {int(train_y.sum()):,}"
    )

    print("Training LGBMRanker...")
    ranker = train_ranker(train_x, train_y, train_groups, args)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{args.tid}.txt"
    ranker.booster_.save_model(str(model_path))
    save_feature_importance(ranker, model_dir / f"{args.tid}_feature_importance.csv")

    eval_dataset_name = resolve_eval_dataset_name(args.eval_dataset, args.eval_dataset_name)
    prediction_mode = resolve_prediction_mode(args.eval_dataset, args.predict_mode)
    print(f"Loading eval conversations: {eval_dataset_name} [{args.eval_split}]")
    eval_dataset = load_dataset(eval_dataset_name, split=args.eval_split)
    eval_examples = iter_query_examples(
        dataset=eval_dataset,
        track_store=track_store,
        prediction_mode=prediction_mode,
        include_user_profile=args.include_user_profile,
        include_goal_text=args.include_goal_text,
        with_labels=False,
    )
    print(f"Prediction examples: {len(eval_examples):,} ({prediction_mode})")

    predictions = predict_examples(
        examples=eval_examples,
        track_store=track_store,
        retriever=retriever,
        feature_builder=feature_builder,
        ranker=ranker,
        top_k=args.top_k,
    )

    output_path = Path(args.output_dir) / args.eval_dataset / f"{args.tid}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)
    print(f"Saved predictions to {output_path}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
