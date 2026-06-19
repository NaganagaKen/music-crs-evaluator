from __future__ import annotations

import argparse
import json
import re
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
TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_'+-]*")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "you",
}


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def flatten_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        values: list[str] = []
        for nested in value.values():
            values.extend(flatten_text_values(nested))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for nested in value:
            values.extend(flatten_text_values(nested))
        return values
    text = value_to_text(value).strip()
    return [text] if text else []


def normalize_label(value: Any) -> str:
    text = value_to_text(value).strip().casefold()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def attribute_token(prefix: str, value: Any) -> str:
    label = normalize_label(value)
    return f"__{prefix}__:{label}" if label else ""


def word_tokens(prefix: str, value: Any, max_tokens: int) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for text in flatten_text_values(value):
        for raw_token in TOKEN_PATTERN.findall(text.casefold()):
            token = raw_token.strip("_'+-")
            if len(token) < 2 or token in STOPWORDS or token in seen:
                continue
            seen.add(token)
            tokens.append(f"__{prefix}__:{token}")
            if 0 <= max_tokens <= len(tokens):
                return tokens
    return tokens


def phrase_token(prefix: str, value: Any, max_words: int = 6) -> str:
    words = [
        token.strip("_'+-")
        for text in flatten_text_values(value)
        for token in TOKEN_PATTERN.findall(text.casefold())
    ]
    words = [word for word in words if len(word) >= 2 and word not in STOPWORDS]
    if not words or len(words) > max_words:
        return ""
    return f"__{prefix}_phrase__:{'_'.join(words)}"


def unique_tokens(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


def unique_token_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for relation, token in pairs:
        key = (relation, token)
        if token and key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def parse_year(value: Any) -> int | None:
    text = value_to_text(value)
    match = re.search(r"(\d{4})", text)
    if not match:
        return None
    year = int(match.group(1))
    if 1800 <= year <= 2100:
        return year
    return None


def release_tokens(value: Any) -> list[str]:
    year = parse_year(value)
    if year is None:
        return []
    decade = (year // 10) * 10
    return [
        f"__release_year__:{year}",
        f"__release_decade__:{decade}s",
    ]


def popularity_token(value: Any, bin_size: int) -> str:
    try:
        popularity = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(popularity):
        return ""
    lower = int(popularity // bin_size) * bin_size
    upper = lower + bin_size - 1
    return f"__popularity_bin__:{lower}_{upper}"


def duration_token(value: Any, bin_seconds: int) -> str:
    try:
        duration_ms = int(value)
    except (TypeError, ValueError):
        return ""
    if duration_ms <= 0:
        return ""
    duration_seconds = duration_ms // 1000
    lower = (duration_seconds // bin_seconds) * bin_seconds
    upper = lower + bin_seconds - 1
    return f"__duration_bin__:{lower}_{upper}s"


def isrc_tokens(values: Any) -> list[str]:
    tokens: list[str] = []
    for isrc in flatten_text_values(values):
        label = normalize_label(isrc).upper()
        if len(label) >= 2:
            tokens.append(f"__isrc_country__:{label[:2].casefold()}")
        if len(label) >= 5:
            tokens.append(f"__isrc_prefix__:{label[:5].casefold()}")
    return unique_tokens(tokens)


def profile_tokens(profile: Any, include_user_id: bool) -> list[str]:
    if not isinstance(profile, dict):
        return []
    tokens: list[str] = []
    categorical_fields = [
        "age_group",
        "country_code",
        "country_name",
        "gender",
        "preferred_language",
        "preferred_musical_culture",
        "user_split",
    ]
    for field in categorical_fields:
        token = attribute_token(f"profile_{field}", profile.get(field))
        if token:
            tokens.append(token)
    if include_user_id:
        token = attribute_token("user", profile.get("user_id"))
        if token:
            tokens.append(token)
    tokens.extend(word_tokens("profile_text", profile.get("preferred_musical_culture"), 8))
    tokens.extend(word_tokens("text", profile.get("preferred_musical_culture"), 8))
    return unique_tokens(tokens)


def goal_tokens(goal: Any, max_goal_tokens: int) -> list[str]:
    if not isinstance(goal, dict):
        return []
    tokens: list[str] = []
    for field in ["category", "specificity"]:
        token = attribute_token(f"goal_{field}", goal.get(field))
        if token:
            tokens.append(token)
    tokens.extend(word_tokens("goal_text", goal.get("listener_goal"), max_goal_tokens))
    tokens.extend(word_tokens("text", goal.get("listener_goal"), max_goal_tokens))
    return unique_tokens(tokens)


def session_date_tokens(session_date: Any) -> list[str]:
    year = parse_year(session_date)
    if year is None:
        return []
    decade = (year // 10) * 10
    return [f"__session_year__:{year}", f"__session_decade__:{decade}s"]


def user_text_tokens(text: Any, max_user_text_tokens: int) -> list[str]:
    return unique_tokens(
        word_tokens("user_text", text, max_user_text_tokens)
        + word_tokens("text", text, max_user_text_tokens)
    )


def first_content(rows: list[dict[str, Any]], role: str) -> str:
    for row in rows:
        if row.get("role") == role:
            return value_to_text(row.get("content"))
    return ""


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


def context_tokens_for_turn(
    item: dict[str, Any],
    conversations: list[dict[str, Any]],
    target_turn_number: int,
    max_user_text_tokens: int,
    max_goal_tokens: int,
    include_user_id_token: bool,
) -> list[str]:
    tokens: list[str] = []
    tokens.extend(
        profile_tokens(
            item.get("user_profile"),
            include_user_id=include_user_id_token,
        )
    )
    tokens.extend(goal_tokens(item.get("conversation_goal"), max_goal_tokens))
    tokens.extend(session_date_tokens(item.get("session_date")))

    user_rows = [
        row
        for row in conversations
        if row.get("role") == "user"
        and row.get("turn_number") is not None
        and int(row["turn_number"]) <= target_turn_number
    ]
    user_rows.sort(key=lambda row: int(row["turn_number"]))
    for row in user_rows:
        tokens.extend(user_text_tokens(row.get("content"), max_user_text_tokens))
    return unique_tokens(tokens)


def load_dataset_split(
    dataset_name: str,
    split: str,
    arrow_path: str | None,
) -> Dataset:
    return (
        Dataset.from_file(arrow_path)
        if arrow_path is not None
        else load_dataset(dataset_name, split=split)
    )


def load_training_sequences(
    dataset_name: str,
    split: str,
    arrow_path: str | None,
    max_user_text_tokens: int,
    max_goal_tokens: int,
    include_user_id_token: bool,
) -> tuple[list[list[str]], list[list[str]], dict[str, int]]:
    dataset = load_dataset_split(dataset_name, split, arrow_path)
    session_sequences: list[list[str]] = []
    context_sequences: list[list[str]] = []
    relation_counts: Counter[str] = Counter()

    for item in tqdm(dataset, desc="Building train context corpus"):
        conversations = list(item["conversations"])
        music_sequence = extract_music_sequence(conversations)
        if music_sequence:
            session_sequences.append(music_sequence)

        turn_numbers = sorted(
            {
                int(row["turn_number"])
                for row in conversations
                if row.get("role") == "user" and row.get("turn_number") is not None
            }
        )
        for turn_number in turn_numbers:
            current_rows = [
                row
                for row in conversations
                if row.get("turn_number") is not None
                and int(row["turn_number"]) == turn_number
            ]
            track_id = first_content(current_rows, "music")
            if not track_id:
                continue
            context_tokens = context_tokens_for_turn(
                item=item,
                conversations=conversations,
                target_turn_number=turn_number,
                max_user_text_tokens=max_user_text_tokens,
                max_goal_tokens=max_goal_tokens,
                include_user_id_token=include_user_id_token,
            )
            for token in context_tokens:
                context_sequences.append([track_id, token])
                relation_counts[token.split(":", maxsplit=1)[0].strip("_")] += 1

    stats = {
        "training_session_count": len(session_sequences),
        "training_interaction_count": int(sum(map(len, session_sequences))),
        "training_context_sequence_count": len(context_sequences),
        **{
            f"training_context_{relation}_relation_count": int(count)
            for relation, count in relation_counts.items()
        },
    }
    return session_sequences, context_sequences, stats


def load_eval_context_tokens(
    dataset_name: str,
    split: str,
    arrow_path: str | None,
    max_user_text_tokens: int,
    max_goal_tokens: int,
    include_user_id_token: bool,
) -> dict[tuple[str, int], list[str]]:
    dataset = load_dataset_split(dataset_name, split, arrow_path)
    contexts: dict[tuple[str, int], list[str]] = {}

    for item in tqdm(dataset, desc="Building eval query contexts"):
        session_id = value_to_text(item["session_id"])
        conversations = list(item["conversations"])
        turn_numbers = sorted(
            {
                int(row["turn_number"])
                for row in conversations
                if row.get("role") == "user" and row.get("turn_number") is not None
            }
        )
        for turn_number in turn_numbers:
            contexts[(session_id, turn_number)] = context_tokens_for_turn(
                item=item,
                conversations=conversations,
                target_turn_number=turn_number,
                max_user_text_tokens=max_user_text_tokens,
                max_goal_tokens=max_goal_tokens,
                include_user_id_token=include_user_id_token,
            )
    return contexts


def load_metadata_sequences(
    dataset_name: str,
    split: str,
    arrow_path: str | None,
    max_tags_per_track: int,
    max_name_tokens_per_field: int,
    popularity_bin_size: int,
    duration_bin_seconds: int,
) -> tuple[list[str], list[list[str]], dict[str, int]]:
    dataset = load_dataset_split(dataset_name, split, arrow_path)
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

    for row in tqdm(dataset, desc="Building rich metadata corpus"):
        track_id = value_to_text(row["track_id"])
        if not track_id or track_id in seen_track_ids:
            continue
        seen_track_ids.add(track_id)
        track_ids.append(track_id)

        attribute_tokens: list[tuple[str, str]] = []
        attribute_tokens.extend(
            ("artist_id", token)
            for value in (row.get("artist_id") or [])
            if (token := attribute_token("artist_id", value))
        )
        attribute_tokens.extend(
            ("album_id", token)
            for value in (row.get("album_id") or [])
            if (token := attribute_token("album_id", value))
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
        max_tag_text_tokens = (
            max_tags_per_track * 2 if max_tags_per_track >= 0 else 64
        )
        attribute_tokens.extend(
            ("text", token)
            for token in word_tokens("text", row.get("tag_list"), max_tag_text_tokens)
        )

        for field in ["track_name", "artist_name", "album_name"]:
            phrase = phrase_token(field, row.get(field))
            if phrase:
                attribute_tokens.append((f"{field}_phrase", phrase))
            attribute_tokens.extend(
                (field, token)
                for token in word_tokens(
                    field,
                    row.get(field),
                    max_name_tokens_per_field,
                )
            )
            attribute_tokens.extend(
                ("text", token)
                for token in word_tokens(
                    "text",
                    row.get(field),
                    max_name_tokens_per_field,
                )
            )

        attribute_tokens.extend(
            ("release", token) for token in release_tokens(row.get("release_date"))
        )
        pop_token = popularity_token(row.get("popularity"), popularity_bin_size)
        if pop_token:
            attribute_tokens.append(("popularity", pop_token))
        dur_token = duration_token(row.get("duration"), duration_bin_seconds)
        if dur_token:
            attribute_tokens.append(("duration", dur_token))
        attribute_tokens.extend(("isrc", token) for token in isrc_tokens(row.get("ISRC")))

        attribute_tokens = unique_token_pairs(attribute_tokens)

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


def load_ground_truth(
    path: str,
    eval_contexts: dict[tuple[str, int], list[str]],
) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or not records:
        raise ValueError("Ground truth must be a non-empty JSON list.")

    required = {"session_id", "user_id", "turn_number", "ground_truth_track_id"}
    missing = required - set(records[0])
    if missing:
        raise ValueError(f"Ground truth is missing fields: {sorted(missing)}")

    normalized = []
    for record in records:
        session_id = value_to_text(record["session_id"])
        turn_number = int(record["turn_number"])
        normalized.append(
            {
                "session_id": session_id,
                "user_id": value_to_text(record["user_id"]),
                "turn_number": turn_number,
                "ground_truth_track_id": value_to_text(
                    record["ground_truth_track_id"]
                ),
                "context_tokens": eval_contexts.get((session_id, turn_number), []),
            }
        )
    normalized.sort(key=lambda row: (row["session_id"], row["turn_number"]))
    return normalized


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def load_or_train_item2vec(
    sequences: list[list[str]],
    model_path: Path,
    model_config: dict[str, Any],
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

    config_path = model_path.with_suffix(model_path.suffix + ".config.json")
    if model_path.exists() and not force_retrain:
        if config_path.exists():
            saved_config = read_json(config_path)
            if saved_config != model_config:
                raise ValueError(
                    f"Saved model config at {config_path} does not match "
                    "current arguments. Use --force_retrain or a new "
                    "--model_path."
                )
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
    write_json(config_path, model_config)
    print(f"Saved Item2Vec model to {model_path}")
    print(f"Saved Item2Vec config to {config_path}")
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


def mean_vector(tokens: list[str], model: Any) -> tuple[np.ndarray | None, int]:
    in_vocab = [token for token in tokens if token in model.wv]
    if not in_vocab:
        return None, 0
    matrix = np.asarray([model.wv[token] for token in in_vocab], dtype=np.float32)
    return matrix.mean(axis=0), len(in_vocab)


def build_user_vector(
    history_track_ids: list[str],
    context_tokens: list[str],
    model: Any,
    history_weight: float,
    context_weight: float,
) -> tuple[np.ndarray | None, str, int, int]:
    history_vector, in_vocab_history_count = mean_vector(history_track_ids, model)
    context_vector, in_vocab_context_count = mean_vector(context_tokens, model)

    components: list[np.ndarray] = []
    weights: list[float] = []
    if history_vector is not None and history_weight > 0:
        components.append(history_vector)
        weights.append(history_weight)
    if context_vector is not None and context_weight > 0:
        components.append(context_vector)
        weights.append(context_weight)

    if not components:
        if not history_track_ids and not context_tokens:
            status = "no_history_or_context"
        elif history_track_ids and not context_tokens:
            status = "all_history_oov"
        elif context_tokens and not history_track_ids:
            status = "all_context_oov"
        else:
            status = "all_query_oov"
        return None, status, in_vocab_history_count, in_vocab_context_count

    user_vector = np.average(np.vstack(components), axis=0, weights=weights)
    user_vector = normalize_vector(user_vector)
    if user_vector is None:
        return (
            None,
            "zero_norm_user_vector",
            in_vocab_history_count,
            in_vocab_context_count,
        )
    if history_vector is not None and context_vector is not None:
        status = "history_and_context"
    elif history_vector is not None:
        status = "history_only"
    else:
        status = "context_only"
    return user_vector, status, in_vocab_history_count, in_vocab_context_count


def build_retrieval(
    records: list[dict[str, Any]],
    model: Any,
    catalog_track_ids: list[str],
    top_k: int,
    batch_size: int,
    history_weight: float,
    context_weight: float,
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
        user_vector, status, in_vocab_history_count, in_vocab_context_count = (
            build_user_vector(
                history_track_ids=history,
                context_tokens=record["context_tokens"],
                model=model,
                history_weight=history_weight,
                context_weight=context_weight,
            )
        )
        status_counts[status] += 1
        contexts.append(
            {
                "record": record,
                "history_track_ids": list(history),
                "user_vector": user_vector,
                "history_count": len(history),
                "context_token_count": len(record["context_tokens"]),
                "in_vocab_history_count": in_vocab_history_count,
                "in_vocab_context_count": in_vocab_context_count,
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
                "predicted_response": "rich-context item2vec cosine retrieval",
            }
        )
        evaluation_rows.append(
            {
                "session_id": record["session_id"],
                "user_id": record["user_id"],
                "turn_number": record["turn_number"],
                "ground_truth_track_id": record["ground_truth_track_id"],
                "history_count": context["history_count"],
                "context_token_count": context["context_token_count"],
                "in_vocab_history_count": context["in_vocab_history_count"],
                "in_vocab_context_count": context["in_vocab_context_count"],
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
            "mean_context_token_count": float(
                turn_rows["context_token_count"].mean()
            ),
            "mean_in_vocab_context_count": float(
                turn_rows["in_vocab_context_count"].mean()
            ),
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
            "Train rich context Item2Vec on train sessions, all-track "
            "metadata, and train turn context, then retrieve by cosine "
            "similarity to history plus current query-context vectors."
        )
    )
    parser.add_argument(
        "--tid",
        default="exp010_rich_item2vec_context_top500",
    )
    parser.add_argument("--dataset_name", default=CHALLENGE_DATASET)
    parser.add_argument("--train_split", default="train")
    parser.add_argument(
        "--train_arrow_path",
        default=None,
        help="Optional local train Arrow file; bypasses Hugging Face loading.",
    )
    parser.add_argument("--eval_split", default="test")
    parser.add_argument(
        "--eval_arrow_path",
        default=None,
        help="Optional local dev/test Arrow file for query context tokens.",
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
        default=20,
        help="Most frequent metadata tags retained per track; -1 keeps all.",
    )
    parser.add_argument(
        "--max_name_tokens_per_field",
        type=int,
        default=8,
        help="Maximum word tokens retained from each name metadata field.",
    )
    parser.add_argument("--max_user_text_tokens", type=int, default=32)
    parser.add_argument("--max_goal_tokens", type=int, default=48)
    parser.add_argument("--popularity_bin_size", type=int, default=10)
    parser.add_argument("--duration_bin_seconds", type=int, default=30)
    parser.add_argument(
        "--include_user_id_token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use user_id as a warm-user context token.",
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
    parser.add_argument("--history_weight", type=float, default=1.0)
    parser.add_argument("--context_weight", type=float, default=0.35)
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
        default="exp/models/exp010/rich_item2vec.model",
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
    if args.history_weight <= 0 and args.context_weight <= 0:
        raise ValueError("At least one query weight must be positive.")

    session_sequences, train_context_sequences, train_context_stats = (
        load_training_sequences(
            args.dataset_name,
            args.train_split,
            args.train_arrow_path,
            args.max_user_text_tokens,
            args.max_goal_tokens,
            args.include_user_id_token,
        )
    )
    catalog_track_ids, metadata_sequences, metadata_stats = (
        load_metadata_sequences(
            dataset_name=args.track_metadata_dataset_name,
            split=args.track_metadata_split,
            arrow_path=args.track_metadata_arrow_path,
            max_tags_per_track=args.max_tags_per_track,
            max_name_tokens_per_field=args.max_name_tokens_per_field,
            popularity_bin_size=args.popularity_bin_size,
            duration_bin_seconds=args.duration_bin_seconds,
        )
    )
    sequences = session_sequences + metadata_sequences + train_context_sequences
    training_track_counts = Counter(
        track_id for sequence in session_sequences for track_id in sequence
    )

    model_config = {
        "dataset_name": args.dataset_name,
        "train_split": args.train_split,
        "train_arrow_path": args.train_arrow_path,
        "track_metadata_dataset_name": args.track_metadata_dataset_name,
        "track_metadata_split": args.track_metadata_split,
        "track_metadata_arrow_path": args.track_metadata_arrow_path,
        "max_tags_per_track": args.max_tags_per_track,
        "max_name_tokens_per_field": args.max_name_tokens_per_field,
        "max_user_text_tokens": args.max_user_text_tokens,
        "max_goal_tokens": args.max_goal_tokens,
        "popularity_bin_size": args.popularity_bin_size,
        "duration_bin_seconds": args.duration_bin_seconds,
        "include_user_id_token": args.include_user_id_token,
        "vector_size": args.vector_size,
        "window": args.window,
        "min_count": args.min_count,
        "negative": args.negative,
        "epochs": args.epochs,
        "workers": args.workers,
        "seed": args.seed,
    }
    model = load_or_train_item2vec(
        sequences=sequences,
        model_path=Path(args.model_path),
        model_config=model_config,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        negative=args.negative,
        epochs=args.epochs,
        workers=args.workers,
        seed=args.seed,
        force_retrain=args.force_retrain,
    )

    eval_contexts = load_eval_context_tokens(
        args.dataset_name,
        args.eval_split,
        args.eval_arrow_path,
        args.max_user_text_tokens,
        args.max_goal_tokens,
        args.include_user_id_token,
    )
    records = load_ground_truth(args.ground_truth_path, eval_contexts)
    retrieval_entries, evaluation_rows, status_counts = build_retrieval(
        records=records,
        model=model,
        catalog_track_ids=catalog_track_ids,
        top_k=args.top_k,
        batch_size=args.batch_size,
        history_weight=args.history_weight,
        context_weight=args.context_weight,
    )
    scores, per_query = compute_recall_scores(evaluation_rows, k_values)

    target_in_vocab_count = int(per_query["target_in_item2vec_vocab"].sum())
    scores["metadata"] = {
        "tid": args.tid,
        "eval_dataset": args.eval_dataset,
        "dataset_name": args.dataset_name,
        "train_split": args.train_split,
        "train_arrow_path": args.train_arrow_path,
        "eval_split": args.eval_split,
        "eval_arrow_path": args.eval_arrow_path,
        "track_metadata_dataset_name": args.track_metadata_dataset_name,
        "track_metadata_split": args.track_metadata_split,
        "track_metadata_arrow_path": args.track_metadata_arrow_path,
        "max_tags_per_track": args.max_tags_per_track,
        "max_name_tokens_per_field": args.max_name_tokens_per_field,
        "max_user_text_tokens": args.max_user_text_tokens,
        "max_goal_tokens": args.max_goal_tokens,
        "popularity_bin_size": args.popularity_bin_size,
        "duration_bin_seconds": args.duration_bin_seconds,
        "include_user_id_token": args.include_user_id_token,
        "ground_truth_path": args.ground_truth_path,
        "model_path": args.model_path,
        "model": "rich_context_word2vec_skipgram",
        "metadata_relations": [
            "artist_id",
            "album_id",
            "tag_list",
            "track_name",
            "artist_name",
            "album_name",
            "release_year",
            "release_decade",
            "popularity_bin",
            "duration_bin",
            "isrc_country",
            "isrc_prefix",
        ],
        "training_context_relations": [
            "user_text",
            "profile",
            "conversation_goal",
            "session_date",
            "user_id",
        ],
        "user_vector": "weighted_mean_of_prior_turn_item_vectors_and_context_tokens",
        "history_weight": args.history_weight,
        "context_weight": args.context_weight,
        "retrieval_metric": "cosine_similarity",
        "exclude_history_items": True,
        "empty_history_strategy": "use_context_tokens_when_available",
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
        **train_context_stats,
        "training_unique_track_count": int(len(training_track_counts)),
        "total_training_sequence_count": int(len(sequences)),
        **metadata_stats,
        "candidate_track_count": int(len(catalog_track_ids)),
        "item2vec_vocab_size": int(len(model.wv)),
        "eval_context_key_count": int(len(eval_contexts)),
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
    write_json(retrieval_path, retrieval_entries)

    score_path = (
        Path(args.score_dir)
        / args.eval_dataset
        / f"{args.tid}_recall.json"
    )
    write_json(score_path, scores)

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
