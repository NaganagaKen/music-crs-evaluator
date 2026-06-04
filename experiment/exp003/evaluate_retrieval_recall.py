from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_K_VALUES = (20, 50, 100, 200, 500)


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def build_prediction_lookup(
    predictions: list[dict[str, Any]],
) -> dict[tuple[str, int], list[str]]:
    lookup: dict[tuple[str, int], list[str]] = {}
    duplicates: list[tuple[str, int]] = []

    for row in predictions:
        key = (str(row["session_id"]), int(row["turn_number"]))
        if key in lookup:
            duplicates.append(key)
        lookup[key] = [str(track_id) for track_id in row.get("predicted_track_ids", [])]

    if duplicates:
        raise ValueError(f"Duplicate predictions found. First duplicate: {duplicates[0]}")
    return lookup


def compute_recall(
    ground_truth: list[dict[str, Any]],
    prediction_lookup: dict[tuple[str, int], list[str]],
    k_values: list[int],
    drop_empty_predictions: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    missing_prediction_count = 0
    dropped_empty_prediction_count = 0
    dropped_user_ids: set[str] = set()
    all_user_ids: set[str] = set()
    evaluated_user_ids: set[str] = set()

    for truth in ground_truth:
        session_id = str(truth["session_id"])
        user_id = str(truth["user_id"])
        turn_number = int(truth["turn_number"])
        ground_truth_track_id = str(truth["ground_truth_track_id"])
        key = (session_id, turn_number)
        predictions = prediction_lookup.get(key)

        all_user_ids.add(user_id)
        if predictions is None:
            missing_prediction_count += 1
            predictions = []

        if drop_empty_predictions and not predictions:
            dropped_empty_prediction_count += 1
            dropped_user_ids.add(user_id)
            continue

        evaluated_user_ids.add(user_id)
        row = {
            "session_id": session_id,
            "user_id": user_id,
            "turn_number": turn_number,
            "ground_truth_track_id": ground_truth_track_id,
            "candidate_count": len(predictions),
        }
        for k in k_values:
            row[f"recall@{k}"] = float(ground_truth_track_id in predictions[:k])
        rows.append(row)

    per_query = pd.DataFrame(rows)
    metric_columns = [f"recall@{k}" for k in k_values]
    if per_query.empty:
        macro = {column: 0.0 for column in metric_columns}
        micro = {column: 0.0 for column in metric_columns}
        turn_wise: dict[str, dict[str, float]] = {}
    else:
        turn_wise_df = per_query.groupby("turn_number", sort=True)[metric_columns].mean()
        macro = turn_wise_df.mean(axis=0).to_dict()
        micro = per_query[metric_columns].mean(axis=0).to_dict()
        turn_wise = {
            str(turn_number): {
                column: float(value)
                for column, value in values.items()
            }
            for turn_number, values in turn_wise_df.to_dict(orient="index").items()
        }

    scores: dict[str, Any] = {
        **{column: float(value) for column, value in macro.items()},
        "micro": {column: float(value) for column, value in micro.items()},
        "turn_wise": turn_wise,
        "metadata": {
            "total_ground_truth_turn_count": int(len(ground_truth)),
            "evaluated_turn_count": int(len(per_query)),
            "dropped_empty_prediction_turn_count": int(dropped_empty_prediction_count),
            "missing_prediction_turn_count": int(missing_prediction_count),
            "total_ground_truth_user_count": int(len(all_user_ids)),
            "evaluated_user_count": int(len(evaluated_user_ids)),
            "dropped_user_count": int(len(dropped_user_ids - evaluated_user_ids)),
            "drop_empty_predictions": bool(drop_empty_predictions),
        },
    }
    return scores, per_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute recall from a saved retrieval JSON."
    )
    parser.add_argument("--input_tid", default="exp003_cf_bpr_l1_top500_none")
    parser.add_argument(
        "--output_tid",
        default=None,
        help="Defaults to '<input_tid>_drop_empty_eval'.",
    )
    parser.add_argument("--eval_dataset", default="devset")
    parser.add_argument("--ground_truth_path", default="exp/ground_truth/devset.json")
    parser.add_argument("--retrieval_dir", default="exp/retrieval")
    parser.add_argument("--score_dir", default="exp/scores")
    parser.add_argument(
        "--retrieval_path",
        default=None,
        help="Override retrieval path. Otherwise uses retrieval_dir/eval_dataset/input_tid.json.",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
    )
    parser.add_argument(
        "--keep_empty_predictions",
        action="store_true",
        help="Keep empty prediction rows in the denominator as zero recall.",
    )
    parser.add_argument(
        "--save_per_query",
        action="store_true",
        help="Also save per session-turn recall rows as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = sorted(set(args.k_values))
    output_tid = args.output_tid or f"{args.input_tid}_drop_empty_eval"
    retrieval_path = (
        Path(args.retrieval_path)
        if args.retrieval_path is not None
        else Path(args.retrieval_dir) / args.eval_dataset / f"{args.input_tid}.json"
    )
    ground_truth_path = Path(args.ground_truth_path)

    ground_truth = load_json(ground_truth_path)
    predictions = load_json(retrieval_path)
    prediction_lookup = build_prediction_lookup(predictions)

    scores, per_query = compute_recall(
        ground_truth=ground_truth,
        prediction_lookup=prediction_lookup,
        k_values=k_values,
        drop_empty_predictions=not args.keep_empty_predictions,
    )
    scores["metadata"].update(
        {
            "input_tid": args.input_tid,
            "output_tid": output_tid,
            "eval_dataset": args.eval_dataset,
            "ground_truth_path": str(ground_truth_path),
            "retrieval_path": str(retrieval_path),
            "k_values": k_values,
            "input_prediction_count": int(len(predictions)),
        }
    )

    score_path = Path(args.score_dir) / args.eval_dataset / f"{output_tid}_recall.json"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open("w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    if args.save_per_query:
        per_query_path = Path(args.score_dir) / args.eval_dataset / f"{output_tid}_per_query.csv"
        per_query.to_csv(per_query_path, index=False)
        print(f"Saved per-query rows to {per_query_path}")

    print(f"Saved recall scores to {score_path}")
    print(
        "Evaluated "
        f"{scores['metadata']['evaluated_turn_count']} / "
        f"{scores['metadata']['total_ground_truth_turn_count']} turns"
    )
    print("Macro recall:")
    for k in k_values:
        print(f"  recall@{k}: {scores[f'recall@{k}']:.6f}")


if __name__ == "__main__":
    main()
