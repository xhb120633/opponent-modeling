#!/usr/bin/env python3
"""Evaluate simple observable-history baselines for opponent next-action prediction."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluate_predictive_action_logits_vllm import balanced_sample, summarize_group


IDENTIFIER_COLUMNS = {
    "match_id",
    "repeat_idx",
    "ego_agent",
    "opponent_family",
    "role",
    "split",
}
LATENT_TARGET_COLUMNS = {"p_self", "p_op"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--matched-llm-rows")
    parser.add_argument(
        "--target-mode",
        choices=("bridge_mode", "realized_action"),
        default="bridge_mode",
    )
    parser.add_argument("--samples-per-split", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-estimators", type=int, default=500)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def target_frame(rows: list[dict[str, Any]], split: str) -> pd.DataFrame:
    records = []
    for row in rows:
        meta = row["metadata"]
        records.append(
            {
                "example_id": str(meta["example_id"]),
                "match_id": str(meta["match_id"]),
                "decision_round": int(meta["decision_round"]),
                "role_expected": str(meta["role"]),
                "opponent_family_expected": str(meta["opponent_family"]),
                "target_action": int(row["target"]["opponent_next_action"]),
                "eval_split": split,
            }
        )
    frame = pd.DataFrame(records)
    if frame["example_id"].duplicated().any():
        raise ValueError(f"Duplicate example ids in {split}")
    return frame


def join_features(targets: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    joined = targets.merge(
        features,
        on=["match_id", "decision_round"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[~joined["_merge"].eq("both"), "example_id"].tolist()
        raise ValueError(f"Missing feature rows: {missing[:5]}")
    joined = joined.drop(columns="_merge")
    if not joined["role_expected"].eq(joined["role"]).all():
        raise ValueError("Role mismatch between frozen JSONL and source features")
    if not joined["opponent_family_expected"].eq(joined["opponent_family"]).all():
        raise ValueError("Opponent-family mismatch between frozen JSONL and source features")
    return joined


def behavior_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = IDENTIFIER_COLUMNS | LATENT_TARGET_COLUMNS | {
        "example_id",
        "target_action",
        "eval_split",
        "role_expected",
        "opponent_family_expected",
        "realized_opponent_action",
    }
    columns = [column for column in frame.columns if column not in excluded]
    nonnumeric = [column for column in columns if not pd.api.types.is_numeric_dtype(frame[column])]
    if nonnumeric:
        raise ValueError(f"Unexpected nonnumeric behavior features: {nonnumeric}")
    return columns


def make_models(numeric: list[str], seed: int, n_estimators: int) -> dict[str, tuple[Any, list[str]]]:
    behavior = numeric + ["role"]
    transformer_scaled = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("role", OneHotEncoder(handle_unknown="ignore"), ["role"]),
        ]
    )
    transformer_tree = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            ("role", OneHotEncoder(handle_unknown="ignore"), ["role"]),
        ]
    )
    return {
        "logistic_behavior": (
            make_pipeline(transformer_scaled, LogisticRegression(C=1.0, max_iter=2000)),
            behavior,
        ),
        "extra_trees_behavior": (
            make_pipeline(
                transformer_tree,
                ExtraTreesClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=10,
                    max_features=0.75,
                    # Avoid platform-specific worker spawning and keep this
                    # small diagnostic exactly reproducible.
                    n_jobs=1,
                    random_state=seed,
                ),
            ),
            behavior,
        ),
    }


def score_row(row: pd.Series, model: str, probability: float) -> dict[str, Any]:
    p1 = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    target = int(row["target_action"])
    prediction = int(p1 >= 0.5)
    correct = int(prediction == target)
    confidence = max(p1, 1.0 - p1)
    return {
        "split": row["eval_split"],
        "model": model,
        "example_id": row["example_id"],
        "match_id": row["match_id"],
        "role": row["role"],
        "opponent_family": row["opponent_family"],
        "target_action": target,
        "p_action_1": p1,
        "predicted_action": prediction,
        "valid_logits": 1,
        "brier": (p1 - target) ** 2,
        "constant_05_brier": 0.25,
        "nll": -(target * math.log(p1) + (1 - target) * math.log(1.0 - p1)),
        "correct": correct,
        "confidence": confidence,
        "confidence_if_correct": confidence if correct else None,
        "confidence_if_incorrect": confidence if not correct else None,
    }


def validate_matched_ids(eval_frames: dict[str, pd.DataFrame], path: Path) -> None:
    llm = pd.read_csv(path)
    llm = llm[llm["model"].eq("parent")]
    for split, frame in eval_frames.items():
        expected = set(llm.loc[llm["split"].eq(split), "example_id"].astype(str))
        actual = set(frame["example_id"].astype(str))
        if actual != expected:
            raise ValueError(
                f"Matched LLM ids differ for {split}: baseline={len(actual)} llm={len(expected)}"
            )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_feature_importance(models: dict[str, tuple[Any, list[str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, (pipeline, _) in models.items():
        names = pipeline.steps[0][1].get_feature_names_out()
        estimator = pipeline.steps[-1][1]
        if hasattr(estimator, "coef_"):
            raw = np.asarray(estimator.coef_[0], dtype=float)
        else:
            raw = np.asarray(estimator.feature_importances_, dtype=float)
        for feature, value in zip(names, raw):
            rows.append(
                {
                    "model": model_name,
                    "feature": str(feature),
                    "signed_value": float(value),
                    "importance": float(abs(value)),
                }
            )
    return sorted(rows, key=lambda row: (row["model"], -row["importance"], row["feature"]))


def add_realized_action(features: pd.DataFrame) -> pd.DataFrame:
    """Recover action at round t from round t+1's lag-1 observable feature."""
    result = features.sort_values(["match_id", "decision_round"]).copy()
    grouped = result.groupby("match_id", sort=False)
    result["next_decision_round"] = grouped["decision_round"].shift(-1)
    result["realized_opponent_action"] = grouped["opponent_lag1"].shift(-1)
    contiguous = result["next_decision_round"].eq(result["decision_round"] + 1)
    result.loc[~contiguous, "realized_opponent_action"] = np.nan
    return result.drop(columns="next_decision_round")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root) / "predictive_state"
    raw = {
        "train": load_jsonl(data_root / "train.jsonl"),
        "id": balanced_sample(
            load_jsonl(data_root / "val.jsonl"), args.samples_per_split, args.seed
        ),
        "ood": balanced_sample(
            load_jsonl(data_root / "ood.jsonl"), args.samples_per_split, args.seed + 1
        ),
    }
    targets = {split: target_frame(rows, split) for split, rows in raw.items()}
    features = pd.read_csv(args.feature_csv)
    if args.target_mode == "realized_action":
        features = add_realized_action(features)
    frames = {split: join_features(frame, features) for split, frame in targets.items()}
    if args.target_mode == "realized_action":
        for split, frame in frames.items():
            frame = frame.dropna(subset=["realized_opponent_action"]).copy()
            frame["target_action"] = frame["realized_opponent_action"].astype(int)
            frames[split] = frame
        if args.matched_llm_rows:
            raise ValueError("Matched LLM comparison is only valid for bridge_mode targets")
    if args.matched_llm_rows:
        validate_matched_ids({"id": frames["id"], "ood": frames["ood"]}, Path(args.matched_llm_rows))

    numeric = behavior_feature_columns(frames["train"])
    models = make_models(numeric, args.seed, args.n_estimators)
    for model, columns in models.values():
        model.fit(frames["train"][columns], frames["train"]["target_action"])

    scored: list[dict[str, Any]] = []
    for split in ("id", "ood"):
        frame = frames[split]
        probabilities: dict[str, np.ndarray] = {
            "constant_05": np.full(len(frame), 0.5),
            "empirical_frequency": frame["opponent_rate_1"].to_numpy(dtype=float),
        }
        for name, (model, columns) in models.items():
            probabilities[name] = model.predict_proba(frame[columns])[:, 1]
        for name, values in probabilities.items():
            scored.extend(score_row(row, name, p1) for (_, row), p1 in zip(frame.iterrows(), values))

    summary = []
    for (split, model), group in pd.DataFrame(scored).groupby(["split", "model"], sort=True):
        records = group.to_dict("records")
        summary.append({"split": split, "model": model, **summarize_group(records)})

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "matched_rows.csv", scored)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "feature_importance.csv", model_feature_importance(models))
    manifest = {
        "seed": args.seed,
        "target_mode": args.target_mode,
        "samples_per_split": args.samples_per_split,
        "train_examples": len(frames["train"]),
        "behavior_features": numeric + ["role"],
        "excluded_latent_or_metadata_features": sorted(
            LATENT_TARGET_COLUMNS | {"ego_agent", "opponent_family"}
        ),
        "matched_llm_ids_verified": bool(args.matched_llm_rows),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
