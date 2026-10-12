#!/usr/bin/env python3
"""Summarize base-vs-probability-GRPO online evaluation CSVs."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


def first_existing(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def parse_object(value: object) -> dict[str, object] | None:
    if not isinstance(value, str):
        return None
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = value[start : end + 1]
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return {str(key).strip().lower(): item for key, item in parsed.items()}
    return None


def numeric_field(parsed: dict[str, object] | None, key: str) -> float:
    if parsed is None or key not in parsed or isinstance(parsed[key], bool):
        return np.nan
    try:
        return float(parsed[key])
    except (TypeError, ValueError):
        return np.nan

def summarize_file(path: Path, condition: str) -> dict[str, object]:
    frame = pd.read_csv(path)
    role_col = first_existing(frame, ("llm_role", "role"))
    opponent_col = first_existing(frame, ("opponent", "opponent_name", "opponent_type"))
    match_col = first_existing(frame, ("match_id", "repeat", "repetition"))
    round_col = first_existing(frame, ("round", "round_index", "trial"))
    payoff_col = first_existing(frame, ("llm_payoff", "payoff", "reward"))
    probability_col = first_existing(frame, ("opponent_action_1_probability",))

    parsed = frame["llm_final_text"].map(parse_object)
    predicted_probability = parsed.map(
        lambda item: numeric_field(item, "opponent_action_1_probability")
    )
    predicted_action = parsed.map(lambda item: numeric_field(item, "opponent_next_action"))
    best_action = parsed.map(lambda item: numeric_field(item, "my_best_action"))
    final_action = parsed.map(lambda item: numeric_field(item, "final_action"))

    opponent_index = pd.to_numeric(frame["opponent_player_index"], errors="coerce")
    opponent_action = pd.Series(
        np.where(opponent_index == 0, frame["choice_agent0"], frame["choice_agent1"]),
        index=frame.index,
        dtype=float,
    )
    result: dict[str, object] = {
        "condition": condition,
        "file": path.name,
        "n_rows": len(frame),
        "n_matches": frame[match_col].nunique() if match_col else np.nan,
        "n_rounds": frame[round_col].nunique() if round_col else np.nan,
        "role": frame[role_col].dropna().astype(str).iloc[0] if role_col else "unknown",
        "opponent": (
            frame[opponent_col].dropna().astype(str).iloc[0] if opponent_col else "unknown"
        ),
    }

    if payoff_col:
        payoff = pd.to_numeric(frame[payoff_col], errors="coerce")
        result["mean_round_payoff"] = payoff.mean()
        if match_col:
            match_payoff = frame.assign(_payoff=payoff).groupby(match_col)["_payoff"].sum()
            result["mean_match_payoff"] = match_payoff.mean()
            result["win_rate"] = (match_payoff > 0).mean()
            result["tie_rate"] = (match_payoff == 0).mean()

    valid_probability = predicted_probability.between(0, 1)
    valid_prediction = predicted_action.isin([0, 1]) & opponent_action.isin([0, 1])
    valid_actions = best_action.isin([0, 1]) & final_action.isin([0, 1])
    result["probability_parse_rate"] = valid_probability.mean()
    result["structure_parse_rate"] = (
        valid_probability & predicted_action.isin([0, 1]) & valid_actions
    ).mean()
    result["opponent_prediction_accuracy"] = (
        predicted_action[valid_prediction] == opponent_action[valid_prediction]
    ).mean()
    actual_valid = valid_probability & opponent_action.isin([0, 1])
    result["action_brier_score"] = (
        (predicted_probability[actual_valid] - opponent_action[actual_valid]) ** 2
    ).mean()
    result["mean_predicted_p1"] = predicted_probability[actual_valid].mean()
    result["observed_action1_rate"] = opponent_action[actual_valid].mean()

    if probability_col:
        target_probability = pd.to_numeric(frame[probability_col], errors="coerce")
        target_valid = valid_probability & target_probability.between(0, 1)
        result["mean_target_probability"] = target_probability[target_valid].mean()
        result["target_probability_mse"] = (
            (predicted_probability[target_valid] - target_probability[target_valid]) ** 2
        ).mean()

    role = str(result["role"])
    implied_prediction = (predicted_probability > 0.5).astype(float)
    decision_valid = valid_probability & predicted_action.isin([0, 1])
    result["probability_to_prediction_consistency"] = (
        implied_prediction[decision_valid] == predicted_action[decision_valid]
    ).mean()
    expected_best = predicted_action if role == "seeker" else 1 - predicted_action
    best_valid = predicted_action.isin([0, 1]) & best_action.isin([0, 1])
    result["prediction_to_best_action_consistency"] = (
        expected_best[best_valid] == best_action[best_valid]
    ).mean()
    result["best_to_final_action_consistency"] = (
        best_action[valid_actions] == final_action[valid_actions]
    ).mean()

    optimal_action = opponent_action if role == "seeker" else 1 - opponent_action
    action_valid = final_action.isin([0, 1]) & opponent_action.isin([0, 1])
    result["oracle_action_accuracy"] = (
        final_action[action_valid] == optimal_action[action_valid]
    ).mean()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for condition_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        for csv_path in sorted(condition_dir.glob("*.csv")):
            rows.append(summarize_file(csv_path, condition_dir.name))
    summary = pd.DataFrame(rows).sort_values(["opponent", "role", "condition"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
    print(summary.to_csv(index=False))


if __name__ == "__main__":
    main()
