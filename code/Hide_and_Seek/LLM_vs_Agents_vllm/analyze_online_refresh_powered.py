#!/usr/bin/env python3
"""Analyze environment-online probability GRPO rollout progression."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_probability_grpo_eval import numeric_field, parse_object


ITER_PATTERN = re.compile(r"iter(?P<iteration>\d+)")


def load_rollouts(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("iter*/rollouts/*.csv")):
        match = ITER_PATTERN.search(str(path))
        if match is None:
            continue
        rollout_iteration = int(match.group("iteration"))
        frame = pd.read_csv(path)
        parsed = frame["llm_final_text"].map(parse_object)
        frame["predicted_probability"] = parsed.map(
            lambda item: numeric_field(item, "opponent_action_1_probability")
        )
        frame["predicted_action"] = parsed.map(
            lambda item: numeric_field(item, "opponent_next_action")
        )
        frame["best_action"] = parsed.map(lambda item: numeric_field(item, "my_best_action"))
        frame["final_action"] = parsed.map(lambda item: numeric_field(item, "final_action"))
        opponent_index = pd.to_numeric(frame["opponent_player_index"], errors="coerce")
        frame["opponent_action"] = np.where(
            opponent_index.eq(0), frame["choice_agent0"], frame["choice_agent1"]
        )
        frame["target_probability"] = pd.to_numeric(
            frame["opponent_action_1_probability"], errors="coerce"
        )
        frame["llm_payoff"] = pd.to_numeric(frame["llm_payoff"], errors="coerce")
        frame["stage"] = rollout_iteration - 1
        frame["trained_steps"] = frame["stage"] * 30
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No rollout CSVs found under {root}")
    return pd.concat(frames, ignore_index=True)


def add_metrics(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    valid_q = out["predicted_probability"].between(0, 1)
    valid_target = out["target_probability"].between(0, 1)
    valid_action = out["predicted_action"].isin([0, 1])
    valid_final = out["final_action"].isin([0, 1])
    out["structure_valid"] = (
        valid_q
        & valid_action
        & out["best_action"].isin([0, 1])
        & valid_final
    )
    out["target_squared_error"] = np.where(
        valid_q & valid_target,
        (out["predicted_probability"] - out["target_probability"]) ** 2,
        np.nan,
    )
    out["action_squared_error"] = np.where(
        valid_q,
        (out["predicted_probability"] - out["opponent_action"]) ** 2,
        np.nan,
    )
    out["prediction_correct"] = np.where(
        valid_action,
        out["predicted_action"].eq(out["opponent_action"]),
        np.nan,
    )
    implied = out["predicted_probability"].gt(0.5).astype(float)
    out["probability_prediction_consistent"] = np.where(
        valid_q & valid_action, implied.eq(out["predicted_action"]), np.nan
    )
    expected_best = np.where(
        out["llm_role"].eq("seeker"),
        out["predicted_action"],
        1 - out["predicted_action"],
    )
    out["prediction_best_consistent"] = np.where(
        valid_action & out["best_action"].isin([0, 1]),
        out["best_action"].eq(expected_best),
        np.nan,
    )
    out["best_final_consistent"] = np.where(
        out["best_action"].isin([0, 1]) & valid_final,
        out["best_action"].eq(out["final_action"]),
        np.nan,
    )
    return out


def match_table(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["stage", "trained_steps", "llm_role", "opponent_name", "match_id"]
    matches = (
        data.groupby(keys, sort=True)
        .agg(
            match_payoff=("llm_payoff", "sum"),
            mean_round_payoff=("llm_payoff", "mean"),
            target_probability_mse=("target_squared_error", "mean"),
            action_brier=("action_squared_error", "mean"),
            prediction_accuracy=("prediction_correct", "mean"),
            structure_parse_rate=("structure_valid", "mean"),
            mean_q=("predicted_probability", "mean"),
            mean_target=("target_probability", "mean"),
            probability_prediction_consistency=("probability_prediction_consistent", "mean"),
            prediction_best_consistency=("prediction_best_consistent", "mean"),
            best_final_consistency=("best_final_consistent", "mean"),
        )
        .reset_index()
    )
    matches["win"] = matches["match_payoff"].gt(0).astype(float)
    matches["tie"] = matches["match_payoff"].eq(0).astype(float)
    return matches


def summarize(matches: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        matches.groupby(group_cols, sort=True)
        .agg(
            n_matches=("match_id", "size"),
            mean_match_payoff=("match_payoff", "mean"),
            mean_round_payoff=("mean_round_payoff", "mean"),
            win_rate=("win", "mean"),
            tie_rate=("tie", "mean"),
            target_probability_mse=("target_probability_mse", "mean"),
            action_brier=("action_brier", "mean"),
            prediction_accuracy=("prediction_accuracy", "mean"),
            structure_parse_rate=("structure_parse_rate", "mean"),
            mean_q=("mean_q", "mean"),
            mean_target=("mean_target", "mean"),
            probability_prediction_consistency=("probability_prediction_consistency", "mean"),
            prediction_best_consistency=("prediction_best_consistency", "mean"),
            best_final_consistency=("best_final_consistency", "mean"),
        )
        .reset_index()
    )


def bootstrap_stage(matches: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    metrics = [
        "target_probability_mse",
        "mean_round_payoff",
        "win",
        "prediction_accuracy",
    ]
    for (stage, trained_steps), group in matches.groupby(["stage", "trained_steps"], sort=True):
        row: dict[str, float] = {
            "stage": float(stage),
            "trained_steps": float(trained_steps),
            "n_matches": float(len(group)),
        }
        sample_index = rng.integers(0, len(group), size=(n_boot, len(group)))
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            estimates = np.nanmean(values[sample_index], axis=1)
            row[metric] = float(np.nanmean(values))
            row[f"{metric}_ci_low"] = float(np.nanquantile(estimates, 0.025))
            row[f"{metric}_ci_high"] = float(np.nanquantile(estimates, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_progression(
    overall: pd.DataFrame,
    by_role: pd.DataFrame,
    bootstrap: pd.DataFrame,
    path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {"overall": "#202020", "seeker": "#2474A6", "hider": "#D65F3A"}
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), constrained_layout=True)
    panels = [
        ("target_probability_mse", "Policy-probability MSE", axes[0, 0]),
        ("prediction_accuracy", "Opponent-action accuracy", axes[0, 1]),
        ("mean_round_payoff", "Mean online payoff", axes[1, 0]),
        ("win_rate", "Match win rate", axes[1, 1]),
    ]
    for metric, ylabel, ax in panels:
        ax.plot(
            overall["trained_steps"],
            overall[metric],
            marker="o",
            color=colors["overall"],
            linewidth=2,
            label="Overall",
            zorder=3,
        )
        ci_metric = "win" if metric == "win_rate" else metric
        ax.fill_between(
            bootstrap["trained_steps"],
            bootstrap[f"{ci_metric}_ci_low"],
            bootstrap[f"{ci_metric}_ci_high"],
            color=colors["overall"],
            alpha=0.10,
            linewidth=0,
        )
        for role in ("seeker", "hider"):
            subset = by_role[by_role["llm_role"].eq(role)]
            ax.plot(
                subset["trained_steps"],
                subset[metric],
                marker="o",
                linewidth=1.3,
                color=colors[role],
                alpha=0.9,
                label=role.title(),
            )
        ax.set_xlabel("Completed GRPO steps before rollout")
        ax.set_ylabel(ylabel)
        ax.set_xticks(overall["trained_steps"])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        if metric in {"prediction_accuracy", "win_rate"}:
            ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)
            ax.set_ylim(0, 1)
        elif metric == "mean_round_payoff":
            ax.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
            ax.set_ylim(-1, 1)
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper center")
    fig.suptitle(
        "Environment-online probability GRPO pilot\n"
        "Each point is the next refreshed trajectory distribution",
        fontsize=11,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data = add_metrics(load_rollouts(args.run_root))
    matches = match_table(data)
    overall = summarize(matches, ["stage", "trained_steps"])
    by_role = summarize(matches, ["stage", "trained_steps", "llm_role"])
    by_cell = summarize(
        matches, ["stage", "trained_steps", "llm_role", "opponent_name"]
    )
    bootstrap = bootstrap_stage(matches, args.bootstrap, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / "round_level_metrics.csv", index=False)
    matches.to_csv(args.output_dir / "match_level_metrics.csv", index=False)
    overall.to_csv(args.output_dir / "progression_overall.csv", index=False)
    by_role.to_csv(args.output_dir / "progression_by_role.csv", index=False)
    by_cell.to_csv(args.output_dir / "progression_by_role_opponent.csv", index=False)
    bootstrap.to_csv(args.output_dir / "progression_bootstrap_ci.csv", index=False)
    plot_progression(
        overall,
        by_role,
        bootstrap,
        args.output_dir / "online_refresh_progression.png",
    )
    print(overall.to_string(index=False))
    print("\nBy role and opponent:")
    print(by_cell.to_string(index=False))


if __name__ == "__main__":
    main()
