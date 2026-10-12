#!/usr/bin/env python3
"""Summarize and plot iterative online GRPO checkpoint evaluations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_RE = re.compile(r"_(sft_s0300|iter(\d{2})_s(\d{3}))_(seeker|hider|selfplay)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--old-summary-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def checkpoint_info(path: Path) -> tuple[str, int, str, int]:
    match = LABEL_RE.search(path.stem)
    if not match:
        raise ValueError(f"Cannot identify checkpoint from {path.name}")
    label = match.group(1)
    if label == "sft_s0300":
        return label, 0, "00", 0
    iteration = match.group(2)
    inner_step = int(match.group(3))
    cumulative_step = (int(iteration) - 1) * 75 + inner_step
    return label, cumulative_step, iteration, inner_step


def parse_reply(value: object) -> dict[str, object] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        object_match = re.search(r"\{.*\}", text, flags=re.S)
        if not object_match:
            return None
        try:
            parsed = json.loads(object_match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def mean_ci(values: list[float], rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan, np.nan, np.nan
    if len(array) == 1 or np.all(array == array[0]):
        value = float(array.mean())
        return value, value, value
    samples = rng.choice(array, size=(n_boot, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(array.mean()), float(low), float(high)


def summarize_matches(
    match_rows: list[dict[str, float]],
    metadata: dict[str, object],
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, object]:
    result = dict(metadata)
    result["n_matches"] = len(match_rows)
    for metric in (
        "final_payoff",
        "parse_rate",
        "opponent_prediction_acc",
        "my_best_action_acc",
        "final_action_acc",
        "best_response_consistency",
        "win_rate",
    ):
        mean, low, high = mean_ci([row[metric] for row in match_rows], rng, n_boot)
        result[metric] = mean
        result[f"{metric}_ci_low"] = low
        result[f"{metric}_ci_high"] = high
    return result


def summarize_algorithm(path: Path, rng: np.random.Generator, n_boot: int) -> dict[str, object]:
    frame = pd.read_csv(path)
    label, step, iteration, inner_step = checkpoint_info(path)
    role = str(frame["llm_role"].iloc[0])
    per_match: list[dict[str, float]] = []
    for _, match in frame.groupby("match_id", sort=False):
        match = match.sort_values("round")
        scored = match[match["round"] > 0]
        replies = [parse_reply(value) for value in scored["llm_final_text"]]
        player = int(match["llm_player_index"].iloc[0])
        opponent = 1 - player
        own_choice = scored[f"choice_agent{player}"].to_numpy()
        opponent_choice = scored[f"choice_agent{opponent}"].to_numpy()
        oracle = opponent_choice if role == "seeker" else 1 - opponent_choice
        valid = np.asarray([reply is not None for reply in replies])

        def parsed_values(key: str) -> np.ndarray:
            return np.asarray([reply.get(key, -99) if reply else -99 for reply in replies])

        prediction = parsed_values("opponent_next_action")
        best = parsed_values("my_best_action")
        final = parsed_values("final_action")
        per_match.append(
            {
                "final_payoff": float(match["llm_payoff"].sum()),
                "parse_rate": float(valid.mean()),
                "opponent_prediction_acc": float((prediction[valid] == opponent_choice[valid]).mean()),
                "my_best_action_acc": float((best[valid] == oracle[valid]).mean()),
                "final_action_acc": float((own_choice == oracle).mean()),
                "best_response_consistency": float((best[valid] == final[valid]).mean()),
                "win_rate": float(match["llm_payoff"].sum() > 0),
            }
        )
    return summarize_matches(
        per_match,
        {"label": label, "cumulative_step": step, "iteration": iteration, "inner_step": inner_step, "role": role},
        rng,
        n_boot,
    )


def summarize_selfplay(path: Path, rng: np.random.Generator, n_boot: int) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    label, step, iteration, inner_step = checkpoint_info(path)
    role_rows: dict[str, list[dict[str, float]]] = {"seeker": [], "hider": []}
    for _, match in frame.groupby("match_id", sort=False):
        match = match.sort_values("round")
        scored = match[match["round"] > 0]
        choices = {0: scored["choice_agent0"].to_numpy(), 1: scored["choice_agent1"].to_numpy()}
        for player, role in ((0, "seeker"), (1, "hider")):
            replies = [parse_reply(value) for value in scored[f"agent{player}_llm_final_text"]]
            valid = np.asarray([reply is not None for reply in replies])
            opponent_choice = choices[1 - player]
            own_choice = choices[player]
            oracle = opponent_choice if role == "seeker" else 1 - opponent_choice

            def parsed_values(key: str) -> np.ndarray:
                return np.asarray([reply.get(key, -99) if reply else -99 for reply in replies])

            prediction = parsed_values("opponent_next_action")
            best = parsed_values("my_best_action")
            final = parsed_values("final_action")
            payoff = float(match[f"payoff_agent{player}"].sum())
            role_rows[role].append(
                {
                    "final_payoff": payoff,
                    "parse_rate": float(valid.mean()),
                    "opponent_prediction_acc": float((prediction[valid] == opponent_choice[valid]).mean()),
                    "my_best_action_acc": float((best[valid] == oracle[valid]).mean()),
                    "final_action_acc": float((own_choice == oracle).mean()),
                    "best_response_consistency": float((best[valid] == final[valid]).mean()),
                    "win_rate": float(payoff > 0),
                }
            )
    return [
        summarize_matches(
            rows,
            {"label": label, "cumulative_step": step, "iteration": iteration, "inner_step": inner_step, "role": role},
            rng,
            n_boot,
        )
        for role, rows in role_rows.items()
    ]


def make_compact(alg: pd.DataFrame, selfplay: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for step in sorted(alg["cumulative_step"].unique()):
        alg_step = alg[alg["cumulative_step"] == step]
        self_step = selfplay[selfplay["cumulative_step"] == step].set_index("role")
        rows.append(
            {
                "step": step,
                "ToM2_payoff_mean_roles": alg_step["final_payoff"].mean(),
                "ToM2_pred_mean_roles": alg_step["opponent_prediction_acc"].mean(),
                "ToM2_final_acc_mean_roles": alg_step["final_action_acc"].mean(),
                "ToM2_win_rate_mean_roles": alg_step["win_rate"].mean(),
                "selfplay_abs_payoff_asym": abs(self_step.loc["seeker", "final_payoff"] - self_step.loc["hider", "final_payoff"]),
                "selfplay_seeker_payoff": self_step.loc["seeker", "final_payoff"],
                "selfplay_hider_payoff": self_step.loc["hider", "final_payoff"],
            }
        )
    return pd.DataFrame(rows)


def role_plot(ax: plt.Axes, frame: pd.DataFrame, metric: str, ylabel: str, chance: float | None = None) -> None:
    colors = {"seeker": "#1f77b4", "hider": "#d95f02"}
    for role in ("seeker", "hider"):
        part = frame[frame["role"] == role].sort_values("cumulative_step")
        y = part[metric].to_numpy()
        low = part[f"{metric}_ci_low"].to_numpy()
        high = part[f"{metric}_ci_high"].to_numpy()
        ax.plot(part["cumulative_step"], y, marker="o", lw=1.8, ms=4, label=role.capitalize(), color=colors[role])
        ax.fill_between(part["cumulative_step"], low, high, color=colors[role], alpha=0.13, linewidth=0)
    if chance is not None:
        ax.axhline(chance, color="#777777", linestyle="--", linewidth=1, label="Chance")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)


def plot_detailed(alg: pd.DataFrame, selfplay: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.5), sharex=True)
    role_plot(axes[0, 0], alg, "final_payoff", "Mean match payoff", chance=0.0)
    role_plot(axes[0, 1], alg, "opponent_prediction_acc", "Opponent prediction accuracy", chance=0.5)
    role_plot(axes[1, 0], alg, "win_rate", "Match win rate", chance=0.5)
    role_plot(axes[1, 1], selfplay, "final_payoff", "Self-play mean payoff", chance=0.0)
    for ax, title in zip(axes.flat, ("A  ToM2 payoff", "B  ToM2 prediction", "C  ToM2 win rate", "D  Self-play role symmetry")):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Cumulative GRPO updates")
        ax.set_xticks(range(0, 226, 25))
    axes[0, 0].legend(frameon=False, ncol=3, loc="best")
    fig.suptitle("K=16 iterative online GRPO progression", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(compact: pd.DataFrame, old_dir: Path, output: Path) -> None:
    old = pd.read_csv(old_dir / "compact_progress_summary.csv")
    metrics = (
        ("ToM2_payoff_mean_roles", "Role-averaged ToM2 payoff", 0.0),
        ("ToM2_pred_mean_roles", "Role-averaged prediction accuracy", 0.5),
        ("selfplay_abs_payoff_asym", "Self-play payoff asymmetry", 0.0),
    )
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.35), sharex=True)
    for ax, (metric, ylabel, reference) in zip(axes, metrics):
        ax.plot(old["step"], old[metric], marker="o", ms=4, lw=1.7, color="#777777", label="K=4")
        ax.plot(compact["step"], compact[metric], marker="o", ms=4, lw=1.9, color="#0072B2", label="K=16")
        ax.axhline(reference, color="#bbbbbb", linestyle="--", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Cumulative GRPO updates")
        ax.set_xticks(range(0, 226, 50))
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].legend(frameon=False)
    fig.suptitle("Does broader group sampling improve online GRPO?", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_dir = args.input_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    alg_rows = [summarize_algorithm(path, rng, args.bootstrap) for path in sorted((args.input_dir / "alg").glob("*.csv"))]
    self_rows = []
    for path in sorted((args.input_dir / "selfplay").glob("*.csv")):
        self_rows.extend(summarize_selfplay(path, rng, args.bootstrap))
    alg = pd.DataFrame(alg_rows).sort_values(["cumulative_step", "role"])
    selfplay = pd.DataFrame(self_rows).sort_values(["cumulative_step", "role"])
    compact = make_compact(alg, selfplay)
    alg.to_csv(summary_dir / "alg_tom2_progress_summary.csv", index=False)
    selfplay.to_csv(summary_dir / "selfplay_progress_summary.csv", index=False)
    compact.to_csv(summary_dir / "compact_progress_summary.csv", index=False)
    plot_detailed(alg, selfplay, summary_dir / "k16_progression_detailed.png")
    if args.old_summary_dir:
        plot_comparison(compact, args.old_summary_dir, summary_dir / "k4_vs_k16_progression.png")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
