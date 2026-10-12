"""
Summarize LLM-vs-agent Hide-and-Seek runs with two views:

1) Head-to-head: per opponent type, match win rate (final net payoff > 0) and
   how many opponent types are "beaten" (strict majority wins: >50% of matches).
2) Net return: mean (and optional sum) of final match cumulative payoff.

Each row in ALL_MATCHES is one round; one match is identified by match_id.
Final net payoff for a match is taken from the last decision_round row:
llm_known_score_cumulative.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data_original_vllm",
        help="Directory containing ALL_MATCHES_*.csv files.",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="*_ALL_MATCHES_*.csv",
        help="Glob pattern under input-dir.",
    )
    return p.parse_args()


def final_per_match(df: pd.DataFrame) -> pd.DataFrame:
    """One row per match_id with opponent_name and final cumulative payoff."""
    sort_cols = ["match_id", "decision_round"]
    last = df.sort_values(sort_cols).groupby("match_id", as_index=False).tail(1)
    return last[["match_id", "opponent_name", "llm_known_score_cumulative", "llm_model_name"]]


def main() -> None:
    args = parse_args()
    files = sorted(args.input_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.glob!r} in {args.input_dir}")

    for path in files:
        df = pd.read_csv(path)
        if "llm_known_score_cumulative" not in df.columns:
            raise SystemExit(f"{path.name}: missing llm_known_score_cumulative")
        fin = final_per_match(df)
        model = (
            fin["llm_model_name"].dropna().iloc[0]
            if fin["llm_model_name"].notna().any()
            else path.stem
        )
        fin["win"] = fin["llm_known_score_cumulative"] > 0

        by_opp = (
            fin.groupby("opponent_name", sort=True)
            .agg(
                matches=("match_id", "count"),
                wins=("win", "sum"),
                win_rate=("win", "mean"),
                mean_net_payoff=("llm_known_score_cumulative", "mean"),
            )
            .reset_index()
        )
        by_opp["win_rate_pct"] = (100 * by_opp["win_rate"]).round(1)

        n_types = int(by_opp.shape[0])
        beaten_majority = int((by_opp["win_rate"] > 0.5).sum())
        beaten_positive_mean = int((by_opp["mean_net_payoff"] > 0).sum())
        overall_mean_net = float(fin["llm_known_score_cumulative"].mean())
        overall_total_net = float(fin["llm_known_score_cumulative"].sum())
        overall_win_rate = float(fin["win"].mean())

        print("=" * 72)
        print(model)
        print(path.name)
        print(f"matches: {len(fin)}  opponent types: {n_types}")
        print()
        print("[1] Head-to-head (win = final llm_known_score_cumulative > 0)")
        print(f"    overall match win rate: {100 * overall_win_rate:.1f}%")
        print(
            "    opponent types with >50% match wins: "
            f"{beaten_majority} / {n_types}"
        )
        print(
            "    opponent types with mean net payoff > 0: "
            f"{beaten_positive_mean} / {n_types}"
        )
        print()
        print(by_opp.to_string(index=False, formatters={"win_rate": lambda x: f"{x:.3f}"}))
        print()
        print("[2] Net return (final cumulative payoff per match)")
        print(f"    mean net payoff per match: {overall_mean_net:.3f}")
        print(f"    total net payoff (sum over matches): {overall_total_net:.1f}")
        print()


if __name__ == "__main__":
    main()
