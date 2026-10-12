from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from model_output_parse import resolve_action_from_completion


EXPECTED_ROUNDS = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit vLLM ALL_MATCHES CSVs for completeness and parse quality."
    )
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--glob", type=str, default="*_ALL_MATCHES*.csv")
    return p.parse_args()


def _value_counts_prefixed(series: pd.Series, prefix: str) -> dict[str, int]:
    if series is None:
        return {}
    counts = series.fillna("__NA__").astype(str).value_counts(dropna=False)
    return {f"{prefix}_{k}": int(v) for k, v in counts.items()}


def audit_file(path: Path) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(path)
    model_name = (
        str(df["llm_model_name"].dropna().iloc[0])
        if "llm_model_name" in df.columns and df["llm_model_name"].notna().any()
        else str(df["agent0"].dropna().iloc[0])
    )

    required = ["match_id", "decision_round", "choice_agent0", "llm_raw_reply", "llm_parse_mode"]
    missing_cols = [c for c in required if c not in df.columns]

    reparse_actions: list[int | None] = []
    reparse_modes: list[str] = []
    reparse_sources: list[str] = []
    reparse_backends: list[str] = []
    if "llm_raw_reply" in df.columns:
        for raw in df["llm_raw_reply"].fillna("").astype(str):
            action, mode, source, split = resolve_action_from_completion(raw, model_name)
            reparse_actions.append(action)
            reparse_modes.append(mode)
            reparse_sources.append(source)
            reparse_backends.append(split.parse_backend)

        df["_reparse_action"] = reparse_actions
        df["_reparse_mode"] = reparse_modes
        df["_reparse_source"] = reparse_sources
        df["_reparse_backend"] = reparse_backends
    else:
        df["_reparse_action"] = None
        df["_reparse_mode"] = "missing_raw_reply"
        df["_reparse_source"] = "missing_raw_reply"
        df["_reparse_backend"] = "missing_raw_reply"

    if "choice_agent0" in df.columns:
        recorded = pd.to_numeric(df["choice_agent0"], errors="coerce")
        reparsed = pd.to_numeric(df["_reparse_action"], errors="coerce")
        df["_reparse_matches_recorded"] = recorded.eq(reparsed)
    else:
        df["_reparse_matches_recorded"] = False

    match_cols = ["match_id"]
    if "opponent_name" in df.columns:
        match_cols.append("opponent_name")
    per_match = (
        df.groupby(match_cols, dropna=False)
        .agg(
            rows=("decision_round", "count") if "decision_round" in df.columns else ("round", "count"),
            min_round=("decision_round", "min") if "decision_round" in df.columns else ("round", "min"),
            max_round=("decision_round", "max") if "decision_round" in df.columns else ("round", "max"),
            final_llm_payoff=(
                "llm_known_score_cumulative",
                "last",
            )
            if "llm_known_score_cumulative" in df.columns
            else ("payoff_agent0_cumulative", "last"),
            fallback_rows=("llm_parse_mode", lambda x: int((x == "fallback").sum()))
            if "llm_parse_mode" in df.columns
            else ("_reparse_mode", lambda x: 0),
            reparse_fallback_rows=("_reparse_mode", lambda x: int((x == "fallback").sum())),
            reparse_mismatch_rows=("_reparse_matches_recorded", lambda x: int((~x).sum())),
        )
        .reset_index()
    )
    per_match["complete_50_rounds"] = per_match["rows"].eq(EXPECTED_ROUNDS)
    per_match["source_file"] = path.name
    per_match["llm_model_name"] = model_name

    n_rows = int(len(df))
    fallback_rows = int((df["llm_parse_mode"] == "fallback").sum()) if "llm_parse_mode" in df.columns else 0
    reparse_fallback_rows = int((df["_reparse_mode"] == "fallback").sum())
    mismatch_rows = int((~df["_reparse_matches_recorded"]).sum())

    summary = {
        "source_file": path.name,
        "llm_model_name": model_name,
        "rows": n_rows,
        "matches": int(df["match_id"].nunique()) if "match_id" in df.columns else 0,
        "opponents": int(df["opponent_name"].nunique()) if "opponent_name" in df.columns else 0,
        "missing_required_cols": ";".join(missing_cols),
        "runtime_fallback_rows": fallback_rows,
        "runtime_fallback_rate": fallback_rows / n_rows if n_rows else 0.0,
        "reparse_fallback_rows": reparse_fallback_rows,
        "reparse_fallback_rate": reparse_fallback_rows / n_rows if n_rows else 0.0,
        "reparse_mismatch_rows": mismatch_rows,
        "reparse_mismatch_rate": mismatch_rows / n_rows if n_rows else 0.0,
        "incomplete_matches": int((~per_match["complete_50_rounds"]).sum()),
        "mean_final_llm_payoff": float(per_match["final_llm_payoff"].mean()),
        "min_final_llm_payoff": float(per_match["final_llm_payoff"].min()),
        "max_final_llm_payoff": float(per_match["final_llm_payoff"].max()),
    }
    if "llm_parse_mode" in df.columns:
        summary.update(_value_counts_prefixed(df["llm_parse_mode"], "runtime_parse"))
    summary.update(_value_counts_prefixed(df["_reparse_mode"], "reparse_mode"))
    summary.update(_value_counts_prefixed(df["_reparse_source"], "reparse_source"))
    summary.update(_value_counts_prefixed(df["_reparse_backend"], "reparse_backend"))
    return summary, per_match


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.glob!r} in {args.input_dir}")

    summaries = []
    match_frames = []
    for file_path in files:
        summary, per_match = audit_file(file_path)
        summaries.append(summary)
        match_frames.append(per_match)

    summary_df = pd.DataFrame(summaries).fillna(0)
    match_df = pd.concat(match_frames, ignore_index=True)
    summary_path = args.output_dir / "vllm_result_quality_summary.csv"
    match_path = args.output_dir / "vllm_result_quality_by_match.csv"
    summary_df.to_csv(summary_path, index=False)
    match_df.to_csv(match_path, index=False)

    cols = [
        "source_file",
        "rows",
        "matches",
        "runtime_fallback_rows",
        "runtime_fallback_rate",
        "reparse_fallback_rows",
        "reparse_mismatch_rows",
        "incomplete_matches",
        "mean_final_llm_payoff",
    ]
    print(summary_df[cols].to_string(index=False))
    print(f"Saved: {summary_path}")
    print(f"Saved: {match_path}")


if __name__ == "__main__":
    main()
