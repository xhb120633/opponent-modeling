from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from build_grpo_offline_dataset_locked_v1 import strip_structured_instruction
from prompt_templates import (
    PROMPT_VERSION_LOCKED_GRPO_REASONING,
    build_locked_grpo_reasoning_prompt,
    prompt_hash,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build candidate-only GRPO data from online self-play rollout CSVs.")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--candidate-label", default="candidate")
    p.add_argument("--train-frac", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--asymmetry-weight-lambda", type=float, default=0.0)
    p.add_argument("--asymmetry-weight-cap", type=float, default=1.3)
    return p.parse_args()


def role_best_action(opponent_action: int, role: str) -> int:
    return opponent_action if role == "seeker" else 1 - opponent_action


def round_bin(decision_round: int) -> str:
    if decision_round <= 10:
        return "early"
    if decision_round <= 30:
        return "mid"
    return "late"


def row_for_side(row: pd.Series, side: int, candidate_label: str, source_csv: str, source_row_index: int) -> dict[str, Any] | None:
    prefix = f"agent{side}_llm"
    opp_side = 1 - side
    label = str(row.get(f"agent{side}_label", ""))
    backend = str(row.get(f"agent{side}_backend", ""))
    if label != candidate_label and backend != "candidate":
        return None
    role = "seeker" if side == 0 else "hider"
    prompt = row.get(f"{prefix}_prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    decision_round = int(row.get("decision_round", row.get("round", 0) + 1))
    opponent_action = int(row[f"choice_agent{opp_side}"])
    best_action = role_best_action(opponent_action, role)
    state_block = strip_structured_instruction(prompt)
    grpo_prompt = build_locked_grpo_reasoning_prompt(state_block)
    match_id = str(row.get("match_id", ""))
    return {
        "prompt": grpo_prompt,
        "actual_opponent_next_action": opponent_action,
        "actual_my_best_action": best_action,
        "actual_final_action": best_action,
        "current_round_payoff": int(row[f"payoff_agent{side}"]),
        "role": role,
        "opponent_name": str(row.get("frozen_label", row.get(f"agent{opp_side}_label", "frozen"))),
        "decision_round": decision_round,
        "round_bin": round_bin(decision_round),
        "match_id": match_id,
        "repeat_idx": int(row.get("repeat_idx", -1)),
        "example_id": f"{match_id}:r{decision_round:03d}:agent{side}",
        "source_csv": source_csv,
        "source_row_index": source_row_index,
        "prompt_version": PROMPT_VERSION_LOCKED_GRPO_REASONING,
        "prompt_hash": prompt_hash(PROMPT_VERSION_LOCKED_GRPO_REASONING),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.csv")):
        df = pd.read_csv(path)
        for idx, row in df.iterrows():
            for side in (0, 1):
                item = row_for_side(row, side, args.candidate_label, path.name, int(idx))
                if item is not None:
                    rows.append(item)

    if not rows:
        raise RuntimeError(f"No candidate rows found in {input_dir} for candidate_label={args.candidate_label!r}")

    frame = pd.DataFrame(rows)
    role_counts = frame["role"].value_counts().to_dict()
    if role_counts.get("seeker", 0) != role_counts.get("hider", 0):
        raise RuntimeError(f"Role imbalance in online self-play data: {role_counts}")

    frame["role_reward_weight"] = 1.0
    role_payoff_means: dict[str, float] = {}
    asymmetry_weights = {"seeker": 1.0, "hider": 1.0}
    if args.asymmetry_weight_lambda > 0:
        match_payoffs = (
            frame.groupby(["role", "match_id"], dropna=False)["current_round_payoff"]
            .sum()
            .reset_index(name="match_payoff")
        )
        role_payoff_means = {
            str(role): float(value)
            for role, value in match_payoffs.groupby("role")["match_payoff"].mean().to_dict().items()
        }
        seeker_mean = role_payoff_means.get("seeker", 0.0)
        hider_mean = role_payoff_means.get("hider", 0.0)
        n_rounds_est = max(1.0, float(frame["decision_round"].max()))
        seeker_deficit = max(0.0, hider_mean - seeker_mean) / n_rounds_est
        hider_deficit = max(0.0, seeker_mean - hider_mean) / n_rounds_est
        cap = max(1.0, float(args.asymmetry_weight_cap))
        asymmetry_weights = {
            "seeker": min(cap, 1.0 + args.asymmetry_weight_lambda * seeker_deficit),
            "hider": min(cap, 1.0 + args.asymmetry_weight_lambda * hider_deficit),
        }
        frame["role_reward_weight"] = frame["role"].map(asymmetry_weights).fillna(1.0)

    frame["split"] = "val"
    split_manifest: dict[str, dict[str, list[int]]] = {}
    for role, role_frame in frame.groupby("role", sort=True):
        repeats = sorted(role_frame["repeat_idx"].unique().tolist())
        shuffled = pd.Series(repeats).sample(frac=1.0, random_state=args.seed).tolist()
        if len(shuffled) <= 1:
            train_repeats = set(shuffled)
        else:
            n_train = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * args.train_frac))))
            train_repeats = set(shuffled[:n_train])
        role_mask = frame["role"].eq(role)
        frame.loc[role_mask & frame["repeat_idx"].isin(train_repeats), "split"] = "train"
        split_manifest[str(role)] = {
            "train": sorted(map(int, train_repeats)),
            "val": sorted(map(int, set(repeats) - train_repeats)),
        }

    train_rows = frame[frame["split"].eq("train")].drop(columns=["split"]).to_dict("records")
    val_rows = frame[frame["split"].eq("val")].drop(columns=["split"]).to_dict("records")
    write_jsonl(output_root / "train.jsonl", train_rows)
    write_jsonl(output_root / "val.jsonl", val_rows)

    manifest = {
        "source": str(input_dir),
        "target": str(output_root),
        "candidate_label": args.candidate_label,
        "prompt_version": PROMPT_VERSION_LOCKED_GRPO_REASONING,
        "prompt_hash": prompt_hash(PROMPT_VERSION_LOCKED_GRPO_REASONING),
        "examples": len(rows),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "role_counts": role_counts,
        "role_payoff_means": role_payoff_means,
        "asymmetry_weight_lambda": args.asymmetry_weight_lambda,
        "asymmetry_weight_cap": args.asymmetry_weight_cap,
        "role_reward_weights": asymmetry_weights,
        "splits_by_role_repeat": split_manifest,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
