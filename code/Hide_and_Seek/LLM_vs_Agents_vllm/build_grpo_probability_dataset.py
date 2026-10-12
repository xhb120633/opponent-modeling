from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from probability_tom_utils import probability_01
from prompt_templates import PROMPT_VERSION_LOCKED_PROBABILITY, prompt_hash


TOM_OPPONENTS = {"ToM0", "ToM1", "ToM2", "ToM3", "ToM4", "ToM5"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build probability-target GRPO data from online ToM matches.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def round_bin(decision_round: int) -> str:
    if decision_round <= 10:
        return "early"
    if decision_round <= 30:
        return "mid"
    return "late"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(args.input_dir).glob("*.csv")):
        frame = pd.read_csv(path)
        for source_index, row in frame.iterrows():
            opponent = str(row.get("opponent_name", ""))
            if opponent not in TOM_OPPONENTS:
                continue
            target_probability = probability_01(row.get("opponent_action_1_probability"))
            prompt = row.get("llm_prompt")
            if target_probability is None or not isinstance(prompt, str) or not prompt.strip():
                continue
            role = str(row["llm_role"])
            decision_round = int(row.get("decision_round", int(row["round"]) + 1))
            match_id = str(row["match_id"])
            rows.append(
                {
                    "prompt": prompt,
                    "actual_opponent_action_probability": target_probability,
                    "actual_opponent_next_action": int(row[f"choice_agent{int(row['opponent_player_index'])}"]),
                    "role": role,
                    "opponent_name": opponent,
                    "decision_round": decision_round,
                    "round_bin": round_bin(decision_round),
                    "match_id": match_id,
                    "repeat_idx": int(row["repeat_idx"]),
                    "example_id": f"{match_id}:r{decision_round:03d}",
                    "source_csv": path.name,
                    "source_row_index": int(source_index),
                    "prompt_version": PROMPT_VERSION_LOCKED_PROBABILITY,
                    "prompt_hash": prompt_hash(PROMPT_VERSION_LOCKED_PROBABILITY),
                    "role_reward_weight": 1.0,
                }
            )

    if not rows:
        raise RuntimeError("No ToM rows with valid policy probabilities were found.")

    data = pd.DataFrame(rows)
    data["split"] = "val"
    split_manifest: dict[str, dict[str, list[int]]] = {}
    for (role, opponent), group in data.groupby(["role", "opponent_name"], sort=True):
        repeats = sorted(group["repeat_idx"].unique().tolist())
        shuffled = pd.Series(repeats).sample(frac=1.0, random_state=args.seed + len(split_manifest)).tolist()
        if len(shuffled) <= 1:
            train_repeats = set(shuffled)
        else:
            n_train = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * args.train_frac))))
            train_repeats = set(shuffled[:n_train])
        mask = data["role"].eq(role) & data["opponent_name"].eq(opponent)
        data.loc[mask & data["repeat_idx"].isin(train_repeats), "split"] = "train"
        split_manifest[f"{role}:{opponent}"] = {
            "train": sorted(map(int, train_repeats)),
            "val": sorted(map(int, set(repeats) - train_repeats)),
        }

    train = data[data["split"].eq("train")].drop(columns=["split"]).to_dict("records")
    val = data[data["split"].eq("val")].drop(columns=["split"]).to_dict("records")
    output = Path(args.output_root)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "val.jsonl", val)
    manifest = {
        "prompt_version": PROMPT_VERSION_LOCKED_PROBABILITY,
        "prompt_hash": prompt_hash(PROMPT_VERSION_LOCKED_PROBABILITY),
        "examples": len(data),
        "train_examples": len(train),
        "val_examples": len(val),
        "counts_by_role_opponent": {
            f"{role}:{opponent}": int(count)
            for (role, opponent), count in data.groupby(["role", "opponent_name"]).size().items()
        },
        "target_probability_summary": data["actual_opponent_action_probability"].describe().to_dict(),
        "splits_by_role_opponent": split_manifest,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()