from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prompt_templates import (
    PROMPT_VERSION_LOCKED_PREDICTIVE_STATE,
    build_locked_predictive_state_prompt,
    build_observable_state_block,
    prompt_hash,
)


DEFAULT_SOURCE = "analysis/predictive_state_identifiability_v1/predictive_state_features.csv"
DEFAULT_OUTPUT = "data_predictive_state_bridge_v1"
SPLIT_MAP = {"train": "train", "id_val": "val", "ood_family": "ood"}


def render_completion_and_value_spans(
    target: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    pieces = ["{"]
    spans: list[dict[str, Any]] = []
    for index, (key, value) in enumerate(target.items()):
        if index:
            pieces.append(",")
        pieces.append(json.dumps(str(key), ensure_ascii=False))
        pieces.append(":")
        value_text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        start = sum(len(piece) for piece in pieces)
        pieces.append(value_text)
        end = sum(len(piece) for piece in pieces)
        if isinstance(value, str) and value_text.startswith('"') and value_text.endswith('"'):
            value_start, value_end = start + 1, end - 1
        else:
            value_start, value_end = start, end
        spans.append(
            {
                "field": str(key),
                "value": value,
                "char_start": value_start,
                "char_end": value_end,
                "include_quotes": False,
            }
        )
    pieces.append("}")
    return "".join(pieces), spans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a leakage-audited predictive-state bridge SFT dataset."
    )
    parser.add_argument("--source-csv", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--max-per-stratum",
        type=int,
        default=200,
        help="Maximum training rows per role/opponent/round/confidence stratum; <=0 keeps all.",
    )
    parser.add_argument("--val-max-per-stratum", type=int, default=25)
    parser.add_argument("--ood-max-per-stratum", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probability_bin(value: float) -> int:
    return int(np.clip(np.floor(float(value) * 10.0 + 0.5), 0, 10))


def round_bin(decision_round: int) -> str:
    if decision_round <= 10:
        return "early"
    if decision_round <= 30:
        return "middle"
    return "late"


def confidence_stratum(p_self: float, p_op: float) -> str:
    confidence = max(abs(float(p_self) - 0.5), abs(float(p_op) - 0.5))
    if confidence < 0.1:
        return "near_half"
    if confidence < 0.3:
        return "medium"
    return "high"


def validate_source(data: pd.DataFrame) -> None:
    required = {
        "match_id",
        "repeat_idx",
        "ego_agent",
        "opponent_family",
        "role",
        "decision_round",
        "p_self",
        "p_op",
        "self_lag1",
        "opponent_lag1",
        "payoff_lag1",
        "split",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    unknown_splits = set(data["split"].unique()).difference(SPLIT_MAP)
    if unknown_splits:
        raise ValueError(f"Unknown source splits: {sorted(unknown_splits)}")
    if data.groupby("match_id")["split"].nunique().max() != 1:
        raise ValueError("A match appears in more than one source split.")


def add_observable_histories(data: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for match_id, match in data.groupby("match_id", sort=True):
        ordered = match.sort_values("decision_round")
        expected = list(range(1, len(ordered) + 1))
        actual = ordered["decision_round"].astype(int).tolist()
        if actual != expected:
            raise ValueError(f"Non-contiguous decision rounds for {match_id}: {actual[:5]}")

        self_actions: list[int] = []
        opponent_actions: list[int] = []
        payoffs: list[int] = []
        for _, row in ordered.iterrows():
            decision_round = int(row["decision_round"])
            if decision_round > 1:
                self_actions.append(int(row["self_lag1"]))
                opponent_actions.append(int(row["opponent_lag1"]))
                payoffs.append(int(row["payoff_lag1"]))
            state_block = build_observable_state_block(
                role=str(row["role"]),
                self_actions=self_actions,
                opponent_actions=opponent_actions,
                payoffs=payoffs,
            )
            record = row.to_dict()
            record["state_block"] = state_block
            record["prompt"] = build_locked_predictive_state_prompt(state_block)
            record["dataset_split"] = SPLIT_MAP[str(row["split"])]
            record["round_bin"] = round_bin(decision_round)
            record["confidence_stratum"] = confidence_stratum(row["p_self"], row["p_op"])
            records.append(record)
    return pd.DataFrame(records)


def sample_balanced(
    data: pd.DataFrame,
    max_per_stratum: int,
    seed: int,
    val_max_per_stratum: int | None = None,
    ood_max_per_stratum: int | None = None,
) -> pd.DataFrame:
    caps = {
        "train": max_per_stratum,
        "val": max_per_stratum if val_max_per_stratum is None else val_max_per_stratum,
        "ood": max_per_stratum if ood_max_per_stratum is None else ood_max_per_stratum,
    }
    if all(cap <= 0 for cap in caps.values()):
        return data.copy()
    keys = ["dataset_split", "role", "opponent_family", "round_bin", "confidence_stratum"]
    sampled = []
    for key, group in data.groupby(keys, sort=True, dropna=False):
        cap = caps[str(key[0])]
        if cap <= 0 or len(group) <= cap:
            sampled.append(group)
            continue
        key_seed = int(
            hashlib.sha1(f"{seed}|{key}".encode("utf-8")).hexdigest()[:8], 16
        )
        sampled.append(group.sample(n=cap, random_state=key_seed))
    return pd.concat(sampled, ignore_index=True).sort_values(
        ["dataset_split", "match_id", "decision_round"]
    )


def make_target(row: pd.Series) -> dict[str, int]:
    belief_bin = probability_bin(row["p_op"])
    action_bin = probability_bin(row["p_self"])
    opponent_action = int(action_bin >= 6)
    best_action = opponent_action if row["role"] == "seeker" else 1 - opponent_action
    return {
        "opponent_belief_my_action_bin": belief_bin,
        "opponent_action_bin": action_bin,
        "opponent_next_action": opponent_action,
        "my_best_action": best_action,
        "final_action": best_action,
    }


def make_record(row: pd.Series) -> dict[str, Any]:
    target = make_target(row)
    completion, spans = render_completion_and_value_spans(target)
    prompt_digest = prompt_hash(PROMPT_VERSION_LOCKED_PREDICTIVE_STATE)
    example_id = f"{row['match_id']}__r{int(row['decision_round']):03d}"
    return {
        "messages": [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": completion},
        ],
        "prompt": row["prompt"],
        "completion": completion,
        "target": target,
        "loss_mask_type": "assistant_value_only",
        "loss_value_spans": spans,
        "metadata": {
            "example_id": example_id,
            "match_id": row["match_id"],
            "repeat_idx": int(row["repeat_idx"]),
            "ego_agent": row["ego_agent"],
            "opponent_family": row["opponent_family"],
            "role": row["role"],
            "decision_round": int(row["decision_round"]),
            "round_bin": row["round_bin"],
            "confidence_stratum": row["confidence_stratum"],
            "target_mode": "predictive_state",
            "prompt_version": PROMPT_VERSION_LOCKED_PREDICTIVE_STATE,
            "prompt_hash": prompt_digest,
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def leakage_audit(records: list[dict[str, Any]]) -> None:
    forbidden = ("opponent_family", "oracle", "internal_states", "current oracle")
    for record in records:
        prompt = record["prompt"].lower()
        hits = [token for token in forbidden if token in prompt]
        if hits:
            raise ValueError(
                f"Prompt leakage in {record['metadata']['example_id']}: {hits}"
            )


def build_dataset(
    source_csv: Path,
    max_per_stratum: int,
    seed: int,
    val_max_per_stratum: int | None = None,
    ood_max_per_stratum: int | None = None,
) -> pd.DataFrame:
    source = pd.read_csv(source_csv)
    validate_source(source)
    enriched = add_observable_histories(source)
    return sample_balanced(
        enriched,
        max_per_stratum=max_per_stratum,
        seed=seed,
        val_max_per_stratum=val_max_per_stratum,
        ood_max_per_stratum=ood_max_per_stratum,
    )


def main() -> None:
    args = parse_args()
    source_csv = Path(args.source_csv).resolve()
    data = build_dataset(
        source_csv,
        args.max_per_stratum,
        args.seed,
        val_max_per_stratum=args.val_max_per_stratum,
        ood_max_per_stratum=args.ood_max_per_stratum,
    )
    counts = data["dataset_split"].value_counts().sort_index().to_dict()
    matches = {
        split: sorted(group["match_id"].unique().tolist())
        for split, group in data.groupby("dataset_split", sort=True)
    }
    print(f"[bridge-data] source={source_csv}")
    print(f"[bridge-data] examples={len(data)} counts={counts}")
    print(f"[bridge-data] match_counts={ {key: len(value) for key, value in matches.items()} }")
    if args.dry_run:
        return

    output_dir = Path(args.output_dir).resolve() / "predictive_state"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}
    for split in ("train", "val", "ood"):
        split_rows = data[data["dataset_split"].eq(split)]
        records = [make_record(row) for _, row in split_rows.iterrows()]
        leakage_audit(records)
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, records)
        output_hashes[path.name] = sha256_file(path)

    summary = (
        data.groupby(
            ["dataset_split", "role", "opponent_family", "round_bin", "confidence_stratum"],
            dropna=False,
        )
        .agg(examples=("match_id", "size"), matches=("match_id", "nunique"))
        .reset_index()
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    manifest = {
        "dataset": "predictive_state_bridge_v1",
        "prompt_version": PROMPT_VERSION_LOCKED_PREDICTIVE_STATE,
        "prompt_hash": prompt_hash(PROMPT_VERSION_LOCKED_PREDICTIVE_STATE),
        "source_csv": str(source_csv),
        "source_sha256": sha256_file(source_csv),
        "seed": args.seed,
        "max_per_stratum": args.max_per_stratum,
        "val_max_per_stratum": args.val_max_per_stratum,
        "ood_max_per_stratum": args.ood_max_per_stratum,
        "counts": {key: int(value) for key, value in counts.items()},
        "match_counts": {key: len(value) for key, value in matches.items()},
        "matches": matches,
        "output_sha256": output_hashes,
        "leakage_audit": "passed_behavior_only_prompt",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"[bridge-data] wrote={output_dir}")


if __name__ == "__main__":
    main()
