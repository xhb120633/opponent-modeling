#!/usr/bin/env python3
"""Build teacher-forced recurrent belief-transition SFT data.

The current opponent action is a target only.  Each prompt contains the prior
oracle belief and the immediately preceding observable transition.  At online
evaluation the same slot is filled with the LLM's own previous prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from belief_transition_prompt import PROMPT_VERSION, build_belief_transition_prompt


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(rounds: pd.DataFrame) -> None:
    required = {
        "match_id", "repeat_idx", "llm_role", "llm_player_index",
        "decision_round", "choice_agent0", "choice_agent1", "opponent_action",
        "llm_payoff", "oracle_tom_p_self", "opponent_name", "seed_base",
        "match_seed",
    }
    missing = required - set(rounds.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for match_id, match in rounds.groupby("match_id", sort=True):
        actual = sorted(match["decision_round"].astype(int).tolist())
        if actual != list(range(1, len(actual) + 1)):
            raise ValueError(f"Non-contiguous rounds in {match_id}")
        if match["llm_role"].nunique() != 1 or match["llm_player_index"].nunique() != 1:
            raise ValueError(f"Inconsistent match metadata in {match_id}")


def build_examples(
    rounds: pd.DataFrame, val_repeat_mod: int = 5, probability_clip: float = 0.05,
    state_source: str = "oracle",
) -> dict[str, list[dict]]:
    if val_repeat_mod < 2:
        raise ValueError("val_repeat_mod must be >= 2")
    if state_source not in {"oracle", "model"}:
        raise ValueError("state_source must be oracle or model")
    if state_source == "model" and "predicted_opponent_action_1_probability" not in rounds:
        raise ValueError("Model-state source lacks predicted probabilities")
    validate_source(rounds)
    outputs: dict[str, list[dict]] = {"train": [], "val": []}
    for match_id, match in rounds.groupby("match_id", sort=True):
        ordered = match.sort_values("decision_round")
        previous_belief = 0.5
        previous_observation: tuple[int, int, int] | None = None
        for _, row in ordered.iterrows():
            if previous_observation is None:
                self_action = opponent_action = payoff = None
            else:
                self_action, opponent_action, payoff = previous_observation
            prompt = build_belief_transition_prompt(
                role=str(row["llm_role"]),
                decision_round=int(row["decision_round"]),
                previous_belief_p1=previous_belief,
                previous_self_action=self_action,
                previous_opponent_action=opponent_action,
                previous_payoff=payoff,
            )
            target_action = int(row["opponent_action"])
            raw_target_probability = float(row["oracle_tom_p_self"])
            target_probability = min(
                1.0 - probability_clip, max(probability_clip, raw_target_probability)
            )
            if not 0.0 <= target_probability <= 1.0:
                raise ValueError(f"Invalid target probability in {match_id}")
            split = "val" if int(row["repeat_idx"]) % val_repeat_mod == 0 else "train"
            example_id = f"{match_id}__r{int(row['decision_round']):03d}"
            outputs[split].append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": str(target_action)},
                ],
                "prompt": prompt,
                "completion": str(target_action),
                "target": {
                    "realized_opponent_action": target_action,
                    "soft_opponent_action_1_probability": target_probability,
                },
                "soft_target_p1": target_probability,
                "loss_mask_type": "assistant_value_only",
                "loss_value_spans": [{
                    "field": "realized_opponent_action", "value": target_action,
                    "char_start": 0, "char_end": 1, "message_index": 1,
                }],
                "metadata": {
                    "example_id": example_id,
                    "match_id": str(match_id),
                    "repeat_idx": int(row["repeat_idx"]),
                    "role": str(row["llm_role"]),
                    "opponent_family": str(row["opponent_name"]),
                    "decision_round": int(row["decision_round"]),
                    "seed_base": int(row["seed_base"]),
                    "match_seed": int(row["match_seed"]),
                    "target_mode": "soft_realized_action",
                    "teacher": "tom_generating_p_self",
                    "prompt_version": PROMPT_VERSION,
                    "state_training": (
                        "teacher_forced_previous_oracle_belief"
                        if state_source == "oracle"
                        else "dagger_previous_model_belief"
                    ),
                    "deployment_state": "previous_llm_prediction",
                    "no_current_or_future_action_in_prompt": True,
                    "no_future_leakage": True,
                },
            })
            llm_index = int(row["llm_player_index"])
            previous_observation = (
                int(row[f"choice_agent{llm_index}"]), target_action, int(row["llm_payoff"])
            )
            previous_belief = (
                target_probability
                if state_source == "oracle"
                else float(row["predicted_opponent_action_1_probability"])
            )
            if not 0.0 <= previous_belief <= 1.0:
                raise ValueError(f"Invalid previous {state_source} belief in {match_id}")
    if not outputs["train"] or not outputs["val"]:
        raise ValueError("Complete-match train/validation split is empty")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-rounds", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--val-repeat-mod", type=int, default=5)
    parser.add_argument("--probability-clip", type=float, default=0.05)
    parser.add_argument("--state-source", choices=["oracle", "model"], default="oracle")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = pd.read_csv(args.oracle_rounds)
    if not 0.0 < args.probability_clip < 0.5:
        raise ValueError("--probability-clip must lie strictly between 0 and 0.5")
    outputs = build_examples(
        source, args.val_repeat_mod, args.probability_clip, args.state_source
    )
    summary = {
        split: {
            "n_rows": len(records),
            "n_matches": len({row["metadata"]["match_id"] for row in records}),
        }
        for split, records in outputs.items()
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return 0
    output = args.output_root / "soft_realized_action"
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, records in outputs.items():
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        hashes[path.name] = sha256(path)
    manifest = {
        "dataset": f"{PROMPT_VERSION}_{args.state_source}_state",
        "teacher": "tom_generating_p_self",
        "prompt_mode": "transition",
        "prompt_version": PROMPT_VERSION,
        "source": str(args.oracle_rounds),
        "source_sha256": sha256(args.oracle_rounds),
        "source_audit": {
            "seed_audited": True,
            "opponents": sorted(source["opponent_name"].astype(str).unique().tolist()),
            "roles": sorted(source["llm_role"].astype(str).unique().tolist()),
            "n_matches": int(source["match_id"].nunique()),
        },
        "split_rule": f"val iff repeat_idx % {args.val_repeat_mod} == 0",
        "information_boundary": "prior model belief plus immediately preceding observable transition",
        "teacher_forcing_warning": "Synthetic dev evaluation must recursively feed back model predictions.",
        "state_source": args.state_source,
        "probability_clip": float(args.probability_clip),
        "summary": summary,
        "output_sha256": hashes,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
