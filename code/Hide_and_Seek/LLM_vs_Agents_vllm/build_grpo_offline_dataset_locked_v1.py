from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prompt_templates import (
    LOCKED_STRUCTURED_INSTRUCTION,
    PROMPT_VERSION_LOCKED_GRPO_REASONING,
    build_locked_grpo_reasoning_prompt,
    prompt_hash,
)

DEFAULT_INPUT_ROOT = "data_sft_locked_v1_qwen25_7b/structured_tom"
DEFAULT_OUTPUT_ROOT = "data_grpo_offline_locked_v1_qwen25_7b/structured_reasoning_gate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build offline GRPO data from locked-v1 structured SFT examples. "
            "The output has prompts plus oracle fields for rule-based rewards; "
            "it does not include teacher completions."
        )
    )
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-examples-per-split", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def strip_structured_instruction(prompt: str) -> str:
    prompt = str(prompt).rstrip()
    marker = "\n\n" + LOCKED_STRUCTURED_INSTRUCTION
    if marker in prompt:
        return prompt.rsplit(marker, 1)[0].rstrip()
    if LOCKED_STRUCTURED_INSTRUCTION in prompt:
        return prompt.rsplit(LOCKED_STRUCTURED_INSTRUCTION, 1)[0].rstrip()
    return prompt


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_grpo_row(row: dict[str, Any]) -> dict[str, Any]:
    target = row.get("target") or {}
    metadata = row.get("metadata") or {}
    state_block = strip_structured_instruction(row.get("prompt", ""))
    prompt = build_locked_grpo_reasoning_prompt(state_block)
    return {
        "prompt": prompt,
        "actual_opponent_next_action": int(target["opponent_next_action"]),
        "actual_my_best_action": int(target["my_best_action"]),
        "actual_final_action": int(target["final_action"]),
        "role": metadata.get("role", ""),
        "opponent_name": metadata.get("opponent_name", ""),
        "decision_round": int(metadata.get("decision_round", -1)),
        "round_bin": metadata.get("round_bin", ""),
        "match_id": metadata.get("match_id", ""),
        "repeat_idx": metadata.get("repeat_idx", ""),
        "example_id": metadata.get("example_id", ""),
        "source_csv": metadata.get("source_csv", ""),
        "source_row_index": metadata.get("source_row_index", ""),
        "prompt_version": PROMPT_VERSION_LOCKED_GRPO_REASONING,
        "prompt_hash": prompt_hash(PROMPT_VERSION_LOCKED_GRPO_REASONING),
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    manifest: dict[str, Any] = {
        "source": str(input_root),
        "target": str(output_root),
        "prompt_version": PROMPT_VERSION_LOCKED_GRPO_REASONING,
        "prompt_hash": prompt_hash(PROMPT_VERSION_LOCKED_GRPO_REASONING),
        "splits": {},
        "reward_note": "Rule-based GRPO reward uses oracle opponent/action fields; no teacher completion is included.",
    }
    for split in splits:
        in_path = input_root / f"{split}.jsonl"
        if not in_path.is_file():
            raise FileNotFoundError(in_path)
        rows = [make_grpo_row(row) for row in load_jsonl(in_path)]
        if args.max_examples_per_split is not None:
            rows = rows[: args.max_examples_per_split]
        manifest["splits"][split] = {"examples": len(rows), "path": str((output_root / f"{split}.jsonl").resolve())}
        if not args.dry_run:
            write_jsonl(output_root / f"{split}.jsonl", rows)
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
