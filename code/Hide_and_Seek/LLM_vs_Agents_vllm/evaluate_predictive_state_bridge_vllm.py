#!/usr/bin/env python3
"""Matched parent-vs-bridge generation evaluation on frozen predictive-state data."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


FIELDS = (
    "opponent_belief_my_action_bin",
    "opponent_action_bin",
    "opponent_next_action",
    "my_best_action",
    "final_action",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--parent-adapter-path", required=True)
    parser.add_argument("--bridge-adapter-path", required=True)
    parser.add_argument("--candidate-label", default="bridge25")
    parser.add_argument(
        "--extra-candidate",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Additional candidate adapter; may be repeated.",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-split", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def extract_json_object(text: str) -> dict[str, Any] | None:
    value = str(text).strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
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
            return parsed
    return None


def parse_bin(value: Any, upper: int) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if str(parsed) != str(value).strip() and not isinstance(value, int):
        return None
    return parsed if 0 <= parsed <= upper else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def balanced_sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Round-robin across the documented evaluation strata."""
    if n <= 0 or n >= len(rows):
        return list(rows)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = row.get("metadata", {})
        key = tuple(str(meta.get(k, "")) for k in ("role", "opponent_family", "round_bin", "confidence_stratum"))
        groups[key].append(row)
    rng = random.Random(seed)
    keys = sorted(groups)
    for key in keys:
        rng.shuffle(groups[key])
    selected: list[dict[str, Any]] = []
    while len(selected) < n:
        added = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                added = True
                if len(selected) == n:
                    break
        if not added:
            break
    return selected


def render_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def parse_extra_candidates(values: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label.strip() or not path.strip():
            raise ValueError(f"Expected LABEL=PATH for --extra-candidate, got: {value!r}")
        parsed.append((label.strip(), path.strip()))
    labels = [label for label, _ in parsed]
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate --extra-candidate labels are not allowed.")
    return parsed


def score_generation(source: dict[str, Any], model_label: str, raw_text: str, split: str) -> dict[str, Any]:
    target = source["target"]
    meta = source.get("metadata", {})
    parsed = extract_json_object(raw_text)
    values: dict[str, int | None] = {}
    for field in FIELDS:
        upper = 10 if field.endswith("_bin") else 1
        values[field] = parse_bin(parsed.get(field), upper) if parsed is not None else None
    valid = parsed is not None and all(values[field] is not None for field in FIELDS)
    p_pred = None if values["opponent_action_bin"] is None else values["opponent_action_bin"] / 10.0
    p_true = int(target["opponent_action_bin"]) / 10.0
    belief_pred = (
        None if values["opponent_belief_my_action_bin"] is None else values["opponent_belief_my_action_bin"] / 10.0
    )
    belief_true = int(target["opponent_belief_my_action_bin"]) / 10.0
    return {
        "split": split,
        "model": model_label,
        "example_id": meta.get("example_id", ""),
        "match_id": meta.get("match_id", ""),
        "role": meta.get("role", ""),
        "opponent_family": meta.get("opponent_family", ""),
        "round_bin": meta.get("round_bin", ""),
        "confidence_stratum": meta.get("confidence_stratum", ""),
        "valid_json": int(parsed is not None),
        "valid_all_fields": int(valid),
        "opponent_action_soft_mse": None if p_pred is None else (p_pred - p_true) ** 2,
        "constant_05_soft_mse": (0.5 - p_true) ** 2,
        "belief_soft_mse": None if belief_pred is None else (belief_pred - belief_true) ** 2,
        "direction_correct": (
            None
            if values["opponent_next_action"] is None
            else int(values["opponent_next_action"] == int(target["opponent_next_action"]))
        ),
        "best_action_correct": (
            None if values["my_best_action"] is None else int(values["my_best_action"] == int(target["my_best_action"]))
        ),
        "final_action_correct": (
            None if values["final_action"] is None else int(values["final_action"] == int(target["final_action"]))
        ),
        "bin_direction_consistent": (
            None
            if values["opponent_action_bin"] is None or values["opponent_next_action"] is None
            else int(values["opponent_next_action"] == int(values["opponent_action_bin"] >= 6))
        ),
        "best_final_consistent": (
            None
            if values["my_best_action"] is None or values["final_action"] is None
            else int(values["my_best_action"] == values["final_action"])
        ),
        **{f"pred_{field}": values[field] for field in FIELDS},
        **{f"target_{field}": int(target[field]) for field in FIELDS},
        "raw_text": raw_text,
    }


def mean_present(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None and math.isfinite(float(row[field]))]
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_fields = (
        "valid_json",
        "valid_all_fields",
        "opponent_action_soft_mse",
        "constant_05_soft_mse",
        "belief_soft_mse",
        "direction_correct",
        "best_action_correct",
        "final_action_correct",
        "bin_direction_consistent",
        "best_final_consistent",
    )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["model"])].append(row)
    output = []
    for (split, model), group in sorted(groups.items()):
        item: dict[str, Any] = {"split": split, "model": model, "n": len(group)}
        for field in metric_fields:
            item[field] = mean_present(group, field)
        baseline = item["constant_05_soft_mse"]
        mse = item["opponent_action_soft_mse"]
        item["opponent_action_skill_vs_05"] = None if not baseline else 1.0 - mse / baseline
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root) / "predictive_state"
    split_rows = {
        "id": balanced_sample(load_jsonl(data_root / "val.jsonl"), args.samples_per_split, args.seed),
        "ood": balanced_sample(load_jsonl(data_root / "ood.jsonl"), args.samples_per_split, args.seed + 1),
    }
    for split, rows in split_rows.items():
        print(f"[bridge-eval] split={split} selected={len(rows)}")
    if args.dry_run:
        return

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=args.model_name,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
        trust_remote_code=True,
        enable_lora=True,
        max_loras=1,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, seed=args.seed)
    candidate_specs = [(args.candidate_label, args.bridge_adapter_path), *parse_extra_candidates(args.extra_candidate)]
    if len({label for label, _ in candidate_specs}) != len(candidate_specs):
        raise ValueError("Candidate labels must be unique.")
    adapters = [("parent", LoRARequest("parent_adapter", 1, args.parent_adapter_path))]
    adapters.extend(
        (label, LoRARequest(f"candidate_adapter_{index}", index + 2, path))
        for index, (label, path) in enumerate(candidate_specs)
    )
    scored: list[dict[str, Any]] = []
    for model_label, lora_request in adapters:
        for split, sources in split_rows.items():
            for start in range(0, len(sources), args.batch_size):
                batch = sources[start : start + args.batch_size]
                prompts = [render_prompt(tokenizer, row["prompt"]) for row in batch]
                generated = llm.generate(prompts, sampling, lora_request=lora_request)
                for source, item in zip(batch, generated):
                    text = item.outputs[0].text if item.outputs else ""
                    scored.append(score_generation(source, model_label, text, split))
                print(
                    f"[bridge-eval] model={model_label} split={split} "
                    f"progress={min(start + len(batch), len(sources))}/{len(sources)}"
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(scored)
    write_csv(output_dir / "matched_rows.csv", scored)
    write_csv(output_dir / "summary.csv", summary)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
