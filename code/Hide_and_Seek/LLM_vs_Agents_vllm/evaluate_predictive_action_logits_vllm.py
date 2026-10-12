#!/usr/bin/env python3
"""Audit binary next-action confidence from model logits on frozen ID/OOD histories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


PROBE_SUFFIX = """Before choosing, predict the opponent's current action from the observable history above.

Answer with exactly one digit: 0 or 1."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--parent-adapter-path", required=True)
    parser.add_argument("--candidate", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--data-subdir", default="predictive_state")
    parser.add_argument("--target-field", default="opponent_next_action")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-split", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def balanced_sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Match the existing bridge evaluator's documented round-robin sampling."""
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


def build_probe_prompt(prompt: str) -> str:
    """Remove the five-field probability/action interface while preserving history."""
    markers = ("\nBefore choosing,", "\nUse integer probability bins:", "\nRequired fields:")
    cut_points = [prompt.find(marker) for marker in markers if prompt.find(marker) >= 0]
    if not cut_points:
        raise ValueError("Could not find the predictive-state instruction suffix.")
    prefix = prompt[: min(cut_points)].rstrip()
    return f"{prefix}\n\n{PROBE_SUFFIX}"


def render_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def parse_candidates(values: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label.strip() or not path.strip():
            raise ValueError(f"Expected LABEL=PATH for --candidate, got {value!r}")
        parsed.append((label.strip(), path.strip()))
    labels = [label for label, _ in parsed]
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate candidate labels are not allowed.")
    return parsed


def binary_token_classes(tokenizer: Any) -> dict[int, int]:
    """Map single-token plain/space-prefixed variants to binary classes."""
    mapping: dict[int, int] = {}
    for target, variants in ((0, ("0", " 0")), (1, ("1", " 1"))):
        for variant in variants:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(token_ids) == 1:
                token_id = int(token_ids[0])
                previous = mapping.get(token_id)
                if previous is not None and previous != target:
                    raise ValueError(f"Token id {token_id} maps to both binary classes.")
                mapping[token_id] = target
    if set(mapping.values()) != {0, 1}:
        raise ValueError(f"Tokenizer does not expose single-token 0/1 choices: {mapping}")
    return mapping


def _as_logprob(value: Any) -> float:
    raw = getattr(value, "logprob", value)
    return float(raw)


def binary_probability(
    logprobs: Mapping[int, Any] | None, token_classes: Mapping[int, int]
) -> tuple[float | None, dict[str, Any]]:
    if not logprobs:
        return None, {"found_class0": 0, "found_class1": 0, "prob_mass": 0.0}
    masses = {0: 0.0, 1: 0.0}
    found = {0: 0, 1: 0}
    for raw_token_id, value in logprobs.items():
        token_id = int(raw_token_id)
        target = token_classes.get(token_id)
        if target is None:
            continue
        masses[target] += math.exp(_as_logprob(value))
        found[target] += 1
    total = masses[0] + masses[1]
    if not found[0] or not found[1] or total <= 0.0 or not math.isfinite(total):
        return None, {
            "found_class0": found[0],
            "found_class1": found[1],
            "prob_mass": total,
        }
    return masses[1] / total, {
        "found_class0": found[0],
        "found_class1": found[1],
        "prob_mass": total,
    }


def score_probability(
    source: dict[str, Any],
    model_label: str,
    split: str,
    p1: float | None,
    diagnostics: dict[str, Any],
    target_field: str = "opponent_next_action",
) -> dict[str, Any]:
    meta = source.get("metadata", {})
    target = int(source["target"][target_field])
    if p1 is None:
        prediction = None
        brier = None
        nll = None
        correct = None
        confidence = None
    else:
        clipped = min(max(float(p1), 1e-12), 1.0 - 1e-12)
        prediction = int(p1 >= 0.5)
        brier = (p1 - target) ** 2
        nll = -(target * math.log(clipped) + (1 - target) * math.log(1.0 - clipped))
        correct = int(prediction == target)
        confidence = max(p1, 1.0 - p1)
    return {
        "split": split,
        "model": model_label,
        "example_id": meta.get("example_id", ""),
        "match_id": meta.get("match_id", ""),
        "role": meta.get("role", ""),
        "opponent_family": meta.get("opponent_family", ""),
        "round_bin": meta.get("round_bin", ""),
        "confidence_stratum": meta.get("confidence_stratum", ""),
        "target_action": target,
        "p_action_1": p1,
        "predicted_action": prediction,
        "valid_logits": int(p1 is not None),
        "brier": brier,
        "constant_05_brier": 0.25,
        "nll": nll,
        "correct": correct,
        "confidence": confidence,
        "confidence_if_correct": confidence if correct == 1 else None,
        "confidence_if_incorrect": confidence if correct == 0 else None,
        **diagnostics,
    }


def mean_present(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None and math.isfinite(float(row[field]))]
    return sum(values) / len(values) if values else None


def expected_calibration_error(group: list[dict[str, Any]], bins: int = 10) -> float | None:
    valid = [row for row in group if row.get("p_action_1") is not None]
    if not valid:
        return None
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        bin_index = min(int(float(row["p_action_1"]) * bins), bins - 1)
        buckets[bin_index].append(row)
    total = len(valid)
    return sum(
        len(bucket) / total
        * abs(float(mean_present(bucket, "p_action_1")) - float(mean_present(bucket, "target_action")))
        for bucket in buckets.values()
    )


def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    item = {
        "n": len(group),
        "valid_logits": mean_present(group, "valid_logits"),
        "brier": mean_present(group, "brier"),
        "constant_05_brier": 0.25,
        "nll": mean_present(group, "nll"),
        "accuracy": mean_present(group, "correct"),
        "mean_p_action_1": mean_present(group, "p_action_1"),
        "target_action_1_rate": mean_present(group, "target_action"),
        "mean_confidence": mean_present(group, "confidence"),
        "confidence_correct": mean_present(group, "confidence_if_correct"),
        "confidence_incorrect": mean_present(group, "confidence_if_incorrect"),
        "ece_10": expected_calibration_error(group, bins=10),
    }
    item["brier_skill_vs_05"] = None if item["brier"] is None else 1.0 - float(item["brier"]) / 0.25
    item["confidence_gap"] = (
        None
        if item["mean_confidence"] is None or item["accuracy"] is None
        else float(item["mean_confidence"]) - float(item["accuracy"])
    )
    return item


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["model"])].append(row)
    output = []
    for (split, model), group in sorted(groups.items()):
        output.append({"split": split, "model": model, **summarize_group(group)})
    return output


def summarize_breakdowns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for field in ("role", "opponent_family", "round_bin"):
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(row["split"], row["model"], str(row.get(field, "")))].append(row)
        for (split, model, value), group in sorted(groups.items()):
            output.append(
                {
                    "split": split,
                    "model": model,
                    "breakdown": field,
                    "value": value,
                    **summarize_group(group),
                }
            )
    return output


def calibration_rows(rows: list[dict[str, Any]], bins: int = 10) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        p1 = row.get("p_action_1")
        if p1 is None:
            continue
        bin_index = min(int(float(p1) * bins), bins - 1)
        groups[(row["split"], row["model"], bin_index)].append(row)
    for (split, model, bin_index), group in sorted(groups.items()):
        output.append(
            {
                "split": split,
                "model": model,
                "bin": bin_index,
                "bin_left": bin_index / bins,
                "bin_right": (bin_index + 1) / bins,
                "n": len(group),
                "mean_p_action_1": mean_present(group, "p_action_1"),
                "empirical_action_1_rate": mean_present(group, "target_action"),
                "brier": mean_present(group, "brier"),
            }
        )
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
    data_root = Path(args.data_root) / args.data_subdir
    split_rows = {
        "id": balanced_sample(load_jsonl(data_root / "val.jsonl"), args.samples_per_split, args.seed),
        "ood": balanced_sample(load_jsonl(data_root / "ood.jsonl"), args.samples_per_split, args.seed + 1),
    }
    for split, rows in split_rows.items():
        print(f"[logit-audit] split={split} selected={len(rows)}")
        for row in rows:
            build_probe_prompt(row["prompt"])
    if args.dry_run:
        return

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    candidates = parse_candidates(args.candidate)
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
    token_classes = binary_token_classes(tokenizer)
    allowed_token_ids = sorted(token_classes)
    print(f"[logit-audit] token_classes={token_classes}")
    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        min_tokens=1,
        logprobs=len(allowed_token_ids),
        allowed_token_ids=allowed_token_ids,
        seed=args.seed,
    )
    adapters = [("parent", LoRARequest("parent_adapter", 1, args.parent_adapter_path))]
    adapters.extend(
        (label, LoRARequest(f"candidate_adapter_{index}", index + 2, path))
        for index, (label, path) in enumerate(candidates, start=1)
    )
    scored: list[dict[str, Any]] = []
    for model_label, lora_request in adapters:
        for split, sources in split_rows.items():
            for start in range(0, len(sources), args.batch_size):
                batch = sources[start : start + args.batch_size]
                prompts = [render_prompt(tokenizer, build_probe_prompt(row["prompt"])) for row in batch]
                generated = llm.generate(prompts, sampling, lora_request=lora_request)
                for source, item in zip(batch, generated):
                    first_logprobs = None
                    if item.outputs and item.outputs[0].logprobs:
                        first_logprobs = item.outputs[0].logprobs[0]
                    p1, diagnostics = binary_probability(first_logprobs, token_classes)
                    scored.append(
                        score_probability(
                            source,
                            model_label,
                            split,
                            p1,
                            diagnostics,
                            target_field=args.target_field,
                        )
                    )
                print(
                    f"[logit-audit] model={model_label} split={split} "
                    f"progress={min(start + len(batch), len(sources))}/{len(sources)}"
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(scored)
    breakdown = summarize_breakdowns(scored)
    calibration = calibration_rows(scored)
    write_csv(output_dir / "matched_rows.csv", scored)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "breakdown.csv", breakdown)
    write_csv(output_dir / "calibration.csv", calibration)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "probe_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "probe_suffix": PROBE_SUFFIX,
                "samples_per_split": args.samples_per_split,
                "data_subdir": args.data_subdir,
                "target_field": args.target_field,
                "seed": args.seed,
                "token_classes": token_classes,
                "candidate_labels": [label for label, _ in candidates],
            },
            handle,
            indent=2,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
