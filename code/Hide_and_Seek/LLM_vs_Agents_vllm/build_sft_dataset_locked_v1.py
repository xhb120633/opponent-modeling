from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from hashlib import sha1
from pathlib import Path
from typing import Any

import pandas as pd

from prompt_templates import (
    OPPONENT_CANDIDATES,
    PROMPT_VERSION_LOCKED_STRUCTURED,
    build_locked_structured_prompt,
    prompt_hash,
)


DEFAULT_INPUT_DIRS = (
    "data_original_vllm_qwen25_scale",
    "data_original_vllm_qwen25_scale_hider_tom012",
    "data_original_vllm_qwen25_scale_hider_tom12",
    "data_original_vllm_qwen25_7b_hider_ql_rb_wsls_sft",
)
DEFAULT_SOURCE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TARGET_MODES = ("structured_tom", "action_only")
PROMPT_VERSION_ACTION_ONLY = "locked_action_only_v1"

ACTION_ONLY_INSTRUCTION = """
Use the same game-state information and payoff history to choose the action that maximizes your payoff under your role and the scoring rule.

Return exactly one JSON object, with no extra text:
{
  "final_action": 0 or 1
}
""".strip()


def action_only_prompt_hash() -> str:
    payload = "\n\n".join([PROMPT_VERSION_ACTION_ONLY, ACTION_ONLY_INSTRUCTION])
    return sha1(payload.encode("utf-8")).hexdigest()[:12]


def build_action_only_prompt(state_block: str) -> str:
    return f"{state_block.rstrip()}\n\n{ACTION_ONLY_INSTRUCTION}".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build locked-v1 SFT datasets from saved hide-and-seek battle histories. "
            "The output uses the exact locked_structured_v1 prompt family and "
            "assistant-only structured JSON targets."
        )
    )
    parser.add_argument(
        "--input-csv",
        action="append",
        default=None,
        help="Specific ALL_MATCHES CSV. Can be repeated. If omitted, --input-dir is scanned.",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        default=None,
        help=(
            "Directory searched for input CSVs. Can be repeated. Defaults to the "
            "Qwen2.5 seeker scaling and hider scaling directories."
        ),
    )
    parser.add_argument("--input-glob", default="*ALL_MATCHES_vllm*.csv")
    parser.add_argument(
        "--source-model",
        default=DEFAULT_SOURCE_MODEL,
        help=(
            "Only include histories from this source LLM. Use 'all' to include every "
            "matching CSV in the input directories."
        ),
    )
    parser.add_argument(
        "--opponents",
        default="QL,RB,WSLS,ToM0,ToM1,ToM2",
        help="Comma-separated opponent mixture to include when present.",
    )
    parser.add_argument("--min-decision-round", type=int, default=1)
    parser.add_argument("--max-decision-round", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default="data_sft_locked_v1_qwen25_7b",
        help="Directory for train/val/test JSONL files and summaries.",
    )
    parser.add_argument(
        "--target-mode",
        choices=("both",) + TARGET_MODES,
        default="both",
        help="Build structured ToM targets, action-only targets, or both.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and summarize rows but do not write files.",
    )
    return parser.parse_args()


def safe_model_slug(model_name: str) -> str:
    return model_name.replace("/", "-").replace(":", "-")


def filename_model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def discover_input_csvs(args: argparse.Namespace) -> list[Path]:
    if args.input_csv:
        paths = [Path(p) for p in args.input_csv]
    else:
        input_dirs = [Path(p) for p in (args.input_dir or DEFAULT_INPUT_DIRS)]
        candidates: list[Path] = []
        for input_dir in input_dirs:
            candidates.extend(sorted(input_dir.glob(args.input_glob)))
        if args.source_model.lower() == "all":
            paths = candidates
        else:
            tail = args.source_model.split("/")[-1]
            slug = safe_model_slug(args.source_model)
            paths = [p for p in candidates if tail in p.name or slug in p.name]

    paths = [p.resolve() for p in paths]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing input CSV(s): " + ", ".join(missing))
    if not paths:
        raise FileNotFoundError("No input CSVs found for SFT dataset construction.")
    return paths


def strip_old_response_constraints(prompt: str) -> str:
    marker = "\nReply with ONLY one character:"
    if marker in prompt:
        return prompt.split(marker, 1)[0].rstrip()
    return prompt.rstrip()


def infer_role(row: pd.Series) -> str:
    role = str(row.get("llm_role", "")).strip().lower()
    if role in ("seeker", "hider"):
        return role
    prompt = str(row.get("llm_prompt", ""))
    if "You are the HIDER" in prompt:
        return "hider"
    return "seeker"


def infer_opponent_action(row: pd.Series, role: str) -> int | None:
    if "opponent_choice_from_history" in row and pd.notna(row["opponent_choice_from_history"]):
        return parse_binary(row["opponent_choice_from_history"])

    if "opponent_player_index" in row and pd.notna(row["opponent_player_index"]):
        idx = parse_binary(row["opponent_player_index"])
        if idx in (0, 1):
            return parse_binary(row.get(f"choice_agent{idx}"))

    fallback_col = "choice_agent0" if role == "hider" else "choice_agent1"
    return parse_binary(row.get(fallback_col))


def parse_binary(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    text = str(value or "").strip()
    if text in ("0", "1"):
        return int(text)
    match = re.search(r"\b([01])\b", text)
    if match:
        return int(match.group(1))
    return None


def best_action_for_role(opponent_action: int, role: str) -> int:
    return 1 - opponent_action if role == "hider" else opponent_action


def round_bin(decision_round: Any) -> str:
    try:
        r = int(decision_round)
    except (TypeError, ValueError):
        return "unknown"
    if r <= 10:
        return "early"
    if r <= 30:
        return "mid"
    return "late"


def stable_id(parts: list[Any], n: int = 16) -> str:
    payload = "|".join(str(p) for p in parts)
    return sha1(payload.encode("utf-8")).hexdigest()[:n]


def read_labeled_rows(args: argparse.Namespace, paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    opponents = {x.strip() for x in args.opponents.split(",") if x.strip()}
    unknown = sorted(opponents - set(OPPONENT_CANDIDATES))
    if unknown:
        raise ValueError(f"Unknown opponent labels: {unknown}")

    required = {"llm_prompt", "opponent_name", "decision_round", "choice_agent0", "choice_agent1"}
    for path in paths:
        df = pd.read_csv(path)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        df = df.reset_index(names="source_row_index")
        df["source_csv"] = path.name
        df["source_csv_path"] = str(path)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data[data["opponent_name"].isin(opponents)].copy()
    data = data[pd.to_numeric(data["decision_round"], errors="coerce") >= args.min_decision_round]
    if args.max_decision_round is not None:
        data = data[pd.to_numeric(data["decision_round"], errors="coerce") <= args.max_decision_round]

    records: list[dict[str, Any]] = []
    for idx, row in data.iterrows():
        role = infer_role(row)
        opponent_action = infer_opponent_action(row, role)
        if opponent_action not in (0, 1):
            continue
        best_action = best_action_for_role(int(opponent_action), role)
        state_block = strip_old_response_constraints(str(row["llm_prompt"]))
        prompt = build_locked_structured_prompt(state_block)
        source_csv = str(row.get("source_csv", ""))
        match_id = str(row.get("match_id", f"{source_csv}:match{row.get('llm_match_idx', '')}"))
        repeat_idx = row.get("repeat_idx", "")
        decision_round = int(row["decision_round"])
        example_id = stable_id([source_csv, row.get("source_row_index", idx), match_id, decision_round])
        group_id = stable_id([source_csv, match_id, role, row["opponent_name"]], n=20)

        target_structured = {
            "opponent_strategy": str(row["opponent_name"]),
            "opponent_next_action": int(opponent_action),
            "my_best_action": int(best_action),
            "final_action": int(best_action),
        }
        target_action_only = {"final_action": int(best_action)}
        records.append(
            {
                "example_id": example_id,
                "group_id": group_id,
                "source_csv": source_csv,
                "source_row_index": int(row.get("source_row_index", idx)),
                "source_model": str(row.get("llm_model_name", row.get("base_llm_name", ""))),
                "match_id": match_id,
                "repeat_idx": int(repeat_idx) if str(repeat_idx).strip().lstrip("-").isdigit() else repeat_idx,
                "decision_round": decision_round,
                "round_bin": round_bin(decision_round),
                "role": role,
                "opponent_name": str(row["opponent_name"]),
                "state_block": state_block,
                "prompt": prompt,
                "target_structured": target_structured,
                "target_action_only": target_action_only,
                "prompt_version": PROMPT_VERSION_LOCKED_STRUCTURED,
                "prompt_hash": prompt_hash(),
            }
        )

    out = pd.DataFrame.from_records(records)
    if out.empty:
        raise ValueError("No valid labeled SFT examples after filtering.")
    return out


def assign_splits(data: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    train_ratio = args.train_ratio / ratio_sum
    val_ratio = args.val_ratio / ratio_sum

    group_to_split: dict[str, str] = {}
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for (role, opponent, group_id), _ in data.groupby(["role", "opponent_name", "group_id"], dropna=False):
        strata[(str(role), str(opponent))].append(str(group_id))

    for stratum, groups in strata.items():
        groups = sorted(set(groups))
        seed_payload = f"{args.seed}|{stratum[0]}|{stratum[1]}"
        rng = random.Random(int(sha1(seed_payload.encode("utf-8")).hexdigest()[:8], 16))
        rng.shuffle(groups)
        n = len(groups)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_val = min(max(n_val, 1), n - n_train - 1)
        else:
            n_train = max(1, n - 1)
            n_val = 0
        for i, group_id in enumerate(groups):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            group_to_split[group_id] = split

    return data["group_id"].map(group_to_split)


def prompt_info_for_mode(target_mode: str) -> tuple[str, str]:
    if target_mode == "structured_tom":
        return PROMPT_VERSION_LOCKED_STRUCTURED, prompt_hash()
    if target_mode == "action_only":
        return PROMPT_VERSION_ACTION_ONLY, action_only_prompt_hash()
    raise ValueError(target_mode)


def prompt_for_mode(row: pd.Series, target_mode: str) -> str:
    if target_mode == "structured_tom":
        return row["prompt"]
    if target_mode == "action_only":
        return build_action_only_prompt(row["state_block"])
    raise ValueError(target_mode)


def render_completion_and_value_spans(target: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    pieces = ["{"]
    spans: list[dict[str, Any]] = []
    for i, (key, value) in enumerate(target.items()):
        if i:
            pieces.append(",")
        pieces.append(json.dumps(str(key), ensure_ascii=False))
        pieces.append(":")
        value_text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        start = sum(len(p) for p in pieces)
        pieces.append(value_text)
        end = sum(len(p) for p in pieces)
        if isinstance(value, str) and value_text.startswith('"') and value_text.endswith('"'):
            value_start = start + 1
            value_end = end - 1
        else:
            value_start = start
            value_end = end
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


def make_sft_record(row: pd.Series, target_mode: str) -> dict[str, Any]:
    target_key = "target_structured" if target_mode == "structured_tom" else "target_action_only"
    target = row[target_key]
    prompt = prompt_for_mode(row, target_mode)
    prompt_version, prompt_digest = prompt_info_for_mode(target_mode)
    completion, loss_value_spans = render_completion_and_value_spans(target)
    metadata = {
        "example_id": row["example_id"],
        "group_id": row["group_id"],
        "source_csv": row["source_csv"],
        "source_row_index": int(row["source_row_index"]),
        "source_model": row["source_model"],
        "match_id": row["match_id"],
        "repeat_idx": row["repeat_idx"],
        "decision_round": int(row["decision_round"]),
        "round_bin": row["round_bin"],
        "role": row["role"],
        "opponent_name": row["opponent_name"],
        "target_mode": target_mode,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_digest,
    }
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        "prompt": prompt,
        "completion": completion,
        "target": target,
        "loss_mask_type": "assistant_value_only",
        "loss_value_spans": loss_value_spans,
        "metadata": metadata,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_dataset(data: pd.DataFrame, args: argparse.Namespace, target_mode: str) -> dict[str, Any]:
    out_dir = Path(args.output_dir) / target_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_rows = data[data["split"].eq(split)].copy()
        records = [make_sft_record(row, target_mode) for _, row in split_rows.iterrows()]
        write_jsonl(out_dir / f"{split}.jsonl", records)
        counts[split] = len(records)

    summary = summarize(data)
    summary.to_csv(out_dir / "summary.csv", index=False)
    manifest = {
        "target_mode": target_mode,
        "prompt_version": prompt_info_for_mode(target_mode)[0],
        "prompt_hash": prompt_info_for_mode(target_mode)[1],
        "source_model_filter": args.source_model,
        "opponents": sorted({x.strip() for x in args.opponents.split(",") if x.strip()}),
        "min_decision_round": args.min_decision_round,
        "max_decision_round": args.max_decision_round,
        "seed": args.seed,
        "counts": counts,
        "summary_csv": str((out_dir / "summary.csv").resolve()),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in data.groupby(["split", "role", "opponent_name", "round_bin"], dropna=False):
        split, role, opponent, rb = keys
        rows.append(
            {
                "split": split,
                "role": role,
                "opponent_name": opponent,
                "round_bin": rb,
                "examples": len(sub),
                "groups": sub["group_id"].nunique(),
                "matches": sub["match_id"].nunique(),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "role", "opponent_name", "round_bin"])


def print_plan(data: pd.DataFrame, manifests: list[dict[str, Any]] | None = None) -> None:
    print(f"examples={len(data)} groups={data['group_id'].nunique()} matches={data['match_id'].nunique()}")
    print(f"prompt_version={PROMPT_VERSION_LOCKED_STRUCTURED} prompt_hash={prompt_hash()}")
    print("by split:", Counter(data["split"]))
    print("by role/opponent:")
    for (role, opponent), sub in data.groupby(["role", "opponent_name"], dropna=False):
        print(f"  {role:6s} {opponent:5s} examples={len(sub):5d} groups={sub['group_id'].nunique():3d}")
    if manifests:
        print("written:")
        for manifest in manifests:
            print(f"  {manifest['target_mode']}: {manifest['counts']}")


def main() -> None:
    args = parse_args()
    paths = discover_input_csvs(args)
    data = read_labeled_rows(args, paths)
    data["split"] = assign_splits(data, args)
    modes = TARGET_MODES if args.target_mode == "both" else (args.target_mode,)

    print("input_csvs:")
    for path in paths:
        print(f"  {path}")
    print_plan(data)
    if args.dry_run:
        return

    manifests = [write_dataset(data, args, mode) for mode in modes]
    print_plan(data, manifests=manifests)


if __name__ == "__main__":
    main()
