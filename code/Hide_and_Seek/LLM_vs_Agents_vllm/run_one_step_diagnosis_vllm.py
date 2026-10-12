from __future__ import annotations

import argparse
import ast
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import pandas as pd


from prompt_templates import OPPONENT_CANDIDATES, PROMPT_VERSION_LOCKED_STRUCTURED, build_locked_structured_prompt, prompt_hash

ACTION_LABELS = {0: "Tree", 1: "House"}


@dataclass(frozen=True)
class DiagnosisParse:
    opponent_strategy: str | None
    opponent_next_action: int | None
    my_best_action: int | None
    parse_ok: bool
    parse_backend: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline one-step-ahead diagnosis on saved hide-and-seek prefixes. "
            "Each CSV row becomes one batched vLLM diagnostic query."
        )
    )
    parser.add_argument("--model-name", required=True, help="HF model name used for diagnostic inference.")
    parser.add_argument(
        "--input-csv",
        action="append",
        default=None,
        help="Raw ALL_MATCHES CSV to diagnose. If omitted, auto-discovered from --input-dir.",
    )
    parser.add_argument(
        "--input-dir",
        default="data_original_vllm_qwen25_scale",
        help="Directory searched when --input-csv is omitted.",
    )
    parser.add_argument(
        "--input-glob",
        default="*ALL_MATCHES_vllm*.csv",
        help="Glob pattern under --input-dir for auto-discovery.",
    )
    parser.add_argument(
        "--output-dir",
        default="data_diagnosis_vllm_qwen25_3b_32b",
        help="Directory for row-level diagnosis CSVs and summaries.",
    )
    parser.add_argument("--output-tag", default="", help="Optional filename tag.")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--batch-collect-ms", type=int, default=20)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--parallel-prompts", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lora-path", default="", help="Optional vLLM LoRA adapter path.")
    parser.add_argument("--lora-name", default="sft_adapter", help="Name assigned to the LoRA adapter in vLLM.")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="low")
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Debug only. Production diagnosis should leave this unset to cover every row.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore any existing output rows and rerun all prompts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load inputs and print planned counts without importing vLLM or generating.",
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
        input_dir = Path(args.input_dir)
        model_tail = args.model_name.split("/")[-1]
        candidates = sorted(input_dir.glob(args.input_glob))
        paths = [p for p in candidates if model_tail in p.name]
        if not paths:
            slug = safe_model_slug(args.model_name)
            paths = [p for p in candidates if slug in p.name]

    paths = [p.resolve() for p in paths]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing input CSV(s): " + ", ".join(missing))
    if not paths:
        raise FileNotFoundError(
            f"No input CSVs found for model={args.model_name!r} in {args.input_dir!r}"
        )
    return paths


def make_diagnosis_id(source_csv: Path, row_index: int, row: pd.Series) -> str:
    payload = "|".join(
        [
            source_csv.name,
            str(row_index),
            str(row.get("match_id", "")),
            str(row.get("opponent_name", "")),
            str(row.get("repeat_idx", "")),
            str(row.get("decision_round", "")),
        ]
    )
    return sha1(payload.encode("utf-8")).hexdigest()[:16]


def read_inputs(paths: list[Path], limit_rows: int | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required = {
        "llm_prompt",
        "opponent_name",
        "decision_round",
        "repeat_idx",
        "match_id",
        "choice_agent0",
        "choice_agent1",
    }
    for path in paths:
        df = pd.read_csv(path)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        df = df.reset_index(names="source_row_index")
        df["source_csv"] = path.name
        df["source_csv_path"] = str(path)
        df["diagnosis_id"] = [
            make_diagnosis_id(path, int(idx), row) for idx, row in df.iterrows()
        ]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if "opponent_choice_from_history" in out.columns:
        out["label_opponent_next_action"] = pd.to_numeric(
            out["opponent_choice_from_history"], errors="coerce"
        )
    else:
        out["label_opponent_next_action"] = pd.to_numeric(out["choice_agent1"], errors="coerce")
    out["label_my_best_action"] = out.apply(label_best_action, axis=1)
    out["round_bin"] = out["decision_round"].apply(round_bin)
    out = out.dropna(subset=["llm_prompt", "label_opponent_next_action", "label_my_best_action"])
    out["label_opponent_next_action"] = out["label_opponent_next_action"].astype(int)
    out["label_my_best_action"] = out["label_my_best_action"].astype(int)
    if limit_rows is not None:
        out = out.head(limit_rows).copy()
    return out


def infer_role(row: pd.Series) -> str:
    prompt = str(row.get("llm_prompt", ""))
    if "You are the HIDER" in prompt:
        return "hider"
    return "seeker"


def label_best_action(row: pd.Series) -> int:
    opp = int(row.get("opponent_choice_from_history", row.get("choice_agent1")))
    role = infer_role(row)
    if role == "hider":
        return 1 - opp
    return opp


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


def strip_old_response_constraints(prompt: str) -> str:
    marker = "\nReply with ONLY one character:"
    if marker in prompt:
        return prompt.split(marker, 1)[0].rstrip()
    return prompt.rstrip()


def build_prompt(row: pd.Series) -> str:
    original_prompt = strip_old_response_constraints(str(row["llm_prompt"]))
    return build_locked_structured_prompt(original_prompt)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = stripped[start : end + 1]
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_strategy(value: Any, raw_text: str = "") -> str | None:
    text = str(value or "").strip()
    aliases = {
        "tom0": "ToM0",
        "tom 0": "ToM0",
        "level0": "ToM0",
        "level 0": "ToM0",
        "tom1": "ToM1",
        "tom 1": "ToM1",
        "level1": "ToM1",
        "level 1": "ToM1",
        "tom2": "ToM2",
        "tom 2": "ToM2",
        "level2": "ToM2",
        "level 2": "ToM2",
        "q-learning": "QL",
        "q learning": "QL",
        "qlearner": "QL",
        "ql": "QL",
        "biased random": "RB",
        "random bias": "RB",
        "rb": "RB",
        "win-stay/lose-shift": "WSLS",
        "win stay lose shift": "WSLS",
        "wsls": "WSLS",
    }
    low = re.sub(r"[_\-]+", " ", text.lower())
    low_compact = re.sub(r"\s+", "", low)
    for cand in OPPONENT_CANDIDATES:
        if text == cand or low_compact == cand.lower():
            return cand
    if low in aliases:
        return aliases[low]
    if low_compact in aliases:
        return aliases[low_compact]

    search = raw_text.lower()
    for key, cand in sorted(aliases.items(), key=lambda kv: -len(kv[0])):
        if key in search:
            return cand
    for cand in OPPONENT_CANDIDATES:
        if cand.lower() in search:
            return cand
    return None


def _parse_binary(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    text = str(value or "").strip()
    if text in ("0", "1"):
        return int(text)
    m = re.search(r"\b([01])\b", text)
    if m:
        return int(m.group(1))
    if re.search(r"\btree\b", text, flags=re.IGNORECASE):
        return 0
    if re.search(r"\bhouse\b", text, flags=re.IGNORECASE):
        return 1
    return None


def _field_from_json(parsed: dict[str, Any], names: tuple[str, ...]) -> Any:
    lowered = {str(k).lower().strip(): v for k, v in parsed.items()}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def parse_diagnosis(raw_text: str) -> DiagnosisParse:
    parsed = _extract_json_object(raw_text)
    if parsed is not None:
        strategy_value = _field_from_json(
            parsed, ("opponent_strategy", "strategy", "opponent_policy")
        )
        opponent_action_value = _field_from_json(
            parsed,
            (
                "opponent_next_action",
                "predicted_opponent_action",
                "opponent_action",
                "next_opponent_action",
            ),
        )
        best_action_value = _field_from_json(
            parsed, ("my_best_action", "best_action", "my_action", "optimal_action")
        )
        strategy = _normalize_strategy(strategy_value, raw_text)
        opp_action = _parse_binary(opponent_action_value)
        best_action = _parse_binary(best_action_value)
        return DiagnosisParse(
            opponent_strategy=strategy,
            opponent_next_action=opp_action,
            my_best_action=best_action,
            parse_ok=strategy is not None and opp_action is not None and best_action is not None,
            parse_backend="json",
        )

    strategy = _normalize_strategy("", raw_text)
    opp_match = re.search(
        r"opponent(?:_next)?_?action[^01]*(0|1)|opponent[^.\n]*(?:tree|house|0|1)",
        raw_text,
        flags=re.IGNORECASE,
    )
    best_match = re.search(
        r"(?:my_?best_?action|best_?action|my action|optimal action)[^01]*(0|1)",
        raw_text,
        flags=re.IGNORECASE,
    )
    opp_action = _parse_binary(opp_match.group(0) if opp_match else None)
    best_action = _parse_binary(best_match.group(0) if best_match else None)
    return DiagnosisParse(
        opponent_strategy=strategy,
        opponent_next_action=opp_action,
        my_best_action=best_action,
        parse_ok=strategy is not None and opp_action is not None and best_action is not None,
        parse_backend="regex",
    )


def build_output_row(row: pd.Series, raw_reply: str, diagnostic_model: str) -> dict[str, Any]:
    parsed = parse_diagnosis(raw_reply)
    label_opp = int(row["label_opponent_next_action"])
    label_best = int(row["label_my_best_action"])
    role = infer_role(row)
    pred_best_from_pred_opp = (
        None
        if parsed.opponent_next_action is None
        else (1 - parsed.opponent_next_action if role == "hider" else parsed.opponent_next_action)
    )
    original_action = _parse_binary(row.get("choice_agent0"))
    return {
        "diagnosis_id": row["diagnosis_id"],
        "diagnostic_model": diagnostic_model,
        "source_llm_model": row.get("agent0", row.get("llm_model_name", "")),
        "source_csv": row.get("source_csv", ""),
        "source_row_index": int(row.get("source_row_index", -1)),
        "match_id": row.get("match_id", ""),
        "repeat_idx": row.get("repeat_idx", ""),
        "opponent_name": row.get("opponent_name", ""),
        "decision_round": row.get("decision_round", ""),
        "round_bin": row.get("round_bin", ""),
        "role": role,
        "label_opponent_strategy": row.get("opponent_name", ""),
        "label_opponent_next_action": label_opp,
        "label_my_best_action": label_best,
        "original_llm_action": original_action,
        "pred_opponent_strategy": parsed.opponent_strategy,
        "pred_opponent_next_action": parsed.opponent_next_action,
        "pred_my_best_action": parsed.my_best_action,
        "pred_best_action_from_pred_opponent": pred_best_from_pred_opp,
        "strategy_correct": parsed.opponent_strategy == row.get("opponent_name", ""),
        "opponent_action_correct": parsed.opponent_next_action == label_opp,
        "my_best_action_correct": parsed.my_best_action == label_best,
        "best_response_consistent": parsed.my_best_action == pred_best_from_pred_opp,
        "original_action_correct": original_action == label_best,
        "diagnosis_parse_ok": parsed.parse_ok,
        "diagnosis_parse_backend": parsed.parse_backend,
        "prompt_version": PROMPT_VERSION_LOCKED_STRUCTURED,
        "prompt_hash": prompt_hash(),
        "diagnosis_raw_reply": raw_reply,
    }


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        existing = pd.read_csv(path, usecols=["diagnosis_id"])
    except Exception:
        return set()
    return set(existing["diagnosis_id"].astype(str))


def summarize(output_csv: Path, summary_csv: Path) -> None:
    if not output_csv.exists():
        return
    df = pd.read_csv(output_csv)
    bool_cols = [
        "strategy_correct",
        "opponent_action_correct",
        "my_best_action_correct",
        "best_response_consistent",
        "original_action_correct",
        "diagnosis_parse_ok",
    ]
    for col in bool_cols:
        df[col] = df[col].astype(float)
    group_cols = ["diagnostic_model", "opponent_name", "round_bin"]
    by_bin = (
        df.groupby(group_cols, dropna=False)[bool_cols]
        .mean()
        .reset_index()
        .assign(n=df.groupby(group_cols, dropna=False).size().to_numpy())
    )
    overall = (
        df.groupby(["diagnostic_model", "opponent_name"], dropna=False)[bool_cols]
        .mean()
        .reset_index()
        .assign(round_bin="overall")
        .assign(n=df.groupby(["diagnostic_model", "opponent_name"], dropna=False).size().to_numpy())
    )
    summary = pd.concat([overall, by_bin], ignore_index=True)
    summary.to_csv(summary_csv, index=False)


def run_generation(args: argparse.Namespace, pending: pd.DataFrame, output_csv: Path) -> None:
    from vllm_batcher import DecodeOptions, VLLMBatchedGenerator

    generator = VLLMBatchedGenerator(
        model_name=args.model_name,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        batch_collect_ms=args.batch_collect_ms,
        max_batch_size=args.max_batch_size,
        seed=args.seed,
        lora_path=args.lora_path or None,
        lora_name=args.lora_name,
    )
    decode_options = DecodeOptions(
        enable_thinking=args.enable_thinking,
        reasoning_effort=args.reasoning_effort,
        apply_chat_template=True,
    )

    try:
        total = len(pending)
        for start in range(0, total, args.chunk_size):
            chunk = pending.iloc[start : start + args.chunk_size].copy()
            rows_by_pos = {pos: row for pos, (_, row) in enumerate(chunk.iterrows())}
            results: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=max(1, args.parallel_prompts)) as ex:
                fut_map = {
                    ex.submit(generator.generate_one, build_prompt(row), decode_options): pos
                    for pos, row in rows_by_pos.items()
                }
                for fut in as_completed(fut_map):
                    pos = fut_map[fut]
                    row = rows_by_pos[pos]
                    raw_reply = fut.result()
                    results.append(build_output_row(row, raw_reply, args.model_name))
            results.sort(key=lambda item: item["source_row_index"])
            append_rows(output_csv, results)
            print(
                f"[diagnosis] saved chunk rows={len(results)} "
                f"progress={min(start + len(chunk), total)}/{total} output={output_csv}"
            )
    finally:
        generator.close()


def main() -> None:
    args = parse_args()
    input_csvs = discover_input_csvs(args)
    data = read_inputs(input_csvs, limit_rows=args.limit_rows)
    output_dir = Path(args.output_dir)
    tag = f"_{args.output_tag}" if args.output_tag else ""
    out_base = f"{safe_model_slug(args.model_name)}_one_step_diagnosis{tag}.csv"
    output_csv = (output_dir / out_base).resolve()
    summary_csv = output_csv.with_name(output_csv.stem + "_summary.csv")

    if not args.force:
        done = existing_ids(output_csv)
        if done:
            data = data[~data["diagnosis_id"].astype(str).isin(done)].copy()

    print(
        "[diagnosis] plan: "
        f"model={args.model_name} input_csvs={len(input_csvs)} "
        f"pending_rows={len(data)} output={output_csv}"
    )
    for opp, n in data.groupby("opponent_name").size().sort_index().items():
        print(f"[diagnosis] pending opponent={opp} rows={n}")

    if args.dry_run:
        return
    if data.empty:
        print("[diagnosis] no pending rows; writing summary from existing output.")
        summarize(output_csv, summary_csv)
        return

    run_generation(args, data, output_csv)
    summarize(output_csv, summary_csv)
    print(f"[diagnosis] saved summary: {summary_csv}")


if __name__ == "__main__":
    main()
