from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_output_parse import extract_structured_object
from probability_tom_utils import binary_action, predicted_action, probability_01, role_best_response


CONDITIONS = ("free", "oracle_p_op", "shuffled_p_op", "oracle_p_self")
RECURSIVE_MARKER = "\n\nBefore choosing, update a recursive model of the opponent"
BELIEF_FIELD = "opponent_belief_my_action_1_probability"
ACTION_PROBABILITY_FIELD = "opponent_action_1_probability"

DIAGNOSTIC_INSTRUCTION = """
Infer the opponent recursively from the complete observed match history.

Return exactly one JSON object, with no extra text:
{
  "opponent_belief_my_action_1_probability": a number from 0.0 to 1.0,
  "opponent_action_1_probability": a number from 0.0 to 1.0,
  "opponent_next_action": 0 or 1,
  "my_best_action": 0 or 1,
  "final_action": 0 or 1
}

The first probability is the opponent's current belief that you will choose action 1.
The second probability is the probability that the opponent itself will choose action 1.
Use the role and scoring rule above when choosing my_best_action.
final_action should match my_best_action.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal prefix intervention for recursive probability ToM.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--lora-name", default="intervention_adapter")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def behavior_only_state(prompt: str) -> str:
    state = str(prompt).split(RECURSIVE_MARKER, 1)[0].rstrip()
    pattern = re.compile(
        r"Round (\d+): your prior estimate P\(opponent expects you choose 1\) was [^;]+; "
        r"your prior estimate P\(opponent chooses 1\) was [^;]+; "
        r"you chose ([01]); opponent chose ([^;]+); your payoff was ([^\r\n]+)"
    )
    return pattern.sub(
        lambda match: (
            f"Round {match.group(1)}: you chose {match.group(2)}, "
            f"opponent chose {match.group(3)}, your payoff was {match.group(4)}"
        ),
        state,
    )


def diagnostic_prompt(prompt: str) -> str:
    return f"{behavior_only_state(prompt)}\n\n{DIAGNOSTIC_INSTRUCTION}"


def add_shuffled_beliefs(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["shuffled_p_op"] = np.nan
    out["shuffled_donor_match_id"] = ""
    group_cols = ["llm_role", "opponent_name", "decision_round"]
    for _, group in out.groupby(group_cols, sort=False):
        indices = list(group.sort_values(["repeat_idx", "match_id"]).index)
        if len(indices) < 2:
            donors = indices
        else:
            donors = indices[1:] + indices[:1]
        for target_idx, donor_idx in zip(indices, donors):
            out.at[target_idx, "shuffled_p_op"] = out.at[
                donor_idx, "opponent_belief_llm_action_1_probability"
            ]
            out.at[target_idx, "shuffled_donor_match_id"] = out.at[donor_idx, "match_id"]
    return out


def assistant_prefix(condition: str, row: pd.Series) -> str | None:
    if condition == "free":
        return None
    if condition == "oracle_p_op":
        value = float(row["opponent_belief_llm_action_1_probability"])
        return f'{{\n  "{BELIEF_FIELD}": {value:.8f},\n  "{ACTION_PROBABILITY_FIELD}":'
    if condition == "shuffled_p_op":
        value = float(row["shuffled_p_op"])
        return f'{{\n  "{BELIEF_FIELD}": {value:.8f},\n  "{ACTION_PROBABILITY_FIELD}":'
    if condition == "oracle_p_self":
        value = float(row["opponent_action_1_probability"])
        return f'{{\n  "{ACTION_PROBABILITY_FIELD}": {value:.8f},\n  "opponent_next_action":'
    raise ValueError(f"Unknown condition: {condition!r}")


def render_prompt(tokenizer: Any, user_prompt: str, prefix: str | None) -> str:
    if prefix is None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": prefix},
        ],
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    if prefix.rstrip() not in rendered:
        raise RuntimeError("Assistant prefill was not preserved by the chat template.")
    return rendered


def parse_result(raw_completion: str, prefix: str | None) -> dict[str, Any]:
    reconstructed = f"{prefix or ''}{raw_completion}"
    parsed = extract_structured_object(reconstructed)
    return {
        "reconstructed_completion": reconstructed,
        "parsed": parsed,
        "predicted_p_op": probability_01(parsed.get(BELIEF_FIELD)) if parsed else None,
        "predicted_p_self": probability_01(parsed.get(ACTION_PROBABILITY_FIELD)) if parsed else None,
        "declared_opponent_action": binary_action(parsed.get("opponent_next_action")) if parsed else None,
        "declared_best_action": binary_action(parsed.get("my_best_action")) if parsed else None,
        "declared_final_action": binary_action(parsed.get("final_action")) if parsed else None,
    }


def output_row(row: pd.Series, condition: str, raw_completion: str, prefix: str | None) -> dict[str, Any]:
    result = parse_result(raw_completion, prefix)
    target_p_op = float(row["opponent_belief_llm_action_1_probability"])
    target_p_self = float(row["opponent_action_1_probability"])
    target_opponent_action = predicted_action(target_p_self)
    target_best_action = role_best_response(target_opponent_action, str(row["llm_role"]))
    predicted_opponent_action = (
        predicted_action(result["predicted_p_self"])
        if result["predicted_p_self"] is not None
        else None
    )
    predicted_best_action = (
        role_best_response(predicted_opponent_action, str(row["llm_role"]))
        if predicted_opponent_action is not None
        else None
    )
    return {
        "model_label": row["model_label"],
        "condition": condition,
        "match_id": row["match_id"],
        "repeat_idx": int(row["repeat_idx"]),
        "decision_round": int(row["decision_round"]),
        "llm_role": row["llm_role"],
        "opponent_name": row["opponent_name"],
        "target_p_op": target_p_op,
        "target_p_self": target_p_self,
        "shuffled_p_op": float(row["shuffled_p_op"]),
        "shuffled_donor_match_id": row["shuffled_donor_match_id"],
        "predicted_p_op": result["predicted_p_op"],
        "predicted_p_self": result["predicted_p_self"],
        "target_opponent_action": target_opponent_action,
        "predicted_opponent_action": predicted_opponent_action,
        "declared_opponent_action": result["declared_opponent_action"],
        "target_best_action": target_best_action,
        "predicted_best_action": predicted_best_action,
        "declared_best_action": result["declared_best_action"],
        "declared_final_action": result["declared_final_action"],
        "parse_ok": (
            result["predicted_p_self"] is not None
            and result["declared_opponent_action"] is not None
            and result["declared_best_action"] is not None
            and result["declared_final_action"] is not None
        ),
        "raw_completion": raw_completion,
        "assistant_prefix": prefix or "",
        "reconstructed_completion": result["reconstructed_completion"],
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["p_self_sqerr"] = (data["predicted_p_self"] - data["target_p_self"]) ** 2
    data["p_op_sqerr"] = (data["predicted_p_op"] - data["target_p_op"]) ** 2
    data["direction_correct"] = data["predicted_opponent_action"] == data["target_opponent_action"]
    data["prediction_declaration_consistent"] = (
        data["declared_opponent_action"] == data["predicted_opponent_action"]
    )
    data["best_response_consistent"] = data["declared_best_action"] == data["predicted_best_action"]
    data["final_consistent"] = data["declared_final_action"] == data["declared_best_action"]
    data["final_action_correct"] = data["declared_final_action"] == data["target_best_action"]
    metrics = [
        "parse_ok",
        "p_op_sqerr",
        "p_self_sqerr",
        "direction_correct",
        "prediction_declaration_consistent",
        "best_response_consistent",
        "final_consistent",
        "final_action_correct",
    ]
    groups = ["model_label", "condition", "llm_role", "opponent_name"]
    summary = data.groupby(groups, dropna=False)[metrics].mean().reset_index()
    summary["n"] = data.groupby(groups, dropna=False).size().to_numpy()
    overall = data.groupby(["model_label", "condition"], dropna=False)[metrics].mean().reset_index()
    overall["llm_role"] = "all"
    overall["opponent_name"] = "all"
    overall["n"] = data.groupby(["model_label", "condition"], dropna=False).size().to_numpy()
    return pd.concat([overall, summary], ignore_index=True)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    required = {
        "llm_prompt",
        "llm_role",
        "opponent_name",
        "match_id",
        "repeat_idx",
        "decision_round",
        "opponent_belief_llm_action_1_probability",
        "opponent_action_1_probability",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    frame = frame.dropna(subset=list(required)).copy()
    if args.limit_rows is not None:
        frame = frame.head(args.limit_rows).copy()
    frame["model_label"] = args.model_label
    frame = add_shuffled_beliefs(frame)

    plan = [
        (row_idx, condition)
        for row_idx in frame.index
        for condition in CONDITIONS
    ]
    print(
        f"[oracle-intervention] model={args.model_label} rows={len(frame)} "
        f"conditions={len(CONDITIONS)} prompts={len(plan)}"
    )
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
    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    lora_request = LoRARequest(args.lora_name, 1, args.lora_path)
    outputs: list[dict[str, Any]] = []
    for start in range(0, len(plan), args.batch_size):
        batch_plan = plan[start : start + args.batch_size]
        rendered: list[str] = []
        metadata: list[tuple[pd.Series, str, str | None]] = []
        for row_idx, condition in batch_plan:
            row = frame.loc[row_idx]
            prefix = assistant_prefix(condition, row)
            rendered.append(render_prompt(tokenizer, diagnostic_prompt(row["llm_prompt"]), prefix))
            metadata.append((row, condition, prefix))
        generated = llm.generate(rendered, sampling, lora_request=lora_request)
        for item, (row, condition, prefix) in zip(generated, metadata):
            text = item.outputs[0].text if item.outputs else ""
            outputs.append(output_row(row, condition, text, prefix))
        print(f"[oracle-intervention] progress={min(start + len(batch_plan), len(plan))}/{len(plan)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{args.model_label}_oracle_intervention_rows.csv"
    summary_csv = output_dir / f"{args.model_label}_oracle_intervention_summary.csv"
    result = pd.DataFrame(outputs)
    result.to_csv(output_csv, index=False)
    summarize(result).to_csv(summary_csv, index=False)
    print(f"[oracle-intervention] saved rows={output_csv} summary={summary_csv}")


if __name__ == "__main__":
    main()
