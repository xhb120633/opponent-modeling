from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


from prompt_templates import (
    PROMPT_VERSION_LOCKED_ACTION_ONLY,
    PROMPT_VERSION_LOCKED_STRUCTURED,
    build_locked_action_only_prompt,
    build_locked_structured_prompt,
    prompt_hash,
)

PROMPT_MODES = ("neutral_behavior", "recursive_prompt", PROMPT_VERSION_LOCKED_STRUCTURED, PROMPT_VERSION_LOCKED_ACTION_ONLY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline pass@K support audit for hide-and-seek prefixes."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default="data_pass64_support_vllm")
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--prompt-mode", choices=PROMPT_MODES, required=True)
    parser.add_argument("--opponent", default="ToM2")
    parser.add_argument("--samples-per-prefix", type=int, default=64)
    parser.add_argument("--prefixes-per-bin", type=int, default=20)
    parser.add_argument("--round-bins", default="early,mid,late")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lora-path", default="", help="Optional vLLM LoRA adapter path.")
    parser.add_argument("--lora-name", default="sft_adapter", help="Name assigned to the LoRA adapter in vLLM.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def round_bin(decision_round: Any) -> str:
    r = int(decision_round)
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


def extract_json_object(text: str) -> dict[str, Any] | None:
    t = str(text).strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    start = t.find("{")
    end = t.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = t[start : end + 1]
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


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
    m = re.search(r"\b([01])\b", text)
    if m:
        return int(m.group(1))
    return None


def recursive_marker_score(text: str) -> int:
    t = str(text).lower()
    patterns = [
        r"opponent (?:is )?(?:modeling|predicting|expecting|anticipating)",
        r"(?:expects|predicts|anticipates) (?:me|my|that i)",
        r"thinks (?:i|i will|my)",
        r"my (?:pattern|strategy|policy|previous choices|recent choices)",
        r"counter[- ]?(?:predict|predicting|prediction|strategy)",
        r"exploit(?:s|ing)? my",
        r"what (?:i|the seeker) (?:will|would) choose",
    ]
    return sum(bool(re.search(p, t)) for p in patterns)


def build_prompt(row: pd.Series, prompt_mode: str) -> str:
    state = strip_old_response_constraints(str(row["llm_prompt"]))
    if prompt_mode == PROMPT_VERSION_LOCKED_STRUCTURED:
        return build_locked_structured_prompt(state)
    if prompt_mode == PROMPT_VERSION_LOCKED_ACTION_ONLY:
        return build_locked_action_only_prompt(state)
    base = f"""You are choosing the next action in a repeated hide-and-seek game.

Use only the game-state prompt below. It contains the complete history available before the current round.

Game-state prompt:
```text
{state}
```
"""
    if prompt_mode == "neutral_behavior":
        instruction = """
Explain the opponent's behavioral pattern only from the observed history, then predict the opponent's next action and choose your action.

Return exactly one JSON object, with no extra text:
{
  "opponent_behavior_model": "short explanation",
  "opponent_next_action": 0 or 1,
  "my_best_action": 0 or 1,
  "final_action": 0 or 1
}
""".strip()
    elif prompt_mode == "recursive_prompt":
        instruction = """
Explicitly reason about how the opponent may be modeling your recent policy. Then predict what the opponent expects you to choose, predict the opponent's next action, and choose your action.

Return exactly one JSON object, with no extra text:
{
  "opponent_behavior_model": "short explanation",
  "opponent_model_of_me": "what the opponent likely thinks I will do",
  "opponent_expected_my_action": 0 or 1,
  "opponent_next_action": 0 or 1,
  "my_best_action": 0 or 1,
  "final_action": 0 or 1
}
""".strip()
    else:
        raise ValueError(prompt_mode)
    return f"{base}\n{instruction}"


def model_key(model_name: str) -> str:
    return (model_name or "").strip().lower()


def is_qwen_hybrid(model_name: str) -> bool:
    k = model_key(model_name)
    if not k.startswith("qwen/"):
        return False
    if "qwen2.5" in k or "qwen2-" in k or "/qwen2/" in k:
        return False
    return any(marker in k for marker in ("qwen3", "qwq"))


def render_prompt(tokenizer: Any, model_name: str, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"enable_thinking": False} if is_qwen_hybrid(model_name) else {}
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def infer_role_from_prompt(prompt: str) -> str:
    return "hider" if "You are the HIDER" in str(prompt) else "seeker"


def best_action_for_role(opponent_action: int, role: str) -> int:
    return 1 - opponent_action if role == "hider" else opponent_action


def normalize_strategy(value: Any) -> str | None:
    text = str(value or "").strip()
    compact = re.sub(r"[\s_\-]+", "", text.lower())
    for cand in ("ToM0", "ToM1", "ToM2", "QL", "RB", "WSLS"):
        if compact == cand.lower():
            return cand
    aliases = {"qlearning": "QL", "biasedrandom": "RB", "winstayloseshift": "WSLS"}
    return aliases.get(compact)


def load_prefixes(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input_csv)
    df = df[df["opponent_name"].eq(args.opponent)].copy()
    if df.empty:
        raise ValueError(f"No rows for opponent={args.opponent!r} in {args.input_csv}")
    df["round_bin"] = df["decision_round"].apply(round_bin)
    bins = [x.strip() for x in args.round_bins.split(",") if x.strip()]
    selected = []
    for b in bins:
        sub = df[df["round_bin"].eq(b)].copy()
        if sub.empty:
            continue
        n = min(args.prefixes_per_bin, len(sub))
        selected.append(sub.sample(n=n, random_state=args.seed + len(selected)))
    out = pd.concat(selected, ignore_index=True)
    out = out.reset_index(drop=True)
    out["prefix_id"] = [
        f"{args.opponent}_{r.round_bin}_rep{int(r.repeat_idx):03d}_round{int(r.decision_round):02d}_{i:04d}"
        for i, r in out.iterrows()
    ]
    out["label_opponent_next_action"] = pd.to_numeric(
        out.get("opponent_choice_from_history", out["choice_agent1"]), errors="coerce"
    ).astype(int)
    out["role"] = out["llm_prompt"].map(infer_role_from_prompt)
    out["label_my_best_action"] = [
        best_action_for_role(int(opp), str(role))
        for opp, role in zip(out["label_opponent_next_action"], out["role"])
    ]
    return out


def summarize(samples: pd.DataFrame) -> pd.DataFrame:
    def pass_rate(x: pd.Series) -> float:
        return bool(x.any())

    summary = (
        samples.groupby(["model_name", "prompt_mode", "opponent", "round_bin", "prefix_id"], dropna=False)
        .agg(
            samples=("sample_idx", "size"),
            sample_action_acc=("final_action_correct", "mean"),
            sample_opp_pred_acc=("opponent_prediction_correct", "mean"),
            pass_action=("final_action_correct", pass_rate),
            pass_opp_pred=("opponent_prediction_correct", pass_rate),
            pass_minimal=("minimal_useful_sample", pass_rate),
            pass_recursive_correct=("recursive_and_correct", pass_rate),
            reward_variance=("final_action_correct", "var"),
            action_entropy=("pred_final_action", entropy_binary),
            recursive_marker_any=("recursive_marker", "mean"),
        )
        .reset_index()
    )
    overall = (
        summary.groupby(["model_name", "prompt_mode", "opponent", "round_bin"], dropna=False)
        .agg(
            prefixes=("prefix_id", "size"),
            sample_action_acc=("sample_action_acc", "mean"),
            sample_opp_pred_acc=("sample_opp_pred_acc", "mean"),
            pass_action=("pass_action", "mean"),
            pass_opp_pred=("pass_opp_pred", "mean"),
            pass_minimal=("pass_minimal", "mean"),
            pass_recursive_correct=("pass_recursive_correct", "mean"),
            reward_variance=("reward_variance", "mean"),
            action_entropy=("action_entropy", "mean"),
            recursive_marker_rate=("recursive_marker_any", "mean"),
        )
        .reset_index()
    )
    all_rows = (
        summary.groupby(["model_name", "prompt_mode", "opponent"], dropna=False)
        .agg(
            prefixes=("prefix_id", "size"),
            sample_action_acc=("sample_action_acc", "mean"),
            sample_opp_pred_acc=("sample_opp_pred_acc", "mean"),
            pass_action=("pass_action", "mean"),
            pass_opp_pred=("pass_opp_pred", "mean"),
            pass_minimal=("pass_minimal", "mean"),
            pass_recursive_correct=("pass_recursive_correct", "mean"),
            reward_variance=("reward_variance", "mean"),
            action_entropy=("action_entropy", "mean"),
            recursive_marker_rate=("recursive_marker_any", "mean"),
        )
        .reset_index()
    )
    all_rows["round_bin"] = "overall"
    return pd.concat([all_rows, overall], ignore_index=True)


def entropy_binary(values: pd.Series) -> float:
    vals = [v for v in values if v in (0, 1)]
    if not vals:
        return 0.0
    p = sum(v == 1 for v in vals) / len(vals)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def main() -> None:
    args = parse_args()
    prefixes = load_prefixes(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model_name.replace("/", "-").replace(":", "-")
    tag = f"_{args.output_tag}" if args.output_tag else ""
    stem = f"{safe_model}_{args.opponent}_{args.prompt_mode}_pass{args.samples_per_prefix}{tag}"
    samples_path = out_dir / f"{stem}_samples.csv"
    summary_path = out_dir / f"{stem}_summary.csv"

    print(
        f"[pass64] model={args.model_name} mode={args.prompt_mode} opponent={args.opponent} "
        f"prefixes={len(prefixes)} samples_per_prefix={args.samples_per_prefix} "
        f"output={samples_path}"
    )
    print(prefixes.groupby("round_bin").size().to_string())
    if args.dry_run:
        return

    from vllm import LLM, SamplingParams

    engine_kwargs = dict(
        model=args.model_name,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )
    lora_request = None
    if args.lora_path:
        from vllm.lora.request import LoRARequest

        engine_kwargs.update(enable_lora=True, max_loras=1)
        lora_request = LoRARequest(args.lora_name, 1, args.lora_path)
    llm = LLM(**engine_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        n=args.samples_per_prefix,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    rows: list[dict[str, Any]] = []
    for start in range(0, len(prefixes), args.batch_size):
        chunk = prefixes.iloc[start : start + args.batch_size].copy()
        prompts = [
            render_prompt(tokenizer, args.model_name, build_prompt(row, args.prompt_mode))
            for _, row in chunk.iterrows()
        ]
        if lora_request is None:
            outputs = llm.generate(prompts, sampling_params)
        else:
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        for row_pos, (_, prefix_row) in enumerate(chunk.iterrows()):
            out = outputs[row_pos]
            for sample_idx, completion in enumerate(out.outputs):
                text = getattr(completion, "text", "")
                parsed = extract_json_object(text)
                pred_strategy = normalize_strategy(parsed.get("opponent_strategy")) if parsed else None
                pred_opp = parse_binary(parsed.get("opponent_next_action")) if parsed else None
                pred_best = parse_binary(parsed.get("my_best_action")) if parsed else None
                pred_final = parse_binary(parsed.get("final_action")) if parsed else None
                label_opp = int(prefix_row["label_opponent_next_action"])
                label_best = int(prefix_row["label_my_best_action"])
                explanation = json.dumps(parsed, ensure_ascii=True) if parsed else str(text)
                marker_score = recursive_marker_score(explanation)
                opp_ok = pred_opp == label_opp
                final_ok = pred_final == label_best
                role = str(prefix_row.get("role", "seeker"))
                pred_best_from_opp = None if pred_opp is None else best_action_for_role(pred_opp, role)
                consistent = pred_best is not None and pred_best == pred_best_from_opp and pred_final == pred_best
                rows.append(
                    {
                        "model_name": args.model_name,
                        "prompt_mode": args.prompt_mode,
                        "opponent": args.opponent,
                        "prefix_id": prefix_row["prefix_id"],
                        "source_csv": Path(args.input_csv).name,
                        "match_id": prefix_row.get("match_id", ""),
                        "repeat_idx": prefix_row.get("repeat_idx", ""),
                        "decision_round": prefix_row.get("decision_round", ""),
                        "round_bin": prefix_row.get("round_bin", ""),
                        "role": prefix_row.get("role", ""),
                        "sample_idx": sample_idx,
                        "prompt_version": args.prompt_mode,
                        "prompt_hash": prompt_hash(args.prompt_mode) if args.prompt_mode in (PROMPT_VERSION_LOCKED_STRUCTURED, PROMPT_VERSION_LOCKED_ACTION_ONLY) else "",
                        "label_opponent_next_action": label_opp,
                        "label_my_best_action": label_best,
                        "pred_opponent_strategy": pred_strategy,
                        "pred_opponent_next_action": pred_opp,
                        "pred_my_best_action": pred_best,
                        "pred_final_action": pred_final,
                        "opponent_prediction_correct": opp_ok,
                        "final_action_correct": final_ok,
                        "prediction_final_consistent": consistent,
                        "strategy_correct": pred_strategy == args.opponent,
                        "recursive_marker_score": marker_score,
                        "recursive_marker": marker_score >= 2,
                        "minimal_useful_sample": bool(opp_ok and final_ok and consistent),
                        "recursive_and_correct": bool(opp_ok and final_ok and consistent and ((pred_strategy == "ToM2") if args.prompt_mode == PROMPT_VERSION_LOCKED_STRUCTURED else (False if args.prompt_mode == PROMPT_VERSION_LOCKED_ACTION_ONLY else marker_score >= 2))),
                        "parse_ok": parsed is not None and pred_final is not None,
                        "raw_reply": text,
                    }
                )
        print(f"[pass64] generated prefixes {start + len(chunk)}/{len(prefixes)}")

    samples = pd.DataFrame(rows)
    samples.to_csv(samples_path, index=False)
    summary = summarize(samples)
    summary.to_csv(summary_path, index=False)
    print(f"[pass64] saved {samples_path}")
    print(f"[pass64] saved {summary_path}")
    print(summary[summary["round_bin"].eq("overall")].to_string(index=False))


if __name__ == "__main__":
    main()
