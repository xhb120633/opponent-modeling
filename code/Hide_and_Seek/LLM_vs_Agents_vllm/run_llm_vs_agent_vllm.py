import argparse
import ast
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tomsup as ts
from tomsup.agent import TOM

from llm_tomsup_agent_vllm import VLLMAgent
from mixed_strategy_agents import StrategyMixtureAgent
from probability_tom_utils import (
    extract_tom_action_probability,
    extract_tom_opponent_belief_probability,
)


def _is_qwen_hybrid_thinking(model_name: str) -> bool:
    k = (model_name or "").strip().lower()
    if not k.startswith("qwen/"):
        return False
    if "qwen2.5" in k or "qwen2-" in k or "/qwen2/" in k:
        return False
    return any(marker in k for marker in ("qwen3", "qwq"))


def _resolve_token_budget(model_name: str, enable_thinking: bool, max_tokens: int, max_model_len: int):
    """
    Completion budget policy:
    - Qwen3 hybrid + thinking ON  -> max_tokens>=8192, max_model_len>=16384
    - Qwen3 hybrid + thinking OFF -> defaults to max_tokens=64 for short-answer
      baselines, but honors larger explicit budgets for structured JSON prompts.
    - gpt-oss                     -> max_tokens>=8192 when reasoning (always)
    """
    mlow = model_name.lower()
    uses_reasoning_budget = "openai/gpt-oss" in mlow or enable_thinking

    if _is_qwen_hybrid_thinking(model_name):
        if enable_thinking:
            uses_reasoning_budget = True
            if max_tokens < 8192:
                print(
                    f"[config] Qwen hybrid enable_thinking={enable_thinking}: "
                    f"max_tokens {max_tokens} -> 8192"
                )
                max_tokens = 8192
            if max_model_len < 16384:
                print(
                    f"[config] Qwen hybrid enable_thinking={enable_thinking}: "
                    f"max_model_len {max_model_len} -> 16384"
                )
                max_model_len = 16384
        else:
            uses_reasoning_budget = False
            if max_model_len < 4096:
                print(
                    f"[config] Qwen hybrid enable_thinking={enable_thinking}: "
                    f"max_model_len {max_model_len} -> 4096"
                )
                max_model_len = 4096
        return max_tokens, max_model_len, uses_reasoning_budget

    if uses_reasoning_budget:
        floor_new = 8192
        if max_tokens < floor_new:
            print(
                f"[config] raising --max-tokens from {max_tokens} to {floor_new} "
                "(gpt-oss or --enable-thinking: reasoning shares completion budget)"
            )
            max_tokens = floor_new
        min_ctx = 16384
        if max_model_len < min_ctx:
            print(
                f"[config] raising --max-model-len from {max_model_len} to {min_ctx} "
                "(match history + single-turn reasoning completion)"
            )
            max_model_len = min_ctx

    return max_tokens, max_model_len, uses_reasoning_budget


def _safe_to_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def enrich_results(results, llm_agent, opponent_name, llm_role="seeker", llm_player_index=0):
    enriched = results.copy()
    enriched["decision_round"] = enriched["round"] + 1
    enriched["payoff_agent0_cumulative"] = enriched["payoff_agent0"].cumsum()
    enriched["payoff_agent1_cumulative"] = enriched["payoff_agent1"].cumsum()
    llm_payoff_col = f"payoff_agent{llm_player_index}"
    opponent_player_index = 1 - llm_player_index
    opponent_payoff_col = f"payoff_agent{opponent_player_index}"
    enriched["llm_role"] = llm_role
    enriched["observer_role"] = llm_role
    enriched["llm_player_index"] = llm_player_index
    enriched["opponent_player_index"] = opponent_player_index
    enriched["llm_payoff"] = enriched[llm_payoff_col]
    enriched["opponent_payoff"] = enriched[opponent_payoff_col]
    enriched["llm_score_cumulative"] = enriched["llm_payoff"].cumsum()
    enriched["opponent_score_cumulative"] = enriched["opponent_payoff"].cumsum()

    if llm_agent.history is not None and not llm_agent.history.empty:
        llm_history = llm_agent.history.copy().rename(
            columns={
                "round": "decision_round",
                "choice": "llm_choice_from_history",
                "op_choice": "llm_observed_op_choice",
                "payoff": "llm_recorded_payoff",
                "predicted_opponent_belief_my_action_1_probability": "llm_predicted_opponent_belief_my_action_1_probability",
                "predicted_opponent_action_1_probability": "llm_predicted_opponent_action_1_probability",
                "raw_reply": "llm_raw_reply",
                "cot_text": "llm_cot_text",
                "final_text": "llm_final_text",
                "split_parse_backend": "llm_split_parse_backend",
                "action_parse_source": "llm_action_parse_source",
                "parse_mode": "llm_parse_mode",
                "enable_thinking": "llm_enable_thinking",
                "prompt_mode": "llm_prompt_mode",
                "prompt": "llm_prompt",
                "prompt_version": "llm_prompt_version",
                "prompt_hash": "llm_prompt_hash",
                "model_name": "llm_model_name",
                "strategy": "llm_strategy",
                "match_idx": "llm_match_idx",
            }
        )
        llm_history["llm_known_score_cumulative"] = (
            llm_history["llm_recorded_payoff"].fillna(0).astype(int).cumsum()
        )
        keep_cols = [
            "decision_round",
            "llm_choice_from_history",
            "llm_observed_op_choice",
            "llm_recorded_payoff",
            "llm_predicted_opponent_belief_my_action_1_probability",
            "llm_predicted_opponent_action_1_probability",
            "llm_raw_reply",
            "llm_cot_text",
            "llm_final_text",
            "llm_split_parse_backend",
            "llm_action_parse_source",
            "llm_parse_mode",
            "llm_enable_thinking",
            "llm_prompt_mode",
            "llm_prompt",
            "llm_prompt_version",
            "llm_prompt_hash",
            "llm_model_name",
            "llm_strategy",
            "llm_match_idx",
            "llm_known_score_cumulative",
        ]
        enriched = enriched.merge(llm_history[keep_cols], on="decision_round", how="left")

    opponent_choices = []
    opponent_internal_states = []
    opponent_action_1_probabilities = []
    opponent_belief_llm_action_1_probabilities = []
    opponent_active_strategies = []
    opponent_candidate_choices = []
    opponent_history_parsed = []
    opponent_history_col = f"history_agent{opponent_player_index}"
    for value in enriched.get(opponent_history_col, pd.Series(index=enriched.index, dtype=object)):
        parsed = _safe_to_dict(value)
        opponent_history_parsed.append(parsed)
        if parsed is None:
            opponent_choices.append(None)
            opponent_internal_states.append(None)
            opponent_action_1_probabilities.append(None)
            opponent_belief_llm_action_1_probabilities.append(None)
            opponent_active_strategies.append(None)
            opponent_candidate_choices.append(None)
            continue
        opponent_choices.append(parsed.get("choice"))
        opponent_active_strategies.append(parsed.get("active_strategy"))
        candidate_choices = parsed.get("candidate_choices")
        opponent_candidate_choices.append(
            json.dumps(_to_jsonable(candidate_choices), ensure_ascii=True, sort_keys=True)
            if candidate_choices is not None
            else None
        )
        internal_states = parsed.get("internal_states")
        if internal_states is None:
            opponent_internal_states.append(None)
            opponent_action_1_probabilities.append(None)
            opponent_belief_llm_action_1_probabilities.append(None)
        else:
            opponent_internal_states.append(
                json.dumps(_to_jsonable(internal_states), ensure_ascii=True, sort_keys=True)
            )
            opponent_action_1_probabilities.append(extract_tom_action_probability(internal_states))
            opponent_belief_llm_action_1_probabilities.append(
                extract_tom_opponent_belief_probability(internal_states)
            )

    enriched["opponent_name"] = opponent_name
    enriched["opponent_choice_from_history"] = opponent_choices
    enriched["opponent_active_strategy"] = opponent_active_strategies
    enriched["opponent_candidate_choices_json"] = opponent_candidate_choices
    enriched["opponent_internal_states_json"] = opponent_internal_states
    enriched["opponent_action_1_probability"] = opponent_action_1_probabilities
    enriched["opponent_belief_llm_action_1_probability"] = (
        opponent_belief_llm_action_1_probabilities
    )
    enriched["opponent_history_source_col"] = opponent_history_col
    enriched["opponent_history_parsed_json"] = [
        json.dumps(_to_jsonable(item), ensure_ascii=True, sort_keys=True) if item is not None else None
        for item in opponent_history_parsed
    ]
    enriched["history_agent1_parsed"] = [
        json.dumps(_to_jsonable(item), ensure_ascii=True, sort_keys=True) if item is not None else None
        for item in opponent_history_parsed
    ]
    return enriched


def build_opponents():
    return {
        "ToM0": lambda: TOM(level=0, save_history=True),
        "ToM1": lambda: TOM(level=1, save_history=True),
        "ToM2": lambda: TOM(level=2, save_history=True),
        "ToM3": lambda: TOM(level=3, save_history=True),
        "ToM4": lambda: TOM(level=4, save_history=True),
        "ToM5": lambda: TOM(level=5, save_history=True),
        "QL": lambda: ts.QL(learning_rate=0.5, b_temp=1, save_history=True),
        "RB": lambda: ts.RB(bias=0.7, save_history=True),
        "WSLS": lambda: ts.WSLS(save_history=True),
        "Mix60ToM2_40RB": lambda: StrategyMixtureAgent(
            "Mix60ToM2_40RB", [("ToM2", 0.6), ("RB", 0.4)], save_history=True
        ),
        "Mix60ToM2_40WSLS": lambda: StrategyMixtureAgent(
            "Mix60ToM2_40WSLS", [("ToM2", 0.6), ("WSLS", 0.4)], save_history=True
        ),
        "Mix60ToM3_40ToM2": lambda: StrategyMixtureAgent(
            "Mix60ToM3_40ToM2", [("ToM3", 0.6), ("ToM2", 0.4)], save_history=True
        ),
        "Trans25ToM2_RB": lambda: StrategyMixtureAgent(
            "Trans25ToM2_RB", [("ToM2", 1.0), ("RB", 1.0)], switch_round=25, save_history=True
        ),
        "Trans25RB_ToM2": lambda: StrategyMixtureAgent(
            "Trans25RB_ToM2", [("RB", 1.0), ("ToM2", 1.0)], switch_round=25, save_history=True
        ),
    }


def run_single_match(
    generator,
    model_name,
    opponent_name,
    opponent_fn,
    p_matrix,
    n_rounds,
    repeat_idx,
    enable_thinking,
    reasoning_effort,
    prompt_mode,
    llm_role,
):
    if llm_role not in ("seeker", "hider"):
        raise ValueError(f"llm_role must be seeker|hider, got {llm_role!r}")
    llm_agent = VLLMAgent(
        model_name=model_name,
        generator=generator,
        save_history=True,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        prompt_mode=prompt_mode,
        llm_role=llm_role,
    )
    opponent = opponent_fn()
    llm_player_index = 0 if llm_role == "seeker" else 1
    if llm_role == "hider":
        agent0 = opponent
        agent1 = llm_agent
    else:
        agent0 = llm_agent
        agent1 = opponent
    results = ts.compete(
        agent0,
        agent1,
        p_matrix=p_matrix,
        n_rounds=n_rounds,
        save_history=True,
        verbose=False,
    )
    results = enrich_results(
        results,
        llm_agent=llm_agent,
        opponent_name=opponent_name,
        llm_role=llm_role,
        llm_player_index=llm_player_index,
    )
    results["agent0"] = model_name if llm_player_index == 0 else opponent_name
    results["agent1"] = opponent_name if llm_player_index == 0 else model_name
    results["llm_name"] = model_name
    results["base_llm_name"] = model_name
    results["repeat_idx"] = repeat_idx
    results["match_id"] = f"{model_name}_as_{llm_role}_vs_{opponent_name}_rep{repeat_idx}"
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--n-rounds", type=int, default=50)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument(
        "--repeat-offset",
        type=int,
        default=0,
        help="Offset added to repeat_idx and match_id; useful for sharded reruns.",
    )
    parser.add_argument("--output-dir", type=str, default="data_original_vllm")
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--parallel-matches", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="vLLM max sequence length. Auto-raised with gpt-oss / --enable-thinking so prompt+reasoning fits.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max new tokens per LLM call. For gpt-oss / --enable-thinking, raised to at least 8192 "
        "(reasoning uses the same budget).",
    )
    parser.add_argument("--batch-collect-ms", type=int, default=20)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lora-path", type=str, default="", help="Optional vLLM LoRA adapter path.")
    parser.add_argument("--lora-name", type=str, default="sft_adapter", help="Name assigned to the LoRA adapter in vLLM.")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--reasoning-effort", type=str, choices=["low", "medium", "high"], default="low")
    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=[
            "baseline",
            "structured_behavior",
            "structured_behavior_brief",
            "structured_rule",
            "structured_strategy",
            "locked_structured_v1",
            "locked_action_only_v1",
            "locked_prediction_only_v1",
            "locked_probability_v2",
            "locked_recursive_probability_v3",
            "locked_predictive_state_v1",
        ],
        default="baseline",
        help="Prompt/action format for online play. baseline preserves the original action-only prompt.",
    )
    parser.add_argument(
        "--llm-role",
        type=str,
        choices=["seeker", "hider"],
        default="seeker",
        help="Role played by the LLM. seeker preserves the original agent0 configuration; hider swaps LLM to agent1.",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default="",
        help="Comma-separated opponent subset, e.g. ToM2 or ToM0,ToM1,ToM2. Empty means all opponents.",
    )
    parser.add_argument(
        "--no-checkpoint-matches",
        action="store_true",
        help="Disable per-match checkpoint CSVs. By default each completed match is saved immediately.",
    )
    return parser.parse_args()


def _checkpoint_path(checkpoint_dir: Path, safe_model: str, suffix: str, repeat_idx: int, opponent_name: str) -> Path:
    safe_opp = opponent_name.replace("/", "-").replace(":", "-")
    return checkpoint_dir / f"{safe_model}_rep{repeat_idx:03d}_{safe_opp}{suffix}.part.csv"


def _write_checkpoint(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def main():
    from vllm_batcher import VLLMBatchedGenerator

    args = parse_args()
    out_dir = Path(args.output_dir)
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    safe_model = args.model_name.replace("/", "-").replace(":", "-")
    planned_path = (out_dir / f"{safe_model}_ALL_MATCHES_vllm{suffix}.csv").resolve()
    checkpoint_enabled = not args.no_checkpoint_matches
    checkpoint_dir = out_dir / ".match_checkpoints"
    print(
        "[run] audit: "
        f"model_name={args.model_name!r} "
        f"enable_thinking={bool(args.enable_thinking)} "
        f"reasoning_effort={args.reasoning_effort!r} "
        f"prompt_mode={args.prompt_mode!r} "
        f"llm_role={args.llm_role!r} "
        f"repeat_offset={args.repeat_offset} "
        f"output_tag={args.output_tag!r} "
        f"checkpoint_matches={checkpoint_enabled} "
        f"planned_csv={planned_path} "
        f"lora_path={args.lora_path!r}"
    )
    job = os.environ.get("SLURM_JOB_ID", "")
    arr = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    if job or arr:
        print(f"[run] audit: SLURM_JOB_ID={job or '-'} SLURM_ARRAY_TASK_ID={arr or '-'}")

    opponents = build_opponents()
    if args.opponents.strip():
        keep = [x.strip() for x in args.opponents.split(",") if x.strip()]
        unknown = sorted(set(keep) - set(opponents))
        if unknown:
            raise SystemExit(f"Unknown opponent(s): {unknown}; valid={sorted(opponents)}")
        opponents = {name: opponents[name] for name in keep}
    custom_matrix = np.array([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]])
    p_matrix = ts.PayoffMatrix(name="custom", predefined=custom_matrix)

    max_tokens, max_model_len, uses_reasoning_budget = _resolve_token_budget(
        args.model_name,
        args.enable_thinking,
        args.max_tokens,
        args.max_model_len,
    )

    generator = VLLMBatchedGenerator(
        model_name=args.model_name,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=max_tokens,
        batch_collect_ms=args.batch_collect_ms,
        max_batch_size=args.max_batch_size,
        seed=args.seed,
        lora_path=args.lora_path or None,
        lora_name=args.lora_name,
    )
    print(
        f"[run] audit: max_tokens={max_tokens} max_model_len={max_model_len} "
        f"uses_reasoning_budget={uses_reasoning_budget}"
    )
    generator.audit_thinking_template()

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        for rep in range(args.n_repeats):
            repeat_idx = args.repeat_offset + rep
            for opp_name, opp_fn in opponents.items():
                jobs.append((repeat_idx, opp_name, opp_fn))

        all_results = []
        pending_jobs = []
        for rep, opp_name, opp_fn in jobs:
            checkpoint_path = _checkpoint_path(checkpoint_dir, safe_model, suffix, rep, opp_name)
            if checkpoint_enabled and checkpoint_path.is_file():
                df = pd.read_csv(checkpoint_path)
                all_results.append(df)
                print(f"[checkpoint] loaded rep={rep} opp={opp_name}: {checkpoint_path}")
            else:
                pending_jobs.append((rep, opp_name, opp_fn))

        with ThreadPoolExecutor(max_workers=max(1, args.parallel_matches)) as ex:
            fut_map = {
                ex.submit(
                    run_single_match,
                    generator,
                    args.model_name,
                    opp_name,
                    opp_fn,
                    p_matrix,
                    args.n_rounds,
                    rep,
                    args.enable_thinking,
                    args.reasoning_effort,
                    args.prompt_mode,
                    args.llm_role,
                ): (rep, opp_name)
                for rep, opp_name, opp_fn in pending_jobs
            }
            for fut in as_completed(fut_map):
                rep, opp_name = fut_map[fut]
                df = fut.result()
                all_results.append(df)
                if checkpoint_enabled:
                    checkpoint_path = _checkpoint_path(checkpoint_dir, safe_model, suffix, rep, opp_name)
                    _write_checkpoint(df, checkpoint_path)
                    print(f"[checkpoint] saved rep={rep} opp={opp_name}: {checkpoint_path}")
                llm_score = int(df["llm_payoff"].sum())
                opp_score = int(df["opponent_payoff"].sum())
                print(f"[done] rep={rep} opp={opp_name} llm={llm_score} opp={opp_score}")

        final_df = pd.concat(all_results, ignore_index=True)
        out_path = planned_path
        if os.environ.get("VLLM_REFUSE_CSV_OVERWRITE", "").strip() in ("1", "true", "yes"):
            if out_path.is_file():
                raise SystemExit(
                    f"Refusing to overwrite existing CSV (set VLLM_REFUSE_CSV_OVERWRITE=0 or remove file): {out_path}"
                )
        final_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
    finally:
        generator.close()


if __name__ == "__main__":
    main()
