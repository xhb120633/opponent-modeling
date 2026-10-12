"""Run hide-and-seek LLM-vs-agent games through Azure OpenAI chat deployments."""

from __future__ import annotations

import argparse
import ast
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tomsup as ts
from tomsup.agent import TOM

from azure_chat_generator import AzureChatGenerator
from llm_tomsup_agent_vllm import VLLMAgent
from mixed_strategy_agents import StrategyMixtureAgent


DEFAULT_ENV_FILE = os.environ.get("TOM_AZURE_ENV_FILE")


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


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



def hide_and_seek_payoff_matrix():
    """agent0=seeker, agent1=hider; matching gives seeker +1, hider -1."""
    matrix = np.array(
        [
            [[1, -1], [-1, 1]],
            [[-1, 1], [1, -1]],
        ]
    )
    return ts.PayoffMatrix(name="hide_and_seek_matching_pennies", predefined=matrix)

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

    opponent_player_index = 1 - llm_player_index
    opponent_history_col = f"history_agent{opponent_player_index}"
    opponent_choices = []
    opponent_internal_states = []
    opponent_active_strategies = []
    opponent_candidate_choices = []
    opponent_history_parsed = []
    for value in enriched.get(opponent_history_col, pd.Series(index=enriched.index, dtype=object)):
        parsed = _safe_to_dict(value)
        opponent_history_parsed.append(parsed)
        if parsed is None:
            opponent_choices.append(None)
            opponent_internal_states.append(None)
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
        opponent_internal_states.append(
            json.dumps(_to_jsonable(internal_states), ensure_ascii=True, sort_keys=True)
            if internal_states is not None
            else None
        )

    enriched["opponent_name"] = opponent_name
    enriched["opponent_choice_from_history"] = opponent_choices
    enriched["opponent_active_strategy"] = opponent_active_strategies
    enriched["opponent_candidate_choices_json"] = opponent_candidate_choices
    enriched["opponent_internal_states_json"] = opponent_internal_states
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


def run_single_match(
    generator,
    deployment,
    opponent_name,
    opponent_fn,
    p_matrix,
    n_rounds,
    repeat_idx,
    prompt_mode,
    llm_role,
    reasoning_effort,
):
    if llm_role not in ("seeker", "hider"):
        raise ValueError(f"llm_role must be seeker|hider, got {llm_role!r}")
    llm_agent = VLLMAgent(
        model_name=deployment,
        generator=generator,
        save_history=True,
        enable_thinking=False,
        reasoning_effort=reasoning_effort,
        prompt_mode=prompt_mode,
        llm_role=llm_role,
    )
    opponent = opponent_fn()
    llm_player_index = 0 if llm_role == "seeker" else 1
    agent0, agent1 = (opponent, llm_agent) if llm_role == "hider" else (llm_agent, opponent)
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
    results["agent0"] = deployment if llm_player_index == 0 else opponent_name
    results["agent1"] = opponent_name if llm_player_index == 0 else deployment
    results["llm_name"] = deployment
    results["base_llm_name"] = deployment
    results["repeat_idx"] = repeat_idx
    results["match_id"] = f"{deployment}_as_{llm_role}_vs_{opponent_name}_rep{repeat_idx}"
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", type=str, default=os.environ.get("AZURE_DEPLOYMENT_TOP_LEVEL", ""))
    parser.add_argument("--env-file", type=str, default=DEFAULT_ENV_FILE)
    parser.add_argument("--api-version", type=str, default=None)
    parser.add_argument("--endpoint", type=str, default=None)
    parser.add_argument("--n-rounds", type=int, default=50)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--repeat-offset", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="data_azure_gpt54")
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--prompt-mode", type=str, default="locked_structured_v1")
    parser.add_argument("--llm-role", type=str, choices=["seeker", "hider"], default="seeker")
    parser.add_argument("--opponents", type=str, default="ToM0,ToM1,ToM2")
    parser.add_argument("--max-completion-tokens", type=int, default=384)
    parser.add_argument("--reasoning-effort", type=str, default="service_default")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--send-temperature", action="store_true")
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-delay-s", type=float, default=2.0)
    parser.add_argument("--sleep-between-calls-s", type=float, default=0.0)
    parser.add_argument("--write-match-checkpoints", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.deployment:
        raise SystemExit("--deployment is required unless AZURE_DEPLOYMENT_TOP_LEVEL is set")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    opponents = build_opponents()
    selected_opponents = [name.strip() for name in args.opponents.split(",") if name.strip()]
    unknown = [name for name in selected_opponents if name not in opponents]
    if unknown:
        raise SystemExit(f"Unknown opponents: {unknown}. Available: {sorted(opponents)}")

    p_matrix = hide_and_seek_payoff_matrix()
    generator = AzureChatGenerator(
        deployment=args.deployment,
        env_file=args.env_file,
        api_version=args.api_version,
        endpoint=args.endpoint,
        max_completion_tokens=args.max_completion_tokens,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        send_temperature=args.send_temperature,
        timeout_s=args.request_timeout_s,
        max_retries=args.max_retries,
        retry_base_delay_s=args.retry_base_delay_s,
        sleep_between_calls_s=args.sleep_between_calls_s,
    )

    tag = f"_{args.output_tag}" if args.output_tag else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = (
        f"{_safe_slug(args.deployment)}_{args.llm_role}_{'_'.join(selected_opponents)}_"
        f"{args.prompt_mode}{tag}_{timestamp}"
    )
    csv_path = output_dir / f"{stem}.csv"
    summary_path = output_dir / f"{stem}_summary.csv"
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / f"{stem}_matches"
    if args.write_match_checkpoints or args.resume_completed:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    print(
        f"[azure-run] deployment={args.deployment} role={args.llm_role} opponents={selected_opponents} "
        f"rounds={args.n_rounds} repeats={args.n_repeats} prompt={args.prompt_mode} "
        f"max_completion_tokens={args.max_completion_tokens} reasoning_effort={args.reasoning_effort}"
    )
    for opponent_name in selected_opponents:
        for local_rep in range(args.n_repeats):
            repeat_idx = args.repeat_offset + local_rep
            match_stem = f"{_safe_slug(args.deployment)}_{args.llm_role}_{opponent_name}_rep{repeat_idx}"
            match_path = checkpoint_dir / f"{match_stem}.csv"
            if args.resume_completed and match_path.exists():
                print(f"[resume] {match_path}", flush=True)
                all_results.append(pd.read_csv(match_path))
                continue
            print(f"[match] role={args.llm_role} opponent={opponent_name} repeat={repeat_idx}", flush=True)
            match_df = run_single_match(
                generator=generator,
                deployment=args.deployment,
                opponent_name=opponent_name,
                opponent_fn=opponents[opponent_name],
                p_matrix=p_matrix,
                n_rounds=args.n_rounds,
                repeat_idx=repeat_idx,
                prompt_mode=args.prompt_mode,
                llm_role=args.llm_role,
                reasoning_effort=args.reasoning_effort,
            )
            if args.write_match_checkpoints or args.resume_completed:
                match_df.to_csv(match_path, index=False)
                print(f"[checkpoint] wrote {match_path}", flush=True)
            all_results.append(match_df)

    out = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    usage = generator.usage_totals
    out["azure_deployment"] = args.deployment
    out["azure_reasoning_effort"] = args.reasoning_effort
    out["azure_prompt_tokens_total_run"] = usage.prompt_tokens
    out["azure_completion_tokens_total_run"] = usage.completion_tokens
    out["azure_total_tokens_total_run"] = usage.total_tokens
    out["azure_reasoning_tokens_total_run"] = usage.reasoning_tokens
    out["azure_cached_prompt_tokens_total_run"] = usage.cached_prompt_tokens

    out.to_csv(csv_path, index=False)

    summary = (
        out.groupby(["azure_deployment", "llm_role", "opponent_name", "repeat_idx"], dropna=False)
        .agg(
            final_payoff=("llm_payoff", "sum"),
            mean_payoff=("llm_payoff", "mean"),
            win_rate=("llm_payoff", lambda s: float((s > 0).mean())),
            parse_success=("llm_parse_mode", lambda s: float((s != "fallback").mean())),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {summary_path}")
    print(
        f"[usage] calls={usage.calls} prompt_tokens={usage.prompt_tokens} "
        f"completion_tokens={usage.completion_tokens} reasoning_tokens={usage.reasoning_tokens} "
        f"total_tokens={usage.total_tokens} cached_prompt_tokens={usage.cached_prompt_tokens}"
    )


if __name__ == "__main__":
    main()