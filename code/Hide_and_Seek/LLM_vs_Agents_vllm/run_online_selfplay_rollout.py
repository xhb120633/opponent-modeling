from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tomsup as ts

from llm_tomsup_agent_vllm import VLLMAgent
from openai_chat_generator import OpenAIChatGenerator


def payoff_matrix() -> ts.PayoffMatrix:
    custom_matrix = np.array([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]])
    return ts.PayoffMatrix(name="custom", predefined=custom_matrix)


def make_generator(base_url: str, model: str, args: argparse.Namespace) -> OpenAIChatGenerator:
    return OpenAIChatGenerator(
        model=model,
        base_url=base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
    )


def make_agent(
    model_name: str,
    generator: OpenAIChatGenerator,
    role: str,
    prompt_mode: str,
    label: str,
) -> VLLMAgent:
    agent = VLLMAgent(
        model_name=model_name,
        generator=generator,
        save_history=True,
        enable_thinking=False,
        reasoning_effort="low",
        prompt_mode=prompt_mode,
        llm_role=role,
    )
    agent.strategy = label
    return agent


def merge_history(results: pd.DataFrame, agent: VLLMAgent, player_index: int, prefix: str) -> pd.DataFrame:
    if agent.history is None or agent.history.empty:
        return results
    hist = agent.history.copy().rename(
        columns={
            "round": "decision_round",
            "choice": f"{prefix}_choice_from_history",
            "op_choice": f"{prefix}_observed_op_choice",
            "payoff": f"{prefix}_recorded_payoff",
            "raw_reply": f"{prefix}_raw_reply",
            "cot_text": f"{prefix}_cot_text",
            "final_text": f"{prefix}_final_text",
            "split_parse_backend": f"{prefix}_split_parse_backend",
            "action_parse_source": f"{prefix}_action_parse_source",
            "parse_mode": f"{prefix}_parse_mode",
            "enable_thinking": f"{prefix}_enable_thinking",
            "prompt_mode": f"{prefix}_prompt_mode",
            "prompt": f"{prefix}_prompt",
            "prompt_version": f"{prefix}_prompt_version",
            "prompt_hash": f"{prefix}_prompt_hash",
            "model_name": f"{prefix}_model_name",
            "strategy": f"{prefix}_strategy",
            "match_idx": f"{prefix}_match_idx",
        }
    )
    hist[f"{prefix}_known_score_cumulative"] = hist[f"{prefix}_recorded_payoff"].fillna(0).astype(int).cumsum()
    keep = [c for c in hist.columns if c == "decision_round" or c.startswith(f"{prefix}_")]
    out = results.merge(hist[keep], on="decision_round", how="left")
    out[f"{prefix}_player_index"] = player_index
    return out


def enrich(
    results: pd.DataFrame,
    agent0: VLLMAgent,
    agent1: VLLMAgent,
    repeat_idx: int,
    args: argparse.Namespace,
) -> pd.DataFrame:
    out = results.copy()
    out["decision_round"] = out["round"] + 1
    out["payoff_agent0_cumulative"] = out["payoff_agent0"].cumsum()
    out["payoff_agent1_cumulative"] = out["payoff_agent1"].cumsum()
    out["agent0_label"] = args.agent0_label
    out["agent1_label"] = args.agent1_label
    out["agent0_role"] = "seeker"
    out["agent1_role"] = "hider"
    out["agent0_backend"] = "candidate" if args.candidate_role == "seeker" else "frozen"
    out["agent1_backend"] = "candidate" if args.candidate_role == "hider" else "frozen"
    out["agent0_prompt_mode"] = args.prompt_mode
    out["agent1_prompt_mode"] = args.prompt_mode
    out["candidate_role"] = args.candidate_role
    out["candidate_label"] = args.candidate_label
    out["frozen_label"] = args.frozen_label
    out["repeat_idx"] = repeat_idx
    out["match_id"] = (
        f"{args.candidate_label}_as_{args.candidate_role}_vs_"
        f"{args.frozen_label}_rep{repeat_idx:03d}"
    )
    out = merge_history(out, agent0, player_index=0, prefix="agent0_llm")
    out = merge_history(out, agent1, player_index=1, prefix="agent1_llm")
    return out


def run_single_match(
    args: argparse.Namespace,
    candidate_generator: OpenAIChatGenerator,
    frozen_generator: OpenAIChatGenerator,
    repeat_idx: int,
) -> pd.DataFrame:
    if args.candidate_role == "seeker":
        agent0 = make_agent(args.candidate_model, candidate_generator, "seeker", args.prompt_mode, args.candidate_label)
        agent1 = make_agent(args.frozen_model, frozen_generator, "hider", args.prompt_mode, args.frozen_label)
    else:
        agent0 = make_agent(args.frozen_model, frozen_generator, "seeker", args.prompt_mode, args.frozen_label)
        agent1 = make_agent(args.candidate_model, candidate_generator, "hider", args.prompt_mode, args.candidate_label)
    results = ts.compete(
        agent0,
        agent1,
        p_matrix=payoff_matrix(),
        n_rounds=args.n_rounds,
        save_history=True,
        verbose=False,
    )
    return enrich(results, agent0=agent0, agent1=agent1, repeat_idx=repeat_idx, args=args)


def checkpoint_path(out_dir: Path, args: argparse.Namespace, repeat_idx: int) -> Path:
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    return out_dir / ".match_checkpoints" / (
        f"{args.candidate_label}_as_{args.candidate_role}_vs_{args.frozen_label}_"
        f"rep{repeat_idx:03d}{suffix}.part.csv"
    )


def write_checkpoint(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect online self-play rollouts from candidate vs frozen vLLM servers.")
    p.add_argument("--candidate-base-url", required=True)
    p.add_argument("--frozen-base-url", required=True)
    p.add_argument("--candidate-model", default="candidate")
    p.add_argument("--frozen-model", default="frozen")
    p.add_argument("--candidate-label", default="candidate")
    p.add_argument("--frozen-label", default="frozen")
    p.add_argument("--candidate-role", choices=["seeker", "hider"], required=True)
    p.add_argument("--agent0-label", default="")
    p.add_argument("--agent1-label", default="")
    p.add_argument("--prompt-mode", default="locked_structured_v1")
    p.add_argument("--n-rounds", type=int, default=50)
    p.add_argument("--n-repeats", type=int, default=8)
    p.add_argument("--repeat-offset", type=int, default=0)
    p.add_argument("--parallel-matches", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--timeout-s", type=float, default=180.0)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--output-dir", default="data_online_selfplay_rollouts")
    p.add_argument("--output-tag", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.agent0_label:
        args.agent0_label = args.candidate_label if args.candidate_role == "seeker" else args.frozen_label
    if not args.agent1_label:
        args.agent1_label = args.candidate_label if args.candidate_role == "hider" else args.frozen_label

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    out_path = out_dir / f"{args.candidate_label}_as_{args.candidate_role}_vs_{args.frozen_label}{suffix}.csv"
    print(
        "[online-selfplay] audit: "
        f"candidate={args.candidate_label}:{args.candidate_role}:{args.candidate_base_url} "
        f"frozen={args.frozen_label}:{args.frozen_base_url} rounds={args.n_rounds} "
        f"repeats={args.n_repeats} out={out_path}"
    )

    candidate_generator = make_generator(args.candidate_base_url, args.candidate_model, args)
    frozen_generator = make_generator(args.frozen_base_url, args.frozen_model, args)
    try:
        all_results = []
        pending = []
        for rep in range(args.n_repeats):
            repeat_idx = args.repeat_offset + rep
            cpath = checkpoint_path(out_dir, args, repeat_idx)
            if cpath.is_file():
                all_results.append(pd.read_csv(cpath))
                print(f"[checkpoint] loaded rep={repeat_idx}: {cpath}")
            else:
                pending.append(repeat_idx)

        with ThreadPoolExecutor(max_workers=max(1, args.parallel_matches)) as ex:
            futures = {
                ex.submit(run_single_match, args, candidate_generator, frozen_generator, rep): rep
                for rep in pending
            }
            for fut in as_completed(futures):
                rep = futures[fut]
                df = fut.result()
                all_results.append(df)
                cpath = checkpoint_path(out_dir, args, rep)
                write_checkpoint(df, cpath)
                p0 = int(df["payoff_agent0"].sum())
                p1 = int(df["payoff_agent1"].sum())
                print(f"[done] rep={rep} agent0={p0} agent1={p1} saved={cpath}")

        final = pd.concat(all_results, ignore_index=True)
        final.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
    finally:
        candidate_generator.close()
        frozen_generator.close()


if __name__ == "__main__":
    main()
