from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tomsup as ts

from openai_chat_generator import OpenAIChatGenerator
from run_llm_vs_agent_vllm import build_opponents, run_single_match


def payoff_matrix() -> ts.PayoffMatrix:
    matrix = np.array([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]])
    return ts.PayoffMatrix(name="custom", predefined=matrix)


def checkpoint_path(root: Path, role: str, opponent: str, repeat_idx: int) -> Path:
    return root / ".match_checkpoints" / f"{role}_{opponent}_rep{repeat_idx:03d}.part.csv"


def write_checkpoint(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect current-policy rollouts against frozen tomsup agents.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="candidate")
    parser.add_argument("--opponents", default="ToM0,ToM1,ToM2")
    parser.add_argument("--roles", default="seeker,hider")
    parser.add_argument("--prompt-mode", default="locked_probability_v2")
    parser.add_argument("--n-rounds", type=int, default=50)
    parser.add_argument("--n-repeats-per-cell", type=int, default=2)
    parser.add_argument("--parallel-matches", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-tag", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    opponents = build_opponents()
    requested_opponents = [item.strip() for item in args.opponents.split(",") if item.strip()]
    requested_roles = [item.strip() for item in args.roles.split(",") if item.strip()]
    unknown = sorted(set(requested_opponents) - set(opponents))
    if unknown:
        raise ValueError(f"Unknown opponents: {unknown}")
    if set(requested_roles) - {"seeker", "hider"}:
        raise ValueError(f"Roles must be seeker,hider; got {requested_roles}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = OpenAIChatGenerator(
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
    )
    tasks = [
        (role, opponent, repeat_idx)
        for role in requested_roles
        for opponent in requested_opponents
        for repeat_idx in range(args.n_repeats_per_cell)
    ]
    print(
        "[online-tom] "
        f"model={args.model} roles={requested_roles} opponents={requested_opponents} "
        f"matches={len(tasks)} rounds={args.n_rounds} prompt={args.prompt_mode}"
    )

    def run(task: tuple[str, str, int]) -> pd.DataFrame:
        role, opponent, repeat_idx = task
        return run_single_match(
            generator=generator,
            model_name=args.model,
            opponent_name=opponent,
            opponent_fn=opponents[opponent],
            p_matrix=payoff_matrix(),
            n_rounds=args.n_rounds,
            repeat_idx=repeat_idx,
            enable_thinking=False,
            reasoning_effort="low",
            prompt_mode=args.prompt_mode,
            llm_role=role,
        )

    try:
        completed = []
        pending = []
        for task in tasks:
            role, opponent, repeat_idx = task
            path = checkpoint_path(output_dir, role, opponent, repeat_idx)
            if path.is_file():
                completed.append(pd.read_csv(path))
            else:
                pending.append(task)

        with ThreadPoolExecutor(max_workers=max(1, args.parallel_matches)) as executor:
            futures = {executor.submit(run, task): task for task in pending}
            for future in as_completed(futures):
                role, opponent, repeat_idx = futures[future]
                frame = future.result()
                write_checkpoint(frame, checkpoint_path(output_dir, role, opponent, repeat_idx))
                completed.append(frame)
                print(
                    f"[online-tom] done role={role} opponent={opponent} rep={repeat_idx} "
                    f"payoff={int(frame['llm_payoff'].sum())}"
                )

        result = pd.concat(completed, ignore_index=True)
        suffix = f"_{args.output_tag}" if args.output_tag else ""
        output_path = output_dir / f"candidate_vs_tom_online{suffix}.csv"
        result.to_csv(output_path, index=False)
        expected_rows = len(tasks) * args.n_rounds
        if len(result) != expected_rows:
            raise RuntimeError(f"Expected {expected_rows} rollout rows, found {len(result)}")
        valid_targets = pd.to_numeric(result["opponent_action_1_probability"], errors="coerce").between(0, 1)
        if not valid_targets.all():
            raise RuntimeError(f"Missing probability targets in {(~valid_targets).sum()} rollout rows")
        print(f"[online-tom] saved={output_path} rows={len(result)}")
    finally:
        generator.close()


if __name__ == "__main__":
    main()
