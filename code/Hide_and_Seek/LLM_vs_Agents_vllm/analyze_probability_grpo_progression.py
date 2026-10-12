#!/usr/bin/env python3
"""Summarize a flat directory of probability-GRPO progression evaluations."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from analyze_probability_grpo_eval import summarize_file


STEP_PATTERN = re.compile(r"prob_progress_s(?P<step>\d+)_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for csv_path in sorted(args.root.glob("*.csv")):
        match = STEP_PATTERN.search(csv_path.name)
        if match is None:
            continue
        step = int(match.group("step"))
        row = summarize_file(csv_path, f"step{step}")
        row["step"] = step
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["step", "opponent", "role"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
    print(summary.to_csv(index=False))


if __name__ == "__main__":
    main()
