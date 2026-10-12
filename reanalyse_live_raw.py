"""Recalculate audited Figure 2B/2C live-play summaries from released rounds.

This reads only the packaged, compressed per-round data. It does not replay
model inference or independently regenerate the opponent's internal states.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def csv_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".gz":
        text = gzip.decompress(path.read_bytes()).decode("utf-8-sig")
    else:
        text = path.read_text(encoding="utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def figure2a() -> None:
    table = csv_rows(ROOT / "raw" / "figure2a_qwen25_tom2_match_scores.csv")
    plotted = {
        row["label"]: float(row["value"])
        for row in csv_rows(ROOT / "data" / "plotted_values.csv")
        if row["figure"] == "2" and row["panel"] == "a"
    }
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in table:
        groups[row["plot_label"]].append(row)
    assert len(table) == 280 and set(groups) == set(plotted)
    for label, group in sorted(groups.items()):
        assert sorted(int(row["rep"]) for row in group) == list(range(20))
        mean = sum(float(row["final_payoff"]) for row in group) / 20
        assert abs(mean - plotted[label]) < 1e-9
        print(f"Figure 2A {label}: 20 matches; mean payoff {mean:.3f}")


def figure2b() -> None:
    root = ROOT / "raw" / "figure2b_tom2_hider"
    audit_root = ROOT / "results" / "audited_live_cells"
    for label in ("oss_low", "oss_medium", "oss_high", "gemma_thinking"):
        audit = json.loads((audit_root / f"{label}_tom2.json").read_text(encoding="utf-8"))
        table = csv_rows(root / f"{label}.csv.gz")
        scores: dict[int, float] = defaultdict(float)
        correct = 0
        for row in table:
            payoff = float(row["llm_payoff"])
            scores[int(row["repeat_idx"])] += payoff
            correct += payoff > 0
        assert len(table) == 1000 and sorted(scores) == list(range(20))
        assert [scores[k] for k in range(20)] == audit["match_scores"]
        mean = sum(scores.values()) / 20
        assert abs(mean - audit["mean_final_payoff"]) < 1e-9
        print(f"Figure 2B {label}: {len(table)} rounds; "
              f"mean hider payoff {mean:.3f}; correct-action fraction {correct / 1000:.3f}")


def figure2c() -> None:
    for role in ("seeker", "hider"):
        root = ROOT / "raw" / f"figure2c_gpt54_{role}"
        audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
        summary = {
            (row["arm"], int(row["rep"])): row
            for row in csv_rows(root / "match_summary.csv")
        }
        assert audit["audit_passed"] and len(summary) == 40
        for arm in ("action_only", "structured"):
            payoffs = []
            accuracies = []
            for rep in range(20):
                table = csv_rows(root / "matches" / f"{arm}_{rep:02d}" / "rounds.csv.gz")
                assert len(table) == 50
                payoff = sum(float(row["llm_payoff"]) for row in table)
                accuracy = sum(float(row["llm_payoff"]) > 0 for row in table) / 50
                expected = summary[(arm, rep)]
                assert abs(payoff - float(expected["payoff"])) < 1e-9
                assert abs(accuracy - float(expected["correct_action_fraction"])) < 1e-9
                payoffs.append(payoff)
                accuracies.append(accuracy)
            mean = sum(payoffs) / 20
            fraction = sum(accuracies) / 20
            assert abs(mean - audit["metrics"][arm]["mean"]) < 1e-9
            assert abs(fraction - audit["metrics"][arm]["correct_action_fraction"]) < 1e-9
            print(f"Figure 2C {role} {arm}: 1000 rounds; "
                  f"mean payoff {mean:.3f}; correct-action fraction {fraction:.3f}")


if __name__ == "__main__":
    figure2a()
    figure2b()
    figure2c()
