"""Recalculate four published live-play cells from released per-match scores."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CASES = (
    ("oss_low_tom2.json", "gpt-oss-20b (effort=low) hider", 20260923),
    ("oss_medium_tom2.json", "gpt-oss-20b (effort=medium) hider", 20260925),
    ("oss_high_tom2.json", "gpt-oss-20b (effort=high) hider", 20260923),
    ("gemma_thinking_tom2.json", "gemma-4-31B-it (thinking) hider", 20260925),
)


def close(actual: float, expected: str, tolerance: float = 0.00051) -> None:
    assert expected != "", "A plotted value is missing"
    assert abs(actual - float(expected)) <= tolerance, (actual, expected)


def main() -> None:
    with (ROOT / "data" / "plotted_values.csv").open(newline="", encoding="utf-8") as stream:
        main_rows = list(csv.DictReader(stream))
    for filename, main_label, seed in CASES:
        record = json.loads((ROOT / "results" / "audited_live_cells" / filename).read_text(encoding="utf-8"))
        scores = np.asarray(record["match_scores"], dtype=int)
        assert record["status"] == "audited" and record["audit_passed"] is True
        assert len(scores) == record["n_matches"] == 20
        assert record["n_rows"] == record["unique_match_round_keys"] == 1000
        assert record["strict"] + record["fallback_random"] == 1000
        assert len(record["fallback_rows"]) == record["fallback_random"]
        payoff = float(scores.mean())
        rng = np.random.default_rng(seed)
        bootstrap = rng.choice(scores, size=(20000, 20), replace=True).mean(axis=1)
        low, high = (float(value) for value in np.quantile(bootstrap, [0.025, 0.975]))
        recorded_ci = record.get("ci95_match_bootstrap", record.get("ci95_match_bootstrap_payoff"))
        close(payoff, str(record["mean_final_payoff"]))
        close(low, str(recorded_ci[0]))
        close(high, str(recorded_ci[1]))

        # Each of 50 zero-sum rounds pays +1 for a correct hider action, -1 otherwise.
        correct_fraction = 0.5 + payoff / 100
        main = [row for row in main_rows if row["figure"] == "2" and row["panel"] == "b" and row["label"] == main_label]
        assert len(main) == 1, (filename, "Figure 2B row missing")
        close(correct_fraction, main[0]["value"])
        close(0.5 + low / 100, main[0]["ci_low"])
        close(0.5 + high / 100, main[0]["ci_high"])

        print(f"{filename}: payoff {payoff:.1f} [{low:.1f}, {high:.1f}], "
              f"correct action {correct_fraction:.3f}, strict {record['strict']}/1000")

    print("PASS: four audited 20-match cells reproduce their Figure 2B values.")
    print("This is a statistical reanalysis of released scores, not a rerun or semantic re-audit of model outputs.")


if __name__ == "__main__":
    main()
