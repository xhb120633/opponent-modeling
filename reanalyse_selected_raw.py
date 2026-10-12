"""Recalculate selected Figure 3/4 points from released row-level records.

This checks statistics and packaged input hashes. It does not rerun models or
independently validate simulator targets against excluded original histories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PANELS = ("3a", "3b", "3d", "3e", "3f", "4d", "4e")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def plotted_values():
    with (ROOT / "data/plotted_values.csv").open(newline="", encoding="utf-8") as stream:
        return {(row["figure"], row["panel"], row["label"]): float(row["value"])
                for row in csv.DictReader(stream)}


def check(values, figure: str, panel: str, label: str, calculated: float) -> None:
    published = values[(figure, panel, label)]
    if abs(calculated - published) > 1e-12:
        raise AssertionError((figure, panel, label, calculated, published))
    print(f"Figure {figure}{panel} {label}: {calculated:.8f} PASS")


def t04(selected: set[str], values) -> None:
    root = ROOT / "raw/figure3_t04"
    audit = read_json(root / "audit_metrics.json")
    assert audit["audit_passed"] is True
    for stage in ("action", "belief"):
        path = root / stage / "responses.jsonl"
        assert digest(path) == audit["stages"][stage]["responses_sha256"]
        assert len(read_jsonl(path)) == 240
    for panel, stage, labels in (
        ("a", "action", ("original_qwen", "original_parent", "constant", "frequency")),
        ("b", "belief", ("original_qwen", "original_parent", "constant", "frequency")),
        ("d", "belief", ("original_qwen", "base_selected", "original_parent",
                          "parent_selected", "constant", "frequency")),
    ):
        if "3" + panel not in selected:
            continue
        for label in labels:
            rows = audit["stages"][stage]["rows"][label]
            assert len(rows) == 60 and all(row.get("valid", True) for row in rows)
            mse = sum((row["p"] - row["target"]) ** 2 for row in rows) / 60
            assert abs(mse - audit["stages"][stage]["metrics"][label]["mse"]) < 1e-12
            check(values, "3", panel, label, mse)


def r5(selected: set[str], values) -> None:
    root = ROOT / "raw/figure3_r5"
    paths = {
        "original_A500": root / "known/initial_A_fresh.jsonl",
        "known": root / "known/eval_fresh.jsonl",
        "mixed": root / "mixed/eval_fresh.jsonl",
    }
    rows = {arm: read_jsonl(path) for arm, path in paths.items()}
    for arm, records in rows.items():
        assert len(records) == 1200 and len({(r["information"], r["match_id"], r["round"]) for r in records}) == 1200, arm
    audit = read_json(root / "audit_20260918.json")
    assert audit["status"] == "independently_audited"

    def loss(row):
        return (row["p"] - row["q"]) ** 2 if row["valid"] else 1.0

    if "3e" in selected:
        for arm in paths:
            for information in ("known", "unknown"):
                subset = [r for r in rows[arm] if r["information"] == information and r["round"] > 1]
                assert len(subset) == 540
                mse = sum(map(loss, subset)) / 540
                expected = audit["metrics"][arm]["fresh"][information]["all"]["mse"]
                assert abs(mse - expected) < 1e-12
                check(values, "3", "e", f"{arm}_{information}", mse)
        reference = [r for r in rows["original_A500"] if r["information"] == "unknown" and r["round"] > 1]
        check(values, "3", "e", "constant", sum((0.5 - r["q"]) ** 2 for r in reference) / 540)

    if "3f" in selected:
        known = {(r["match_id"], r["round"]): r for r in rows["known"] if r["information"] == "unknown" and r["round"] > 1}
        mixed = {(r["match_id"], r["round"]): r for r in rows["mixed"] if r["information"] == "unknown" and r["round"] > 1}
        assert known.keys() == mixed.keys() and len(known) == 540
        for role in (0, 1):
            for level in (0, 1, 2):
                pairs = [(known[key], mixed[key]) for key in known if known[key]["role"] == role and known[key]["level"] == level]
                assert len(pairs) == 90
                assert all(a["q"] == b["q"] for a, b in pairs)
                prefix = f"A{role}_ToM{level}"
                check(values, "3", "f", prefix + "_mixed_minus_known",
                      sum(loss(b) - loss(a) for a, b in pairs) / 90)
                check(values, "3", "f", prefix + "_mixed_minus_constant",
                      sum(loss(b) - (0.5 - b["q"]) ** 2 for _, b in pairs) / 90)


def c03(selected: set[str], values) -> None:
    root = ROOT / "raw/figure4_intervention"
    audit = read_json(root / "audit_results.json")
    assert audit["passed"] is True
    for name, expected in audit["output_hashes"].items():
        assert digest(root / name) == expected, name
    grouped = defaultdict(list)
    for row in read_json(root / "scored_rows.json"):
        grouped[row["condition"]].append(row)
    for condition in ("b_self", "b_true", "b_shuffled", "q_self", "q_true"):
        assert len(grouped[condition]) == 54 and all(row["valid"] for row in grouped[condition])

    def average(condition: str, metric: str):
        return sum(row[metric] for row in grouped[condition]) / 54

    if "4d" in selected:
        for label, condition in (("self", "b_self"), ("true", "b_true"), ("shuffled", "b_shuffled")):
            mse = average(condition, "mse")
            assert abs(mse - audit["groups"][condition]["all"]["penalized_mse"]) < 1e-12
            check(values, "4", "d", label, mse)
    if "4e" in selected:
        for label, left, right in (("b_true-b_self", "b_true", "b_self"),
                                   ("b_true-b_shuffled", "b_true", "b_shuffled"),
                                   ("q_true-q_self", "q_true", "q_self")):
            check(values, "4", "e", label, average(left, "regret") - average(right, "regret"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("all",) + PANELS, default="all")
    args = parser.parse_args()
    selected = set(PANELS) if args.panel == "all" else {args.panel}
    values = plotted_values()
    if selected & {"3a", "3b", "3d"}:
        t04(selected, values)
    if selected & {"3e", "3f"}:
        r5(selected, values)
    if selected & {"4d", "4e"}:
        c03(selected, values)
    print("PASS: selected row-level recalculation matches the published plotted values.")
    print("This does not rerun inference or independently re-audit the original simulator targets.")


if __name__ == "__main__":
    main()
