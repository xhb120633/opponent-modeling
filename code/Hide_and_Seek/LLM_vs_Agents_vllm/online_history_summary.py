"""No-leakage summaries shared by training and genuine online evaluation."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np


ROUND_RE = re.compile(
    r"^Round\s+\d+:\s+you chose\s+([01]),\s+opponent chose\s+([01]),\s+"
    r"your payoff was\s+(-?\d+)\s*$",
    flags=re.MULTILINE,
)


def _rate(values: Sequence[int]) -> float:
    return float(np.mean(values)) if values else 0.5


def _stay_rate(values: Sequence[int]) -> float:
    if len(values) <= 1:
        return 0.5
    return float(np.mean(np.asarray(values[1:]) == np.asarray(values[:-1])))


def deterministic_summary(
    self_actions: Sequence[int], opponent_actions: Sequence[int], payoffs: Sequence[int]
) -> str:
    if not (len(self_actions) == len(opponent_actions) == len(payoffs)):
        raise ValueError("History columns must have equal lengths")
    recent = list(opponent_actions[-10:])
    pairs = list(zip(self_actions[:-1], opponent_actions[1:]))
    conditional = {}
    for action in (0, 1):
        following = [int(opponent) for previous_self, opponent in pairs if previous_self == action]
        conditional[action] = (len(following), _rate(following))
    recent_text = "".join(str(int(value)) for value in recent) if recent else "none"
    return (
        "Deterministic observable-history summary (computed only from completed rounds):\n"
        f"- completed rounds: {len(opponent_actions)}\n"
        f"- opponent action-1 rate, all rounds: {_rate(opponent_actions):.3f}\n"
        f"- opponent action-1 rate, last 10: {_rate(recent):.3f}\n"
        f"- opponent repeat rate, all transitions: {_stay_rate(opponent_actions):.3f}\n"
        f"- opponent repeat rate, last 10: {_stay_rate(recent):.3f}\n"
        f"- P(opponent next action was 1 after your previous action 0): {conditional[0][1]:.3f} "
        f"from {conditional[0][0]} observations\n"
        f"- P(opponent next action was 1 after your previous action 1): {conditional[1][1]:.3f} "
        f"from {conditional[1][0]} observations\n"
        f"- your mean payoff: {(float(np.mean(payoffs)) if payoffs else 0.0):.3f}\n"
        f"- opponent's most recent actions, oldest to newest: {recent_text}\n"
        "Treat this as a lossless aid for extracting simple behavioral statistics; "
        "use the full history for context."
    )


def parse_completed_history(prompt: str) -> tuple[list[int], list[int], list[int]]:
    rows = ROUND_RE.findall(prompt)
    return (
        [int(row[0]) for row in rows],
        [int(row[1]) for row in rows],
        [int(row[2]) for row in rows],
    )


def augment_probe_prompt(prompt: str) -> str:
    """Insert the deterministic summary immediately before the prediction request."""
    marker = "\n\nBefore choosing, predict the opponent's current action"
    if prompt.count(marker) != 1:
        raise ValueError("Expected exactly one prediction-request marker")
    if "Deterministic observable-history summary" in prompt:
        raise ValueError("Prompt already contains a deterministic summary")
    self_actions, opponent_actions, payoffs = parse_completed_history(prompt)
    summary = deterministic_summary(self_actions, opponent_actions, payoffs)
    return prompt.replace(marker, f"\n\n{summary}{marker}", 1)

