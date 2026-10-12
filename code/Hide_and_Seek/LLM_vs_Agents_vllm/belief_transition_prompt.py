"""Shared prompt contract for an LLM-native recurrent belief transition."""

from __future__ import annotations

import math


PROMPT_VERSION = "llm_native_belief_transition_v1"


def build_belief_transition_prompt(
    *,
    role: str,
    decision_round: int,
    previous_belief_p1: float,
    previous_self_action: int | None = None,
    previous_opponent_action: int | None = None,
    previous_payoff: int | None = None,
) -> str:
    if role not in {"hider", "seeker"}:
        raise ValueError(f"Invalid role {role!r}")
    if int(decision_round) < 1:
        raise ValueError("decision_round must be positive")
    probability = float(previous_belief_p1)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("previous_belief_p1 must lie in [0, 1]")
    observation = (previous_self_action, previous_opponent_action, previous_payoff)
    if any(value is None for value in observation) and not all(
        value is None for value in observation
    ):
        raise ValueError("Previous observation must be completely present or absent")
    if all(value is None for value in observation):
        observation_text = "none (first decision)"
    else:
        self_action, opponent_action, payoff = map(int, observation)
        if self_action not in {0, 1} or opponent_action not in {0, 1}:
            raise ValueError("Previous actions must be binary")
        observation_text = (
            f"self_action={self_action}; opponent_action={opponent_action}; "
            f"payoff={payoff}"
        )
    return (
        "Update your persistent belief about the opponent.\n"
        "Predict the probability that the opponent's NEXT action is 1.\n"
        "Return exactly one token: 0 or 1. Your probability distribution over "
        "those two tokens is the prediction; do not explain.\n"
        f"interface_version: {PROMPT_VERSION}\n"
        f"role: {role}\n"
        f"decision_round: {int(decision_round)}\n"
        f"previous_belief_opponent_action_1: {probability:.6f}\n"
        f"new_observation: {observation_text}"
    )
