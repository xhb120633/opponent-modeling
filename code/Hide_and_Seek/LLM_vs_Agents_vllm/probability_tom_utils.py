from __future__ import annotations

from typing import Any

import numpy as np


def probability_01(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        return None
    return probability


def binary_action(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        action = int(value)
    except (TypeError, ValueError):
        return None
    return action if action in (0, 1) and float(value) == action else None


def predicted_action(probability: float) -> int:
    return int(probability > 0.5)


def role_best_response(opponent_action: int, role: str) -> int:
    if role == "seeker":
        return opponent_action
    if role == "hider":
        return 1 - opponent_action
    raise ValueError(f"Unknown role: {role!r}")


def extract_tom_action_probability(internal_states: Any) -> float | None:
    try:
        value = np.asarray(internal_states["own_states"]["p_self"]).reshape(-1)[0]
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return probability_01(value)


def extract_tom_opponent_belief_probability(internal_states: Any) -> float | None:
    try:
        value = np.asarray(internal_states["own_states"]["p_op"]).reshape(-1)[0]
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return probability_01(value)


def recursive_probability_reward_parts(
    parsed: dict[str, Any] | None,
    target_belief_probability: Any,
    target_action_probability: Any,
    role: str,
) -> dict[str, Any]:
    parts: dict[str, Any] = {
        "valid_format": 0.0,
        "recursive_brier_skill": 0.0,
        "action_brier_skill": 0.0,
        "prediction_decision_consistency": 0.0,
        "best_response_consistency": 0.0,
        "final_action_consistency": 0.0,
        "total": -0.75,
        "predicted_belief_probability": None,
        "target_belief_probability": probability_01(target_belief_probability),
        "predicted_action_probability": None,
        "target_action_probability": probability_01(target_action_probability),
    }
    if (
        parsed is None
        or parts["target_belief_probability"] is None
        or parts["target_action_probability"] is None
    ):
        return parts

    recursive_q = probability_01(parsed.get("opponent_belief_my_action_1_probability"))
    action_q = probability_01(parsed.get("opponent_action_1_probability"))
    declared = binary_action(parsed.get("opponent_next_action"))
    best = binary_action(parsed.get("my_best_action"))
    final = binary_action(parsed.get("final_action"))
    if recursive_q is None or action_q is None or declared is None or best is None or final is None:
        return parts

    target_belief = float(parts["target_belief_probability"])
    target_action = float(parts["target_action_probability"])
    implied = predicted_action(action_q)
    expected_best = role_best_response(declared, role)
    parts["predicted_belief_probability"] = recursive_q
    parts["predicted_action_probability"] = action_q
    parts["valid_format"] = 0.05
    parts["recursive_brier_skill"] = 0.25 - (recursive_q - target_belief) ** 2
    parts["action_brier_skill"] = 0.25 - (action_q - target_action) ** 2
    if declared == implied:
        parts["prediction_decision_consistency"] = 0.05
    if best == expected_best:
        parts["best_response_consistency"] = 0.05
    if final == best:
        parts["final_action_consistency"] = 0.05
    parts["total"] = sum(
        float(parts[key])
        for key in (
            "valid_format",
            "recursive_brier_skill",
            "action_brier_skill",
            "prediction_decision_consistency",
            "best_response_consistency",
            "final_action_consistency",
        )
    )
    return parts

def recursive_probability_v4_reward_parts(
    parsed: dict[str, Any] | None,
    target_belief_probability: Any,
    target_action_probability: Any,
    role: str,
) -> dict[str, Any]:
    """Reward calibrated prediction and role-aware exploitation over a 0.5 policy.

    The recursive belief remains a weak auxiliary target. The opponent-action
    probability receives calibration and confidence-weighted direction rewards.
    Action reward is gated on a correct declared opponent prediction so a direct
    history-to-action shortcut cannot earn the positive exploitation bonus.
    """
    parts: dict[str, Any] = {
        "valid_format": 0.0,
        "recursive_calibration": 0.0,
        "action_calibration": 0.0,
        "action_direction_skill": 0.0,
        "declared_prediction_skill": 0.0,
        "gated_action_regret": 0.0,
        "prediction_decision_consistency": 0.0,
        "best_response_consistency": 0.0,
        "final_action_consistency": 0.0,
        "target_confidence": None,
        "total": -3.0,
        "predicted_belief_probability": None,
        "target_belief_probability": probability_01(target_belief_probability),
        "predicted_action_probability": None,
        "target_action_probability": probability_01(target_action_probability),
    }
    if (
        parsed is None
        or parts["target_belief_probability"] is None
        or parts["target_action_probability"] is None
    ):
        return parts

    recursive_q = probability_01(parsed.get("opponent_belief_my_action_1_probability"))
    action_q = probability_01(parsed.get("opponent_action_1_probability"))
    declared = binary_action(parsed.get("opponent_next_action"))
    best = binary_action(parsed.get("my_best_action"))
    final = binary_action(parsed.get("final_action"))
    if recursive_q is None or action_q is None or declared is None or best is None or final is None:
        return parts

    target_belief = float(parts["target_belief_probability"])
    target_action = float(parts["target_action_probability"])
    target_direction = predicted_action(target_action)
    implied = predicted_action(action_q)
    target_best = role_best_response(target_direction, role)
    declared_best = role_best_response(declared, role)
    confidence = 2.0 * abs(target_action - 0.5)

    if action_q > 0.5:
        predicted_side = 1
    elif action_q < 0.5:
        predicted_side = 0
    else:
        predicted_side = None

    parts["predicted_belief_probability"] = recursive_q
    parts["predicted_action_probability"] = action_q
    parts["target_confidence"] = confidence
    parts["valid_format"] = 0.02
    parts["recursive_calibration"] = -0.25 * (recursive_q - target_belief) ** 2
    parts["action_calibration"] = -(action_q - target_action) ** 2

    if confidence > 0.0 and predicted_side is not None:
        parts["action_direction_skill"] = 0.5 * confidence * (
            1.0 if predicted_side == target_direction else -1.0
        )
    if confidence > 0.0:
        parts["declared_prediction_skill"] = 0.25 * confidence * (
            1.0 if declared == target_direction else -1.0
        )
        parts["gated_action_regret"] = confidence * (
            1.0 if declared == target_direction and final == target_best else -1.0
        )

    if declared == implied:
        parts["prediction_decision_consistency"] = 0.02
    if best == declared_best:
        parts["best_response_consistency"] = 0.02
    if final == best:
        parts["final_action_consistency"] = 0.02
    parts["total"] = sum(
        float(parts[key])
        for key in (
            "valid_format",
            "recursive_calibration",
            "action_calibration",
            "action_direction_skill",
            "declared_prediction_skill",
            "gated_action_regret",
            "prediction_decision_consistency",
            "best_response_consistency",
            "final_action_consistency",
        )
    )
    return parts

def probability_reward_parts(parsed: dict[str, Any] | None, target_probability: Any, role: str) -> dict[str, Any]:
    parts: dict[str, Any] = {
        "valid_format": 0.0,
        "prediction_brier_skill": 0.0,
        "prediction_decision_consistency": 0.0,
        "best_response_consistency": 0.0,
        "final_action_consistency": 0.0,
        "total": -0.75,
        "predicted_probability": None,
        "target_probability": probability_01(target_probability),
    }
    if parsed is None or parts["target_probability"] is None:
        return parts

    q = probability_01(parsed.get("opponent_action_1_probability"))
    declared = binary_action(parsed.get("opponent_next_action"))
    best = binary_action(parsed.get("my_best_action"))
    final = binary_action(parsed.get("final_action"))
    if q is None or declared is None or best is None or final is None:
        return parts

    target = float(parts["target_probability"])
    implied = predicted_action(q)
    expected_best = role_best_response(declared, role)
    parts["predicted_probability"] = q
    parts["valid_format"] = 0.05
    parts["prediction_brier_skill"] = 0.25 - (q - target) ** 2
    if declared == implied:
        parts["prediction_decision_consistency"] = 0.05
    if best == expected_best:
        parts["best_response_consistency"] = 0.05
    if final == best:
        parts["final_action_consistency"] = 0.05
    parts["total"] = sum(
        float(parts[key])
        for key in (
            "valid_format",
            "prediction_brier_skill",
            "prediction_decision_consistency",
            "best_response_consistency",
            "final_action_consistency",
        )
    )
    return parts