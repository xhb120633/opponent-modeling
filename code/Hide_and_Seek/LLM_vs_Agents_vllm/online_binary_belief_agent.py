"""Frozen belief-only agent for genuine closed-loop hide-and-seek matches."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
import pandas as pd

from evaluate_predictive_action_logits_vllm import build_probe_prompt
from online_history_summary import deterministic_summary
from prompt_templates import build_locked_predictive_state_prompt, build_observable_state_block


class BinaryProbabilityGenerator(Protocol):
    def predict_one(self, prompt: str) -> tuple[float, dict]: ...


def recover_binary_probability_from_complement(
    p1: float | None, diagnostics: dict, n_allowed_token_ids: int
) -> tuple[float | None, dict]:
    """Recover a pruned binary logprob when the allowed set is exactly {0, 1}."""
    out = dict(diagnostics)
    out["recovered_from_complement"] = 0
    if p1 is not None or n_allowed_token_ids != 2:
        return p1, out
    found0 = int(out.get("found_class0", 0))
    found1 = int(out.get("found_class1", 0))
    mass = float(out.get("prob_mass", float("nan")))
    if not math.isfinite(mass) or mass < 0.0 or mass > 1.0 + 1e-6:
        return None, out
    mass = min(mass, 1.0)
    if (found0, found1) == (1, 0):
        out["recovered_from_complement"] = 1
        return 1.0 - mass, out
    if (found0, found1) == (0, 1):
        out["recovered_from_complement"] = 1
        return mass, out
    return None, out


class OnlineBinaryBeliefAgent:
    """Use model logits only for belief; apply the role-specific best response externally."""

    def __init__(
        self,
        model_name: str,
        generator: BinaryProbabilityGenerator,
        role: str,
        prompt_mode: str = "raw",
        history_ablation: str = "none",
        ablation_seed: int = 20260824,
    ):
        if role not in {"seeker", "hider"}:
            raise ValueError(f"role must be seeker|hider, got {role!r}")
        if prompt_mode not in {"raw", "summary"}:
            raise ValueError(f"prompt_mode must be raw|summary, got {prompt_mode!r}")
        if history_ablation not in {"none", "shuffle", "delete"}:
            raise ValueError(
                f"history_ablation must be none|shuffle|delete, got {history_ablation!r}"
            )
        self.model_name = model_name
        self.generator = generator
        self.role = role
        self.prompt_mode = prompt_mode
        self.history_ablation = history_ablation
        self.ablation_seed = int(ablation_seed)
        self.strategy = (
            f"OnlineBinaryBelief_{model_name}_{role}_{prompt_mode}_{history_ablation}"
        )
        self.choice = None
        self.round_idx = 0
        self._save_history = True
        self.history = pd.DataFrame()

    @property
    def save_history(self):
        return self._save_history

    def reset(self, save_history=None):
        self.choice = None
        self.round_idx = 0
        if save_history is not None:
            self._save_history = bool(save_history)
        self.history = pd.DataFrame() if self._save_history else None

    def _update_previous_feedback(self, op_choice, p_matrix, agent):
        if self.history is None or self.history.empty or op_choice is None:
            return
        last_idx = self.history.index[-1]
        if pd.isna(self.history.loc[last_idx, "payoff"]):
            self.history.loc[last_idx, "op_choice"] = int(op_choice)
            self.history.loc[last_idx, "payoff"] = int(
                p_matrix.payoff(self.choice, op_choice, agent)
            )

    def _completed_history(self) -> tuple[list[int], list[int], list[int]]:
        if self.history is None or self.history.empty:
            return [], [], []
        complete = self.history.dropna(subset=["choice", "op_choice", "payoff"])
        return (
            complete["choice"].astype(int).tolist(),
            complete["op_choice"].astype(int).tolist(),
            complete["payoff"].astype(int).tolist(),
        )

    def build_prompt(self) -> str:
        self_actions, opponent_actions, payoffs = self._completed_history()
        if self.history_ablation == "delete":
            self_actions, opponent_actions, payoffs = [], [], []
        elif self.history_ablation == "shuffle" and len(self_actions) > 1:
            rng = np.random.default_rng(self.ablation_seed + 1009 * len(self_actions))
            order = rng.permutation(len(self_actions))
            self_actions = [self_actions[index] for index in order]
            opponent_actions = [opponent_actions[index] for index in order]
            payoffs = [payoffs[index] for index in order]
        state = build_observable_state_block(
            role=self.role,
            self_actions=self_actions,
            opponent_actions=opponent_actions,
            payoffs=payoffs,
        )
        if self.prompt_mode == "summary":
            state = f"{state}\n\n{deterministic_summary(self_actions, opponent_actions, payoffs)}"
        return build_probe_prompt(build_locked_predictive_state_prompt(state))

    def compete(self, p_matrix=None, agent=0, op_choice=None):
        self.round_idx += 1
        self._update_previous_feedback(op_choice, p_matrix, agent)
        prompt = self.build_prompt()
        p1, diagnostics = self.generator.predict_one(prompt)
        predicted_action = int(float(p1) >= 0.5)
        action = predicted_action if self.role == "seeker" else 1 - predicted_action
        self.choice = int(action)
        if self.history is not None:
            row = pd.DataFrame(
                [{
                    "round": self.round_idx,
                    "choice": self.choice,
                    "op_choice": None,
                    "payoff": None,
                    "predicted_opponent_action_1_probability": float(p1),
                    "predicted_opponent_action": predicted_action,
                    "binary_prob_mass": diagnostics.get("prob_mass"),
                    "found_class0": diagnostics.get("found_class0"),
                    "found_class1": diagnostics.get("found_class1"),
                    "recovered_from_complement": diagnostics.get("recovered_from_complement", 0),
                    "prompt": prompt,
                    "model_name": self.model_name,
                    "strategy": self.strategy,
                    "prompt_mode": self.prompt_mode,
                    "history_ablation": self.history_ablation,
                }]
            )
            self.history = row if self.history.empty else pd.concat([self.history, row], ignore_index=True)
        return self.choice
