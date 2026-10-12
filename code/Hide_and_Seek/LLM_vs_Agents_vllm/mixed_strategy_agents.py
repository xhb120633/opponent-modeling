from __future__ import annotations

import random
from typing import Any

import pandas as pd
import tomsup as ts
from tomsup.agent import TOM


def make_algorithm_agent(name: str):
    if name.startswith("ToM"):
        return TOM(level=int(name[3:]), save_history=True)
    if name == "QL":
        return ts.QL(learning_rate=0.5, b_temp=1, save_history=True)
    if name == "RB":
        return ts.RB(bias=0.7, save_history=True)
    if name == "WSLS":
        return ts.WSLS(save_history=True)
    raise ValueError(f"Unknown algorithm agent: {name}")


class StrategyMixtureAgent:
    """
    Online OOD opponent that hides a non-stationary policy behind the same
    two-action game interface. Component policies all observe the realized
    interaction history; on each round a scheduler chooses which component's
    proposed action is actually played.
    """

    def __init__(
        self,
        name: str,
        components: list[tuple[str, float]],
        switch_round: int | None = None,
        save_history: bool = False,
    ):
        if not components:
            raise ValueError("StrategyMixtureAgent requires at least one component")
        self.name = name
        self.components = [(agent_name, float(weight)) for agent_name, weight in components]
        self.switch_round = switch_round
        self.choice = None
        self.strategy = name
        self.history = pd.DataFrame() if save_history else None
        self._save_history = save_history
        self.round_idx = 0
        self.match_idx = 1
        self._agents = {agent_name: make_algorithm_agent(agent_name) for agent_name, _ in self.components}

    def reset(self, save_history=None):
        self.choice = None
        self.round_idx = 0
        if self.history is not None and not self.history.empty:
            self.match_idx += 1
        if save_history is None:
            save_history = self._save_history
        else:
            self._save_history = save_history
        self.history = pd.DataFrame() if save_history else None
        self._agents = {agent_name: make_algorithm_agent(agent_name) for agent_name, _ in self.components}

    @property
    def save_history(self):
        return self._save_history

    @property
    def local_history(self):
        if self.history is None:
            return []
        return self.history.to_dict("records")

    def _append_history_row(self, row: dict[str, Any]):
        if self.history is None:
            return
        row_df = pd.DataFrame([row])
        if self.history.empty:
            self.history = row_df
        else:
            self.history = pd.concat([self.history, row_df], ignore_index=True)

    def _update_previous_round_feedback(self, op_choice, p_matrix, agent):
        if self.history is None or self.history.empty:
            return
        if self.choice is None or op_choice is None or p_matrix is None:
            return
        last_idx = self.history.index[-1]
        if pd.isna(self.history.loc[last_idx, "payoff"]):
            payoff = p_matrix.payoff(self.choice, op_choice, agent)
            self.history.loc[last_idx, "op_choice"] = op_choice
            self.history.loc[last_idx, "payoff"] = payoff

    def _select_component(self) -> str:
        if self.switch_round is not None:
            if len(self.components) != 2:
                raise ValueError("transition mixture expects exactly two components")
            return self.components[0][0] if self.round_idx <= self.switch_round else self.components[1][0]
        total = sum(max(0.0, weight) for _, weight in self.components)
        if total <= 0:
            raise ValueError(f"Mixture weights must sum positive: {self.components}")
        threshold = random.random() * total
        cumulative = 0.0
        for agent_name, weight in self.components:
            cumulative += max(0.0, weight)
            if threshold <= cumulative:
                return agent_name
        return self.components[-1][0]

    def compete(self, p_matrix=None, agent=0, op_choice=None):
        self.round_idx += 1
        self._update_previous_round_feedback(op_choice=op_choice, p_matrix=p_matrix, agent=agent)

        candidate_choices = {}
        for agent_name, component in self._agents.items():
            candidate_choices[agent_name] = component.compete(
                p_matrix=p_matrix,
                agent=agent,
                op_choice=op_choice,
            )

        active = self._select_component()
        action = int(candidate_choices[active])
        self.choice = action

        # Align every component to the realized action history before the next
        # feedback update; unplayed candidate actions remain diagnostic only.
        for component in self._agents.values():
            component.choice = action
            if getattr(component, "history", None) is not None and not component.history.empty:
                component.history.loc[component.history.index[-1], "choice"] = action

        self._append_history_row(
            {
                "round": self.round_idx,
                "choice": action,
                "op_choice": None,
                "payoff": None,
                "active_strategy": active,
                "candidate_choices": dict(candidate_choices),
                "components": dict(self.components),
                "switch_round": self.switch_round,
                "strategy": self.strategy,
                "match_idx": self.match_idx,
            }
        )
        return action
