import pandas as pd
import random
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from model_output_parse import extract_structured_object, resolve_action_from_completion
from probability_tom_utils import probability_01
from prompt_templates import (
    PROMPT_VERSION_LOCKED_ACTION_ONLY,
    PROMPT_VERSION_LOCKED_PREDICTION_ONLY,
    PROMPT_VERSION_LOCKED_PROBABILITY,
    PROMPT_VERSION_LOCKED_PREDICTIVE_STATE,
    PROMPT_VERSION_LOCKED_RECURSIVE_PROBABILITY,
    PROMPT_VERSION_LOCKED_STRUCTURED,
    build_locked_action_only_prompt,
    build_locked_prediction_only_prompt,
    build_locked_probability_prompt,
    build_locked_predictive_state_prompt,
    build_observable_state_block,
    build_locked_recursive_probability_prompt,
    build_locked_structured_prompt,
    prompt_hash,
)
try:
    from vllm_batcher import DecodeOptions
except ModuleNotFoundError:
    @dataclass(frozen=True)
    class DecodeOptions:
        enable_thinking: Optional[bool] = None
        reasoning_effort: Optional[str] = None
        apply_chat_template: bool = True

if TYPE_CHECKING:
    from vllm_batcher import VLLMBatchedGenerator
else:
    VLLMBatchedGenerator = object


class VLLMAgent:
    """
    Prompt/content/decision logic intentionally mirrors the original ollama agent.
    Only backend inference path changes to vLLM batched generation.
    """

    def __init__(
        self,
        model_name: str,
        generator: VLLMBatchedGenerator,
        save_history: bool = False,
        enable_thinking: bool = False,
        reasoning_effort: str = "low",
        prompt_mode: str = "baseline",
        llm_role: str = "seeker",
    ):
        self.model_name = model_name
        self.generator = generator
        self.choice = None
        self.strategy = f"VLLMAgent_{model_name}"
        self.prompt_mode = prompt_mode
        if llm_role not in ("seeker", "hider"):
            raise ValueError(f"llm_role must be seeker|hider, got {llm_role!r}")
        self.llm_role = llm_role
        self.match_idx = 1
        self.decode_options = DecodeOptions(
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            apply_chat_template=True,
        )

        self.history = pd.DataFrame() if save_history else None
        self._save_history = save_history
        self.round_idx = 0

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

    @property
    def save_history(self):
        return self._save_history

    @property
    def local_history(self):
        if self.history is None:
            return []
        return self.history.to_dict("records")

    def _append_history_row(self, row: dict):
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

    def _format_history(self, n_rounds: int = None) -> str:
        if self.history is None or self.history.empty:
            return "No previous rounds."

        recent = self.history if n_rounds is None else self.history.tail(n_rounds)
        lines = []
        for _, row in recent.iterrows():
            op_choice_text = row["op_choice"]
            if pd.isna(op_choice_text):
                op_choice_text = "not available yet"

            payoff_text = row["payoff"]
            if pd.isna(payoff_text):
                payoff_text = "not available yet"

            if self.prompt_mode in (
                PROMPT_VERSION_LOCKED_RECURSIVE_PROBABILITY,
                PROMPT_VERSION_LOCKED_PREDICTIVE_STATE,
            ):
                belief = row.get("predicted_opponent_belief_my_action_1_probability")
                action_probability = row.get("predicted_opponent_action_1_probability")
                belief_text = "not available" if pd.isna(belief) else f"{float(belief):.4f}"
                action_probability_text = (
                    "not available"
                    if pd.isna(action_probability)
                    else f"{float(action_probability):.4f}"
                )
                lines.append(
                    f"Round {int(row['round'])}: "
                    f"your prior estimate P(opponent expects you choose 1) was {belief_text}; "
                    f"your prior estimate P(opponent chooses 1) was {action_probability_text}; "
                    f"you chose {int(row['choice'])}; "
                    f"opponent chose {op_choice_text}; "
                    f"your payoff was {payoff_text}"
                )
            else:
                lines.append(
                    f"Round {int(row['round'])}: "
                    f"you chose {int(row['choice'])}, "
                    f"opponent chose {op_choice_text}, "
                    f"your payoff was {payoff_text}"
                )

        return "\n".join(lines)

    def _build_state_block(self, op_choice=None, p_matrix=None, agent=0) -> str:
        if self.prompt_mode == PROMPT_VERSION_LOCKED_PREDICTIVE_STATE:
            history = self.history
            if history is None or history.empty:
                self_actions: list[int] = []
                opponent_actions: list[int] = []
                payoffs: list[int] = []
            else:
                self_actions = history["choice"].dropna().astype(int).tolist()
                opponent_actions = history["op_choice"].dropna().astype(int).tolist()
                payoffs = history["payoff"].dropna().astype(int).tolist()
            return build_observable_state_block(
                role=self.llm_role,
                self_actions=self_actions,
                opponent_actions=opponent_actions,
                payoffs=payoffs,
            )
        history_text = self._format_history(n_rounds=None)
        completed_rounds = 0 if self.history is None else len(self.history)
        known_payoffs = (
            0
            if self.history is None or self.history.empty
            else self.history["payoff"].dropna().astype(int).sum()
        )
        decision_round = self.round_idx

        if self.llm_role == "hider":
            role_lines = """
You are the HIDER.
The opponent is the SEEKER.
""".strip()
            goal_lines = """
Your goal is to maximize your total score across rounds by predicting where the opponent will seek and hiding elsewhere.
Use the outcomes of previous rounds to improve your future hiding choices.
""".strip()
            scoring_lines = """
- If the opponent guesses your hiding location, you lose (-1).
- If the opponent guesses incorrectly, you win (+1).
""".strip()
            feedback_line = "Your payoff tells you whether your previous hiding choice was found or avoided the seeker."
        else:
            role_lines = """
You are the SEEKER.
The opponent is the HIDER.
""".strip()
            goal_lines = """
Your goal is to maximize your total score across rounds by predicting where the opponent will hide.
Use the outcomes of previous rounds to improve your future guesses.
""".strip()
            scoring_lines = """
- If you guess correctly, you win (+1).
- If you guess incorrectly, you lose (-1).
""".strip()
            feedback_line = "Your payoff tells you whether your previous guess was correct or incorrect."

        return f"""
You are playing a repeated hide-and-seek game.

{role_lines}

This is a multi-round interactive game against the SAME opponent.
This match is persistent across rounds. Do not treat each round as a fresh one-shot task.
The opponent is another agent that can reason about your past behavior and adapt strategically.

{goal_lines}

Important:
- {feedback_line}
- Use payoff feedback to update your belief about the opponent.
- The opponent may also adapt to your behavior over time and may be modeling you.
- Therefore, do not assume the opponent is fixed. Try to infer how they are responding to you.
- The history below is the complete match history so far for this same opponent.

Actions:
0 = Tree
1 = House

Scoring:
{scoring_lines}

Current decision:
- You are choosing for round {decision_round}.
- Completed rounds against this same opponent: {completed_rounds}.
- Your cumulative known score so far: {known_payoffs}.

Complete match history:
{history_text}
""".strip()

    def _baseline_response_instruction(self) -> str:
        return """
Reply with ONLY one character:
0 or 1

Do not explain.
Do not output any other words.
""".strip()

    def _structured_response_instruction(self) -> str:
        if self.prompt_mode == "structured_strategy":
            pattern_field = '"opponent_strategy": one of "ToM0", "ToM1", "ToM2", "QL", "RB", "WSLS"'
            pattern_note = """
- First infer which candidate strategy best explains the opponent: ToM0, ToM1, ToM2, QL, RB, or WSLS.
""".rstrip()
        else:
            pattern_field = '"opponent_pattern": a short behavioral description'
            pattern_note = """
- Describe the opponent's behavioral pattern directly; do not force it into a fixed strategy label.
""".rstrip()
            if self.prompt_mode == "structured_behavior_brief":
                pattern_field = '"opponent_pattern": at most 8 words'
                pattern_note = """
- Briefly describe the opponent's behavioral pattern in at most 8 words.
- Do not include step-by-step reasoning, markdown, tables, or paragraphs.
""".rstrip()

        if self.prompt_mode in ("structured_rule", "structured_strategy"):
            if self.llm_role == "hider":
                rule_note = """
- You are the HIDER. If you predict opponent_next_action = 0, your best action is 1.
- You are the HIDER. If you predict opponent_next_action = 1, your best action is 0.
- Therefore final_action must differ from opponent_next_action.
""".rstrip()
            else:
                rule_note = """
- You are the SEEKER. If you predict opponent_next_action = 0, your best action is 0.
- You are the SEEKER. If you predict opponent_next_action = 1, your best action is 1.
- Therefore final_action must match opponent_next_action.
""".rstrip()
        else:
            rule_note = """
- Use your role and the scoring rule to choose the action that maximizes payoff.
""".rstrip()

        compact_note = ""
        if self.prompt_mode == "structured_behavior_brief":
            compact_note = """
The JSON must be one line and under 60 words total.
Every string value must be a short phrase.
""".strip()

        return f"""
Before choosing, make an explicit private prediction and response plan.
{pattern_note}
- Predict the opponent's next hiding action.
{rule_note}
{compact_note}

Return exactly one JSON object, with no extra text:
{{
  {pattern_field},
  "opponent_next_action": 0 or 1,
  "my_best_action": 0 or 1,
  "final_action": 0 or 1
}}
""".strip()

    def build_prompt(self, op_choice=None, p_matrix=None, agent=0) -> str:
        state = self._build_state_block(op_choice=op_choice, p_matrix=p_matrix, agent=agent)
        if self.prompt_mode == "baseline":
            instruction = self._baseline_response_instruction()
        elif self.prompt_mode == PROMPT_VERSION_LOCKED_STRUCTURED:
            return build_locked_structured_prompt(state)
        elif self.prompt_mode == PROMPT_VERSION_LOCKED_ACTION_ONLY:
            return build_locked_action_only_prompt(state)
        elif self.prompt_mode == PROMPT_VERSION_LOCKED_PREDICTION_ONLY:
            return build_locked_prediction_only_prompt(state)
        elif self.prompt_mode == PROMPT_VERSION_LOCKED_PROBABILITY:
            return build_locked_probability_prompt(state)
        elif self.prompt_mode == PROMPT_VERSION_LOCKED_RECURSIVE_PROBABILITY:
            return build_locked_recursive_probability_prompt(state)
        elif self.prompt_mode == PROMPT_VERSION_LOCKED_PREDICTIVE_STATE:
            return build_locked_predictive_state_prompt(state)
        elif self.prompt_mode in (
            "structured_behavior",
            "structured_behavior_brief",
            "structured_rule",
            "structured_strategy",
        ):
            instruction = self._structured_response_instruction()
        else:
            raise ValueError(f"Unknown prompt_mode: {self.prompt_mode!r}")
        return f"{state}\n\n{instruction}".strip()

    def compete(self, p_matrix=None, agent=0, op_choice=None):
        self.round_idx += 1

        self._update_previous_round_feedback(
            op_choice=op_choice,
            p_matrix=p_matrix,
            agent=agent,
        )
        prompt = self.build_prompt(op_choice=op_choice, p_matrix=p_matrix, agent=agent)
        raw_reply = self.generator.generate_one(prompt, decode_options=self.decode_options)
        prediction_only_role = (
            self.llm_role if self.prompt_mode == PROMPT_VERSION_LOCKED_PREDICTION_ONLY else None
        )
        action, parse_mode, action_parse_source, split = resolve_action_from_completion(
            raw_reply,
            self.model_name,
            prefer_structured=self.prompt_mode != "baseline",
            prediction_only_role=prediction_only_role,
        )
        if action is None:
            action = random.choice([0, 1])
            parse_mode = "fallback"
            action_parse_source = "fallback_random"

        structured = extract_structured_object(split.answer_text) or extract_structured_object(raw_reply)
        recursive_belief_probability = (
            probability_01(structured.get("opponent_belief_my_action_1_probability"))
            if structured is not None
            else None
        )
        opponent_action_probability = (
            probability_01(structured.get("opponent_action_1_probability"))
            if structured is not None
            else None
        )
        if structured is not None and self.prompt_mode == PROMPT_VERSION_LOCKED_PREDICTIVE_STATE:
            belief_bin = structured.get("opponent_belief_my_action_bin")
            action_bin = structured.get("opponent_action_bin")
            try:
                belief_bin = int(belief_bin)
                recursive_belief_probability = belief_bin / 10.0 if 0 <= belief_bin <= 10 else None
            except (TypeError, ValueError):
                recursive_belief_probability = None
            try:
                action_bin = int(action_bin)
                opponent_action_probability = action_bin / 10.0 if 0 <= action_bin <= 10 else None
            except (TypeError, ValueError):
                opponent_action_probability = None

        self.choice = action
        self._append_history_row(
            {
                "round": self.round_idx,
                "choice": action,
                "op_choice": None,
                "payoff": None,
                "predicted_opponent_belief_my_action_1_probability": recursive_belief_probability,
                "predicted_opponent_action_1_probability": opponent_action_probability,
                "raw_reply": raw_reply,
                "cot_text": split.cot_text,
                "final_text": split.answer_text,
                "split_parse_backend": split.parse_backend,
                "action_parse_source": action_parse_source,
                "parse_mode": parse_mode,
                "enable_thinking": bool(self.decode_options.enable_thinking),
                "prompt_mode": self.prompt_mode,
                "prompt": prompt,
                "prompt_version": self.prompt_mode,
                "prompt_hash": prompt_hash(self.prompt_mode) if self.prompt_mode in (PROMPT_VERSION_LOCKED_STRUCTURED, PROMPT_VERSION_LOCKED_ACTION_ONLY, PROMPT_VERSION_LOCKED_PREDICTION_ONLY, PROMPT_VERSION_LOCKED_PROBABILITY, PROMPT_VERSION_LOCKED_RECURSIVE_PROBABILITY, PROMPT_VERSION_LOCKED_PREDICTIVE_STATE) else "",
                "model_name": self.model_name,
                "strategy": self.strategy,
                "match_idx": self.match_idx,
            }
        )
        return action
