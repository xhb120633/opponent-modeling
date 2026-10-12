from __future__ import annotations

import hashlib
from collections.abc import Sequence

PROMPT_VERSION_LOCKED_STRUCTURED = "locked_structured_v1"
PROMPT_VERSION_LOCKED_ACTION_ONLY = "locked_action_only_v1"
PROMPT_VERSION_LOCKED_PREDICTION_ONLY = "locked_prediction_only_v1"
PROMPT_VERSION_LOCKED_GRPO_REASONING = "locked_grpo_reasoning_gate_v1"
PROMPT_VERSION_LOCKED_PROBABILITY = "locked_probability_v2"
PROMPT_VERSION_LOCKED_RECURSIVE_PROBABILITY = "locked_recursive_probability_v3"
PROMPT_VERSION_LOCKED_PREDICTIVE_STATE = "locked_predictive_state_v1"

OPPONENT_CANDIDATES = ("ToM0", "ToM1", "ToM2", "QL", "RB", "WSLS")

STRATEGY_GLOSSARY = """
Candidate opponent_strategy values:
- RB: biased random policy with a stable location preference.
- QL: reinforcement-learning policy that updates action values from payoff.
- WSLS: win-stay/lose-shift policy; repeat after success and switch after failure.
- ToM0: level-0 theory-of-mind policy using observed behavior without recursive belief modeling.
- ToM1: level-1 recursive theory-of-mind policy that models a ToM0-like player.
- ToM2: level-2 recursive theory-of-mind policy that models a ToM1-like player.
""".strip()

LOCKED_STRUCTURED_SCHEMA = """
Return exactly one JSON object, with no extra text:
{
  "opponent_strategy": one of "ToM0", "ToM1", "ToM2", "QL", "RB", "WSLS",
  "opponent_next_action": 0 or 1,
  "my_best_action": 0 or 1,
  "final_action": 0 or 1
}
""".strip()

LOCKED_STRUCTURED_INSTRUCTION = f"""
Before choosing, infer the opponent's policy from the same game-state information and payoff history.

{STRATEGY_GLOSSARY}

Required fields:
- opponent_strategy: the candidate strategy that best explains the opponent.
- opponent_next_action: your prediction of the opponent's current-round action, where 0 = Tree and 1 = House.
- my_best_action: the action that maximizes your payoff under your role and the scoring rule.
- final_action: the action you will actually play; it should match my_best_action.

{LOCKED_STRUCTURED_SCHEMA}
""".strip()

LOCKED_ACTION_ONLY_SCHEMA = """
Return exactly one JSON object, with no extra text:
{
  "final_action": 0 or 1
}
""".strip()

LOCKED_ACTION_ONLY_INSTRUCTION = f"""
Choose the action that maximizes your payoff under your role and the scoring rule.

Required field:
- final_action: the action you will actually play, where 0 = Tree and 1 = House.

{LOCKED_ACTION_ONLY_SCHEMA}
""".strip()

LOCKED_PREDICTION_ONLY_SCHEMA = """
Return exactly one JSON object, with no extra text:
{
  "opponent_next_action": 0 or 1
}
""".strip()

LOCKED_PREDICTION_ONLY_INSTRUCTION = f"""
Predict the opponent's current-round action from the same game-state information and payoff history.

Required field:
- opponent_next_action: your prediction of the opponent's current-round action, where 0 = Tree and 1 = House.

Do not choose your own action. Your action will be computed from this prediction and your role by the game wrapper.

{LOCKED_PREDICTION_ONLY_SCHEMA}
""".strip()



LOCKED_GRPO_REASONING_SCHEMA = """
Return exactly one JSON object, with no extra text:
{
  "reasoning": "brief opponent model and decision rationale",
  "opponent_next_action": 0 or 1,
  "my_best_action": 0 or 1,
  "final_action": 0 or 1
}
""".strip()

LOCKED_GRPO_REASONING_INSTRUCTION = f"""
Before choosing, briefly reason about how the opponent is likely to choose in this round.

The reasoning field is for your own concise opponent model. The scored fields are:
- opponent_next_action: your prediction of the opponent's current-round action, where 0 = Tree and 1 = House.
- my_best_action: the action that maximizes your payoff under your role and the scoring rule.
- final_action: the action you will actually play; it should match my_best_action.

{LOCKED_GRPO_REASONING_SCHEMA}
""".strip()
LOCKED_PROBABILITY_SCHEMA = """
Return exactly one JSON object, with no extra text:
{
  "opponent_strategy": one of "ToM0", "ToM1", "ToM2", "QL", "RB", "WSLS",
  "opponent_action_1_probability": a number from 0.0 to 1.0,
  "opponent_next_action": 0 or 1,
  "my_best_action": 0 or 1,
  "final_action": 0 or 1
}
""".strip()

LOCKED_PROBABILITY_INSTRUCTION = f"""
Before choosing, infer the opponent's policy from the same game-state information and payoff history.

{STRATEGY_GLOSSARY}

Required fields:
- opponent_strategy: the candidate strategy that best explains the opponent.
- opponent_action_1_probability: your estimated probability that the opponent chooses 1 = House this round.
- opponent_next_action: 1 when that probability is greater than 0.5, otherwise 0.
- my_best_action: the best response to your predicted opponent action under your role and the scoring rule.
- final_action: the action you will actually play; it should match my_best_action.

{LOCKED_PROBABILITY_SCHEMA}
""".strip()

LOCKED_RECURSIVE_PROBABILITY_SCHEMA = """
Return exactly one JSON object, with no extra text:
{
  "opponent_strategy": one of "ToM0", "ToM1", "ToM2", "QL", "RB", "WSLS",
  "opponent_belief_my_action_1_probability": a number from 0.0 to 1.0,
  "opponent_action_1_probability": a number from 0.0 to 1.0,
  "opponent_next_action": 0 or 1,
  "my_best_action": 0 or 1,
  "final_action": 0 or 1
}
""".strip()

LOCKED_RECURSIVE_PROBABILITY_INSTRUCTION = f"""
Before choosing, update a recursive model of the opponent from the complete match history.

{STRATEGY_GLOSSARY}

Reason in this order:
1. Estimate what the opponent currently believes about your action.
2. Estimate how the opponent will act given that belief and its role.
3. Choose your role-conditioned best response.

Required fields:
- opponent_strategy: the candidate strategy that best explains the opponent.
- opponent_belief_my_action_1_probability: your estimate of the probability the opponent assigns to you choosing 1 = House this round.
- opponent_action_1_probability: your estimate of the probability that the opponent chooses 1 = House this round.
- opponent_next_action: 1 when opponent_action_1_probability is greater than 0.5, otherwise 0.
- my_best_action: the best response to your predicted opponent action under your role and the scoring rule.
- final_action: the action you will actually play; it should match my_best_action.

Previous belief estimates shown in the history are your own earlier estimates, not revealed ground truth. Update them using subsequent observed actions and payoffs.

{LOCKED_RECURSIVE_PROBABILITY_SCHEMA}
""".strip()

LOCKED_PREDICTIVE_STATE_SCHEMA = """
Return exactly one JSON object, with no extra text:
{
  "opponent_belief_my_action_bin": an integer from 0 to 10,
  "opponent_action_bin": an integer from 0 to 10,
  "opponent_next_action": 0 or 1,
  "my_best_action": 0 or 1,
  "final_action": 0 or 1
}
""".strip()

LOCKED_PREDICTIVE_STATE_INSTRUCTION = f"""
Before choosing, update a predictive model of this opponent from the complete match history.

Reason in this order:
1. Estimate what the opponent currently expects you to do.
2. Predict the opponent's current action from that estimate and the observed history.
3. Choose your role-conditioned best response.

Use integer probability bins: 0 means 0%, 5 means 50%, and 10 means 100%.

Required fields:
- opponent_belief_my_action_bin: the probability bin for the opponent expecting you to choose 1 = House this round.
- opponent_action_bin: the probability bin for the opponent choosing 1 = House this round.
- opponent_next_action: 1 when opponent_action_bin is 6 through 10, otherwise 0.
- my_best_action: the best response to opponent_next_action under your role and the scoring rule.
- final_action: the action you will actually play; it should match my_best_action.

Do not name or classify the opponent's algorithm. The history contains only observable actions and payoffs; it does not reveal the opponent's internal state.

{LOCKED_PREDICTIVE_STATE_SCHEMA}
""".strip()

def prompt_hash(prompt_version: str = PROMPT_VERSION_LOCKED_STRUCTURED) -> str:
    if prompt_version == PROMPT_VERSION_LOCKED_PREDICTIVE_STATE:
        payload = "\n\n".join(
            [
                prompt_version,
                LOCKED_PREDICTIVE_STATE_SCHEMA,
                LOCKED_PREDICTIVE_STATE_INSTRUCTION,
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    if prompt_version == PROMPT_VERSION_LOCKED_ACTION_ONLY:
        payload = "\n\n".join(
            [
                prompt_version,
                LOCKED_ACTION_ONLY_SCHEMA,
                LOCKED_ACTION_ONLY_INSTRUCTION,
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    if prompt_version == PROMPT_VERSION_LOCKED_PREDICTION_ONLY:
        payload = "\n\n".join(
            [
                prompt_version,
                LOCKED_PREDICTION_ONLY_SCHEMA,
                LOCKED_PREDICTION_ONLY_INSTRUCTION,
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    if prompt_version == PROMPT_VERSION_LOCKED_GRPO_REASONING:
        payload = "\n\n".join(
            [
                prompt_version,
                LOCKED_GRPO_REASONING_SCHEMA,
                LOCKED_GRPO_REASONING_INSTRUCTION,
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    if prompt_version == PROMPT_VERSION_LOCKED_RECURSIVE_PROBABILITY:
        payload = "\n\n".join(
            [
                prompt_version,
                STRATEGY_GLOSSARY,
                LOCKED_RECURSIVE_PROBABILITY_SCHEMA,
                LOCKED_RECURSIVE_PROBABILITY_INSTRUCTION,
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    if prompt_version == PROMPT_VERSION_LOCKED_PROBABILITY:
        payload = "\n\n".join(
            [prompt_version, STRATEGY_GLOSSARY, LOCKED_PROBABILITY_SCHEMA, LOCKED_PROBABILITY_INSTRUCTION]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    payload = "\n\n".join(
        [
            prompt_version,
            STRATEGY_GLOSSARY,
            LOCKED_STRUCTURED_SCHEMA,
            LOCKED_STRUCTURED_INSTRUCTION,
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def build_locked_structured_prompt(state_block: str) -> str:
    return f"{state_block.rstrip()}\n\n{LOCKED_STRUCTURED_INSTRUCTION}".strip()


def build_locked_action_only_prompt(state_block: str) -> str:
    return f"{state_block.rstrip()}\n\n{LOCKED_ACTION_ONLY_INSTRUCTION}".strip()


def build_locked_prediction_only_prompt(state_block: str) -> str:
    return f"{state_block.rstrip()}\n\n{LOCKED_PREDICTION_ONLY_INSTRUCTION}".strip()

def build_locked_grpo_reasoning_prompt(state_block: str) -> str:
    return f"{state_block.rstrip()}\n\n{LOCKED_GRPO_REASONING_INSTRUCTION}".strip()




def build_locked_recursive_probability_prompt(state_block: str) -> str:
    return f"{state_block.rstrip()}\n\n{LOCKED_RECURSIVE_PROBABILITY_INSTRUCTION}".strip()


def build_locked_predictive_state_prompt(state_block: str) -> str:
    return f"{state_block.rstrip()}\n\n{LOCKED_PREDICTIVE_STATE_INSTRUCTION}".strip()


def build_observable_state_block(
    *,
    role: str,
    self_actions: Sequence[int],
    opponent_actions: Sequence[int],
    payoffs: Sequence[int],
) -> str:
    """Render the locked state using only information observable before a decision."""
    if role not in {"seeker", "hider"}:
        raise ValueError(f"Unknown role: {role}")
    if not (len(self_actions) == len(opponent_actions) == len(payoffs)):
        raise ValueError("Observable history columns must have equal length.")

    if role == "hider":
        role_lines = "You are the HIDER.\nThe opponent is the SEEKER."
        goal_lines = (
            "Your goal is to maximize your total score across rounds by predicting where "
            "the opponent will seek and hiding elsewhere.\n"
            "Use the outcomes of previous rounds to improve your future hiding choices."
        )
        scoring_lines = (
            "- If the opponent guesses your hiding location, you lose (-1).\n"
            "- If the opponent guesses incorrectly, you win (+1)."
        )
        feedback_line = (
            "Your payoff tells you whether your previous hiding choice was found or avoided the seeker."
        )
    else:
        role_lines = "You are the SEEKER.\nThe opponent is the HIDER."
        goal_lines = (
            "Your goal is to maximize your total score across rounds by predicting where "
            "the opponent will hide.\n"
            "Use the outcomes of previous rounds to improve your future guesses."
        )
        scoring_lines = (
            "- If you guess correctly, you win (+1).\n"
            "- If you guess incorrectly, you lose (-1)."
        )
        feedback_line = "Your payoff tells you whether your previous guess was correct or incorrect."

    if self_actions:
        history_text = "\n".join(
            f"Round {round_idx}: you chose {int(self_action)}, opponent chose "
            f"{int(opponent_action)}, your payoff was {int(payoff)}"
            for round_idx, (self_action, opponent_action, payoff) in enumerate(
                zip(self_actions, opponent_actions, payoffs), start=1
            )
        )
    else:
        history_text = "No completed rounds yet."

    completed_rounds = len(self_actions)
    decision_round = completed_rounds + 1
    known_payoffs = sum(int(value) for value in payoffs)
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

def build_locked_probability_prompt(state_block: str) -> str:
    return f"{state_block.rstrip()}\n\n{LOCKED_PROBABILITY_INSTRUCTION}".strip()
