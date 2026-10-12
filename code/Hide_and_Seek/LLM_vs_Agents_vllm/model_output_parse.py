"""
Split raw vLLM decoder text into chain-of-thought (where applicable) and answer text.

Supports:
- OpenAI gpt-oss (Harmony): <|channel|>analysis / final blocks
- Google Gemma 4 native thinking: <|channel|>thought ... then <|channel|> / <|message|>
- Qwen extended reasoning: <think>...</think>
- Legacy Qwen2: <思考>...</思考> (see _LEGACY_THINK)
"""

from __future__ import annotations

import re
import json
import ast
from dataclasses import dataclass

_HARMONY_FINAL = re.compile(
    r"<\|channel\|\>final(?:[^\n<]*)?<\|message\|\>([\s\S]*?)"
    r"(?:<\|return\|\>|<\|end\|\>|<\|start\|\>|<\|call\|\>|$)",
    re.IGNORECASE,
)
_HARMONY_ANALYSIS = re.compile(
    r"<\|channel\|\>analysis(?:[^\n<]*)?<\|message\|\>([\s\S]*?)"
    r"(?:<\|end\|\>|<\|start\|\>|$)",
    re.IGNORECASE,
)
# Qwen3 / Gemma extended thinking: <think>...</think> (split strings avoid editor mangling).
_RT_OPEN = "<" + "redacted" + "_" + "thinking" + ">"
_RT_CLOSE = "</" + "redacted" + "_" + "thinking" + ">"
_REDACTED_THINK = re.compile(
    re.escape(_RT_OPEN) + r"([\s\S]*?)" + re.escape(_RT_CLOSE),
    re.IGNORECASE | re.DOTALL,
)
# Opening tag may live in chat-template prefix (not in vLLM completion text).
_REDACTED_THINK_CLOSE = re.compile(
    re.escape(_RT_CLOSE) + r"\s*([\s\S]*)",
    re.IGNORECASE | re.DOTALL,
)
# Older Qwen2-style blocks: <思考>...</思考> (Unicode names, not the word "think").
_LT_OPEN = "<" + "\u601d\u8003" + ">"
_LT_CLOSE = "</" + "\u601d\u8003" + ">"
_LEGACY_THINK = re.compile(
    re.escape(_LT_OPEN) + r"([\s\S]*?)" + re.escape(_LT_CLOSE),
    re.IGNORECASE | re.DOTALL,
)
_GPT_OSS_BARE_ANALYSIS = re.compile(r"(?i)^analysis")
# Offline vLLM often emits bare Harmony (no <|channel|> tokens): analysis{CoT}assistantfinal{0|1}
_GPT_OSS_BARE_FINAL = re.compile(r"(?i)assistantfinal\s*([01])\s*$")
_GPT_OSS_BARE_FINAL_ANY = re.compile(r"(?i)assistantfinal\s*([01])")
# Truncated bare Harmony: model ran out of tokens mid-analysis but often states intent at the tail.
_HARMONY_TAIL_CHOICE = re.compile(
    r"(?i)(?:pick|choose|guess|select|answer(?: with| is)?|output|return|go with)\s+(?:is\s+)?([01])\s*[.!]?\s*$"
)
_HARMONY_LETS_CHOICE = re.compile(r"(?i)let'?s\s+(?:pick|choose|guess|go with)\s+([01])\s*[.!]?\s*$")
_HARMONY_EXPLICIT_TAIL_CHOICE = re.compile(
    r"(?i)(?:thus\s+)?(?:final\s+)?(?:answer|decision|choice|output)\s*(?:is|:)?\s*([01])\s*[.!]?\s*$"
)


def _is_gpt_oss(model_l: str) -> bool:
    return "openai/gpt-oss" in model_l or "/gpt-oss" in model_l


def _is_qwen_hybrid(model_l: str) -> bool:
    if not model_l.startswith("qwen/"):
        return False
    if "qwen2.5" in model_l or "qwen2-" in model_l or "/qwen2/" in model_l:
        return False
    return any(marker in model_l for marker in ("qwen3", "qwq"))


def _strip_redacted_open(text: str) -> str:
    t = text.strip()
    if t.lower().startswith(_RT_OPEN.lower()):
        return t[len(_RT_OPEN) :].lstrip()
    return t


def _parse_redacted_thinking(text: str) -> ParsedModelOutput | None:
    """
    Qwen3 hybrid thinking blocks. Completion often omits the opening tag (in template prefix).
    """
    m = _REDACTED_THINK.search(text)
    if m:
        return ParsedModelOutput(
            raw_text=text,
            cot_text=m.group(1).strip(),
            answer_text=_REDACTED_THINK.sub("", text, count=1).strip(),
            parse_backend="redacted_thinking",
        )

    cm = _REDACTED_THINK_CLOSE.search(text)
    if cm:
        cot = _strip_redacted_open(text[: cm.start()].strip())
        ans = cm.group(1).strip()
        backend = "redacted_thinking" if ans else "redacted_thinking_truncated"
        return ParsedModelOutput(
            raw_text=text,
            cot_text=cot,
            answer_text=ans,
            parse_backend=backend,
        )

    if text.strip().lower().startswith(_RT_OPEN.lower()):
        cot = _strip_redacted_open(text.strip())
        return ParsedModelOutput(
            raw_text=text,
            cot_text=cot,
            answer_text="",
            parse_backend="redacted_thinking_truncated",
        )

    return None


def _qwen_thinking_tail_action(cot_text: str) -> str:
    """When thinking ran out of tokens before </think>, read intent from tail."""
    if not cot_text:
        return ""
    tail = cot_text[-500:].strip()
    last_line = tail.rsplit("\n", 1)[-1].strip()
    if last_line in ("0", "1"):
        return last_line
    for pat in (_HARMONY_EXPLICIT_TAIL_CHOICE, _HARMONY_LETS_CHOICE, _HARMONY_TAIL_CHOICE):
        m = pat.search(tail)
        if m:
            return m.group(1)
    return ""


def _harmony_bare_tail_action(cot_text: str) -> str:
    """Last-resort: read intended 0/1 from the final lines of truncated CoT only."""
    if not cot_text:
        return ""
    tail = cot_text[-500:].strip()
    last_line = tail.rsplit("\n", 1)[-1].strip()
    if last_line in ("0", "1"):
        return last_line
    for pat in (_HARMONY_EXPLICIT_TAIL_CHOICE, _HARMONY_LETS_CHOICE, _HARMONY_TAIL_CHOICE):
        m = pat.search(tail)
        if m:
            return m.group(1)
    return ""


# Gemma 4: asymmetric channel markers (not XML). Thought body until next channel or message.
_GEMMA_CHANNEL_THOUGHT = re.compile(
    r"<\|channel\|\>thought\s*(?:\n)?([\s\S]*?)(?=<\|channel\|\>|<\|message\|\>|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedModelOutput:
    """Raw model string plus fields used for logging and action parsing."""

    raw_text: str
    cot_text: str
    answer_text: str
    parse_backend: str


def parse_model_output(raw_text: str, model_name: str = "") -> ParsedModelOutput:
    """
    Derive CoT and user-facing answer text from one completion string.

    parse_backend: harmony | harmony_truncated | harmony_bare | harmony_bare_heuristic |
    harmony_bare_truncated | gemma_channel_thought | redacted_thinking | legacy_think | plain
    """
    text = raw_text if isinstance(raw_text, str) else str(raw_text)
    model_l = (model_name or "").lower()

    analysis_matches = list(_HARMONY_ANALYSIS.finditer(text))
    cot_h = "\n\n".join(
        m.group(1).strip() for m in analysis_matches if m.group(1).strip()
    ).strip()
    fm = _HARMONY_FINAL.search(text)
    if analysis_matches or fm:
        if fm:
            return ParsedModelOutput(
                raw_text=text,
                cot_text=cot_h,
                answer_text=fm.group(1).strip(),
                parse_backend="harmony",
            )
        return ParsedModelOutput(
            raw_text=text,
            cot_text=cot_h,
            answer_text="",
            parse_backend="harmony_truncated",
        )

    if _is_gpt_oss(model_l) and _GPT_OSS_BARE_ANALYSIS.match(text):
        body = _GPT_OSS_BARE_ANALYSIS.sub("", text, count=1).lstrip()
        fm_bare = _GPT_OSS_BARE_FINAL.search(body)
        if fm_bare:
            return ParsedModelOutput(
                raw_text=text,
                cot_text=body[: fm_bare.start()].strip(),
                answer_text=fm_bare.group(1),
                parse_backend="harmony_bare",
            )
        fm_any = _GPT_OSS_BARE_FINAL_ANY.search(body)
        if fm_any:
            return ParsedModelOutput(
                raw_text=text,
                cot_text=body[: fm_any.start()].strip(),
                answer_text=fm_any.group(1),
                parse_backend="harmony_bare",
            )
        tail_action = _harmony_bare_tail_action(body)
        if tail_action:
            return ParsedModelOutput(
                raw_text=text,
                cot_text=body,
                answer_text=tail_action,
                parse_backend="harmony_bare_heuristic",
            )
        return ParsedModelOutput(
            raw_text=text,
            cot_text=body,
            answer_text="",
            parse_backend="harmony_bare_truncated",
        )

    if "google/gemma-4" in model_l:
        gm = _GEMMA_CHANNEL_THOUGHT.search(text)
        if gm:
            cot_g = gm.group(1).strip()
            ans_g = text[gm.end() :].strip()
            return ParsedModelOutput(
                raw_text=text,
                cot_text=cot_g,
                answer_text=ans_g,
                parse_backend="gemma_channel_thought",
            )

    rm = _parse_redacted_thinking(text)
    if rm is not None:
        if not rm.answer_text and rm.cot_text and _is_qwen_hybrid(model_l):
            tail_action = _qwen_thinking_tail_action(rm.cot_text)
            if tail_action:
                return ParsedModelOutput(
                    raw_text=text,
                    cot_text=rm.cot_text,
                    answer_text=tail_action,
                    parse_backend="redacted_thinking_heuristic",
                )
        return rm

    lm = _LEGACY_THINK.search(text)
    if lm:
        cot = lm.group(1).strip()
        ans = _LEGACY_THINK.sub("", text, count=1).strip()
        return ParsedModelOutput(
            raw_text=text,
            cot_text=cot,
            answer_text=ans,
            parse_backend="legacy_think",
        )

    return ParsedModelOutput(
        raw_text=text,
        cot_text="",
        answer_text=text.strip(),
        parse_backend="plain",
    )


def parse_action(text: str) -> tuple[int, str]:
    t = text.strip()
    if t == "0":
        return 0, "strict"
    if t == "1":
        return 1, "strict"
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines and lines[-1] in ("0", "1"):
        return int(lines[-1]), "strict"
    if "0" in t and "1" not in t:
        return 0, "repaired"
    if "1" in t and "0" not in t:
        return 1, "repaired"
    raise ValueError(f"Cannot parse action from response: {text!r}")


def _extract_json_object(text: str) -> dict | None:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    start = t.find("{")
    end = t.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = t[start : end + 1]
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_structured_object(text: str) -> dict | None:
    """Return the first structured JSON-like object from a completion."""
    return _extract_json_object(text)

def parse_structured_action(text: str) -> tuple[int, str]:
    parsed = _extract_json_object(text)
    if parsed is None:
        raise ValueError(f"Cannot parse structured action from response: {text!r}")
    lowered = {str(k).strip().lower(): v for k, v in parsed.items()}
    for key in ("final_action", "my_best_action", "best_action", "my_action", "action"):
        if key not in lowered:
            continue
        value = lowered[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value in (0, 1):
            return value, f"structured_json:{key}"
        if isinstance(value, float) and value in (0.0, 1.0):
            return int(value), f"structured_json:{key}"
        value_text = str(value).strip()
        if value_text in ("0", "1"):
            return int(value_text), f"structured_json:{key}"
    raise ValueError(f"Cannot parse final_action from structured response: {text!r}")



def parse_prediction_only_action(text: str, llm_role: str) -> tuple[int, str]:
    parsed = _extract_json_object(text)
    if parsed is None:
        raise ValueError(f"Cannot parse prediction-only response: {text!r}")
    lowered = {str(k).strip().lower(): v for k, v in parsed.items()}
    for key in ("opponent_next_action", "predicted_opponent_action", "opponent_action", "prediction"):
        if key not in lowered:
            continue
        value = lowered[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value in (0, 1):
            pred = int(value)
        elif isinstance(value, float) and value in (0.0, 1.0):
            pred = int(value)
        else:
            value_text = str(value).strip()
            if value_text not in ("0", "1"):
                continue
            pred = int(value_text)
        role = (llm_role or "seeker").strip().lower()
        if role == "seeker":
            return pred, f"prediction_only_json:{key}"
        if role == "hider":
            return 1 - pred, f"prediction_only_json:{key}"
        raise ValueError(f"llm_role must be seeker|hider, got {llm_role!r}")
    raise ValueError(f"Cannot parse opponent_next_action from prediction-only response: {text!r}")

def resolve_action_from_completion(
    raw_text: str,
    model_name: str = "",
    prefer_structured: bool = False,
    prediction_only_role: str | None = None,
) -> tuple[int | None, str, str, ParsedModelOutput]:
    """
    Split completion then derive action 0/1.

    Returns (action or None, parse_mode, action_parse_source, split).
    """
    split = parse_model_output(raw_text, model_name)
    if prediction_only_role:
        for label, cand in (
            ("final_text", split.answer_text.strip()),
            ("raw_text", split.raw_text.strip()),
        ):
            if not cand:
                continue
            try:
                action, mode = parse_prediction_only_action(cand, prediction_only_role)
                return action, mode, label, split
            except ValueError:
                continue
    if prefer_structured:
        for label, cand in (
            ("final_text", split.answer_text.strip()),
            ("raw_text", split.raw_text.strip()),
        ):
            if not cand:
                continue
            try:
                action, mode = parse_structured_action(cand)
                return action, mode, label, split
            except ValueError:
                continue
    for label, cand in (
        ("final_text", split.answer_text.strip()),
        ("raw_text", split.raw_text.strip()),
    ):
        if not cand:
            continue
        try:
            action, mode = parse_action(cand)
            return action, mode, label, split
        except ValueError:
            continue
    return None, "fallback", "fallback_random", split
