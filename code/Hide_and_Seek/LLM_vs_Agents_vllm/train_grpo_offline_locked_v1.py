from __future__ import annotations

import inspect
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from probability_tom_utils import (
    probability_reward_parts,
    recursive_probability_reward_parts,
    recursive_probability_v4_reward_parts,
)

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_flash_attn_2_available
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser, get_peft_config

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hide_seek_grpo")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)


@dataclass
class ScriptArguments:
    train_file: str = "data_grpo_offline_locked_v1_qwen25_7b/structured_reasoning_gate/train.jsonl"
    eval_file: str | None = "data_grpo_offline_locked_v1_qwen25_7b/structured_reasoning_gate/val.jsonl"
    tokenizer_name_or_path: str | None = None
    sft_adapter_path: str | None = None
    max_train_examples: int | None = None
    max_eval_examples: int | None = None


class StepMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return
        logger.info("[step %s] metrics: %s", state.global_step, logs)


def _config_path_from_argv(argv: list[str]) -> str | None:
    if "--config" not in argv:
        return None
    idx = argv.index("--config")
    if idx + 1 >= len(argv):
        return None
    candidate = argv[idx + 1]
    return candidate if os.path.isfile(candidate) else None


def _merge_yaml_extras(config_path: str, model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig) -> None:
    try:
        import yaml
    except ImportError:
        return
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    for key in ("train_file", "eval_file", "tokenizer_name_or_path", "sft_adapter_path", "max_train_examples", "max_eval_examples"):
        if key in raw and raw[key] is not None:
            setattr(script_args, key, raw[key])
    for key in ("generation_kwargs", "chat_template_kwargs"):
        if key in raw and raw[key] is not None:
            setattr(training_args, key, raw[key])
    for key in ("use_flash_attn", "use_cache", "task_type", "lora_bias"):
        if key in raw and raw[key] is not None:
            setattr(model_args, key, raw[key])


def _resolve_torch_dtype(dtype_value: Any) -> Any:
    if dtype_value is None or isinstance(dtype_value, torch.dtype):
        return dtype_value
    if isinstance(dtype_value, str):
        normalized = dtype_value.replace("torch.", "").strip()
        if normalized.lower() == "auto":
            return "auto"
        return getattr(torch, normalized, dtype_value)
    return dtype_value


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def _extract_first_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_str = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : idx + 1])
                        break
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _binary(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value in (0, 1):
        return int(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    text = str(value).strip()
    if text in ("0", "1"):
        return int(text)
    return None


_TRACE_HANDLES: dict[str, Any] = {}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _optional_at(values: Any, idx: int) -> Any:
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        return values[idx] if idx < len(values) else None
    return values


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _reward_trace_handle():
    if not _env_flag("GRPO_TRACE_REWARDS", False):
        return None
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    if rank not in _TRACE_HANDLES:
        output_dir = os.environ.get("GRPO_OUTPUT_DIR") or os.environ.get("OUTPUT_DIR") or "grpo_reward_trace"
        trace_dir = Path(output_dir) / "reward_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"rank{rank}.jsonl"
        _TRACE_HANDLES[rank] = trace_path.open("a", encoding="utf-8", buffering=1)
        logger.info("Writing GRPO reward trace to %s", trace_path)
    return _TRACE_HANDLES[rank]


def _write_reward_trace(record: dict[str, Any]) -> None:
    handle = _reward_trace_handle()
    if handle is None:
        return
    handle.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")


def gated_tom_reward(
    completions,
    actual_opponent_next_action,
    actual_my_best_action,
    actual_final_action,
    role=None,
    opponent_name=None,
    decision_round=None,
    example_id=None,
    match_id=None,
    repeat_idx=None,
    round_bin=None,
    role_reward_weight=None,
    source_csv=None,
    source_row_index=None,
    prompt_version=None,
    prompt_hash=None,
    prompt=None,
    **kwargs,
):
    rewards: list[float] = []
    global_step = kwargs.get("global_step", "NA")
    stats = {"valid": 0, "pred": 0, "best": 0, "final": 0}
    part_sums = {"valid_format": 0.0, "opponent_prediction": 0.0, "best_action": 0.0, "final_action": 0.0}
    for idx, (completion, gt_opp, gt_best, gt_final) in enumerate(
        zip(completions, actual_opponent_next_action, actual_my_best_action, actual_final_action)
    ):
        raw_completion = _completion_text(completion)
        parsed = _extract_first_json(raw_completion)
        pred = best = final = None
        gt_opp_i = _maybe_int(gt_opp)
        gt_best_i = _maybe_int(gt_best)
        gt_final_i = _maybe_int(gt_final)
        reward_parts = {
            "valid_format": 0.0,
            "opponent_prediction": 0.0,
            "best_action": 0.0,
            "final_action": 0.0,
            "total": 0.0,
        }
        correct = {
            "valid_format": False,
            "opponent_prediction": False,
            "best_action": False,
            "final_action": False,
        }
        if parsed is not None:
            pred = _binary(parsed.get("opponent_next_action"))
            best = _binary(parsed.get("my_best_action"))
            final = _binary(parsed.get("final_action"))
            if pred in (0, 1) and best in (0, 1) and final in (0, 1):
                reward_parts["valid_format"] = 0.05
                reward_parts["total"] = 0.05
                correct["valid_format"] = True
                stats["valid"] += 1
                if pred == gt_opp_i:
                    reward_parts["opponent_prediction"] = 0.95
                    reward_parts["total"] = 1.0
                    correct["opponent_prediction"] = True
                    stats["pred"] += 1
                    if best == gt_best_i:
                        reward_parts["best_action"] = 0.5
                        reward_parts["total"] += 0.5
                        correct["best_action"] = True
                        stats["best"] += 1
                    if final == gt_final_i and best == gt_best_i:
                        reward_parts["final_action"] = 0.5
                        reward_parts["total"] += 0.5
                        correct["final_action"] = True
                        stats["final"] += 1
        for key in part_sums:
            part_sums[key] += reward_parts[key]
        try:
            weight = float(_optional_at(role_reward_weight, idx) or 1.0)
        except Exception:
            weight = 1.0
        reward_parts["role_reward_weight"] = weight
        reward_parts["weighted_total"] = reward_parts["total"] * weight
        rewards.append(float(reward_parts["weighted_total"]))
        if _env_flag("GRPO_TRACE_REWARDS", False):
            record = {
                "global_step": global_step,
                "rank": os.environ.get("RANK"),
                "local_rank": os.environ.get("LOCAL_RANK"),
                "pid": os.getpid(),
                "sample_index_in_reward_call": idx,
                "example_id": _optional_at(example_id, idx),
                "match_id": _optional_at(match_id, idx),
                "role": _optional_at(role, idx),
                "opponent_name": _optional_at(opponent_name, idx),
                "decision_round": _optional_at(decision_round, idx),
                "repeat_idx": _optional_at(repeat_idx, idx),
                "round_bin": _optional_at(round_bin, idx),
                "role_reward_weight": _optional_at(role_reward_weight, idx),
                "source_csv": _optional_at(source_csv, idx),
                "source_row_index": _optional_at(source_row_index, idx),
                "prompt_version": _optional_at(prompt_version, idx),
                "prompt_hash": _optional_at(prompt_hash, idx),
                "parsed": parsed,
                "parsed_actions": {
                    "opponent_next_action": pred,
                    "my_best_action": best,
                    "final_action": final,
                },
                "oracle": {
                    "opponent_next_action": gt_opp_i,
                    "my_best_action": gt_best_i,
                    "final_action": gt_final_i,
                },
                "correct": correct,
                "reward_parts": reward_parts,
                "raw_completion": raw_completion if _env_flag("GRPO_TRACE_RAW_COMPLETION", True) else None,
            }
            if _env_flag("GRPO_TRACE_INCLUDE_PROMPT", False):
                record["prompt"] = _optional_at(prompt, idx)
            _write_reward_trace(record)
    if random.random() < 0.05:
        logger.info(
            "[reward step %s] n=%d mean=%.3f valid=%d pred=%d best=%d final=%d parts=%s",
            global_step,
            len(rewards),
            sum(rewards) / max(1, len(rewards)),
            stats["valid"],
            stats["pred"],
            stats["best"],
            stats["final"],
            {k: round(v, 3) for k, v in part_sums.items()},
        )
    return rewards


def probability_tom_reward(
    completions,
    actual_opponent_action_probability,
    role,
    actual_opponent_next_action=None,
    opponent_name=None,
    decision_round=None,
    example_id=None,
    match_id=None,
    repeat_idx=None,
    round_bin=None,
    role_reward_weight=None,
    source_csv=None,
    source_row_index=None,
    prompt_version=None,
    prompt_hash=None,
    prompt=None,
    **kwargs,
):
    rewards: list[float] = []
    global_step = kwargs.get("global_step", "NA")
    part_sums: dict[str, float] = {}
    for idx, (completion, target_probability) in enumerate(zip(completions, actual_opponent_action_probability)):
        raw_completion = _completion_text(completion)
        parsed = _extract_first_json(raw_completion)
        sample_role = str(_optional_at(role, idx))
        parts = probability_reward_parts(parsed, target_probability, sample_role)
        try:
            weight = float(_optional_at(role_reward_weight, idx) or 1.0)
        except Exception:
            weight = 1.0
        parts["role_reward_weight"] = weight
        parts["weighted_total"] = float(parts["total"]) * weight
        rewards.append(float(parts["weighted_total"]))
        for key, value in parts.items():
            if isinstance(value, (int, float)) and key not in {
                "predicted_probability", "target_probability", "role_reward_weight", "weighted_total", "total"
            }:
                part_sums[key] = part_sums.get(key, 0.0) + float(value)
        if _env_flag("GRPO_TRACE_REWARDS", False):
            record = {
                "reward_mode": "probability_brier",
                "global_step": global_step,
                "rank": os.environ.get("RANK"),
                "local_rank": os.environ.get("LOCAL_RANK"),
                "pid": os.getpid(),
                "sample_index_in_reward_call": idx,
                "example_id": _optional_at(example_id, idx),
                "match_id": _optional_at(match_id, idx),
                "role": sample_role,
                "opponent_name": _optional_at(opponent_name, idx),
                "decision_round": _optional_at(decision_round, idx),
                "repeat_idx": _optional_at(repeat_idx, idx),
                "round_bin": _optional_at(round_bin, idx),
                "source_csv": _optional_at(source_csv, idx),
                "source_row_index": _optional_at(source_row_index, idx),
                "prompt_version": _optional_at(prompt_version, idx),
                "prompt_hash": _optional_at(prompt_hash, idx),
                "actual_opponent_next_action": _optional_at(actual_opponent_next_action, idx),
                "parsed": parsed,
                "reward_parts": parts,
                "raw_completion": raw_completion if _env_flag("GRPO_TRACE_RAW_COMPLETION", True) else None,
            }
            if _env_flag("GRPO_TRACE_INCLUDE_PROMPT", False):
                record["prompt"] = _optional_at(prompt, idx)
            _write_reward_trace(record)
    if random.random() < 0.05:
        logger.info(
            "[probability reward step %s] n=%d mean=%.3f parts=%s",
            global_step,
            len(rewards),
            sum(rewards) / max(1, len(rewards)),
            {key: round(value, 3) for key, value in part_sums.items()},
        )
    return rewards

def recursive_probability_tom_reward(
    completions,
    actual_opponent_belief_my_action_probability,
    actual_opponent_action_probability,
    role,
    actual_opponent_next_action=None,
    opponent_name=None,
    decision_round=None,
    example_id=None,
    match_id=None,
    repeat_idx=None,
    round_bin=None,
    role_reward_weight=None,
    source_csv=None,
    source_row_index=None,
    prompt_version=None,
    prompt_hash=None,
    prompt=None,
    **kwargs,
):
    rewards: list[float] = []
    global_step = kwargs.get("global_step", "NA")
    reward_mode = os.environ.get("GRPO_REWARD_MODE", "recursive_probability").strip().lower()
    if reward_mode == "recursive_probability_v4":
        reward_parts_fn = recursive_probability_v4_reward_parts
        trace_reward_mode = "recursive_probability_v4"
    else:
        reward_parts_fn = recursive_probability_reward_parts
        trace_reward_mode = "recursive_probability_brier"
    part_sums: dict[str, float] = {}
    targets = zip(
        completions,
        actual_opponent_belief_my_action_probability,
        actual_opponent_action_probability,
    )
    for idx, (completion, target_belief, target_action) in enumerate(targets):
        raw_completion = _completion_text(completion)
        parsed = _extract_first_json(raw_completion)
        sample_role = str(_optional_at(role, idx))
        parts = reward_parts_fn(
            parsed,
            target_belief,
            target_action,
            sample_role,
        )
        try:
            weight = float(_optional_at(role_reward_weight, idx) or 1.0)
        except Exception:
            weight = 1.0
        parts["role_reward_weight"] = weight
        parts["weighted_total"] = float(parts["total"]) * weight
        rewards.append(float(parts["weighted_total"]))
        excluded = {
            "predicted_belief_probability",
            "target_belief_probability",
            "predicted_action_probability",
            "target_action_probability",
            "target_confidence",
            "role_reward_weight",
            "weighted_total",
            "total",
        }
        for key, value in parts.items():
            if isinstance(value, (int, float)) and key not in excluded:
                part_sums[key] = part_sums.get(key, 0.0) + float(value)
        if _env_flag("GRPO_TRACE_REWARDS", False):
            record = {
                "reward_mode": trace_reward_mode,
                "global_step": global_step,
                "rank": os.environ.get("RANK"),
                "local_rank": os.environ.get("LOCAL_RANK"),
                "pid": os.getpid(),
                "sample_index_in_reward_call": idx,
                "example_id": _optional_at(example_id, idx),
                "match_id": _optional_at(match_id, idx),
                "role": sample_role,
                "opponent_name": _optional_at(opponent_name, idx),
                "decision_round": _optional_at(decision_round, idx),
                "repeat_idx": _optional_at(repeat_idx, idx),
                "round_bin": _optional_at(round_bin, idx),
                "source_csv": _optional_at(source_csv, idx),
                "source_row_index": _optional_at(source_row_index, idx),
                "prompt_version": _optional_at(prompt_version, idx),
                "prompt_hash": _optional_at(prompt_hash, idx),
                "actual_opponent_next_action": _optional_at(actual_opponent_next_action, idx),
                "parsed": parsed,
                "reward_parts": parts,
                "raw_completion": raw_completion if _env_flag("GRPO_TRACE_RAW_COMPLETION", True) else None,
            }
            if _env_flag("GRPO_TRACE_INCLUDE_PROMPT", False):
                record["prompt"] = _optional_at(prompt, idx)
            _write_reward_trace(record)
    if random.random() < 0.05:
        logger.info(
            "[recursive probability reward step %s] n=%d mean=%.3f parts=%s",
            global_step,
            len(rewards),
            sum(rewards) / max(1, len(rewards)),
            {key: round(value, 3) for key, value in part_sums.items()},
        )
    return rewards

def get_checkpoint(training_args: GRPOConfig) -> str | None:
    if os.path.isdir(training_args.output_dir):
        return get_last_checkpoint(training_args.output_dir)
    return None


def _rewards_are_unscaled(value: Any) -> bool:
    if value is False:
        return True
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"false", "none", "no", "off", "0"}
    return False


def _load_json_dataset(path: str, split_name: str, max_examples: int | None):
    dataset = load_dataset("json", data_files={split_name: path}, split=split_name)
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    return dataset


def train(model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig) -> None:
    scale_rewards_value = getattr(training_args, "scale_rewards", None)
    if not _rewards_are_unscaled(scale_rewards_value):
        raise ValueError(
            "This experiment requires scale_rewards=False. Do not use std-normalized GRPO rewards for hide-and-seek."
        )
    logger.info("scale_rewards=%r (locked; reward function also uses no std normalization)", scale_rewards_value)

    tokenizer_name = script_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_cache=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_kwargs = dict(getattr(training_args, "generation_kwargs", None) or {})
    stop_ids = list(gen_kwargs.get("stop_token_ids") or [])
    for tid in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")):
        if isinstance(tid, int) and tid >= 0 and tid not in stop_ids:
            stop_ids.append(tid)
    if stop_ids:
        gen_kwargs["stop_token_ids"] = stop_ids
    if tokenizer.eos_token_id is not None and gen_kwargs.get("eos_token_id") is None:
        gen_kwargs["eos_token_id"] = int(tokenizer.eos_token_id)
    training_args.generation_kwargs = gen_kwargs

    chat_tpl_kwargs = dict(getattr(training_args, "chat_template_kwargs", None) or {})

    train_dataset = _load_json_dataset(script_args.train_file, "train", script_args.max_train_examples)
    eval_dataset = None
    if script_args.eval_file and Path(script_args.eval_file).is_file():
        eval_dataset = _load_json_dataset(script_args.eval_file, "validation", script_args.max_eval_examples)

    def format_prompt(example: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": example["prompt"]}]
        return {
            "prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **chat_tpl_kwargs),
        }

    train_dataset = train_dataset.map(format_prompt)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(format_prompt)

    model_init_kwargs = dict(getattr(model_args, "model_init_kwargs", None) or {})
    requested_attn_impl = getattr(model_args, "attn_implementation", None)
    if requested_attn_impl:
        resolved_attn_impl = requested_attn_impl
        if requested_attn_impl == "flash_attention_2" and not is_flash_attn_2_available():
            resolved_attn_impl = "sdpa"
            logger.warning("FlashAttention2 unavailable; using %s", resolved_attn_impl)
        model_init_kwargs.setdefault("attn_implementation", resolved_attn_impl)
    resolved_dtype = _resolve_torch_dtype(getattr(model_args, "torch_dtype", None))
    if resolved_dtype is not None:
        model_init_kwargs.setdefault("torch_dtype", resolved_dtype)
    model_init_kwargs.setdefault("revision", model_args.model_revision)
    model_init_kwargs.setdefault("trust_remote_code", model_args.trust_remote_code)

    logger.info("Loading base policy model: %s", model_args.model_name_or_path)
    policy_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_init_kwargs)
    peft_config = get_peft_config(model_args)
    if script_args.sft_adapter_path:
        if PeftModel is None:
            raise ImportError("sft_adapter_path requires peft")
        logger.info("Loading trainable SFT adapter: %s", script_args.sft_adapter_path)
        policy_model = PeftModel.from_pretrained(policy_model, script_args.sft_adapter_path, is_trainable=True)
        peft_config = None

    callbacks = [StepMetricsCallback()]
    reward_mode = os.environ.get("GRPO_REWARD_MODE", "gated_action").strip().lower()
    if reward_mode in {"recursive_probability", "recursive_probability_v4"}:
        reward_func = recursive_probability_tom_reward
    elif reward_mode == "probability":
        reward_func = probability_tom_reward
    else:
        reward_func = gated_tom_reward
    logger.info("GRPO reward mode: %s", reward_mode)
    trainer = GRPOTrainer(
        model=policy_model,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    def wrapped_reward(*args, **kwargs):
        kwargs["global_step"] = trainer.state.global_step
        return reward_func(*args, **kwargs)

    trainer.reward_funcs = [wrapped_reward]

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info("Checkpoint detected, resuming from %s", last_checkpoint)

    logger.info("Starting offline GRPO at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    result = trainer.train(resume_from_checkpoint=last_checkpoint)
    metrics = result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if os.environ.get("GRPO_SKIP_FINAL_SAVE", "").strip().lower() in ("1", "true", "yes"):
        logger.info("GRPO_SKIP_FINAL_SAVE set; skipping final save.")
        return
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("Saved offline GRPO model to %s", training_args.output_dir)


def main() -> None:
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    parse_kw: dict[str, Any] = {}
    sig = inspect.signature(parser.parse_args_and_config)
    if "fail_with_unknown_args" in sig.parameters:
        parse_kw["fail_with_unknown_args"] = False
    model_args, script_args, training_args = parser.parse_args_and_config(**parse_kw)

    cfg_path = _config_path_from_argv(sys.argv)
    if cfg_path:
        _merge_yaml_extras(cfg_path, model_args, script_args, training_args)

    if os.environ.get("GRPO_TRAIN_FILE", "").strip():
        script_args.train_file = os.environ["GRPO_TRAIN_FILE"].strip()
    if os.environ.get("GRPO_EVAL_FILE", "").strip():
        script_args.eval_file = os.environ["GRPO_EVAL_FILE"].strip()
    if os.environ.get("GRPO_OUTPUT_DIR", "").strip():
        training_args.output_dir = os.environ["GRPO_OUTPUT_DIR"].strip()
    if os.environ.get("GRPO_SFT_ADAPTER_PATH", "").strip():
        script_args.sft_adapter_path = os.environ["GRPO_SFT_ADAPTER_PATH"].strip()
    if os.environ.get("VLLM_SERVER_BASE_URL", "").strip():
        training_args.vllm_server_base_url = os.environ["VLLM_SERVER_BASE_URL"].strip()
    if os.environ.get("VLLM_SERVER_HOST", "").strip() and hasattr(training_args, "vllm_server_host"):
        training_args.vllm_server_host = os.environ["VLLM_SERVER_HOST"].strip()

    train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
