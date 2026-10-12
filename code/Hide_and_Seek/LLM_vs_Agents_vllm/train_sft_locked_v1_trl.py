from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainingArguments, set_seed

from evaluate_predictive_action_logits_vllm import binary_token_classes


try:
    from trl import SFTConfig, SFTTrainer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script requires TRL. Install/import trl before running SFT.") from exc

try:
    from peft import LoraConfig, PeftModel
except ImportError:  # pragma: no cover
    LoraConfig = None
    PeftModel = None


DEFAULT_DATA_ROOT = "data_sft_locked_v1_qwen25_7b"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TRL SFT for locked-v1 hide-and-seek targets. The custom collator "
            "uses loss_value_spans so loss is computed only on structured output values."
        )
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--target-mode",
        choices=[
            "structured_tom", "action_only", "predictive_state", "realized_action",
            "soft_realized_action",
        ],
        default="structured_tom",
    )
    parser.add_argument(
        "--init-adapter-path",
        default=None,
        help="Optional parent LoRA adapter to continue training from (for bridge SFT).",
    )
    parser.add_argument("--output-dir", default="checkpoints/sft_locked_v1_qwen25_7b_structured_tom")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--num-train-epochs", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--load-best-model-at-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jsonl_dataset(path: Path) -> HFDataset:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                metadata = row.get("metadata", {})
                spans = []
                for span in row["loss_value_spans"]:
                    clean_span = {
                        "char_start": int(span["char_start"]),
                        "char_end": int(span["char_end"]),
                    }
                    if "message_index" in span:
                        clean_span["message_index"] = int(span["message_index"])
                    spans.append(clean_span)
                clean = {
                        "messages": row["messages"],
                        "completion": row["completion"],
                        "loss_value_spans": spans,
                        "example_id": str(metadata.get("example_id", "")),
                    }
                if "soft_target_p1" in row:
                    probability = float(row["soft_target_p1"])
                    if not 0.0 <= probability <= 1.0:
                        raise ValueError(f"Invalid soft_target_p1={probability}")
                    clean["soft_target_p1"] = probability
                rows.append(clean)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return HFDataset.from_list(rows)


@dataclass
class ValueOnlyDataCollator:
    tokenizer: Any
    max_length: int
    last_selected_counts: list[int] | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_input_ids: list[torch.Tensor] = []
        batch_attention_mask: list[torch.Tensor] = []
        batch_labels: list[torch.Tensor] = []
        selected_counts: list[int] = []

        for feature in features:
            text = self.tokenizer.apply_chat_template(
                feature["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            message_starts: dict[int, int] = {}
            cursor = 0
            for message_idx, message in enumerate(feature["messages"]):
                content = str(message.get("content", ""))
                pos = text.find(content, cursor)
                if pos < 0:
                    pos = text.find(content)
                if pos < 0:
                    raise ValueError(f"Could not locate message {message_idx} content inside chat-template text.")
                message_starts[message_idx] = pos
                cursor = pos + len(content)

            enc = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
            )
            input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
            labels = torch.full_like(input_ids, -100)

            global_spans = []
            for span in feature["loss_value_spans"]:
                if "message_index" in span:
                    base = message_starts[int(span["message_index"])]
                else:
                    completion = feature["completion"]
                    base = text.rfind(completion)
                    if base < 0:
                        raise ValueError("Could not locate assistant completion inside chat-template text.")
                global_spans.append((base + int(span["char_start"]), base + int(span["char_end"])))
            selected = 0
            for token_idx, (start, end) in enumerate(enc["offset_mapping"]):
                if start == end:
                    continue
                keep = any(max(start, span_start) < min(end, span_end) for span_start, span_end in global_spans)
                if keep:
                    labels[token_idx] = input_ids[token_idx]
                    selected += 1
            if selected == 0:
                example_id = feature.get("example_id", "<unknown>")
                raise ValueError(
                    f"No value tokens selected for example_id={example_id}; "
                    "increase --max-length or inspect loss_value_spans."
                )

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            selected_counts.append(selected)

        pad_id = self.tokenizer.pad_token_id
        self.last_selected_counts = selected_counts
        batch = {
            "input_ids": pad_sequence(batch_input_ids, batch_first=True, padding_value=pad_id),
            "attention_mask": pad_sequence(batch_attention_mask, batch_first=True, padding_value=0),
            "labels": pad_sequence(batch_labels, batch_first=True, padding_value=-100),
        }
        soft = [feature.get("soft_target_p1") for feature in features]
        if any(value is not None for value in soft):
            if not all(value is not None for value in soft):
                raise ValueError("A batch cannot mix hard and soft targets")
            batch["soft_target_p1"] = torch.tensor(soft, dtype=torch.float32)
        return batch


class SoftBinarySFTTrainer(SFTTrainer):
    """KL-equivalent binary CE over aggregated 0/1 next-token logits."""

    def __init__(self, *args, binary_classes: dict[int, int], **kwargs):
        super().__init__(*args, **kwargs)
        self.class0_ids = [token_id for token_id, target in binary_classes.items() if target == 0]
        self.class1_ids = [token_id for token_id, target in binary_classes.items() if target == 1]
        if not self.class0_ids or not self.class1_ids:
            raise ValueError(f"Incomplete binary token classes: {binary_classes}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "soft_target_p1" not in inputs:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        model_inputs = dict(inputs)
        target = model_inputs.pop("soft_target_p1").to(dtype=torch.float32)
        labels = model_inputs.pop("labels")
        selected = labels.ne(-100)
        counts = selected.sum(dim=1)
        if not torch.all(counts.eq(1)):
            raise ValueError(f"Soft binary loss requires exactly one selected value token: {counts}")
        target_positions = selected.to(dtype=torch.int64).argmax(dim=1)
        if torch.any(target_positions.le(0)):
            raise ValueError("Selected target token must have a causal predecessor")
        outputs = model(**model_inputs)
        batch_index = torch.arange(labels.shape[0], device=labels.device)
        logits = outputs.logits[batch_index, target_positions - 1].float()
        class0 = torch.logsumexp(logits[:, self.class0_ids], dim=1)
        class1 = torch.logsumexp(logits[:, self.class1_ids], dim=1)
        loss = F.binary_cross_entropy_with_logits(class1 - class0, target)
        return (loss, outputs) if return_outputs else loss


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    effective_eval_steps = args.eval_steps
    effective_save_steps = args.save_steps
    if args.max_steps > 0:
        effective_eval_steps = min(effective_eval_steps, args.max_steps)
        effective_save_steps = min(effective_save_steps, args.max_steps)
    if args.load_best_model_at_end:
        effective_save_steps = effective_eval_steps

    kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_steps=effective_eval_steps,
        save_steps=effective_save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=[] if args.report_to == "none" else [args.report_to],
        remove_unused_columns=False,
        seed=args.seed,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    for key, value in {
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "logging_strategy": "steps",
    }.items():
        kwargs[key] = value
    try:
        return SFTConfig(**kwargs)
    except TypeError:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_root) / args.target_mode
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(f"Missing train/val JSONL under {data_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = load_jsonl_dataset(train_path)
    eval_dataset = load_jsonl_dataset(val_path)
    collator = ValueOnlyDataCollator(tokenizer=tokenizer, max_length=args.max_length)
    first_batch = collator([train_dataset[0]])
    print(
        "[sft] value-only collator smoke:",
        f"tokens={int(first_batch['attention_mask'].sum())}",
        f"value_tokens={int(collator.last_selected_counts[0])}",
    )
    if args.dry_run:
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False

    peft_config = None
    if args.init_adapter_path:
        if PeftModel is None:
            raise ImportError("--init-adapter-path requires peft.")
        adapter_path = Path(args.init_adapter_path)
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"Missing parent adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        print(f"[sft] continuing trainable adapter from {adapter_path.resolve()}")
    elif args.use_lora:
        if LoraConfig is None:
            raise ImportError("--use-lora requires peft.")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
        )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    trainer_class = SoftBinarySFTTrainer if args.target_mode == "soft_realized_action" else SFTTrainer
    trainer_kwargs = dict(
        model=model,
        args=build_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    if trainer_class is SoftBinarySFTTrainer:
        trainer_kwargs["binary_classes"] = binary_token_classes(tokenizer)
    trainer = trainer_class(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    run_manifest = {
        "model_name": args.model_name,
        "target_mode": args.target_mode,
        "data_root": str(Path(args.data_root).resolve()),
        "init_adapter_path": (
            str(Path(args.init_adapter_path).resolve()) if args.init_adapter_path else None
        ),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
    }
    with (Path(args.output_dir) / "bridge_run_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(run_manifest, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
