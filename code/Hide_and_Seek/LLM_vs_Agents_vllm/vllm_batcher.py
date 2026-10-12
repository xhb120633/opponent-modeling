import logging
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


def _model_key(model_name: str) -> str:
    return (model_name or "").strip().lower()


def _is_openai_gpt_oss(k: str) -> bool:
    return k.startswith("openai/gpt-oss") or "/gpt-oss" in k


def _is_qwen(k: str) -> bool:
    return k.startswith("qwen/")


def _is_qwen_hybrid_thinking(k: str) -> bool:
    """
    Qwen3 / Qwen3.5 / Qwen3.6 hybrid models: thinking ON unless enable_thinking=False.
    Qwen2.5 Instruct is not hybrid 鈥?no enable_thinking chat-template switch.
    """
    if not k.startswith("qwen/"):
        return False
    if "qwen2.5" in k or "qwen2-" in k or "/qwen2/" in k:
        return False
    return any(marker in k for marker in ("qwen3", "qwq"))


def _is_gemma4(k: str) -> bool:
    return k.startswith("google/") and "gemma-4" in k


def _merge_completion_text(completion: Any) -> str:
    """
    vLLM may split chain-of-thought into `reasoning` / `reasoning_content` and final
    answer into `text` (see https://docs.vllm.ai/en/latest/features/reasoning_outputs.html).
    Offline `LLM.generate` still often puts everything in `text`; when split, we must
    concatenate so downstream Harmony / tag parsers see the full completion.
    """
    if completion is None:
        return ""
    reasoning = (
        getattr(completion, "reasoning", None)
        or getattr(completion, "reasoning_content", None)
        or ""
    )
    reasoning = str(reasoning).strip() if reasoning else ""
    text = str(getattr(completion, "text", None) or "").strip()
    if reasoning and text:
        return f"{reasoning}\n{text}"
    return text or reasoning


@dataclass(frozen=True)
class DecodeOptions:
    enable_thinking: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    apply_chat_template: bool = True


@dataclass
class _Request:
    prompt: str
    future: Future
    decode_options: DecodeOptions


class VLLMBatchedGenerator:
    """
    Single shared vLLM engine + background batching worker.
    Multiple experiment threads submit prompts here and get grouped
    into one generate() call for better throughput.
    """

    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_num_seqs: int = 64,
        max_model_len: int = 4096,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 64,
        enforce_eager: bool = True,
        trust_remote_code: bool = True,
        batch_collect_ms: int = 20,
        max_batch_size: int = 128,
        seed: Optional[int] = None,
        lora_path: Optional[str] = None,
        lora_name: str = "sft_adapter",
    ):
        self._model_name = model_name
        self._mk = _model_key(model_name)
        self._sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        self._lora_request = None
        engine_kwargs = dict(
            model=model_name,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
        )
        if lora_path:
            engine_kwargs.update(enable_lora=True, max_loras=1)
        self._engine = LLM(**engine_kwargs)
        if lora_path:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest(lora_name, 1, lora_path)
        self._tokenizer = self._engine.get_tokenizer()
        self._queue: "queue.Queue[_Request]" = queue.Queue()
        self._batch_collect_s = max(batch_collect_ms, 1) / 1000.0
        self._max_batch_size = max(1, max_batch_size)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def _chat_template_kwargs_for_model(self, decode_options: DecodeOptions) -> Dict[str, bool]:
        """
        Map DecodeOptions.enable_thinking to tokenizer chat_template_kwargs.

        Offline vLLM (LLM.generate on rendered strings) 鈥?same kwargs as vLLM serve
        default-chat-template-kwargs / request extra_body.

        Qwen3 (incl. Qwen3.6): templates use `enable_thinking`. When omitted, Qwen3.x
        often defaults to an *open* thinking scaffold; Qwen3.6 injects `<think>`
        and only closes it when enable_thinking=False is passed.

        Gemma 4: same kw name; template gates `<|turn>system` / `<|think|>` and the
        generation-prompt `<|channel|>thought` prefix (see HF chat_template.jinja).
        """
        et = decode_options.enable_thinking
        if _is_qwen_hybrid_thinking(self._mk):
            if et is None:
                et = False
            return {"enable_thinking": bool(et)}
        if _is_gemma4(self._mk):
            if et is None:
                et = False
            return {"enable_thinking": bool(et)}
        return {}

    def _build_model_input(self, prompt: str, decode_options: DecodeOptions) -> str:
        if not decode_options.apply_chat_template:
            return prompt

        messages = []
        if decode_options.reasoning_effort and _is_openai_gpt_oss(self._mk):
            # gpt-oss: reasoning depth via system instruction (Harmony).
            messages.append({"role": "system", "content": f"Reasoning: {decode_options.reasoning_effort}"})
        messages.append({"role": "user", "content": prompt})

        chat_template_kwargs = self._chat_template_kwargs_for_model(decode_options)

        try:
            if chat_template_kwargs:
                return self._apply_chat_template_with_kwargs(messages, chat_template_kwargs)
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"apply_chat_template failed for model={self._model_name!r}: {exc}"
            ) from exc

    def _apply_chat_template_with_kwargs(
        self, messages: List[dict], chat_template_kwargs: Dict[str, bool]
    ) -> str:
        """Apply chat template; Qwen hybrid must never silently drop enable_thinking."""
        attempts: list[tuple[str, dict]] = []
        if _is_qwen_hybrid_thinking(self._mk) and "enable_thinking" in chat_template_kwargs:
            et = chat_template_kwargs["enable_thinking"]
            attempts.append(("enable_thinking_kw", {"enable_thinking": et}))
        attempts.append(("chat_template_kwargs", chat_template_kwargs))

        last_type_error: TypeError | None = None
        for label, extra in attempts:
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **extra,
                )
            except TypeError as exc:
                last_type_error = exc
                logger.debug(
                    "[vllm_batcher] apply_chat_template %s failed for %r: %s",
                    label,
                    self._model_name,
                    exc,
                )
                continue

        if _is_qwen_hybrid_thinking(self._mk):
            raise RuntimeError(
                f"Qwen hybrid model {self._model_name!r} requires enable_thinking in "
                f"apply_chat_template; all attempts failed: {last_type_error}"
            ) from last_type_error
        if last_type_error is not None:
            logger.warning(
                "[vllm_batcher] apply_chat_template rejected kwargs (%s); "
                "retrying without kwargs. model=%r kwargs=%r",
                last_type_error,
                self._model_name,
                chat_template_kwargs,
            )
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        raise RuntimeError(f"apply_chat_template failed for model={self._model_name!r}")

    def audit_thinking_template(self, sample_user: str = "Reply with 0 or 1 only.") -> None:
        """Log generation-prompt tails for think OFF vs ON (Qwen hybrid / Gemma4)."""
        if not (_is_qwen_hybrid_thinking(self._mk) or _is_gemma4(self._mk)):
            return
        rt_close = "</" + "redacted" + "_" + "thinking" + ">"
        for label, et in (("OFF", False), ("ON", True)):
            opts = DecodeOptions(enable_thinking=et)
            rendered = self._build_model_input(sample_user, opts)
            tail = rendered[-280:]
            has_close = rt_close.lower() in tail.lower()
            print(
                f"[vllm_batcher] template_audit {label} enable_thinking={et} "
                f"has_close_tag={has_close} tail={tail!r}"
            )

    def generate_one(self, prompt: str, decode_options: Optional[DecodeOptions] = None) -> str:
        if decode_options is None:
            decode_options = DecodeOptions()
        fut: Future = Future()
        self._queue.put(_Request(prompt=prompt, future=fut, decode_options=decode_options))
        return fut.result()

    def close(self):
        self._stop_event.set()
        self._worker.join(timeout=3)

    def _collect_batch(self, first_req: _Request) -> List[_Request]:
        batch = [first_req]
        while len(batch) < self._max_batch_size:
            try:
                req = self._queue.get(timeout=self._batch_collect_s)
            except queue.Empty:
                break
            batch.append(req)
        return batch

    def _run_worker(self):
        while not self._stop_event.is_set():
            try:
                first_req = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            batch = self._collect_batch(first_req)
            prompts = [self._build_model_input(item.prompt, item.decode_options) for item in batch]

            try:
                if self._lora_request is None:
                    outputs = self._engine.generate(prompts, self._sampling_params)
                else:
                    outputs = self._engine.generate(
                        prompts,
                        self._sampling_params,
                        lora_request=self._lora_request,
                    )
                for req, out in zip(batch, outputs):
                    comp = out.outputs[0] if out.outputs else None
                    text = _merge_completion_text(comp)
                    req.future.set_result(text)
            except Exception as exc:
                for req in batch:
                    req.future.set_exception(exc)
