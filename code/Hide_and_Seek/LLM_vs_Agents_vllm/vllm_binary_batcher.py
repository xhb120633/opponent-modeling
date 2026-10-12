"""Background-batched normalized 0/1 token probabilities from vLLM."""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass

from vllm import LLM, SamplingParams

from evaluate_predictive_action_logits_vllm import binary_probability, binary_token_classes
from online_binary_belief_agent import recover_binary_probability_from_complement


@dataclass
class _Request:
    prompt: str
    future: Future


class BinaryVLLMBatchedGenerator:
    def __init__(
        self,
        model_name: str,
        lora_path: str,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
        max_num_seqs: int = 64,
        max_model_len: int = 8192,
        batch_collect_ms: int = 20,
        max_batch_size: int = 64,
    ):
        engine_kwargs = dict(
            model=model_name,
            dtype=dtype,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            enforce_eager=True,
            trust_remote_code=True,
            enable_lora=True,
            max_loras=1,
        )
        self._engine = LLM(**engine_kwargs)
        from vllm.lora.request import LoRARequest

        self._lora_request = LoRARequest("online_belief_adapter", 1, lora_path)
        self._tokenizer = self._engine.get_tokenizer()
        self._token_classes = binary_token_classes(self._tokenizer)
        allowed_ids = sorted(self._token_classes)
        self._sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            max_tokens=1,
            min_tokens=1,
            logprobs=len(allowed_ids),
            allowed_token_ids=allowed_ids,
        )
        self._queue = queue.Queue()
        self._collect_s = max(1, batch_collect_ms) / 1000.0
        self._max_batch_size = max(1, max_batch_size)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _render(self, prompt: str) -> str:
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def predict_one(self, prompt: str) -> tuple[float, dict]:
        future = Future()
        self._queue.put(_Request(prompt=prompt, future=future))
        return future.result()

    def close(self):
        self._stop.set()
        self._worker.join(timeout=3)

    def _collect(self, first):
        batch = [first]
        while len(batch) < self._max_batch_size:
            try:
                batch.append(self._queue.get(timeout=self._collect_s))
            except queue.Empty:
                break
        return batch

    def _run(self):
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = self._collect(first)
            try:
                outputs = self._engine.generate(
                    [self._render(item.prompt) for item in batch],
                    self._sampling_params,
                    lora_request=self._lora_request,
                )
                results = []
                for output in outputs:
                    completion = output.outputs[0]
                    logprobs = completion.logprobs[0] if completion.logprobs else None
                    p1, diagnostics = binary_probability(logprobs, self._token_classes)
                    p1, diagnostics = recover_binary_probability_from_complement(
                        p1, diagnostics, len(self._token_classes)
                    )
                    if p1 is None:
                        raise RuntimeError(f"Missing normalized binary logits: {diagnostics}")
                    results.append((float(p1), diagnostics))
                for request, result in zip(batch, results):
                    request.future.set_result(result)
            except Exception as exc:
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(exc)
