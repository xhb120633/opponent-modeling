"""OpenAI-compatible chat generator for local vLLM endpoints."""

from __future__ import annotations

import random
import time
from typing import Optional

from openai import OpenAI


class OpenAIChatGenerator:
    """Small generate_one() adapter matching the interface expected by VLLMAgent."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        max_tokens: int = 384,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = 0.9,
        timeout_s: float = 120.0,
        max_retries: int = 5,
        retry_base_delay_s: float = 1.0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_base_delay_s = retry_base_delay_s
        self.client = OpenAI(base_url=base_url.rstrip("/") + "/v1", api_key=api_key)

    def generate_one(self, prompt: str, decode_options=None) -> str:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "timeout": self.timeout_s,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                retryable = status in (408, 409, 429, 500, 502, 503, 504) or status is None
                if attempt >= self.max_retries or not retryable:
                    raise
                delay = self.retry_base_delay_s * (2**attempt) + random.random()
                time.sleep(delay)
        raise RuntimeError(f"OpenAI-compatible generation failed: {last_error}")

    def close(self) -> None:
        return None
