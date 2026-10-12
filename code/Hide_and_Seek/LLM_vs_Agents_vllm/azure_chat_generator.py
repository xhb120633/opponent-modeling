"""Azure OpenAI generator adapter for the hide-and-seek LLM agent."""

from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openai import AzureOpenAI


@dataclass(frozen=True)
class AzureUsageTotals:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_prompt_tokens: int = 0


def load_env_file(path: Optional[str], override: bool = False) -> None:
    if not path:
        return
    env_path = Path(path).expanduser()
    if not env_path.exists():
        raise FileNotFoundError(f"Azure env file not found: {env_path}")
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


def _usage_attr(obj, name: str, default: int = 0) -> int:
    value = getattr(obj, name, default) if obj is not None else default
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


class AzureChatGenerator:
    """Small generate_one() adapter matching the interface expected by VLLMAgent."""

    def __init__(
        self,
        deployment: str,
        env_file: Optional[str] = None,
        api_version: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        max_completion_tokens: int = 384,
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        send_temperature: bool = False,
        timeout_s: float = 120.0,
        max_retries: int = 5,
        retry_base_delay_s: float = 2.0,
        sleep_between_calls_s: float = 0.0,
    ):
        load_env_file(env_file)
        self.deployment = deployment
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.send_temperature = send_temperature
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_base_delay_s = retry_base_delay_s
        self.sleep_between_calls_s = sleep_between_calls_s
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.reasoning_tokens = 0
        self.cached_prompt_tokens = 0
        self._usage_lock = threading.Lock()

        resolved_endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        resolved_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        resolved_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION")
        if not resolved_endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set")
        if not resolved_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY is not set")
        if not resolved_version:
            raise RuntimeError("AZURE_OPENAI_API_VERSION is not set")

        self.client = AzureOpenAI(
            azure_endpoint=resolved_endpoint,
            api_key=resolved_key,
            api_version=resolved_version,
        )

    @property
    def usage_totals(self) -> AzureUsageTotals:
        return AzureUsageTotals(
            calls=self.calls,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            reasoning_tokens=self.reasoning_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens,
        )

    def _record_usage(self, usage) -> None:
        completion_details = getattr(usage, "completion_tokens_details", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        with self._usage_lock:
            self.calls += 1
            self.prompt_tokens += _usage_attr(usage, "prompt_tokens")
            self.completion_tokens += _usage_attr(usage, "completion_tokens")
            self.total_tokens += _usage_attr(usage, "total_tokens")
            self.reasoning_tokens += _usage_attr(completion_details, "reasoning_tokens")
            self.cached_prompt_tokens += _usage_attr(prompt_details, "cached_tokens")

    def generate_one(self, prompt: str, decode_options=None) -> str:
        kwargs = {
            "model": self.deployment,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": self.max_completion_tokens,
            "timeout": self.timeout_s,
        }
        effort = self.reasoning_effort
        if effort in (None, "", "service_default") and decode_options is not None:
            effort = getattr(decode_options, "reasoning_effort", None)
        if effort not in (None, "", "none", "service_default"):
            kwargs["reasoning_effort"] = effort
        if self.send_temperature and self.temperature is not None:
            kwargs["temperature"] = self.temperature

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                self._record_usage(getattr(response, "usage", None))
                if self.sleep_between_calls_s > 0:
                    time.sleep(self.sleep_between_calls_s)
                message = response.choices[0].message
                return message.content or ""
            except Exception as exc:  # OpenAI SDK exception taxonomy varies by version.
                last_error = exc
                status = getattr(exc, "status_code", None)
                retryable = status in (408, 409, 429, 500, 502, 503, 504) or status is None
                if attempt >= self.max_retries or not retryable:
                    raise
                delay = self.retry_base_delay_s * (2 ** attempt) + random.random()
                time.sleep(delay)
        raise RuntimeError(f"Azure generation failed: {last_error}")

    def close(self) -> None:
        return None