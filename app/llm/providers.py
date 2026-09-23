"""One streaming client for every provider.

Gemini, Groq and Mistral all expose OpenAI-compatible chat completion endpoints, so a single code
path handles them; only base_url / key / model differ (see app/config.py). This keeps the benchmark
fair: same request shape, same timing code, same error handling for every model.

Measured per call: time-to-first-token, total time, prompt/completion tokens (from the API's
`usage`, estimated only if the provider omits it), and a failure category.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import openai
from openai import OpenAI

from app.config import MODELS, ModelSpec

MAX_RETRIES = 3
TIMEOUT_S = 60


@dataclass
class CallStats:
    model: str
    ttft_s: float | None = None
    total_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    usage_estimated: bool = False
    retries: int = 0
    error: str | None = None        # rate_limit | timeout | api_error | empty_output | missing_key
    error_detail: str | None = None
    text: str = ""
    events: list[str] = field(default_factory=list)

    def cost_usd(self, spec: ModelSpec) -> float | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return (self.prompt_tokens * spec.price_in + self.completion_tokens * spec.price_out) / 1e6


def _client(spec: ModelSpec) -> OpenAI:
    return OpenAI(api_key=spec.api_key, base_url=spec.base_url, timeout=TIMEOUT_S, max_retries=0)


def _classify(exc: Exception) -> str:
    if isinstance(exc, openai.RateLimitError):
        return "rate_limit"
    if isinstance(exc, (openai.APITimeoutError, TimeoutError)):
        return "timeout"
    return "api_error"


def stream_chat(model_key: str, messages: list[dict], stats: CallStats | None = None,
                temperature: float = 0.0, max_tokens: int = 1024) -> Iterator[str]:
    """Yield text deltas; fill `stats` as a side effect. Retries only before the first token."""
    spec = MODELS[model_key]
    stats = stats if stats is not None else CallStats(model_key)
    if not spec.api_key:
        stats.error, stats.error_detail = "missing_key", f"{spec.api_key_env} is not set"
        return

    client = _client(spec)
    for attempt in range(MAX_RETRIES + 1):
        start = time.perf_counter()
        got_token = False
        try:
            kwargs = dict(model=spec.model, messages=messages, temperature=temperature,
                          max_tokens=max_tokens, stream=True)
            if spec.provider != "Mistral":  # Mistral sends usage in the last chunk without this flag
                kwargs["stream_options"] = {"include_usage": True}
            for chunk in client.chat.completions.create(**kwargs):
                if chunk.usage:
                    stats.prompt_tokens = chunk.usage.prompt_tokens
                    stats.completion_tokens = chunk.usage.completion_tokens
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    if not got_token:
                        stats.ttft_s = time.perf_counter() - start
                        got_token = True
                    stats.text += delta
                    yield delta
            stats.total_s = time.perf_counter() - start
            stats.error = None if stats.text.strip() else "empty_output"
            break
        except Exception as exc:  # noqa: BLE001 - every failure is recorded, none is hidden
            stats.error, stats.error_detail = _classify(exc), str(exc)[:300]
            stats.events.append(f"attempt {attempt + 1}: {stats.error}")
            if got_token or attempt == MAX_RETRIES or stats.error == "api_error":
                stats.total_s = time.perf_counter() - start
                break
            stats.retries += 1
            time.sleep(min(2 ** attempt * 5, 30))  # backoff for rate limits / timeouts

    if stats.text and (stats.prompt_tokens is None or stats.completion_tokens is None):
        # Rough fallback (~4 chars/token); flagged so the report can exclude or caveat it.
        stats.prompt_tokens = sum(len(m["content"]) for m in messages) // 4
        stats.completion_tokens = len(stats.text) // 4
        stats.usage_estimated = True


def complete(model_key: str, messages: list[dict], **kw) -> CallStats:
    stats = CallStats(model_key)
    for _ in stream_chat(model_key, messages, stats, **kw):
        pass
    return stats
