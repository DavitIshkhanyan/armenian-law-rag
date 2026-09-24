"""One streaming client for every provider.

Gemini, Groq and OpenRouter all expose OpenAI-compatible chat completion endpoints, so a single code
path handles them; only base_url / key / model differ (see app/config.py). This keeps the benchmark
fair: same request shape, same timing code, same error handling for every model.

Measured per call: time-to-first-token, total time, prompt/completion tokens (from the API's
`usage`, estimated only if the provider omits it), and a failure category.
"""
from __future__ import annotations

import threading
import time
from collections import deque
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
    reasoning_tokens: int | None = None  # hidden thinking tokens, billed as output
    finish_reason: str | None = None
    usage_estimated: bool = False
    retries: int = 0
    error: str | None = None        # rate_limit | timeout | overloaded | api_error | empty_output | truncated | missing_key
    error_detail: str | None = None
    text: str = ""
    events: list[str] = field(default_factory=list)

    def cost_usd(self, spec: ModelSpec) -> float | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return (self.prompt_tokens * spec.price_in + self.completion_tokens * spec.price_out) / 1e6


class Pacer:
    """Client-side limiter for free-tier quotas: keeps requests and estimated tokens within a
    sliding 60 s window per model, so the benchmark measures models rather than 429 storms."""

    def __init__(self):
        self._lock = threading.Lock()
        self._log: dict[str, deque] = {}

    def wait(self, spec: ModelSpec, est_tokens: int) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                log = self._log.setdefault(spec.key, deque())
                while log and now - log[0][0] > 60:
                    log.popleft()
                used = sum(t for _, t in log)
                if len(log) < spec.rpm and (spec.tpm is None or not log or used + est_tokens <= spec.tpm):
                    log.append((now, est_tokens))
                    return
                delay = 60 - (now - log[0][0]) + 0.1
            time.sleep(max(delay, 0.1))


PACER = Pacer()


def estimate_tokens(messages: list[dict], max_tokens: int) -> int:
    # Armenian script costs roughly 2-3x more tokens per character than English; ~2.5 chars/token
    # is a conservative estimate across both. Groq counts the full max_tokens reservation against the
    # per-minute budget, so it is included in full.
    return sum(len(m["content"]) for m in messages) * 2 // 5 + max_tokens


def _client(spec: ModelSpec) -> OpenAI:
    return OpenAI(api_key=spec.api_key, base_url=spec.base_url, timeout=TIMEOUT_S, max_retries=0)


RETRYABLE = {"rate_limit", "timeout", "overloaded"}


def _classify(exc: Exception) -> str:
    if isinstance(exc, openai.RateLimitError):
        return "rate_limit"
    if isinstance(exc, (openai.APITimeoutError, TimeoutError)):
        return "timeout"
    msg = str(exc).lower()
    if (isinstance(exc, (openai.InternalServerError, openai.APIConnectionError))
            or "overloaded" in msg or "temporarily" in msg or "unavailable" in msg):
        return "overloaded"  # transient upstream capacity problem (5xx / provider overloaded)
    return "api_error"


def stream_chat(model_key: str, messages: list[dict], stats: CallStats | None = None,
                temperature: float = 0.0, max_tokens: int = 4096, pace: bool = True) -> Iterator[str]:
    """Yield text deltas; fill `stats` as a side effect. Retries only before the first token."""
    spec = MODELS[model_key]
    stats = stats if stats is not None else CallStats(model_key)
    if not spec.api_key:
        stats.error, stats.error_detail = "missing_key", f"{spec.api_key_env} is not set"
        return

    client = _client(spec)
    for attempt in range(MAX_RETRIES + 1):
        if pace:
            PACER.wait(spec, estimate_tokens(messages, max_tokens))
        start = time.perf_counter()
        got_token = False
        try:
            kwargs = dict(model=spec.model, messages=messages, temperature=temperature,
                          max_tokens=max_tokens, stream=True, **spec.extra)
            kwargs["stream_options"] = {"include_usage": True}
            for chunk in client.chat.completions.create(**kwargs):
                if chunk.usage:
                    u = chunk.usage
                    stats.prompt_tokens = u.prompt_tokens
                    # Billed output = total - prompt. Gemini leaves its hidden "thinking" tokens out of
                    # completion_tokens but bills them; Groq and OpenRouter already include reasoning.
                    billed = (u.total_tokens - u.prompt_tokens) if u.total_tokens else None
                    stats.completion_tokens = max(u.completion_tokens or 0, billed or 0)
                    details = getattr(u, "completion_tokens_details", None)
                    visible_gap = stats.completion_tokens - (u.completion_tokens or 0)
                    stats.reasoning_tokens = (getattr(details, "reasoning_tokens", None) if details else None) or (visible_gap or None)
                if chunk.choices and chunk.choices[0].finish_reason:
                    stats.finish_reason = chunk.choices[0].finish_reason
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    if not got_token:
                        stats.ttft_s = time.perf_counter() - start
                        got_token = True
                    stats.text += delta
                    yield delta
            stats.total_s = time.perf_counter() - start
            stats.error = (None if stats.text.strip() else "empty_output")
            if stats.error is None and stats.finish_reason == "length":
                stats.error = "truncated"  # answer cut off by the output limit (malformed output)
            break
        except Exception as exc:  # noqa: BLE001 - every failure is recorded, none is hidden
            stats.error, stats.error_detail = _classify(exc), str(exc)[:300]
            stats.events.append(f"attempt {attempt + 1}: {stats.error}")
            if got_token or attempt == MAX_RETRIES or stats.error not in RETRYABLE:
                stats.total_s = time.perf_counter() - start
                break
            stats.retries += 1
            # Backoff; a per-minute token window needs up to a minute to clear.
            time.sleep(min(2 ** attempt * 10, 60) if stats.error == "rate_limit" else min(2 ** attempt * 5, 30))

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
