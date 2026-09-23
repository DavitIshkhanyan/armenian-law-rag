"""Check that every configured model is reachable with the current keys.

Usage: uv run python -m app.llm.probe
"""
from app.config import MODELS
from app.llm.providers import complete

if __name__ == "__main__":
    for key, spec in MODELS.items():
        s = complete(key, [{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=256, pace=False)
        status = "OK " if not s.error else f"ERR {s.error}"
        print(f"{status:22} {key:14} {spec.provider:8} {spec.model:28} "
              f"ttft={s.ttft_s and round(s.ttft_s, 2)} tokens={s.prompt_tokens}/{s.completion_tokens} "
              f"{(s.error_detail or s.text.strip())[:120]}")
