"""Central configuration: paths, model registry and paid-tier prices."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"

# Source PDFs. Looked up in data/raw/ first, then the project root.
SOURCE_FILES = {
    "hy": "ՀՀ ՕՐԵՆՔԸ ԷԼԵԿՏՐՈՆԱՅԻՆ ՀԱՂՈՐԴԱԿՑՈՒԹՅԱՆ ՄԱՍԻՆ.pdf",
    "en": "elektr_com_en.pdf",
}


def source_path(lang: str) -> Path:
    name = SOURCE_FILES[lang]
    for base in (RAW_DIR, ROOT):
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Source PDF for '{lang}' not found: put '{name}' in {RAW_DIR}")


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
# Pinned for reproducible embeddings; override together with EMBEDDING_MODEL.
EMBEDDING_REVISION = os.getenv("EMBEDDING_REVISION", "5617a9f61b028005a4858fdac845db406aefb181")


@dataclass(frozen=True)
class ModelSpec:
    key: str             # id used in the app / results
    provider: str
    model: str           # provider's model name
    base_url: str        # OpenAI-compatible endpoint
    api_key_env: str
    price_in: float      # USD per 1M prompt tokens at paid rates
    price_out: float     # USD per 1M completion tokens at paid rates
    rpm: int             # free-tier requests/minute we pace to
    tpm: int | None = None           # free-tier tokens/minute, if the provider enforces one
    extra: dict = field(default_factory=dict)  # provider-specific request params
    benchmark: bool = True           # False for the judge-only model

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env) or None


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_URL = "https://api.groq.com/openai/v1"
MISTRAL_URL = "https://api.mistral.ai/v1"

# Paid list prices (USD / 1M tokens) and free-tier limits as published by each provider in
# September 2026; sources in docs/EVALUATION_REPORT.md. Model ids can be overridden via env.
_SPECS = [
    ModelSpec("gemini-flash", "Google", os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
              GEMINI_URL, "GEMINI_API_KEY", 0.75, 3.75, rpm=10),
    ModelSpec("gpt-oss-120b", "Groq", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
              GROQ_URL, "GROQ_API_KEY", 0.15, 0.60, rpm=30, tpm=8000,
              extra={"reasoning_effort": "low"}),
    ModelSpec("mistral-small", "Mistral", os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
              MISTRAL_URL, "MISTRAL_API_KEY", 0.15, 0.60, rpm=60),
    # Judge: a stronger model that is not one of the benchmarked models.
    ModelSpec("judge", "Mistral", os.getenv("JUDGE_MODEL", "mistral-medium-latest"),
              MISTRAL_URL, "MISTRAL_API_KEY", 1.5, 7.5, rpm=60, benchmark=False),
]
MODELS: dict[str, ModelSpec] = {m.key: m for m in _SPECS}
BENCHMARK_MODELS = [m.key for m in _SPECS if m.benchmark]
JUDGE_MODEL = "judge"
