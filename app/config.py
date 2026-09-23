"""Central configuration: paths, model registry and paid-tier prices."""
from __future__ import annotations

import os
from dataclasses import dataclass
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

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env) or None


# Paid list prices (USD / 1M tokens) as published by each provider; see docs/EVALUATION_REPORT.md.
MODELS: dict[str, ModelSpec] = {
    m.key: m
    for m in [
        ModelSpec("gemini-2.5-flash", "Google", "gemini-2.5-flash",
                  "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", 0.30, 2.50),
        ModelSpec("llama-3.3-70b", "Groq", "llama-3.3-70b-versatile",
                  "https://api.groq.com/openai/v1", "GROQ_API_KEY", 0.59, 0.79),
        ModelSpec("mistral-small", "Mistral", "mistral-small-latest",
                  "https://api.mistral.ai/v1", "MISTRAL_API_KEY", 0.10, 0.30),
    ]
}

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-flash")
