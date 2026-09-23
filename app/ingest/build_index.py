"""Build the processed corpus: parse both PDFs -> articles_{hy,en}.json + chunks.json.

Usage: uv run python -m app.ingest.build_index
"""
from __future__ import annotations

import json
from statistics import mean

from app.config import PROCESSED_DIR
from app.ingest.chunk import chunk_all
from app.ingest.parse import parse_all


def write_json(name: str, obj) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def build_corpus() -> list[dict]:
    articles = parse_all()
    for lang, arts in articles.items():
        write_json(f"articles_{lang}.json", arts)
    chunks = chunk_all(articles)
    write_json("chunks.json", chunks)
    for lang in articles:
        lens = [len(c["text"]) for c in chunks if c["lang"] == lang]
        print(f"{lang}: {len(articles[lang])} articles -> {len(lens)} chunks "
              f"(avg {mean(lens):.0f} chars, max {max(lens)})")
    return chunks


if __name__ == "__main__":
    build_corpus()
