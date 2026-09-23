"""Structure-aware chunking.

Rules:
- A chunk never crosses an article boundary (citations are article-level).
- Within an article, whole numbered parts are packed greedily up to MAX_CHARS, so short
  articles stay in one chunk and long ones split on their own part boundaries.
- A single part that is still too long (e.g. the Article 2 definitions list) is split on
  sentence / list-item boundaries.
- Every chunk's embedding text is prefixed with "Article N. Title" so a chunk from the middle of
  an article still carries its topic.
"""
from __future__ import annotations

import re

MAX_CHARS = 1200
# Split long parts at clause ends: ";", ".", ":" and the Armenian full stop "։".
CLAUSE_SPLIT_RE = re.compile(r"(?<=[;.:։])\s+|\n")

LABELS = {"en": ("Article", "Chapter"), "hy": ("Հոդված", "Գլուխ")}


def _split_long(text: str, limit: int) -> list[str]:
    pieces, cur = [], ""
    for clause in CLAUSE_SPLIT_RE.split(text):
        if not clause:
            continue
        if cur and len(cur) + len(clause) + 1 > limit:
            pieces.append(cur)
            cur = clause
        else:
            cur = f"{cur} {clause}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def header(article: dict) -> str:
    art, chap = LABELS[article["lang"]]
    return f"{art} {article['number']}. {article['title']} ({chap} {article['chapter']}: {article['chapter_title']})"


def chunk_article(article: dict) -> list[dict]:
    units: list[tuple[str, str]] = []  # (part number, text)
    for p in article["parts"]:
        if len(p["text"]) <= MAX_CHARS:
            units.append((p["part"], p["text"]))
        else:
            units.extend((p["part"], piece) for piece in _split_long(p["text"], MAX_CHARS))

    groups: list[list[tuple[str, str]]] = []
    size = 0
    for unit in units:
        if groups and size + len(unit[1]) <= MAX_CHARS:
            groups[-1].append(unit)
            size += len(unit[1])
        else:
            groups.append([unit])
            size = len(unit[1])

    head = header(article)
    chunks = []
    for i, group in enumerate(groups):
        parts = list(dict.fromkeys(p for p, _ in group))
        text = "\n".join(t for _, t in group)
        chunks.append({
            "id": f"{article['lang']}-{article['number']}-{i}",
            "lang": article["lang"],
            "article": article["number"],
            "title": article["title"],
            "chapter": article["chapter"],
            "parts": [p for p in parts if p != "0"],
            "text": text,
            "embed_text": f"{head}\n{text}",
        })
    return chunks


def chunk_all(articles_by_lang: dict[str, list[dict]]) -> list[dict]:
    return [c for arts in articles_by_lang.values() for a in arts for c in chunk_article(a)]
