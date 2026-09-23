"""Context assembly: turn retrieved articles into the text block the LLM sees.

- The unit is the article, in retrieval order (most relevant first), labelled with number and title
  so the model can cite it.
- Text is given in the question's language: Armenian questions get the authoritative Armenian
  original, English questions the English translation. If an article exists only in one version
  (Article 17.1 is only in the English text) that version is used.
- Short articles are included whole (definitions and exceptions are often in another part than the
  one that matched); long articles contribute only their matched chunks.
- A character budget caps prompt size so every model receives the same, bounded context.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from app.config import PROCESSED_DIR
from app.retrieval.hybrid import RetrievalResult

FULL_ARTICLE_CHARS = 3500
MAX_CHUNKS_PER_ARTICLE = 3
CONTEXT_BUDGET_CHARS = 14000


@lru_cache(maxsize=1)
def _articles() -> dict[str, dict[str, dict]]:
    out = {}
    for lang in ("hy", "en"):
        arts = json.loads((PROCESSED_DIR / f"articles_{lang}.json").read_text(encoding="utf-8"))
        out[lang] = {a["number"]: a for a in arts}
    return out


@lru_cache(maxsize=1)
def _chunks_by_article() -> dict[tuple[str, str], list[dict]]:
    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))
    out: dict[tuple[str, str], list[dict]] = {}
    for c in chunks:
        out.setdefault((c["lang"], c["article"]), []).append(c)
    return out


@dataclass
class ContextBlock:
    article: str
    lang: str
    title: str
    text: str
    partial: bool


def _matched_text(lang: str, number: str, chunk_ids: list[str]) -> str:
    """Matched chunks of a long article, mapped into `lang` by part number (or chunk position)."""
    target = _chunks_by_article()[(lang, number)]
    picked: list[dict] = []
    for cid in chunk_ids:
        src_lang, idx = cid.split("-", 1)[0], int(cid.rsplit("-", 1)[1])
        src = next((c for c in _chunks_by_article()[(src_lang, number)] if c["id"] == cid), None)
        match = None
        if src and src["parts"]:
            match = next((c for c in target if set(c["parts"]) & set(src["parts"])), None)
        if match is None and idx < len(target):
            match = target[idx]
        if match and match not in picked:
            picked.append(match)
        if len(picked) >= MAX_CHUNKS_PER_ARTICLE:
            break
    picked.sort(key=lambda c: int(c["id"].rsplit("-", 1)[1]))
    return "\n[...]\n".join(c["text"] for c in picked)


def build_context(result: RetrievalResult) -> tuple[str, list[ContextBlock]]:
    arts = _articles()
    blocks: list[ContextBlock] = []
    used = 0
    for hit in result.articles:
        lang = result.lang if hit.number in arts[result.lang] else ("en" if result.lang == "hy" else "hy")
        art = arts[lang][hit.number]
        if len(art["text"]) <= FULL_ARTICLE_CHARS:
            text, partial = art["text"], False
        else:
            text, partial = _matched_text(lang, hit.number, hit.chunk_ids or [f"{lang}-{hit.number}-0"]), True
        if used + len(text) > CONTEXT_BUDGET_CHARS and blocks:
            break
        blocks.append(ContextBlock(hit.number, lang, art["title"], text, partial))
        used += len(text)

    label = {"en": "Article", "hy": "Հոդված"}
    rendered = "\n\n".join(
        f'<article number="{b.article}" title="{b.title}">\n'
        f"{label[b.lang]} {b.article}. {b.title}\n{b.text}"
        f"{' [excerpt]' if b.partial else ''}\n</article>"
        for b in blocks
    )
    return rendered, blocks
