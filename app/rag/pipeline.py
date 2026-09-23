"""End-to-end RAG: retrieve -> assemble context -> generate (streaming) -> extract citations.

Retrieval and generation are separate calls so the benchmark can retrieve once per question and
send the identical context to every model.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass

from app.llm.providers import CallStats, stream_chat
from app.rag.context import ContextBlock, build_context
from app.rag.prompts import build_messages, is_refusal, parse_citations
from app.retrieval.hybrid import RetrievalResult, get_retriever

TOP_K = 5


@dataclass
class Prepared:
    question: str
    retrieval: RetrievalResult
    context: str
    blocks: list[ContextBlock]
    messages: list[dict]
    retrieval_s: float


def prepare(question: str, top_k: int = TOP_K) -> Prepared:
    t0 = time.perf_counter()
    retrieval = get_retriever().retrieve(question, top_k=top_k)
    context, blocks = build_context(retrieval)
    return Prepared(question, retrieval, context, blocks, build_messages(question, context),
                    time.perf_counter() - t0)


def generate(prep: Prepared, model_key: str) -> Iterator[dict]:
    """Yield {"type": "token"} events, then one {"type": "done"} event with citations and stats."""
    stats = CallStats(model_key)
    for delta in stream_chat(model_key, prep.messages, stats):
        yield {"type": "token", "text": delta}
    context_articles = [b.article for b in prep.blocks]
    cited = parse_citations(stats.text)
    yield {
        "type": "done",
        "answer": stats.text,
        "citations": cited,
        "uncited_context": [a for a in cited if a not in context_articles],  # cited but never retrieved
        "refusal": is_refusal(stats.text),
        "stats": asdict(stats) | {"text": None},
    }


def retrieval_event(prep: Prepared) -> dict:
    return {
        "type": "retrieval",
        "lang": prep.retrieval.lang,
        "retrieval_s": round(prep.retrieval_s, 3),
        "articles": [
            {"number": b.article, "title": b.title, "lang": b.lang, "partial": b.partial, "text": b.text,
             "score": next(h.score for h in prep.retrieval.articles if h.number == b.article)}
            for b in prep.blocks
        ],
    }


def answer(question: str, model_key: str, top_k: int = TOP_K) -> Iterator[dict]:
    prep = prepare(question, top_k)
    yield retrieval_event(prep)
    yield from generate(prep, model_key)
