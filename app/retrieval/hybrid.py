"""Hybrid retrieval: dense (bge-m3, both languages) + BM25 (question language) + explicit
article references, fused with Reciprocal Rank Fusion and aggregated to article level.

Why each signal:
- Dense bge-m3 is cross-lingual: an English question also matches the Armenian original and
  vice versa, so each article gets two chances to be found.
- BM25 catches exact legal terms, numbers and rare words that embeddings blur. Armenian is highly
  inflected (-ը, -ի, -ների, -ում ...), so tokens are prefix-stemmed.
- Users often ask "what does Article 45 say"; an explicit reference is a near-certain signal and
  is forced into the result.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app.retrieval.embeddings import DenseIndex, load_chunks

RRF_K = 60
# Fusion weights per ranked list (dense-hy, dense-en each W_DENSE; BM25 W_BM25). Dense is the
# primary signal: at equal weight BM25 drags paraphrased questions down (recall@5 0.89 vs 1.00 on
# eval/retrieval_dev.json), while a small weight keeps its benefit on exact legal wording.
# See `python -m app.eval.retrieval_eval` and docs/RAG_PIPELINE.md.
W_DENSE = 1.0
W_BM25 = 0.25
STEM_LEN = 6
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
ARMENIAN_RE = re.compile(r"[Ա-֏]")
ARTICLE_REF_RE = re.compile(r"(?:\barticle|\bart\.|հոդված|հոդ\.)\s*(\d{1,2}(?:\.\d)?)", re.IGNORECASE)
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "be", "for", "on", "by", "with", "what",
    "which", "who", "how", "does", "do", "can", "law", "this", "that", "under", "shall", "it", "as",
    "և", "է", "են", "որ", "ինչ", "ինչպես", "ով", "ըստ", "օրենքի", "օրենքով", "սույն", "համար", "կամ", "չի",
}


def detect_lang(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    hy = sum(1 for c in letters if ARMENIAN_RE.match(c))
    return "hy" if hy / len(letters) > 0.3 else "en"


def tokenize(text: str) -> list[str]:
    return [t[:STEM_LEN] for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


@dataclass
class ArticleHit:
    number: str
    score: float
    chunk_ids: list[str] = field(default_factory=list)  # matched chunks, best first
    forced: bool = False


@dataclass
class RetrievalResult:
    query: str
    lang: str
    articles: list[ArticleHit]


class HybridRetriever:
    def __init__(self, chunks: list[dict] | None = None):
        self.chunks = chunks or load_chunks()
        self.dense = DenseIndex(self.chunks)
        self.by_lang = {lang: [i for i, c in enumerate(self.chunks) if c["lang"] == lang] for lang in ("hy", "en")}
        self.bm25 = {
            lang: BM25Okapi([tokenize(self.chunks[i]["embed_text"]) for i in idx])
            for lang, idx in self.by_lang.items()
        }
        self.articles = {c["article"] for c in self.chunks}
        self.lang_articles = {lang: {self.chunks[i]["article"] for i in idx} for lang, idx in self.by_lang.items()}

    def _article_order(self, chunk_ids: list[int]) -> list[str]:
        return list(dict.fromkeys(self.chunks[i]["article"] for i in chunk_ids))

    def _bm25(self, query: str, lang: str, k: int) -> list[int]:
        scores = self.bm25[lang].get_scores(tokenize(query))
        idx = self.by_lang[lang]
        ranked = sorted(range(len(idx)), key=lambda j: -scores[j])[:k]
        return [idx[j] for j in ranked if scores[j] > 0]

    def retrieve(self, query: str, top_k: int = 5, pool: int = 30) -> RetrievalResult:
        lang = detect_lang(query)
        dense = [i for i, _ in self.dense.search(query, k=pool * 2)]
        dense_hy = [i for i in dense if self.chunks[i]["lang"] == "hy"][:pool]
        dense_en = [i for i in dense if self.chunks[i]["lang"] == "en"][:pool]
        ranked_lists = [(W_DENSE, dense_hy), (W_DENSE, dense_en), (W_BM25, self._bm25(query, lang, pool))]

        # Weighted RRF at article level: each list is first collapsed to an article ranking
        # (an article counts once, at its best chunk), so long articles are not favoured.
        scores: dict[str, float] = {}
        chunk_rank: dict[str, dict[str, float]] = {}
        for weight, lst in ranked_lists:
            for rank, a in enumerate(self._article_order(lst)):
                scores[a] = scores.get(a, 0) + weight / (RRF_K + rank + 1)
            for pos, i in enumerate(lst):
                c = self.chunks[i]
                chunk_rank.setdefault(c["article"], {})
                chunk_rank[c["article"]][c["id"]] = chunk_rank[c["article"]].get(c["id"], 0) + 1.0 / (RRF_K + pos + 1)

        # An article present in only one version (Article 17.1 is only in the English text) can
        # appear in just one of the two dense lists; give it the same contribution in the other,
        # otherwise it is structurally outranked by articles that exist in both languages.
        for list_lang, other_list in (("hy", dense_en), ("en", dense_hy)):
            for rank, a in enumerate(self._article_order(other_list)):
                if a not in self.lang_articles[list_lang]:
                    scores[a] = scores.get(a, 0) + W_DENSE / (RRF_K + rank + 1)

        forced = [n for n in dict.fromkeys(ARTICLE_REF_RE.findall(query)) if n in self.articles]
        order = forced + [a for a in sorted(scores, key=lambda a: -scores[a]) if a not in forced]
        hits = []
        for a in order[:max(top_k, len(forced))]:
            ids = sorted(chunk_rank.get(a, {}), key=lambda cid: -chunk_rank[a][cid])
            hits.append(ArticleHit(a, round(scores.get(a, 0.0), 5), ids, forced=a in forced))
        return RetrievalResult(query, lang, hits)


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
