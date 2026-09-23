"""Retrieval-only evaluation: article recall@k for dense, BM25 and hybrid retrieval.

Two sets: the benchmark questions (worded close to the law) and a paraphrase dev set of
colloquial questions (eval/retrieval_dev.json) that was used to choose the fusion weights.

Usage: uv run python -m app.eval.retrieval_eval
"""
from __future__ import annotations

import json

from app.config import EVAL_DIR
from app.retrieval.hybrid import detect_lang, get_retriever


def main() -> None:
    r = get_retriever()
    sets = {
        "benchmark": [(q["question"], q["expected_articles"], q["lang"])
                      for q in json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
                      if q["expected_articles"]],
        "paraphrase_dev": [(d["q"], d["a"], detect_lang(d["q"]))
                           for d in json.loads((EVAL_DIR / "retrieval_dev.json").read_text(encoding="utf-8"))],
    }
    methods = {
        "dense": lambda q, k: r._article_order([i for i, _ in r.dense.search(q, 60)])[:k],
        "bm25": lambda q, k: r._article_order(r._bm25(q, detect_lang(q), 60))[:k],
        "hybrid": lambda q, k: [h.number for h in r.retrieve(q, top_k=k).articles],
    }
    print(f"{'set':16}{'method':8}" + "".join(f"{f'recall@{k}':>11}" for k in (1, 3, 5)) + f"{'hy@5':>8}{'en@5':>8}")
    for set_name, qs in sets.items():
        for name, fn in methods.items():
            row = []
            for k in (1, 3, 5):
                row.append(sum(len(set(a) & set(fn(q, k))) / len(a) for q, a, _ in qs) / len(qs))
            by_lang = []
            for lang in ("hy", "en"):
                sub = [(q, a) for q, a, l in qs if l == lang]
                by_lang.append(sum(len(set(a) & set(fn(q, 5))) / len(a) for q, a in sub) / len(sub))
            print(f"{set_name:16}{name:8}" + "".join(f"{v:11.3f}" for v in row) + "".join(f"{v:8.3f}" for v in by_lang))


if __name__ == "__main__":
    main()
