"""Benchmark: same questions, same retrieved context, several LLMs.

Retrieval runs once per question and the identical prompt goes to every model, so differences
come from the models alone. Phase 1: models answer in parallel threads (one per provider, each paced
to its own free-tier limits). Phase 2: the judge scores the saved answers.

Usage:
  uv run python -m app.eval.run_benchmark                      # all models, all questions
  uv run python -m app.eval.run_benchmark --models nemotron-3-super --limit 3
  uv run python -m app.eval.run_benchmark --rescore eval/results/<run>   # re-judge saved answers

Output: eval/results/<timestamp>/raw.jsonl (one line per model x question), summary.json, summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import statistics
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.config import BENCHMARK_MODELS, EVAL_DIR, JUDGE_MODEL, MODELS, RESULTS_DIR
from app.eval.metrics import ANSWERABLE_TYPES, citation_scores, judge
from app.llm.providers import CallStats, stream_chat
from app.rag.pipeline import prepare
from app.rag.prompts import is_refusal, parse_citations


def load_questions() -> list[dict]:
    return json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))


def _run_one(model: str, q: dict, prep) -> dict:
    stats = CallStats(model)
    for _ in stream_chat(model, prep.messages, stats):
        pass
    cited = parse_citations(stats.text)
    context_articles = [b.article for b in prep.blocks]
    return {
        "model": model, "provider_model": MODELS[model].model,
        "qid": q["id"], "lang": q["lang"], "type": q["type"], "category": q["category"],
        "question": q["question"],
        "retrieved_articles": context_articles,
        "retrieval_recall": (len(set(q["expected_articles"]) & set(context_articles)) / len(q["expected_articles"])
                             if q["expected_articles"] else None),
        "answer": stats.text,
        "citations": cited,
        "cited_not_retrieved": [a for a in cited if a not in context_articles],
        "refusal": is_refusal(stats.text),
        "cost_usd": stats.cost_usd(MODELS[model]),
        **{k: v for k, v in asdict(stats).items() if k not in ("text", "model")},
        **citation_scores(cited, q),
    }


# Armenian costs ~2.5 chars/token, English ~4.5; both caps stay under Groq's 8K tokens per request.
JUDGE_CONTEXT_CHARS = {"hy": 7000, "en": 10000}


def judge_context(prep, row: dict, q: dict) -> str:
    """The part of the retrieved context the judge needs. Articles the answer cites come first (the
    hallucination check is about them), then the ones the answer key expects, then acceptable ones.
    Sending all five articles overflowed Groq's 8K-token per-request limit for Armenian prompts."""
    order = (row["citations"] + q["expected_articles"] + q.get("acceptable_articles", []))
    by_article = {b.article: b for b in prep.blocks}
    blocks = [by_article[a] for a in dict.fromkeys(order) if a in by_article] or prep.blocks[:2]
    budget = JUDGE_CONTEXT_CHARS.get(prep.retrieval.lang, 7000)
    parts, used = [], 0
    for b in blocks:
        text = f"[Article {b.article}. {b.title}]\n{b.text}"
        if used + len(text) > budget:
            if budget - used < 400:
                break
            text = text[: budget - used] + " [...]"  # keep the start of the article rather than drop it
        parts.append(text)
        used += len(text)
    return "\n\n".join(parts)


def score(row: dict, q: dict, prep) -> dict:
    if row["error"] and not row["answer"].strip():
        return row | {"correctness": 0.0, "hallucination": None, "judged": False}
    row = row | judge(q, row["answer"], judge_context(prep, row, q)) | {"judged": True}
    if q["type"] == "out_of_scope" and "correctness" in row:
        # Deterministic guard: an exact refusal on an out-of-scope question is correct by definition.
        row["correctness"] = 1.0 if row["refusal"] else row["correctness"]
    return row


def run(models: list[str], limit: int | None = None, out_dir: Path | None = None) -> Iterator[dict]:
    """Yield progress events; the last one is {"type": "summary", ...}."""
    questions = load_questions()[:limit] if limit else load_questions()
    out_dir = out_dir or RESULTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    yield {"type": "start", "models": models, "n_questions": len(questions), "out_dir": str(out_dir)}

    preps = {}
    for q in questions:
        preps[q["id"]] = prepare(q["question"])
    yield {"type": "retrieval_done"}

    # Phase 1: every model answers every question (models in parallel, each paced to its own limits).
    events: queue.Queue = queue.Queue()
    answers: list[dict] = []
    lock = threading.Lock()

    def worker(model: str):
        for q in questions:
            row = _run_one(model, q, preps[q["id"]])
            with lock:
                answers.append(row)
            events.put({"type": "answered", "model": model, "qid": q["id"], "error": row["error"],
                        "ttft_s": row["ttft_s"]})
        events.put({"type": "model_done", "model": model})

    threads = [threading.Thread(target=worker, args=(m,), daemon=True) for m in models]
    for t in threads:
        t.start()
    done = 0
    while done < len(models):
        ev = events.get()
        done += ev["type"] == "model_done"
        yield ev

    # Phase 2: judge the answers. Kept separate so the judge's rate limit never delays (or skews the
    # timing of) the benchmarked models.
    qs = {q["id"]: q for q in questions}
    rows: list[dict] = []
    for row in answers:
        row = score(row, qs[row["qid"]], preps[row["qid"]])
        rows.append(row)
        with open(out_dir / "raw.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        yield {"type": "progress", "model": row["model"], "qid": row["qid"], "error": row["error"],
               "correctness": row.get("correctness"), "ttft_s": row["ttft_s"],
               "judge_error": row.get("judge_error")}

    summary = summarize(rows, models)
    write_summary(out_dir, summary)
    yield {"type": "summary", "out_dir": str(out_dir), "summary": summary}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


def _pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return round(xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))], 3)


def summarize(rows: list[dict], models: list[str]) -> list[dict]:
    out = []
    for m in models:
        r = [x for x in rows if x["model"] == m]
        if not r:
            continue
        ok = [x for x in r if not x["error"]]
        answerable = [x for x in r if x["type"] in ANSWERABLE_TYPES]
        judged = [x for x in r if x.get("judged") and "hallucination" in x and x["hallucination"] is not None]
        failures: dict[str, int] = {}
        for x in r:
            if x["error"]:
                failures[x["error"]] = failures.get(x["error"], 0) + 1
        out.append({
            "model": m,
            "provider_model": MODELS[m].model,
            "n": len(r),
            "answer_accuracy": _mean([x.get("correctness") for x in r]),
            "accuracy_hy": _mean([x.get("correctness") for x in r if x["category"] == "answerable_hy"]),
            "accuracy_en": _mean([x.get("correctness") for x in r if x["category"] == "answerable_en"]),
            "accuracy_adversarial": _mean([x.get("correctness") for x in r if x["category"] == "adversarial"]),
            "citation_f1": _mean([x["cit_f1"] for x in answerable]),
            "citation_precision": _mean([x["cit_precision"] for x in r if not x["error"]]),
            "citation_recall": _mean([x["cit_recall"] for x in answerable]),
            "cited_not_retrieved": sum(len(x["cited_not_retrieved"]) for x in r),
            "hallucination_rate": _mean([1.0 if x["hallucination"] else 0.0 for x in judged]),
            "refusal_on_out_of_scope": _mean([1.0 if x["refusal"] else 0.0 for x in r if x["type"] == "out_of_scope"]),
            "false_refusal_on_answerable": _mean([1.0 if x["refusal"] else 0.0 for x in answerable]),
            "ttft_p50_s": _pct([x["ttft_s"] for x in ok], 50),
            "ttft_p95_s": _pct([x["ttft_s"] for x in ok], 95),
            "total_p50_s": _pct([x["total_s"] for x in ok], 50),
            "total_p95_s": _pct([x["total_s"] for x in ok], 95),
            "prompt_tokens_avg": _mean([x["prompt_tokens"] for x in ok]),
            "completion_tokens_avg": _mean([x["completion_tokens"] for x in ok]),
            "reasoning_tokens_avg": _mean([x.get("reasoning_tokens") or 0 for x in ok]),
            "tokens_total": sum((x["prompt_tokens"] or 0) + (x["completion_tokens"] or 0) for x in ok),
            "usage_estimated": sum(1 for x in ok if x["usage_estimated"]),
            "cost_usd_total": round(sum(x["cost_usd"] or 0 for x in r), 5),
            "cost_usd_per_1k_questions": (round(1000 * statistics.mean(x["cost_usd"] for x in ok if x["cost_usd"] is not None), 3)
                                          if any(x["cost_usd"] is not None for x in ok) else None),
            "failure_rate": round(1 - len(ok) / len(r), 4),
            "failures": failures,
            "retries": sum(x["retries"] for x in r),
            "judge_errors": sum(1 for x in r if "judge_error" in x),
        })
    return out


def write_summary(out_dir: Path, summary: list[dict]) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if summary:
        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            for s in summary:
                w.writerow(s | {"failures": json.dumps(s["failures"])})
    meta = {"judge_model": MODELS[JUDGE_MODEL].model, "models": {m["model"]: m["provider_model"] for m in summary},
            "finished": datetime.now().isoformat(timespec="seconds")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def rescore(run_dir: Path, only: list[str] | None = None, failed_only: bool = False) -> list[dict]:
    """Re-run the judge on saved answers without re-querying the models. `only` limits it to some
    question ids, `failed_only` to rows where the judge call failed; other scores are kept."""
    qs = {q["id"]: q for q in load_questions()}
    raw = [json.loads(line) for line in (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    redo = lambda r: (not only or r["qid"] in only) and (not failed_only or "judge_error" in r)
    keep = [r for r in raw if not redo(r)]
    raw = [r for r in raw if redo(r)]
    preps = {qid: prepare(q["question"]) for qid, q in qs.items() if any(r["qid"] == qid for r in raw)}
    rows = [score({k: v for k, v in r.items() if k not in ("correctness", "hallucination", "unsupported_claims",
                                                            "judge_rationale", "judge_error", "judge_raw", "judged")},
                  qs[r["qid"]], preps[r["qid"]]) for r in raw]
    rows = sorted(keep + rows, key=lambda r: (r["model"], [q for q in qs].index(r["qid"])))
    (run_dir / "raw.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    models = list(dict.fromkeys(r["model"] for r in rows))
    summary = summarize(rows, models)
    write_summary(run_dir, summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=BENCHMARK_MODELS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--rescore", type=Path)
    ap.add_argument("--only", nargs="*", help="with --rescore: question ids to re-judge")
    ap.add_argument("--failed-only", action="store_true", help="with --rescore: only rows whose judge call failed")
    ap.add_argument("--summarize", type=Path, help="rebuild summary.json/csv of a run from raw.jsonl")
    args = ap.parse_args()
    if args.summarize:
        # Re-derive the deterministic citation fields from the saved answers (e.g. after a citation
        # parser fix); judge scores are left untouched.
        qs = {q["id"]: q for q in load_questions()}
        rows = []
        for r in map(json.loads, (args.summarize / "raw.jsonl").read_text(encoding="utf-8").splitlines()):
            cited = parse_citations(r["answer"])
            r |= {"citations": cited,
                  "cited_not_retrieved": [a for a in cited if a not in r["retrieved_articles"]],
                  "refusal": is_refusal(r["answer"])} | citation_scores(cited, qs[r["qid"]])
            rows.append(r)
        (args.summarize / "raw.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                  encoding="utf-8")
        summary = summarize(rows, list(dict.fromkeys(r["model"] for r in rows)))
        write_summary(args.summarize, summary)
    elif args.rescore:
        summary = rescore(args.rescore, args.only, args.failed_only)
    else:
        t0 = time.time()
        summary = []
        for ev in run(args.models, args.limit):
            if ev["type"] == "answered":
                print(f"[{time.time() - t0:6.0f}s] answered {ev['model']:16} {ev['qid']:7} "
                      f"{'ERR ' + ev['error'] if ev['error'] else 'ok'}", flush=True)
            elif ev["type"] == "progress":
                note = f" judge_error={ev['judge_error'][:80]}" if ev.get("judge_error") else ""
                print(f"[{time.time() - t0:6.0f}s] scored   {ev['model']:16} {ev['qid']:7} "
                      f"score={ev['correctness']}{note}", flush=True)
            elif ev["type"] == "summary":
                summary = ev["summary"]
                print("results:", ev["out_dir"])
    for s in summary:
        print(json.dumps({k: v for k, v in s.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
