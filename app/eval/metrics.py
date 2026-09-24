"""Scoring of one model answer against the ground truth.

Deterministic metrics (no LLM involved):
- citation precision / recall / F1 of cited article numbers vs. the answer key.
- refusal detection for out-of-scope questions (fixed refusal sentence).

LLM-judge metrics (a separate model that is not being benchmarked):
- answer correctness vs. the reference answer and key facts: 0 / 0.5 / 1.
- hallucination: does the answer state anything not supported by the *retrieved context*?
  Judged against the context, not the answer key, so a correct-but-unsupported claim still counts.
"""
from __future__ import annotations

import json
import re

from app.config import JUDGE_MODEL
from app.llm.providers import complete

ANSWERABLE_TYPES = {"factual", "synthesis"}


def citation_scores(cited: list[str], q: dict) -> dict:
    cited_set = set(cited)
    required = set(q["expected_articles"])
    allowed = required | set(q.get("acceptable_articles", []))
    if required:
        recall = len(cited_set & required) / len(required)
        precision = (len(cited_set & allowed) / len(cited_set)) if cited_set else 0.0
    else:
        # Nothing should be required; any citation must at least be one of the acceptable ones.
        recall = None
        precision = (len(cited_set & allowed) / len(cited_set)) if cited_set else 1.0
    f1 = None
    if recall is not None:
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"cit_precision": precision, "cit_recall": recall, "cit_f1": f1,
            "cit_all_required": (required <= cited_set) if required else None}


JUDGE_PROMPT = """You are grading an answer produced by a legal question-answering system about the Law of the Republic of Armenia "On Electronic Communications". Answers may be in Armenian or English.

Question type: {qtype}
- factual / synthesis: the law answers the question; grade against the reference answer and key facts.
- out_of_scope: the law does NOT cover this; a correct answer says so (it may mention related articles) and invents no rules.
- false_premise: the question assumes something the law does not say; a correct answer rejects the premise.

<question>{question}</question>
<reference_answer>{reference}</reference_answer>
<key_facts>{key_facts}</key_facts>
<retrieved_context>
{context}
</retrieved_context>
<answer_to_grade>
{answer}
</answer_to_grade>

Grade two things independently:
1. correctness: 1 = all key facts correct and nothing contradicts the reference; 0.5 = partially correct or missing key facts; 0 = wrong, contradicts the reference, or (for out_of_scope / false_premise) answers as if the law covered it.
2. hallucination: true if the answer states any fact, number, deadline, obligation or article content that is NOT supported by the retrieved context (citation labels themselves do not count). Judge only against the retrieved context, not your own knowledge.

Return only JSON: {{"correctness": 0 | 0.5 | 1, "hallucination": true | false, "unsupported_claims": ["..."], "rationale": "one or two sentences"}}"""


def _parse_json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # reasoning models may inline thoughts
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def judge(q: dict, answer: str, context: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        qtype=q["type"], question=q["question"], reference=q["expected_answer"],
        key_facts="; ".join(q["key_facts"]), context=context, answer=answer,
    )
    for _ in range(2):
        stats = complete(JUDGE_MODEL, [{"role": "user", "content": prompt}], max_tokens=1000)
        if stats.error:
            return {"judge_error": f"{stats.error}: {stats.error_detail}"}
        data = _parse_json(stats.text)
        if data and data.get("correctness") in (0, 0.5, 1) and isinstance(data.get("hallucination"), bool):
            return {"correctness": float(data["correctness"]), "hallucination": data["hallucination"],
                    "unsupported_claims": data.get("unsupported_claims", []),
                    "judge_rationale": data.get("rationale", "")}
    return {"judge_error": "malformed judge output", "judge_raw": stats.text[:500]}
