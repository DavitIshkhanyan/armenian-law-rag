"""Prompt templates. One prompt for every model, so the benchmark compares models, not prompts."""
from __future__ import annotations

import re

REFUSAL = {
    "en": "The provided articles of the Law do not address this question.",
    "hy": "Օրենքի տրամադրված հոդվածները չեն անդրադառնում այս հարցին։",
}

SYSTEM_PROMPT = f"""You are a legal research assistant for the regulatory affairs and legal teams of a telecom operator in Armenia.
You answer questions about the Law of the Republic of Armenia "On Electronic Communications" (HO-176-N).

Rules:
1. Use ONLY the law excerpts given in <context>. Do not use outside knowledge, other laws, or assumptions about current practice.
2. Every factual sentence must end with a citation of the article it comes from, in square brackets:
   English answers: [Article 45] or, for a specific part, [Article 45(2)]. Armenian answers: [Հոդված 45] or [Հոդված 45(2)].
   Cite each article in its own brackets, e.g. [Article 12] [Article 13]. Cite only articles that appear in <context>.
3. Answer in the language of the question (Armenian question -> Armenian answer, English question -> English answer), even if the excerpts are in the other language.
4. If the excerpts do not contain the answer, reply with exactly this sentence and nothing else:
   English: "{REFUSAL['en']}"
   Armenian: "{REFUSAL['hy']}"
   If they answer only part of the question, answer that part and state clearly which part the law excerpts do not cover.
5. Be concise and precise: keep numbers, deadlines, percentages and legal terms exactly as written in the law. No preamble."""


def build_messages(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}"},
    ]


CITATION_RE = re.compile(r"\[\s*(?:Article|Art\.|Հոդված|Հոդ\.)\s*([^\]]+)\]", re.IGNORECASE)
NUM_RE = re.compile(r"\d{1,2}(?:\.\d)?")
# Part / point markers: 49(2)(1), and the unbracketed variants some models write: 49(2)1, 57(2)2).
PAREN_RE = re.compile(r"\([^)]*\)(?:\s*\d+\)?)?")


def parse_citations(answer: str) -> list[str]:
    """Article numbers cited in the answer, in order of first appearance ("45", "17.1")."""
    found: list[str] = []
    for inner in CITATION_RE.findall(answer):
        # Also handles "[Article 12, 13]" or "[Article 12 and Article 13]" if a model ignores the format.
        for num in NUM_RE.findall(PAREN_RE.sub(" ", inner)):
            if num not in found:
                found.append(num)
    return found


def is_refusal(answer: str) -> bool:
    norm = " ".join(answer.split()).strip().rstrip(".։:").lower()
    return any(REFUSAL[l].rstrip(".։:").lower() in norm for l in REFUSAL)
