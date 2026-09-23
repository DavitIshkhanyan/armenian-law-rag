"""Parse the Armenian (authoritative) and English (translation) PDFs of the law into articles.

Output per article: number, title, chapter, section and the body split into top-level parts
("1.", "2.", ...). Parts are what we later chunk on and what citations can point to.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import pymupdf

from app.config import source_path

HEADER_RE = {
    "en": re.compile(r"^Article\s+(\d+(?:\.\d+)?)\.\s*(.*)$"),
    "hy": re.compile(r"^Հոդված\s+(\d+(?:\.\d+)?)\.\s*(.*)$"),
}
CHAPTER_RE = {
    "en": re.compile(r"^C\s*H\s*A\s*P\s*T\s*E\s*R\s+(\d+)\s*$"),
    "hy": re.compile(r"^Գ\s*Լ\s*ՈՒ\s*Խ\s+(\d+)\s*$"),
}
SECTION_RE = {
    "en": re.compile(r"^S\s*E\s*C\s*T\s*I\s*O\s*N\s+(\d+)\s*$"),
    "hy": re.compile(r"^Բ\s*Ա\s*Ժ\s*Ի\s*Ն\s+(\d+)\s*$"),
}
# Where the law text ends (signature block); everything after is dropped.
END_RE = {
    "en": re.compile(r"^President\s*$"),
    "hy": re.compile(r"^Հայաստանի\s*$"),  # "Հայաստանի / Հանրապետության / Նախագահ" signature
}
# A line that is only a list marker: "1.", "(2)", "2)", "(a)", "ա)".
MARKER_ONLY_RE = re.compile(r"^(\(?\d+(?:\.\d+)?[.)]|\(\d+(?:\.\d+)?\)|\(?[a-z]{1,2}\)|[ա-ֆ]{1,2}\))$")
# Start of a new list item / paragraph.
ITEM_START_RE = re.compile(r"^(\d+(?:\.\d+)?\.\s|\(?\d+(?:\.\d+)?\)\s|\(?[a-z]{1,2}\)\s|[ա-ֆ]{1,2}\)\s)")
# Top-level numbered part: "1. ...", "2.1. ..."
PART_RE = re.compile(r"^(\d+(?:\.\d+)?)\.\s")
SENTENCE_END = (".", ";", ":", "։", "`", "՝", ")")


@dataclass
class Article:
    lang: str
    number: str
    title: str
    chapter: str
    chapter_title: str
    section: str | None
    text: str
    parts: list[dict] = field(default_factory=list)


def _page_lines(lang: str) -> list[str]:
    doc = pymupdf.open(source_path(lang))
    pages = list(doc)
    if lang == "hy":
        pages = pages[1:]  # page 0 is arlis.am metadata (number, status, dates)
    lines: list[str] = []
    for page in pages:
        page_lines = [ln.strip() for ln in page.get_text().splitlines()]
        # English pages start with a bare page number.
        while page_lines and (not page_lines[0] or (lang == "en" and page_lines[0].isdigit())):
            page_lines.pop(0)
        lines.extend(page_lines)
    return lines


def _clean(s: str) -> str:
    s = s.replace(" ", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def _merge_markers(lines: list[str]) -> list[str]:
    """Join a marker-only line ("1.", "(2)") with the next non-empty line."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if MARKER_ONLY_RE.match(ln):
            j = i + 1
            while j < len(lines) and not lines[j]:
                j += 1
            if j < len(lines):
                out.append(f"{ln} {lines[j]}")
                i = j + 1
                continue
        out.append(ln)
        i += 1
    return out


def _is_title_line(s: str) -> bool:
    return bool(s) and not s[0].isdigit() and not s.startswith("(")


def _paragraphs(body: list[str]) -> list[str]:
    """Re-flow wrapped PDF lines into paragraphs; a new paragraph starts at a list marker."""
    paras: list[str] = []
    for ln in body:
        if not ln:
            continue
        if not paras or ITEM_START_RE.match(ln):
            paras.append(ln)
        else:
            paras[-1] = f"{paras[-1]} {ln}"
    return [_clean(p) for p in paras]


def _split_parts(paras: list[str]) -> list[dict]:
    """Group paragraphs into top-level numbered parts; unnumbered lead text is part "0"."""
    parts: list[dict] = []
    for p in paras:
        m = PART_RE.match(p)
        if m or not parts:
            parts.append({"part": m.group(1) if m else "0", "text": p})
        else:
            parts[-1]["text"] += "\n" + p
    return parts


def parse(lang: str) -> list[Article]:
    lines = _merge_markers(_page_lines(lang))
    header, chapter_re, section_re, end_re = HEADER_RE[lang], CHAPTER_RE[lang], SECTION_RE[lang], END_RE[lang]

    articles: list[Article] = []
    chapter, chapter_title, section = "", "", None
    body: list[str] = []
    prev_text = ""  # last non-empty line, to reject in-text references like "... and\nArticle 10."
    i = 0

    def flush():
        if articles:
            paras = _paragraphs(body)
            articles[-1].text = "\n".join(paras)
            articles[-1].parts = _split_parts(paras)
        body.clear()

    while i < len(lines):
        ln = lines[i]
        if end_re.match(ln) and articles and articles[-1].number == "67":
            break
        if m := chapter_re.match(ln):
            chapter, section = m.group(1), None
            title_lines = []
            i += 1
            while i < len(lines) and not header.match(lines[i]) and not section_re.match(lines[i]):
                if lines[i]:
                    title_lines.append(lines[i])
                i += 1
            chapter_title = _clean(" ".join(title_lines))
            prev_text = chapter_title
            continue
        if m := section_re.match(ln):
            section = m.group(1)
            i += 1
            # The section title (upper-case lines) follows; it must not end up in an article body.
            while i < len(lines) and not header.match(lines[i]) and (not lines[i] or lines[i].isupper()):
                if lines[i]:
                    prev_text = lines[i]
                i += 1
            continue
        m = header.match(ln)
        if m and (not prev_text or prev_text.endswith(SENTENCE_END) or prev_text == chapter_title or prev_text.isupper()):
            title = m.group(2).strip()
            j = i + 1
            if not title:
                while j < len(lines) and not lines[j]:
                    j += 1
                if j < len(lines) and _is_title_line(lines[j]):
                    title, j = lines[j], j + 1
                else:
                    j = i + 1
            # Long titles wrap onto a continuation line that starts in lower case.
            while title and j < len(lines) and lines[j][:1].islower() and not ITEM_START_RE.match(lines[j]):
                title, j = f"{title} {lines[j]}", j + 1
            if title:
                flush()
                articles.append(Article(lang, m.group(1), _clean(title), chapter, chapter_title, section, ""))
                i = j
                prev_text = title
                continue
        if articles:
            body.append(ln)
        if ln:
            prev_text = ln
        i += 1
    flush()
    return articles


def _num_key(n: str) -> tuple[int, int]:
    a, _, b = n.partition(".")
    return int(a), int(b or 0)


def validate(articles: list[Article]) -> None:
    nums = [a.number for a in articles]
    if nums != sorted(nums, key=_num_key):
        raise ValueError(f"articles out of order: {nums}")
    main = {n for n in nums if "." not in n}
    missing = set(map(str, range(1, 68))) - main
    if missing or len(nums) != len(set(nums)):
        raise ValueError(f"missing={sorted(missing, key=int)} duplicates={len(nums) - len(set(nums))}")
    empty = [a.number for a in articles if len(a.text) < 20]
    if empty:
        raise ValueError(f"empty articles: {empty}")


def parse_all() -> dict[str, list[dict]]:
    out = {}
    for lang in ("hy", "en"):
        arts = parse(lang)
        validate(arts)
        out[lang] = [asdict(a) for a in arts]
    return out


if __name__ == "__main__":
    data = parse_all()
    for lang, arts in data.items():
        extra = [a["number"] for a in arts if "." in a["number"]]
        print(f"{lang}: {len(arts)} articles, extra={extra}, chars={sum(len(a['text']) for a in arts)}")
