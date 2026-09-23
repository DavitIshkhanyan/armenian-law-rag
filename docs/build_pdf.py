"""Render docs/*.md to PDF via headless Chrome.

Usage: uv run --with markdown python docs/build_pdf.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

DOCS = Path(__file__).resolve().parent
CHROME_CANDIDATES = [
    os.getenv("CHROME", ""),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]
CSS = """
body { font: 10.5pt/1.45 -apple-system, "Segoe UI", "Noto Sans Armenian", sans-serif; color: #1d1d1f; max-width: 100%; }
h1 { font-size: 18pt; margin: 0 0 8pt; } h2 { font-size: 13pt; margin: 16pt 0 6pt; border-bottom: 1px solid #ddd; }
h3 { font-size: 11pt; } code, pre { font: 8.5pt/1.35 Menlo, monospace; } pre { background: #f5f6f8; padding: 6pt; white-space: pre-wrap; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 9pt; } th, td { border: 1px solid #ccc; padding: 3pt 5pt; vertical-align: top; }
th { background: #f0f2f5; } hr { border: 0; border-top: 1px solid #ddd; }
@page { size: A4; margin: 16mm 14mm; }
"""


def main() -> None:
    chrome = next((c for c in CHROME_CANDIDATES if c and Path(c).exists()), None)
    if not chrome:
        raise SystemExit("Chrome/Chromium not found; set CHROME=/path/to/chrome")
    for md in sorted(DOCS.glob("*.md")):
        html = markdown.markdown(md.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
        doc = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
            f.write(doc)
        pdf = md.with_suffix(".pdf")
        subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", f"file://{f.name}"], check=True, capture_output=True)
        print("wrote", pdf.relative_to(DOCS.parent))


if __name__ == "__main__":
    main()
