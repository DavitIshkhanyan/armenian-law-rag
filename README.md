# RA Law on Electronic Communications — RAG Q&A and LLM benchmark

An internal assistant that answers questions in Armenian or English about the Law of the Republic of
Armenia "On Electronic Communications" (HO-176-N, [arlis.am/hy/acts/1869](https://www.arlis.am/hy/acts/1869)).
Answers are grounded in the law and cite specific articles. A **Benchmark** tab runs a ground-truth
question set through the same retrieval pipeline on three free-tier LLM providers.

- Design and rationale of the pipeline: [`docs/RAG_PIPELINE.md`](docs/RAG_PIPELINE.md)
- Benchmark results and recommendation: [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md)
- Ground-truth set: [`eval/questions.json`](eval/questions.json) · raw results: [`eval/results/`](eval/results)

## Setup

Requires Python 3.11–3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env          # add GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY
uv run python -m app.llm.probe   # checks every key / model id
```

API keys are read from environment variables only; `.env` is git-ignored.

| Provider | Default model | Key |
|---|---|---|
| Google AI Studio | `gemini-3.8-flash` | `GEMINI_API_KEY` |
| Groq | `openai/gpt-oss-120b` | `GROQ_API_KEY` |
| OpenRouter | `nvidia/nemotron-3-super-120b-a12b:free` | `OPENROUTER_API_KEY` |
| Judge (benchmark only) | `qwen/qwen3.8-27b` on Groq | `GROQ_API_KEY` |

Model ids can be overridden with `GEMINI_MODEL`, `GROQ_MODEL`, `OPENROUTER_MODEL`, `JUDGE_MODEL`.
OpenRouter's free tier allows 50 requests/day without purchased credits — enough for about two full
benchmark runs of the third model per day.

## Run

```bash
uv run uvicorn app.main:app --port 8000
# open http://localhost:8000
```

The first start downloads the `BAAI/bge-m3` embedding model (~2.3 GB) and embeds the 297 chunks (~20 s on CPU);
both are cached afterwards.

Benchmark from the command line (same code as the UI tab):

```bash
uv run python -m app.eval.run_benchmark                     # all models, all 21 questions
uv run python -m app.eval.run_benchmark --models nemotron-3-super --limit 3
uv run python -m app.eval.run_benchmark --rescore eval/results/<run>   # re-judge saved answers
```

## Rebuilding the corpus

The parsed law (`data/processed/*.json`) is committed, so the app runs without the source PDFs.
To rebuild it, put the two PDFs from arlis.am in `data/raw/` (or the project root):

- `ՀՀ ՕՐԵՆՔԸ ԷԼԵԿՏՐՈՆԱՅԻՆ ՀԱՂՈՐԴԱԿՑՈՒԹՅԱՆ ՄԱՍԻՆ.pdf` (Armenian, authoritative)
- `elektr_com_en.pdf` (English translation)

```bash
uv run python -m app.ingest.build_index
```

## Code map

| Stage | Where |
|---|---|
| PDF parsing into articles | `app/ingest/parse.py` |
| Chunking | `app/ingest/chunk.py` |
| Dense index (bge-m3) | `app/retrieval/embeddings.py` |
| Hybrid retrieval, RRF fusion | `app/retrieval/hybrid.py` |
| Context assembly | `app/rag/context.py` |
| Prompt, citation parsing, refusal | `app/rag/prompts.py` |
| Pipeline (retrieve → generate) | `app/rag/pipeline.py` |
| LLM providers, pacing, metrics capture | `app/llm/providers.py` |
| Scoring and judge | `app/eval/metrics.py` |
| Benchmark runner | `app/eval/run_benchmark.py` |
| API / UI | `app/main.py`, `web/` |
