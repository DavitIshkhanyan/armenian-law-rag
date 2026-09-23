"""FastAPI app: static UI + streaming Q&A endpoint (Server-Sent Events).

Run: uv run uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import BENCHMARK_MODELS, MODELS, RESULTS_DIR, ROOT
from app.eval import run_benchmark
from app.rag.pipeline import TOP_K, answer
from app.retrieval.hybrid import get_retriever

WEB_DIR = ROOT / "web"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load bge-m3 and the indexes (and run one query) at startup, not on the first user question.
    get_retriever().dense.search("warm-up", k=1)
    yield


app = FastAPI(title="RA Electronic Communications Law — RAG", lifespan=lifespan)


def sse(events: Iterator[dict]) -> StreamingResponse:
    def gen():
        for ev in events:
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/models")
def models() -> list[dict]:
    return [{"key": m.key, "provider": m.provider, "model": m.model, "available": bool(m.api_key)}
            for m in MODELS.values() if m.key in BENCHMARK_MODELS]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    model: str
    top_k: int = Field(default=TOP_K, ge=1, le=10)


@app.post("/api/ask")
def ask(req: AskRequest) -> StreamingResponse:
    if req.model not in BENCHMARK_MODELS:
        raise HTTPException(400, f"unknown model {req.model}")
    return sse(answer(req.question.strip(), req.model, req.top_k))


class BenchmarkRequest(BaseModel):
    models: list[str] = Field(default_factory=lambda: list(BENCHMARK_MODELS))
    limit: int | None = Field(default=None, ge=1)


@app.post("/api/benchmark")
def benchmark(req: BenchmarkRequest) -> StreamingResponse:
    unknown = [m for m in req.models if m not in BENCHMARK_MODELS]
    if unknown or not req.models:
        raise HTTPException(400, f"unknown or empty models: {unknown}")
    return sse(run_benchmark.run(req.models, req.limit))


@app.get("/api/questions")
def questions() -> list[dict]:
    return [{"id": q["id"], "category": q["category"], "lang": q["lang"], "question": q["question"]}
            for q in run_benchmark.load_questions()]


@app.get("/api/benchmark/runs")
def benchmark_runs() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted((d.name for d in RESULTS_DIR.iterdir() if (d / "summary.json").exists()), reverse=True)


@app.get("/api/benchmark/runs/{name}")
def benchmark_run(name: str) -> dict:
    d = RESULTS_DIR / name
    if not name.replace("-", "").isalnum() or not (d / "summary.json").exists():
        raise HTTPException(404, "run not found")
    raw = [json.loads(line) for line in (d / "raw.jsonl").read_text(encoding="utf-8").splitlines() if line]
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    return {"name": name, "meta": meta, "summary": json.loads((d / "summary.json").read_text(encoding="utf-8")),
            "rows": raw}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
