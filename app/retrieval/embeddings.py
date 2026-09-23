"""Dense index: BAAI/bge-m3 embeddings over all chunks (both languages) held in a numpy matrix.

The corpus is ~300 chunks, so exact cosine search over an in-memory matrix is instant and a
vector database would add moving parts without benefit. Chunk vectors are cached on disk and
keyed by a hash of the chunk texts, so they are recomputed only when the corpus changes.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import numpy as np

from app.config import CACHE_DIR, EMBEDDING_MODEL, EMBEDDING_REVISION, PROCESSED_DIR


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION)


def encode(texts: list[str]) -> np.ndarray:
    vecs = model().encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=len(texts) > 32)
    return np.asarray(vecs, dtype=np.float32)


def load_chunks() -> list[dict]:
    return json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))


class DenseIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        texts = [c["embed_text"] for c in chunks]
        digest = hashlib.sha1((EMBEDDING_MODEL + EMBEDDING_REVISION + "\x00".join(texts)).encode()).hexdigest()[:16]
        path = CACHE_DIR / f"emb-{digest}.npy"
        if path.exists():
            self.matrix = np.load(path)
        else:
            self.matrix = encode(texts)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(path, self.matrix)

    def search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        q = encode([query])[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]
