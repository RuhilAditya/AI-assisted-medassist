"""A tiny persistent vector index (cosine similarity over normalised vectors).

Used for both text (BioBERT embeddings of notes / reference passages) and
images (BiomedCLIP embeddings of X-rays and reports). Brute-force numpy search
is exact and fast enough for this project's scale (<100k items: a few ms).
Swapping in FAISS later only touches `search`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SearchHit:
    id: str
    score: float
    metadata: dict


class VectorIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._ids: list[str] = []
        self._metadata: list[dict] = []

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    def add(self, ids: list[str], vectors: np.ndarray, metadata: list[dict] | None = None) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors of shape (n, {self.dim}), got {vectors.shape}.")
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)
        self._vectors = np.vstack([self._vectors, vectors])
        self._ids.extend(ids)
        self._metadata.extend(metadata or [{} for _ in ids])

    def search(self, query: np.ndarray, k: int = 5, exclude: set[str] | None = None) -> list[SearchHit]:
        if not len(self):
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = self._vectors @ q
        order = np.argsort(-scores)
        hits = []
        for i in order:
            if exclude and self._ids[i] in exclude:
                continue
            hits.append(SearchHit(self._ids[i], float(scores[i]), self._metadata[i]))
            if len(hits) == k:
                break
        return hits

    def search_many(self, queries: np.ndarray, k: int = 5) -> list[list[str]]:
        """Batch search returning only ids (used in evaluation)."""
        q = np.asarray(queries, dtype=np.float32)
        q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
        top = np.argsort(-(q @ self._vectors.T), axis=1)[:, :k]
        return [[self._ids[i] for i in row] for row in top]

    # ---------- persistence ----------
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self._vectors)
        path.with_suffix(".json").write_text(
            json.dumps({"dim": self.dim, "ids": self._ids, "metadata": self._metadata}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> "VectorIndex":
        path = Path(path)
        info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        index = cls(info["dim"])
        index._vectors = np.load(path.with_suffix(".npy")).astype(np.float32)
        index._ids = info["ids"]
        index._metadata = info["metadata"]
        return index
