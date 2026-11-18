"""Clinical text embeddings via Sentence Transformers (BioBERT family)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from medassist.config import get_settings


class ClinicalEmbedder:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import

        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name, device=device or settings.device)

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return L2-normalised embeddings, so dot product == cosine similarity."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


@lru_cache(maxsize=1)
def get_embedder() -> ClinicalEmbedder:
    return ClinicalEmbedder()
