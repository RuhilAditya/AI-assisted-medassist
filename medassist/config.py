"""Central configuration, read from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Keep Hugging Face downloads inside the project (git-ignored) unless overridden.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models_cache"))


@dataclass(frozen=True)
class Settings:
    ollama_url: str
    llm_model: str
    llm_timeout: float
    biomedclip_model: str
    embedding_model: str
    device: str


def _resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    timeout = float(os.getenv("MEDASSIST_LLM_TIMEOUT", "180"))
    if timeout <= 0:
        raise ValueError("MEDASSIST_LLM_TIMEOUT must be greater than zero.")
    return Settings(
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        llm_model=os.getenv("MEDASSIST_LLM_MODEL", "llama3.1:8b"),
        llm_timeout=timeout,
        biomedclip_model=os.getenv(
            "MEDASSIST_BIOMEDCLIP_MODEL",
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        ),
        embedding_model=os.getenv(
            "MEDASSIST_EMBEDDING_MODEL",
            "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        ),
        device=_resolve_device(os.getenv("MEDASSIST_DEVICE", "auto")),
    )
