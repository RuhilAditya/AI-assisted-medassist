"""Minimal Ollama HTTP client (no extra dependency, easy to mock in tests)."""
from __future__ import annotations

from typing import Any

import requests

from medassist.config import Settings, get_settings


class LLMError(RuntimeError):
    """Raised when the local LLM cannot produce a response."""


class OllamaClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def is_available(self) -> bool:
        try:
            requests.get(f"{self.settings.ollama_url}/api/tags", timeout=3).raise_for_status()
            return True
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        response = requests.get(f"{self.settings.ollama_url}/api/tags", timeout=5)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        json_mode: bool = False,
        seed: int | None = 0,
    ) -> str:
        options: dict[str, Any] = {"temperature": temperature, "num_ctx": self.settings.llm_num_ctx}
        if seed is not None:
            options["seed"] = seed  # reproducible evaluation runs
        payload: dict[str, Any] = {
            "model": model or self.settings.llm_model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"
        try:
            response = requests.post(
                f"{self.settings.ollama_url}/api/chat",
                json=payload,
                timeout=self.settings.llm_timeout,
            )
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            raise LLMError(
                f"Could not reach Ollama at {self.settings.ollama_url}. Is it running, "
                f"and is '{payload['model']}' pulled?"
            ) from exc
        except ValueError as exc:
            raise LLMError("Ollama returned invalid JSON.") from exc

        content = (body.get("message") or {}).get("content", "")
        if not content.strip():
            raise LLMError("Ollama returned an empty response.")
        return content.strip()
