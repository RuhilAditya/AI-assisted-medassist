"""Save/load evaluation results so every number in the report is reproducible."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

from medassist.config import PROJECT_ROOT, get_settings

RESULTS_DIR = PROJECT_ROOT / "results"


def save_results(name: str, payload: dict) -> str:
    RESULTS_DIR.mkdir(exist_ok=True)
    settings = get_settings()
    record = {
        "experiment": name,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "device": settings.device,
            "llm_model": settings.llm_model,
            "biomedclip_model": settings.biomedclip_model,
            "embedding_model": settings.embedding_model,
            "python": platform.python_version(),
        },
        **payload,
    }
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return str(path.relative_to(PROJECT_ROOT))


def load_results(name: str) -> dict:
    return json.loads((RESULTS_DIR / f"{name}.json").read_text(encoding="utf-8"))
