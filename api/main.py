"""FastAPI backend.

Phase 0: /health, /status
Phase 1: case creation and de-identified ingestion of notes and images.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from medassist.config import get_settings
from medassist.ingestion.loaders import DataValidationError
from medassist.ingestion.pipeline import ingest_image, ingest_note
from medassist.ingestion.store import CaseNotFoundError, CaseStore
from medassist.llm.ollama_client import OllamaClient

app = FastAPI(
    title="MedAssist API",
    description="Multimodal medical decision-support API (research prototype).",
    version="0.2.0",
)


@lru_cache(maxsize=1)
def get_store() -> CaseStore:
    return CaseStore()


def _not_found(case_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict[str, object]:
    """Report which backing services/models are configured and reachable."""
    settings = get_settings()
    return {
        "device": settings.device,
        "llm_model": settings.llm_model,
        "ollama_reachable": OllamaClient(settings).is_available(),
        "biomedclip_model": settings.biomedclip_model,
        "embedding_model": settings.embedding_model,
    }


@app.post("/cases", status_code=201)
def create_case() -> dict[str, str]:
    return {"case_id": get_store().create_case()}


@app.get("/cases/{case_id}")
def get_case(case_id: str) -> dict:
    try:
        return get_store().get_case(case_id)
    except CaseNotFoundError:
        raise _not_found(case_id) from None


@app.post("/cases/{case_id}/notes", status_code=201)
async def add_note(
    case_id: str,
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
) -> dict:
    """Upload a TXT/PDF note or paste text. Stored only after de-identification."""
    if (file is None) == (not text):
        raise HTTPException(status_code=422, detail="Provide exactly one of 'file' or 'text'.")
    store = get_store()
    try:
        store.get_case(case_id)
        if file is not None:
            return ingest_note(store, case_id, filename=file.filename or "note.txt", content=await file.read())
        return ingest_note(store, case_id, text=text)
    except CaseNotFoundError:
        raise _not_found(case_id) from None
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/cases/{case_id}/images", status_code=201)
async def add_image(case_id: str, file: UploadFile = File(...)) -> dict:
    """Upload DICOM/PNG/JPEG. DICOM tags are anonymised before anything is stored."""
    store = get_store()
    try:
        store.get_case(case_id)
        return ingest_image(store, case_id, file.filename or "image", await file.read())
    except CaseNotFoundError:
        raise _not_found(case_id) from None
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/cases/{case_id}/images/{image_id}")
def get_image(case_id: str, image_id: str) -> FileResponse:
    try:
        return FileResponse(get_store().image_path(case_id, image_id), media_type="image/png")
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found.") from None
