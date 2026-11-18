"""FastAPI backend.

Phase 0: /health, /status
Phase 1: case creation and de-identified ingestion of notes and images.
Phase 2: image analysis (findings, similar cases, heatmaps) and RAG over notes.
"""
from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

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


class NewCase(BaseModel):
    title: str | None = Field(default=None, max_length=120)


@app.post("/cases", status_code=201)
def create_case(body: NewCase | None = None) -> dict[str, str]:
    return {"case_id": get_store().create_case(title=body.title if body else None)}


@app.get("/cases")
def list_cases() -> list[dict]:
    return get_store().list_cases()


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


# ---------------- Phase 2: unimodal analysis ----------------
def _open_case_image(case_id: str, image_id: str) -> Image.Image:
    try:
        return Image.open(get_store().image_path(case_id, image_id))
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found.") from None


@app.post("/cases/{case_id}/images/{image_id}/analyze")
def analyze_case_image(case_id: str, image_id: str, k_similar: int = 3) -> dict:
    """Modality check, chest X-ray finding probabilities, and similar prior cases."""
    from medassist.analysis import analyze_image

    return analyze_image(_open_case_image(case_id, image_id), k_similar=k_similar)


@app.get("/cases/{case_id}/images/{image_id}/heatmap")
def case_image_heatmap(case_id: str, image_id: str, finding: str) -> Response:
    """Occlusion-sensitivity heatmap for one finding, as a PNG overlay."""
    from medassist.analysis import heatmap_image

    try:
        overlay = heatmap_image(_open_case_image(case_id, image_id), finding)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=4, ge=1, le=10)


@app.post("/cases/{case_id}/ask")
def ask_case(case_id: str, body: Question) -> dict:
    """RAG question answering over the case's de-identified notes, with citations."""
    from medassist.analysis import ask_case_notes
    from medassist.llm.ollama_client import LLMError

    try:
        case = get_store().get_case(case_id)
    except CaseNotFoundError:
        raise _not_found(case_id) from None
    try:
        return ask_case_notes(case, body.question, k=body.k)
    except (LLMError, ConnectionError) as exc:
        raise HTTPException(status_code=502, detail=f"LLM unavailable: {exc}") from exc


# ---------------- Phase 3: multimodal assistant ----------------
class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    image_id: str | None = None  # default: the case's most recent image


@app.post("/cases/{case_id}/chat")
def chat(case_id: str, body: ChatRequest) -> dict:
    """Full multimodal answer: routed, evidence-cited, with fused findings,
    cross-modal conflicts, NEWS2 triage and follow-up suggestions."""
    from medassist.fusion.assistant import MedAssistant

    try:
        return MedAssistant(get_store()).chat(case_id, body.question, image_id=body.image_id)
    except CaseNotFoundError:
        raise _not_found(case_id) from None


@app.get("/cases/{case_id}/overview")
def case_overview(case_id: str) -> dict:
    """Findings, conflicts and triage for a case without asking a question (~1 s)."""
    from medassist.fusion.assistant import MedAssistant

    try:
        return MedAssistant(get_store()).overview(case_id)
    except CaseNotFoundError:
        raise _not_found(case_id) from None


@app.post("/cases/{case_id}/chat/stream")
def chat_stream(case_id: str, body: ChatRequest) -> StreamingResponse:
    """Server-sent events: `evidence` (findings, conflicts, triage: ~1 s), then `token`
    chunks of the written answer as the LLM produces them, then `done` (full response)."""
    from medassist.fusion.assistant import MedAssistant

    try:
        get_store().get_case(case_id)
    except CaseNotFoundError:
        raise _not_found(case_id) from None

    def events():
        try:
            for name, payload in MedAssistant(get_store()).chat_stream(case_id, body.question, body.image_id):
                data = payload if name != "token" else {"text": payload}
                yield f"event: {name}\ndata: {json.dumps(data)}\n\n"
        except Exception as exc:  # surface failures to the client instead of a dropped stream
            yield f"event: error\ndata: {json.dumps({'detail': f'{type(exc).__name__}: {exc}'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/cases/{case_id}/chat")
def chat_history(case_id: str) -> list[dict]:
    try:
        return get_store().get_chat(case_id)
    except CaseNotFoundError:
        raise _not_found(case_id) from None


@app.get("/cases/{case_id}/notes/{note_id}")
def get_note(case_id: str, note_id: str) -> dict:
    try:
        case = get_store().get_case(case_id)
    except CaseNotFoundError:
        raise _not_found(case_id) from None
    note = next((n for n in case["notes"] if n["note_id"] == note_id), None)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found.")
    return note


# ---------------- Phase 4: serve the built React app ----------------
UI_DIST = Path(__file__).resolve().parent.parent / "ui" / "dist"
if UI_DIST.is_dir():  # after `npm run build`; in development Vite serves the UI instead
    app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")
