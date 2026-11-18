"""Ingestion pipeline: raw upload -> de-identify -> structure -> store.

Raw uploads exist only in memory; only de-identified output is persisted.
"""
from __future__ import annotations

import io
from dataclasses import asdict

from PIL import Image

from medassist.ingestion.clinical_ner import ClinicalNER, get_ner
from medassist.ingestion.deid import TextDeidentifier, get_deidentifier
from medassist.ingestion.dicom_anon import anonymize_dicom
from medassist.ingestion.loaders import DICOM_SUFFIXES, load_upload
from medassist.ingestion.store import CaseStore
from medassist.ingestion.text_loader import load_text_upload


def ingest_note(
    store: CaseStore,
    case_id: str,
    filename: str | None = None,
    content: bytes | None = None,
    text: str | None = None,
    deidentifier: TextDeidentifier | None = None,
    ner: ClinicalNER | None = None,
) -> dict:
    """Accept either an uploaded file (filename + content) or pasted text."""
    raw = text if text is not None else load_text_upload(filename or "note.txt", content or b"")
    deid = (deidentifier or get_deidentifier()).deidentify(raw)
    # NER runs on the de-identified text, so stored offsets match stored text.
    extraction = (ner or get_ner()).extract(deid.text)
    record = {
        "source_type": "upload" if text is None else "pasted",
        "text": deid.text,
        "deid": {"counts": deid.counts, "findings": [asdict(f) for f in deid.findings]},
        "extraction": extraction.to_dict(),
        "summary": {
            "active_problems": extraction.active("PROBLEM"),
            "negated_findings": extraction.negated(),
            "past_history": extraction.past_history(),
            "family_history": extraction.family_history(),
            "medications": extraction.active("MEDICATION"),
            "tests": extraction.active("TEST"),
            "vitals": extraction.vitals,
        },
    }
    record["note_id"] = store.add_note(case_id, record)
    return record


def ingest_image(store: CaseStore, case_id: str, filename: str, content: bytes) -> dict:
    deid_report = None
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix in DICOM_SUFFIXES or content[128:132] == b"DICM":
        content, report = anonymize_dicom(content)
        deid_report = asdict(report)

    loaded = load_upload(filename, content)
    buffer = io.BytesIO()
    # Colour images (histopathology, photos) keep their colour; radiology is grayscale.
    Image.fromarray(loaded.color if loaded.is_color else loaded.pixels).save(buffer, format="PNG")

    record = {
        "source_format": loaded.source_format,
        "color": loaded.is_color,
        "shape": list(loaded.pixels.shape),
        "metadata": loaded.metadata,
        "deid": deid_report,
        "warnings": (deid_report or {}).get("warnings", []),
    }
    record["image_id"] = store.add_image(case_id, buffer.getvalue(), record)
    return record
