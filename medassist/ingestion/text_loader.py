"""Extract plain text from uploaded clinical documents (TXT / PDF)."""
from __future__ import annotations

import io

from medassist.ingestion.loaders import DataValidationError

MAX_TEXT_BYTES = 10 * 1024 * 1024
TEXT_SUFFIXES = {"txt", "md"}
PDF_SUFFIXES = {"pdf"}


def load_text_upload(name: str, content: bytes) -> str:
    if not content:
        raise DataValidationError("The uploaded document is empty.")
    if len(content) > MAX_TEXT_BYTES:
        raise DataValidationError("Documents must be 10 MB or smaller.")
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""

    if suffix in TEXT_SUFFIXES:
        for encoding in ("utf-8-sig", "utf-16", "latin-1"):
            try:
                return content.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        raise DataValidationError(f"Could not decode '{name}' as text.")

    if suffix in PDF_SUFFIXES:
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except Exception as exc:
            raise DataValidationError(f"Unable to read PDF '{name}'.") from exc
        if not text:
            raise DataValidationError(
                f"'{name}' has no extractable text (it may be a scanned image; OCR is not supported)."
            )
        return text

    raise DataValidationError("Only TXT and PDF documents are supported.")
