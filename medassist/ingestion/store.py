"""File-based case store. It only ever receives de-identified content.

Layout:
  <root>/<case_id>/case.json        case index (notes, images, timestamps)
  <root>/<case_id>/note_<n>.json    de-identified text + entities + de-ID report
  <root>/<case_id>/image_<n>.png    de-identified, normalised image
  <root>/audit.log                  JSON lines: who/what/when, never content

Original filenames are not stored, since they often contain patient names
(e.g. "john_doe_cxr.dcm"); items get generic names instead.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from medassist.config import PROJECT_ROOT

DEFAULT_ROOT = PROJECT_ROOT / "data" / "store"
_CASE_ID_RE = re.compile(r"^[0-9a-f]{12}$")


class CaseNotFoundError(KeyError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CaseStore:
    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ---------- internals ----------
    def _case_dir(self, case_id: str) -> Path:
        if not _CASE_ID_RE.match(case_id):  # also blocks path traversal
            raise CaseNotFoundError(case_id)
        path = self.root / case_id
        if not path.is_dir():
            raise CaseNotFoundError(case_id)
        return path

    def _read_index(self, case_id: str) -> dict:
        return json.loads((self._case_dir(case_id) / "case.json").read_text(encoding="utf-8"))

    def _write_index(self, case_id: str, index: dict) -> None:
        (self._case_dir(case_id) / "case.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    def audit(self, action: str, case_id: str, **details: object) -> None:
        entry = {"time": _now(), "action": action, "case_id": case_id, **details}
        with self._lock, (self.root / "audit.log").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    # ---------- public API ----------
    def create_case(self) -> str:
        case_id = uuid.uuid4().hex[:12]
        (self.root / case_id).mkdir()
        self._write_index(case_id, {"case_id": case_id, "created": _now(), "notes": [], "images": []})
        self.audit("create_case", case_id)
        return case_id

    def add_note(self, case_id: str, record: dict) -> str:
        with self._lock:
            index = self._read_index(case_id)
            note_id = f"note_{len(index['notes']) + 1}"
            path = self._case_dir(case_id) / f"{note_id}.json"
            path.write_text(json.dumps({"note_id": note_id, "added": _now(), **record}, indent=2), encoding="utf-8")
            index["notes"].append(note_id)
            self._write_index(case_id, index)
        self.audit("add_note", case_id, item=note_id, phi_removed=record.get("deid", {}).get("counts", {}))
        return note_id

    def add_image(self, case_id: str, png_bytes: bytes, record: dict) -> str:
        with self._lock:
            index = self._read_index(case_id)
            image_id = f"image_{len(index['images']) + 1}"
            case_dir = self._case_dir(case_id)
            (case_dir / f"{image_id}.png").write_bytes(png_bytes)
            (case_dir / f"{image_id}.json").write_text(
                json.dumps({"image_id": image_id, "added": _now(), **record}, indent=2), encoding="utf-8"
            )
            index["images"].append(image_id)
            self._write_index(case_id, index)
        self.audit("add_image", case_id, item=image_id)
        return image_id

    def get_case(self, case_id: str) -> dict:
        index = self._read_index(case_id)
        case_dir = self._case_dir(case_id)
        notes = [json.loads((case_dir / f"{n}.json").read_text(encoding="utf-8")) for n in index["notes"]]
        images = [json.loads((case_dir / f"{i}.json").read_text(encoding="utf-8")) for i in index["images"]]
        return {**index, "notes": notes, "images": images}

    def image_path(self, case_id: str, image_id: str) -> Path:
        if not re.fullmatch(r"image_\d+", image_id):
            raise CaseNotFoundError(image_id)
        path = self._case_dir(case_id) / f"{image_id}.png"
        if not path.is_file():
            raise CaseNotFoundError(image_id)
        return path
