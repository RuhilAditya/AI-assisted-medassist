"""IU X-Ray (Indiana University Chest X-ray Collection, Open-i).

Each study = one frontal chest X-ray + its radiology report. Ground-truth
finding labels come from the MeSH terms that NLM indexers assigned to each
report, so labels are independent of anything our models produce.

Fields used and their roles:
  indication   -> clinical text input (why the X-ray was ordered)   [model input]
  findings,
  impression   -> the radiologist's report                          [retrieval corpus / reference ONLY]
  MeSH         -> ground-truth labels                               [evaluation ONLY]
Feeding findings/impression to a classifier would be label leakage, because
the MeSH labels were derived from them.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from medassist.config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "iu_xray"
STUDIES_FILE = DATA_DIR / "studies.jsonl"
IMAGES_DIR = DATA_DIR / "images"

# Our finding label -> MeSH main terms that count as positive.
FINDING_MESH = {
    "cardiomegaly": {"Cardiomegaly"},
    "lung opacity": {"Opacity", "Airspace Disease", "Infiltrate", "Consolidation", "Pneumonia", "Density"},
    "atelectasis": {"Pulmonary Atelectasis"},
    "pleural effusion": {"Pleural Effusion"},
    "granuloma": {"Calcified Granuloma", "Granuloma", "Granulomatous Disease"},
    "nodule": {"Nodule"},
    "emphysema": {"Emphysema", "Pulmonary Emphysema", "Pulmonary Disease, Chronic Obstructive"},
    "pulmonary edema": {"Pulmonary Edema", "Pulmonary Congestion"},
    "fracture": {"Fractures, Bone"},
}
FINDINGS = list(FINDING_MESH)
EXCLUDE_MESH = {"Technical Quality of Image Unsatisfactory", "No Indexing"}


@dataclass
class Study:
    uid: str
    image: str  # path relative to DATA_DIR
    indication: str
    findings: str
    impression: str
    mesh: list[str]
    labels: list[str]  # subset of FINDINGS, or ["normal"]
    split: str  # "train" (retrieval knowledge base / probe training) | "test"
    extra: dict = field(default_factory=dict)

    @property
    def report(self) -> str:
        parts = [p for p in (self.findings, self.impression) if p]
        return " ".join(parts)

    @property
    def is_normal(self) -> bool:
        return self.labels == ["normal"]

    def image_path(self) -> Path:
        return DATA_DIR / self.image


def mesh_terms(raw: str) -> list[str]:
    return sorted({t.split("/")[0].strip() for t in (raw or "").split(";") if t.strip()})


def labels_from_mesh(terms: list[str]) -> list[str] | None:
    """Return finding labels, ["normal"], or None if the study should be excluded."""
    if not terms or EXCLUDE_MESH & set(terms):
        return None
    if terms == ["normal"]:
        return ["normal"]
    labels = [name for name, mesh in FINDING_MESH.items() if mesh & set(terms)]
    return labels  # may be [] = abnormal but none of our target findings


def clean_text(text: str | float | None) -> str:
    """Reports are pre-anonymised with 'XXXX' tokens; drop them."""
    if not isinstance(text, str):
        return ""
    return " ".join(w for w in text.split() if "XXXX" not in w).strip()


def load_studies(split: str | None = None) -> list[Study]:
    if not STUDIES_FILE.exists():
        raise FileNotFoundError(
            f"{STUDIES_FILE} not found. Run: python -m scripts.prepare_iu_xray"
        )
    studies = [Study(**json.loads(line)) for line in STUDIES_FILE.read_text(encoding="utf-8").splitlines()]
    return [s for s in studies if split is None or s.split == split]


def save_studies(studies: list[Study]) -> None:
    STUDIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    STUDIES_FILE.write_text("\n".join(json.dumps(asdict(s)) for s in studies), encoding="utf-8")
