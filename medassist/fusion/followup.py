"""Adaptive follow-up suggestions, driven by what is missing or uncertain.

Two kinds:
  requests     information the clinician could add to improve the assessment
  next_questions  useful questions to ask the assistant next (clickable in the UI)
Deterministic so they are always relevant to the actual gaps, never generic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from medassist.fusion import knowledge as kb
from medassist.fusion.conflicts import Conflict
from medassist.fusion.evidence import EvidenceBundle
from medassist.fusion.triage import TriageResult

RESPIRATORY = {"cough", "shortness of breath", "dyspnea", "breathlessness", "fever", "chest pain",
               "wheezing", "hemoptysis", "pleuritic chest pain"}


@dataclass(frozen=True)
class FollowUps:
    requests: list[str] = field(default_factory=list)
    next_questions: list[str] = field(default_factory=list)


def suggest(bundle: EvidenceBundle, conflicts: list[Conflict], triage: TriageResult, route: str) -> FollowUps:
    requests: list[str] = []
    questions: list[str] = []
    active = [t for t, _ in bundle.all_terms("active")]

    if not bundle.has_text:
        requests.append("Add a clinical note with the presenting symptoms, their duration and vital signs.")
    elif triage.news2_score is None:
        missing = [v for v in ("respiratory_rate", "spo2", "blood_pressure", "heart_rate", "temperature")
                   if v not in bundle.vitals]
        requests.append("Record the missing vital signs (" + ", ".join(m.replace("_", " ") for m in missing)
                        + ") so a NEWS2 score can be calculated.")
    if not bundle.has_image and any(kb.term_matches(t, RESPIRATORY) for t in active):
        requests.append("Respiratory symptoms are documented: is a chest X-ray available to upload?")

    for c in conflicts:
        if c.kind == "note_asserts_image_negative":
            requests.append(f"Confirm the X-ray is from the same encounter as the note that mentions {c.finding}.")
        elif c.kind == "image_positive_note_denies":
            requests.append(f"Check the formal radiology report: does it mention {c.finding}?")
        elif c.kind == "modality_mismatch":
            requests.append("Upload the chest X-ray for this case (the current image is a different modality).")

    if bundle.fused:
        top = max(bundle.fused, key=lambda f: f.fused_probability or 0)
        if top.image_positive:
            questions.append(f"Show me where the model sees {top.finding}.")
        questions.append("What is the differential diagnosis given the note and the X-ray?")
    if bundle.has_text and route != "triage":
        questions.append("How urgent is this case, and why?")
    if conflicts and route != "multimodal":
        questions.append("Explain the disagreement between the note and the image.")
    if bundle.has_text and route == "triage":
        questions.append("Summarise the key findings for a handover.")

    return FollowUps(requests=_unique(requests)[:3], next_questions=_unique(questions)[:3])


def _unique(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
