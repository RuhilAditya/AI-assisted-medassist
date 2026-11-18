"""Triage guidance: deterministic, explainable, never delegated to the LLM.

1. NEWS2 (National Early Warning Score 2, Royal College of Physicians, 2017)
   from the vitals found in the notes. Implemented for room air (scale 1); the
   supplemental-oxygen item is unknown from free text and scored 0.
2. Red-flag rules for time-critical presentations that NEWS2 can miss
   (stroke signs, chest pain with breathlessness, haemoptysis, syncope).
3. Image escalation: a positive acute CXR finding together with hypoxia.

Levels: EMERGENCY (immediate) > URGENT (prompt clinician review) >
SOON (same-day review) > ROUTINE. The highest level triggered wins.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from medassist.fusion import knowledge as kb
from medassist.fusion.evidence import EvidenceBundle

LEVELS = ["ROUTINE", "SOON", "URGENT", "EMERGENCY"]
ACUTE_CXR_FINDINGS = {"lung opacity", "pleural effusion", "pulmonary edema"}

RED_FLAGS: list[tuple[str, set[str], str]] = [
    ("EMERGENCY", {"facial droop", "slurred speech"}, "possible stroke (time-critical)"),
    ("EMERGENCY", {"loss of consciousness", "unconscious", "unresponsive"}, "reduced consciousness"),
    ("URGENT", {"hemoptysis"}, "haemoptysis"),
    ("URGENT", {"syncope"}, "syncope"),
    ("URGENT", {"confusion"}, "new confusion"),
]
CHEST_PAIN = {"chest pain", "pleuritic chest pain"}
BREATHLESS = {"shortness of breath", "dyspnea", "breathlessness"}


@dataclass(frozen=True)
class TriageResult:
    level: str
    news2_score: int | None
    news2_components: dict[str, int] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


def _band(value: float, bands: list[tuple[float, float, int]]) -> int:
    for low, high, points in bands:
        if low <= value <= high:
            return points
    return 0


def news2(vitals: dict[str, str], confused: bool) -> tuple[int | None, dict[str, int]]:
    comps: dict[str, int] = {}
    rr = kb._num(vitals.get("respiratory_rate"))
    if rr is not None:
        comps["respiratory_rate"] = _band(rr, [(0, 8, 3), (9, 11, 1), (12, 20, 0), (21, 24, 2), (25, 99, 3)])
    spo2 = kb._num(vitals.get("spo2"))
    if spo2 is not None:
        comps["spo2"] = _band(spo2, [(0, 91, 3), (92, 93, 2), (94, 95, 1), (96, 100, 0)])
    bp = vitals.get("blood_pressure")
    if bp and "/" in bp:
        sbp = kb._num(bp.split("/")[0])
        if sbp is not None:
            comps["systolic_bp"] = _band(sbp, [(0, 90, 3), (91, 100, 2), (101, 110, 1), (111, 219, 0), (220, 400, 3)])
    hr = kb._num(vitals.get("heart_rate"))
    if hr is not None:
        comps["heart_rate"] = _band(hr, [(0, 40, 3), (41, 50, 1), (51, 90, 0), (91, 110, 1), (111, 130, 2), (131, 300, 3)])
    temp_raw = vitals.get("temperature")
    if temp_raw:
        t = kb._num(temp_raw.split()[0])
        if t is not None:
            if temp_raw.endswith("F"):
                t = (t - 32) * 5 / 9
            comps["temperature"] = _band(round(t, 1), [(0, 35.0, 3), (35.1, 36.0, 1), (36.1, 38.0, 0), (38.1, 39.0, 1), (39.1, 45, 2)])
    if confused:
        comps["consciousness"] = 3
    if len(comps) < 3:  # too few observations for a meaningful score
        return None, comps
    return sum(comps.values()), comps


def assess(bundle: EvidenceBundle) -> TriageResult:
    active = {t for t, _ in bundle.all_terms("active")}
    level, reasons, limitations = "ROUTINE", [], []

    def raise_to(new: str, reason: str) -> None:
        nonlocal level
        reasons.append(f"{new}: {reason}")
        if LEVELS.index(new) > LEVELS.index(level):
            level = new

    confused = any(kb.term_matches(t, {"confusion", "altered mental status"}) for t in active)
    score, comps = news2(bundle.vitals, confused)
    if score is None:
        limitations.append("Fewer than 3 vital signs recorded, so NEWS2 could not be calculated.")
    else:
        if score >= 7:
            raise_to("EMERGENCY", f"NEWS2 {score} (>=7, high clinical risk)")
        elif score >= 5:
            raise_to("URGENT", f"NEWS2 {score} (5-6, medium clinical risk)")
        elif 3 in comps.values():
            worst = [k for k, v in comps.items() if v == 3]
            raise_to("URGENT", f"NEWS2 {score} with a single extreme parameter ({', '.join(worst)})")
        elif score >= 1:
            raise_to("SOON", f"NEWS2 {score} (1-4, low clinical risk)")

    for flag_level, terms, label in RED_FLAGS:
        if any(kb.term_matches(t, terms) for t in active):
            raise_to(flag_level, f"red flag: {label}")
    if any(kb.term_matches(t, CHEST_PAIN) for t in active) and any(kb.term_matches(t, BREATHLESS) for t in active):
        raise_to("URGENT", "red flag: chest pain with breathlessness")

    spo2 = kb._num(bundle.vitals.get("spo2"))
    acute = [f for f in bundle.fused if f.finding in ACUTE_CXR_FINDINGS and f.image_positive]
    if acute and spo2 is not None and spo2 < 94:
        names = ", ".join(f.finding for f in acute)
        raise_to("URGENT", f"image suggests {names} with SpO2 {spo2:.0f}%")

    if not bundle.has_text:
        limitations.append("No clinical note: triage is based on the image alone and is unreliable.")
    return TriageResult(level, score, comps, reasons, limitations)
