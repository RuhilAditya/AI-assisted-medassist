"""Clinical knowledge linking note concepts to chest X-ray findings.

Used for three things:
  * fused scores    - a note that asserts or supports a finding shifts the
                      image probability (a log-odds update, see evidence.py)
  * conflicts       - a note that asserts/denies a finding the image disagrees with
  * explanations    - which note terms were used as evidence for which finding

Weights are hand-set log-likelihood ratios (LLR), chosen to be modest and
explainable, NOT learned from data:
  direct assertion  +2.0  (odds x7.4)   e.g. note says "pleural effusion"
  direct negation   -1.5  (odds /4.5)   e.g. note says "no pleural effusion"
  supporting sign   +0.4 each, capped at +1.2 per finding
They are documented as a limitation; a learned model would need paired
notes + images, which public data does not provide at useful scale.
"""
from __future__ import annotations

# Chronic findings persist, so a documented history of their cause counts as an
# assertion ("history of CHF" -> cardiomegaly). For acute findings, history is
# only weak support (a past pneumonia says little about today's film).
CHRONIC_FINDINGS = {"cardiomegaly", "granuloma", "nodule", "emphysema"}

DIRECT_ASSERT_LLR = 2.0
DIRECT_NEGATE_LLR = -1.5
SUPPORT_LLR = 0.4
SUPPORT_CAP = 1.2

# Finding -> canonical note terms that *name* the finding (or its usual cause).
FINDING_TERMS: dict[str, set[str]] = {
    "cardiomegaly": {"cardiomegaly", "enlarged heart", "congestive heart failure", "heart failure",
                     "cardiomyopathy", "chf"},
    "lung opacity": {"pneumonia", "consolidation", "infiltrate", "lung opacity", "airspace disease",
                     "community-acquired pneumonia", "lobar pneumonia", "aspiration pneumonia"},
    "atelectasis": {"atelectasis", "lung collapse", "collapse"},
    "pleural effusion": {"pleural effusion", "effusion", "pleural effusions", "hydrothorax"},
    "granuloma": {"granuloma", "calcified granuloma", "granulomatous disease", "old tuberculosis",
                  "healed tuberculosis", "histoplasmosis"},
    "nodule": {"nodule", "pulmonary nodule", "lung nodule", "lung mass"},
    "emphysema": {"emphysema", "chronic obstructive pulmonary disease", "copd", "hyperinflation"},
    "pulmonary edema": {"pulmonary edema", "pulmonary oedema", "fluid overload", "pulmonary congestion",
                        "acute decompensated heart failure"},
    "fracture": {"fracture", "rib fracture", "fractured rib", "fractures"},
}

# Finding -> symptoms/signs that make it more likely (weak evidence).
SUPPORTING_TERMS: dict[str, set[str]] = {
    "cardiomegaly": {"orthopnea", "leg swelling", "edema", "dyspnea", "shortness of breath",
                     "hypertension", "atrial fibrillation"},
    "lung opacity": {"fever", "productive cough", "cough", "chills", "pleuritic chest pain",
                     "tachypnea", "hypoxia", "crackles"},
    "atelectasis": {"shortness of breath", "hypoxia", "post-operative", "immobility"},
    "pleural effusion": {"shortness of breath", "dyspnea", "pleuritic chest pain", "congestive heart failure",
                         "decreased breath sounds"},
    "granuloma": {"tuberculosis"},
    "nodule": {"weight loss", "hemoptysis", "smoking", "night sweats", "lung cancer"},
    "emphysema": {"wheezing", "breathlessness", "shortness of breath", "smoking", "chronic cough"},
    "pulmonary edema": {"orthopnea", "dyspnea", "shortness of breath", "leg swelling", "crackles",
                        "congestive heart failure"},
    "fracture": {"trauma", "fall", "chest wall pain", "rib pain"},
}

# Vital-sign evidence: (finding, LLR, human-readable reason).
def vital_signals(vitals: dict[str, str]) -> list[tuple[str, float, str]]:
    signals: list[tuple[str, float, str]] = []
    spo2 = _num(vitals.get("spo2"))
    temp = _num((vitals.get("temperature") or "").split()[0] if vitals.get("temperature") else None)
    if temp is not None and (vitals.get("temperature", "").endswith("F")):
        temp = (temp - 32) * 5 / 9
    if spo2 is not None and spo2 < 92:
        for finding in ("lung opacity", "pulmonary edema", "pleural effusion"):
            signals.append((finding, SUPPORT_LLR, f"SpO2 {spo2:.0f}%"))
    if temp is not None and temp >= 38.0:
        signals.append(("lung opacity", SUPPORT_LLR, f"temperature {temp:.1f} C"))
    return signals


def _num(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def term_matches(term: str, vocabulary: set[str]) -> bool:
    """Match canonical note terms against a vocabulary, tolerating modifiers
    ("bilateral pleural effusions" matches "pleural effusion")."""
    term = term.lower().strip()
    return any(v == term or v in term or (len(term) > 4 and term in v) for v in vocabulary)
