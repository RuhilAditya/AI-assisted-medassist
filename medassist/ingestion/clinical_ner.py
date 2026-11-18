"""Clinical entity extraction, normalisation and context (negation etc.).

Pipeline (spaCy):
  scispaCy `en_ner_bc5cdr_md`   -> DISEASE / CHEMICAL entities (trained on BC5CDR)
  medspaCy TargetMatcher         -> symptoms, tests and procedures the model misses
  medspaCy ConText               -> negated / historical / hypothetical / family / uncertain
  medspaCy Sectionizer           -> which note section the entity is in
Plus regex vital-sign extraction (feeds triage in Phase 3).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

LABEL_MAP = {"DISEASE": "PROBLEM", "CHEMICAL": "MEDICATION"}
# Form labels / placeholders the model occasionally tags as entities.
HEADER_NOISE = {"dob", "mrn", "name", "patient", "pt", "uhid", "person", "date", "id"}

# Common abbreviations -> canonical form (normalisation step).
ABBREVIATIONS = {
    "sob": "shortness of breath", "dyspnoea": "dyspnea", "cp": "chest pain",
    "htn": "hypertension", "dm": "diabetes mellitus", "t2dm": "type 2 diabetes mellitus",
    "copd": "chronic obstructive pulmonary disease", "mi": "myocardial infarction",
    "cad": "coronary artery disease", "chf": "congestive heart failure",
    "uti": "urinary tract infection", "tb": "tuberculosis", "pe": "pulmonary embolism",
    "dvt": "deep vein thrombosis", "cva": "stroke", "afib": "atrial fibrillation",
    "af": "atrial fibrillation", "ckd": "chronic kidney disease", "cxr": "chest x-ray",
    "ecg": "electrocardiogram", "ekg": "electrocardiogram", "cbc": "complete blood count",
    "crp": "c-reactive protein", "abg": "arterial blood gas", "loc": "loss of consciousness",
    "n/v": "nausea and vomiting", "ha": "headache", "abd pain": "abdominal pain",
}

SYMPTOM_TERMS = [
    "shortness of breath", "sob", "dyspnea", "dyspnoea", "breathlessness", "cough",
    "productive cough", "dry cough", "fever", "chills", "night sweats", "chest pain",
    "cp", "pleuritic chest pain", "wheezing", "hemoptysis", "fatigue", "weight loss",
    "headache", "dizziness", "syncope", "palpitations", "nausea", "vomiting",
    "abdominal pain", "diarrhea", "confusion", "loss of consciousness", "loc",
    "weakness", "slurred speech", "facial droop", "edema", "leg swelling", "tachypnea",
    "tachycardia", "hypoxia", "sore throat", "rash", "back pain", "joint pain",
]
TEST_TERMS = [
    "chest x-ray", "cxr", "ct scan", "ct chest", "mri", "ultrasound", "ecg", "ekg",
    "electrocardiogram", "cbc", "complete blood count", "crp", "procalcitonin",
    "troponin", "d-dimer", "abg", "blood culture", "sputum culture", "urinalysis",
    "echocardiogram", "spirometry", "covid-19 pcr", "rt-pcr", "hba1c", "lipid panel",
]
PROCEDURE_TERMS = ["intubation", "bronchoscopy", "thoracentesis", "biopsy", "nebulization", "oxygen therapy"]

VITAL_PATTERNS = {
    "blood_pressure": r"\b(?:bp|blood pressure)\s*[:=]?\s*(\d{2,3}\s*/\s*\d{2,3})",
    "heart_rate": r"\b(?:hr|heart rate|pulse)\s*[:=]?\s*(\d{2,3})\b",
    "respiratory_rate": r"\b(?:rr|resp(?:iratory)? rate)\s*[:=]?\s*(\d{1,2})\b",
    "spo2": r"\b(?:spo2|sp02|o2 sat(?:uration)?|sats?)\s*[:=]?\s*(\d{2,3})\s*%?",
    "temperature": r"\b(?:temp(?:erature)?|t)\s*[:=]?\s*(\d{2,3}(?:\.\d)?)\s*°?\s*([cf])?\b",
}


# Context cues the scispaCy model sometimes swallows into an entity
# ("Denies chest pain" tagged as one DISEASE). Trimmed so ConText can see them.
LEADING_CUE_WORDS = {
    "denies", "denied", "deny", "no", "not", "without", "negative", "for", "free", "of",
    "absent", "absence", "history", "hx", "possible", "probable", "suspected", "rule", "out",
    "r/o", "likely", "the", "a", "an",
}


def _register_trim_component() -> str:
    from spacy.language import Language
    from spacy.tokens import Span

    name = "medassist_trim_entity_cues"
    if name in Language.factories:
        return name

    @Language.component(name)
    def trim_entity_cues(doc):
        trimmed = []
        for ent in doc.ents:
            start = ent.start
            while start < ent.end - 1 and (doc[start].lower_ in LEADING_CUE_WORDS or doc[start].is_punct):
                start += 1
            trimmed.append(Span(doc, start, ent.end, label=ent.label_))
        doc.ents = trimmed  # trimming only shrinks spans, so no new overlaps
        return doc

    return name


@dataclass(frozen=True)
class ClinicalEntity:
    text: str
    canonical: str
    label: str  # PROBLEM | MEDICATION | TEST | PROCEDURE
    start: int
    end: int
    negated: bool = False
    historical: bool = False
    hypothetical: bool = False
    family: bool = False
    uncertain: bool = False
    section: str | None = None

    @property
    def is_active(self) -> bool:
        """Present, current, about this patient: what reasoning should rely on."""
        return not (self.negated or self.historical or self.hypothetical or self.family)


@dataclass(frozen=True)
class ClinicalExtraction:
    entities: list[ClinicalEntity] = field(default_factory=list)
    vitals: dict[str, str] = field(default_factory=dict)

    def active(self, label: str | None = None) -> list[str]:
        return sorted({e.canonical for e in self.entities if e.is_active and (label is None or e.label == label)})

    def negated(self) -> list[str]:
        return sorted({e.canonical for e in self.entities if e.negated})

    def past_history(self) -> list[str]:
        """Patient's own historical conditions (often still-relevant comorbidities)."""
        return sorted({
            e.canonical for e in self.entities
            if e.historical and not (e.negated or e.family or e.hypothetical)
        })

    def family_history(self) -> list[str]:
        return sorted({e.canonical for e in self.entities if e.family and not e.negated})

    def to_dict(self) -> dict:
        return {"entities": [asdict(e) for e in self.entities], "vitals": self.vitals}


def normalise(term: str) -> str:
    key = re.sub(r"\s+", " ", term.lower().strip(" .,:;"))
    return ABBREVIATIONS.get(key, key)


def extract_vitals(text: str) -> dict[str, str]:
    vitals: dict[str, str] = {}
    lowered = text.lower()
    for name, pattern in VITAL_PATTERNS.items():
        match = re.search(pattern, lowered)
        if not match:
            continue
        value = match.group(1).replace(" ", "")
        if name == "temperature":
            unit = (match.group(2) or ("f" if float(value) > 45 else "c")).upper()
            value = f"{value} {unit}"
        vitals[name] = value
    return vitals


class ClinicalNER:
    def __init__(self, model: str = "en_ner_bc5cdr_md") -> None:
        import medspacy  # noqa: F401  (registers the medspacy_* factories)
        import spacy
        from medspacy.ner import TargetRule

        self.nlp = spacy.load(model)
        self.nlp.add_pipe(_register_trim_component(), after="ner")
        matcher = self.nlp.add_pipe("medspacy_target_matcher")
        rules = (
            [TargetRule(t, "PROBLEM") for t in SYMPTOM_TERMS]
            + [TargetRule(t, "TEST") for t in TEST_TERMS]
            + [TargetRule(t, "PROCEDURE") for t in PROCEDURE_TERMS]
        )
        matcher.add(rules)
        self.nlp.add_pipe("medspacy_context")
        self.nlp.add_pipe("medspacy_sectionizer")

    def extract(self, text: str) -> ClinicalExtraction:
        if not text or not text.strip():
            return ClinicalExtraction()
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.text.lower().strip(" :.") in HEADER_NOISE:
                continue
            label = LABEL_MAP.get(ent.label_, ent.label_)
            section = getattr(ent._, "section_category", None)
            entities.append(ClinicalEntity(
                text=ent.text,
                canonical=normalise(ent.text),
                label=label,
                start=ent.start_char,
                end=ent.end_char,
                negated=bool(ent._.is_negated),
                historical=bool(ent._.is_historical),
                hypothetical=bool(ent._.is_hypothetical),
                family=bool(ent._.is_family),
                uncertain=bool(ent._.is_uncertain),
                section=section,
            ))
        return ClinicalExtraction(entities=entities, vitals=extract_vitals(text))


@lru_cache(maxsize=1)
def get_ner() -> ClinicalNER:
    return ClinicalNER()
