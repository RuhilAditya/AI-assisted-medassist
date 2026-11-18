"""De-identification of clinical free text (Microsoft Presidio + custom rules).

Follows the spirit of HIPAA Safe Harbor: names, contact details, locations,
IDs and dates (except year) are replaced with typed placeholders such as
[PERSON] or [DATE]. The original PHI values are never logged or returned:
the report only records *what type* was removed and *where*.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

# Eponymous medical terms that NER models often mistake for person names.
MEDICAL_ALLOW_LIST = [
    "Parkinson", "Parkinson's", "Alzheimer", "Alzheimer's", "Crohn", "Crohn's",
    "Hodgkin", "Hodgkin's", "Graves", "Cushing", "Addison", "Down", "Bell's",
    "Babinski", "Murphy", "Kernig", "Brudzinski", "Homan", "Glasgow", "Apgar",
    "Kerley", "Foley", "Swan-Ganz", "Gram", "Doppler", "Holter", "Wells",
    "Tylenol", "Lasix", "Coumadin", "Xarelto", "Ventolin",
]

ENTITY_PLACEHOLDERS = {
    "PERSON": "[PERSON]",
    "PHONE_NUMBER": "[PHONE]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "LOCATION": "[LOCATION]",
    "DATE_TIME": "[DATE]",
    "US_SSN": "[ID]",
    "MEDICAL_RECORD_NUMBER": "[MRN]",
    "IN_AADHAAR": "[ID]",
    "URL": "[URL]",
    "IP_ADDRESS": "[IP]",
    "MEDICAL_LICENSE": "[ID]",
    "AGE_OVER_89": "[AGE>89]",
}


@dataclass(frozen=True)
class PHIFinding:
    entity_type: str
    start: int
    end: int
    score: float


@dataclass(frozen=True)
class DeidResult:
    text: str
    findings: list[PHIFinding] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(f.entity_type for f in self.findings))


def _custom_recognizers():
    from presidio_analyzer import Pattern, PatternRecognizer

    mrn = PatternRecognizer(
        supported_entity="MEDICAL_RECORD_NUMBER",
        patterns=[
            Pattern("mrn_labelled", r"(?i)\b(?:MRN|medical record (?:number|no\.?)|patient id|"
                    r"hospital no\.?|UHID|IP no\.?)\s*[:#]?\s*[A-Z0-9][A-Z0-9-]{3,}\b", 0.9),
        ],
    )
    aadhaar = PatternRecognizer(
        supported_entity="IN_AADHAAR",
        patterns=[Pattern("aadhaar", r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b", 0.6)],
        context=["aadhaar", "aadhar", "uid"],
    )
    # Ages over 89 are identifiers under Safe Harbor.
    age_over_89 = PatternRecognizer(
        supported_entity="AGE_OVER_89",
        patterns=[Pattern("age_90_plus", r"(?i)\b(?:9\d|1[0-1]\d)[- ]?(?:years?|yrs?|y/?o)\b(?:[- ]old)?", 0.8)],
    )
    indian_phone = PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        patterns=[Pattern("in_mobile", r"(?<!\d)(?:\+91[- ]?)?[6-9]\d{4}[- ]?\d{5}(?!\d)", 0.7)],
    )
    return [mrn, aadhaar, age_over_89, indian_phone]


class TextDeidentifier:
    def __init__(self, spacy_model: str = "en_core_web_lg", score_threshold: float = 0.35) -> None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        nlp_engine = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": spacy_model}],
        }).create_engine()
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        for recognizer in _custom_recognizers():
            self.analyzer.registry.add_recognizer(recognizer)
        self.anonymizer = AnonymizerEngine()
        self.score_threshold = score_threshold
        self.entities = list(ENTITY_PLACEHOLDERS)

    def deidentify(self, text: str) -> DeidResult:
        from presidio_anonymizer.entities import OperatorConfig

        if not text or not text.strip():
            return DeidResult(text="")
        results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=self.entities,
            score_threshold=self.score_threshold,
            allow_list=MEDICAL_ALLOW_LIST,
        )
        results = [r for r in results if not _is_non_identifying_time(text, r)]
        results = [_extend_surname_first(text, r) for r in results]
        operators = {
            entity: OperatorConfig("replace", {"new_value": placeholder})
            for entity, placeholder in ENTITY_PLACEHOLDERS.items()
        }
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
        findings = sorted(
            (PHIFinding(r.entity_type, r.start, r.end, round(r.score, 2)) for r in results),
            key=lambda f: f.start,
        )
        return DeidResult(text=anonymized.text, findings=findings)


_CALENDAR_DATE_RE = re.compile(
    r"\b\d{1,4}[/.-]\d{1,2}(?:[/.-]\d{2,4})?\b"            # 14/03/1968, 2024-11-02, 02.11
    r"|\b(?:19|20)\d{2}\b"                                  # bare year 2015
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d"  # Nov 18
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)",
    re.IGNORECASE,
)


def _is_non_identifying_time(text: str, result) -> bool:
    """Keep DATE_TIME spans that are not calendar dates.

    Presidio's DATE_TIME also fires on durations ("5 days"), ages ("56-year-old"),
    times of day ("night"), and even drug names. Safe Harbor only requires removing
    calendar dates (except year); removing the rest destroys clinical meaning, and
    for "No night sweats" it even deletes the negation cue.
    Note: bare years are still redacted (conservative; years alone are allowed by
    Safe Harbor but can identify combined with rare conditions).
    """
    if result.entity_type != "DATE_TIME":
        return False
    return not _CALENDAR_DATE_RE.search(text[result.start:result.end])


_SURNAME_COMMA_RE = re.compile(r"([A-Z][a-zA-Z'-]+),\s*$")


def _extend_surname_first(text: str, result):
    """'Sharma, Ramesh': NER often tags only 'Ramesh'. Pull the surname in."""
    if result.entity_type != "PERSON":
        return result
    match = _SURNAME_COMMA_RE.search(text[max(0, result.start - 40):result.start])
    if match:
        result.start = result.start - (len(text[max(0, result.start - 40):result.start]) - match.start(1))
    return result


@lru_cache(maxsize=1)
def get_deidentifier() -> TextDeidentifier:
    return TextDeidentifier()
