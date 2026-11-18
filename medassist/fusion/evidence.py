"""Evidence bundle: everything the assistant knows about a case, in one place.

Late fusion happens here. Each modality is processed by its specialist model;
the bundle combines them in two ways:
  1. numerically: for every CXR finding, the image probability is updated
     with clinical-text evidence as a log-odds shift (Bayesian-style):
         fused = sigmoid( logit(p_image) + LLR_text )
     where LLR_text sums the knowledge-map weights for the note's assertions,
     negations, supporting symptoms and vitals (see knowledge.py);
  2. symbolically: a labelled, citable text block ([note-1], [image],
     [case-2], [ref-3] ...) that the LLM reasons over.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from medassist.fusion import knowledge as kb


@dataclass(frozen=True)
class NoteEvidence:
    note_id: str
    active: list[str]
    negated: list[str]
    past_history: list[str]
    family_history: list[str]
    medications: list[str]
    tests: list[str]
    vitals: dict[str, str]

    @classmethod
    def from_record(cls, record: dict) -> "NoteEvidence":
        s = record.get("summary", {})
        active = sorted(set(s.get("active_problems", [])))
        return cls(
            note_id=record["note_id"],
            active=active,
            negated=sorted(set(s.get("negated_findings", []))),
            past_history=sorted(set(s.get("past_history", []))),
            family_history=sorted(set(s.get("family_history", []))),
            medications=sorted(set(s.get("medications", []))),
            tests=sorted(set(s.get("tests", []))),
            vitals=dict(s.get("vitals", {})),
        )


@dataclass(frozen=True)
class TextSignal:
    term: str
    llr: float
    kind: str  # "asserted" | "negated" | "supporting" | "vital"
    source: str  # e.g. "note_1"


@dataclass(frozen=True)
class FusedFinding:
    finding: str
    image_probability: float | None
    image_positive: bool | None
    image_level: str | None  # likely | possible | unlikely
    text_llr: float
    fused_probability: float | None
    signals: list[TextSignal] = field(default_factory=list)


@dataclass
class EvidenceBundle:
    notes: list[NoteEvidence] = field(default_factory=list)
    image_id: str | None = None
    image_analysis: dict | None = None
    fused: list[FusedFinding] = field(default_factory=list)
    passages: list[dict] = field(default_factory=list)  # RAG hits: {"text", "source", "score"}

    # ---------- convenience views ----------
    @property
    def has_text(self) -> bool:
        return bool(self.notes)

    @property
    def has_image(self) -> bool:
        return self.image_analysis is not None

    @property
    def is_chest_xray(self) -> bool:
        return bool(self.image_analysis and self.image_analysis.get("modality", {}).get("is_chest_xray"))

    @property
    def vitals(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for note in self.notes:  # later notes override earlier ones
            merged.update(note.vitals)
        return merged

    def all_terms(self, attr: str) -> list[tuple[str, str]]:
        return [(term, n.note_id) for n in self.notes for term in getattr(n, attr)]

    def to_dict(self) -> dict:
        return {
            "notes": [asdict(n) for n in self.notes],
            "image_id": self.image_id,
            "image_analysis": self.image_analysis,
            "fused_findings": [asdict(f) for f in self.fused],
            "passages": self.passages,
            "vitals": self.vitals,
        }

    # ---------- prompt rendering ----------
    def to_prompt(self, include_similar: bool = True) -> str:
        lines: list[str] = []
        for n in self.notes:
            lines.append(f"[{n.note_id}] Clinical note (de-identified, structured):")
            lines.append(f"  Current problems/symptoms: {', '.join(n.active) or 'none recorded'}")
            lines.append(f"  Explicitly absent (negated): {', '.join(n.negated) or 'none'}")
            if n.past_history:
                lines.append(f"  Past history: {', '.join(n.past_history)}")
            if n.family_history:
                lines.append(f"  Family history: {', '.join(n.family_history)}")
            if n.medications:
                lines.append(f"  Medications: {', '.join(n.medications)}")
            if n.vitals:
                lines.append(f"  Vitals: {', '.join(f'{k}={v}' for k, v in n.vitals.items())}")
        if not self.notes:
            lines.append("[note] No clinical note provided.")

        if self.image_analysis:
            mod = self.image_analysis.get("modality", {})
            lines.append(f"[image] Uploaded image ({self.image_id}): automated modality check = "
                         f"{mod.get('modality')} ({mod.get('confidence', 0):.0%}).")
            if self.fused:
                lines.append("  Automated chest X-ray classifier (BiomedCLIP), probabilities 0-1; "
                             "'with context' = image score adjusted by the clinical note:")
                shown, hidden = [], []
                for f in sorted(self.fused, key=lambda x: -(x.fused_probability or 0)):
                    relevant = f.image_level != "unlikely" or any(s.kind in ("asserted", "negated") for s in f.signals)
                    (shown if relevant else hidden).append(f)
                for f in shown:
                    lines.append(f"  - {f.finding}: image {f.image_probability:.2f} ({f.image_level})"
                                 f", with context {f.fused_probability:.2f}")
                if hidden:
                    lines.append(f"  - not detected (all < 0.2): {', '.join(f.finding for f in hidden)}")
            for w in self.image_analysis.get("warnings", []):
                lines.append(f"  Warning: {w}")
            if include_similar:
                for i, c in enumerate(self.image_analysis.get("similar_cases", []), 1):
                    lines.append(f"[case-{i}] Visually similar prior X-ray (different patient), "
                                 f"labels {c['labels']}: \"{c['report'][:300]}\"")
        else:
            lines.append("[image] No image provided.")

        for i, p in enumerate(self.passages, 1):
            lines.append(f"[ref-{i}] Excerpt from {p['source']}: \"{p['text']}\"")
        return "\n".join(lines)


# ---------------- fusion ----------------
def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def text_signals(finding: str, bundle: EvidenceBundle) -> list[TextSignal]:
    signals: list[TextSignal] = []
    for term, src in bundle.all_terms("active"):
        if kb.term_matches(term, kb.FINDING_TERMS[finding]):
            signals.append(TextSignal(term, kb.DIRECT_ASSERT_LLR, "asserted", src))
    for term, src in bundle.all_terms("negated"):
        if kb.term_matches(term, kb.FINDING_TERMS[finding]):
            signals.append(TextSignal(term, kb.DIRECT_NEGATE_LLR, "negated", src))
    support_total = 0.0
    for term, src in bundle.all_terms("past_history"):
        if kb.term_matches(term, kb.FINDING_TERMS[finding]):
            if finding in kb.CHRONIC_FINDINGS:
                signals.append(TextSignal(f"history of {term}", kb.DIRECT_ASSERT_LLR, "asserted", src))
            elif support_total < kb.SUPPORT_CAP:
                signals.append(TextSignal(f"history of {term}", kb.SUPPORT_LLR, "supporting", src))
                support_total += kb.SUPPORT_LLR
    for term, src in bundle.all_terms("active") + bundle.all_terms("past_history"):
        if kb.term_matches(term, kb.SUPPORTING_TERMS[finding]) and support_total < kb.SUPPORT_CAP:
            signals.append(TextSignal(term, kb.SUPPORT_LLR, "supporting", src))
            support_total += kb.SUPPORT_LLR
    for f, llr, reason in kb.vital_signals(bundle.vitals):
        if f == finding and support_total < kb.SUPPORT_CAP:
            signals.append(TextSignal(reason, llr, "vital", "vitals"))
            support_total += llr
    # one direct assertion and one negation are enough; don't double count synonyms
    seen_kinds: set[str] = set()
    deduped = []
    for s in signals:
        if s.kind in ("asserted", "negated"):
            if s.kind in seen_kinds:
                continue
            seen_kinds.add(s.kind)
        deduped.append(s)
    return deduped


def fuse_findings(image_findings: list[dict], bundle: EvidenceBundle) -> list[FusedFinding]:
    fused = []
    for item in image_findings:
        finding, p_img = item["finding"], float(item["probability"])
        signals = text_signals(finding, bundle)
        llr = round(sum(s.llr for s in signals), 3)
        p_fused = 1 / (1 + math.exp(-(_logit(p_img) + llr)))
        level = item.get("level") or ("likely" if item["positive"] else "unlikely")
        fused.append(FusedFinding(finding, round(p_img, 3), level == "likely", level, llr, round(p_fused, 3), signals))
    return fused


def build_bundle(case: dict, image_analysis: dict | None = None, image_id: str | None = None,
                 passages: list[dict] | None = None) -> EvidenceBundle:
    bundle = EvidenceBundle(
        notes=[NoteEvidence.from_record(n) for n in case.get("notes", [])],
        image_id=image_id,
        image_analysis=image_analysis,
        passages=passages or [],
    )
    if image_analysis and image_analysis.get("findings"):
        bundle.fused = fuse_findings(image_analysis["findings"], bundle)
    return bundle
