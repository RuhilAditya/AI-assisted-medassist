"""Cross-modal conflict detection: where text and image evidence disagree.

This is something a unimodal system cannot do by construction, and it is
clinically useful: a discordance is a prompt to double-check (wrong image?
early disease? classifier miss? documentation error?).

Kinds
  note_asserts_image_negative  note states finding F; image classifier does not see it
  image_positive_note_denies   image shows F; note explicitly says "no F"
  clinical_suspicion_unconfirmed strong supporting signs for F, image negative
  unmentioned_image_finding    image shows F with high confidence; note silent (incidental)
  note_internal_inconsistency  one note asserts F, another negates it
  modality_mismatch            question/notes are about the chest but image is not a CXR
"""
from __future__ import annotations

from dataclasses import dataclass

from medassist.fusion import knowledge as kb
from medassist.fusion.evidence import EvidenceBundle

HIGH_CONFIDENCE = 0.75  # "image clearly shows" for unmentioned findings
SUSPICION_LLR = 1.2     # supporting evidence needed before "suspicion unconfirmed" (= the cap)
# Findings where symptoms can reasonably precede or predict a radiographic sign.
SUSPICION_FINDINGS = {"lung opacity", "pleural effusion", "pulmonary edema"}


@dataclass(frozen=True)
class Conflict:
    kind: str
    severity: str  # "high" | "medium" | "info"
    finding: str | None
    message: str
    evidence: list[str]  # citation ids, e.g. ["note_1", "image"]


def detect_conflicts(bundle: EvidenceBundle) -> list[Conflict]:
    conflicts: list[Conflict] = []

    # internal inconsistency across notes (does not need an image)
    for term, src in bundle.all_terms("active"):
        for neg, neg_src in bundle.all_terms("negated"):
            if src != neg_src and (term == neg or kb.term_matches(term, {neg})):
                conflicts.append(Conflict(
                    "note_internal_inconsistency", "medium", None,
                    f"'{term}' is recorded as present in {src} but as absent in {neg_src}.",
                    [src, neg_src]))

    if bundle.has_image and not bundle.is_chest_xray and bundle.has_text:
        mod = bundle.image_analysis.get("modality", {})
        conflicts.append(Conflict(
            "modality_mismatch", "high", None,
            f"The uploaded image looks like '{mod.get('modality')}', not a chest X-ray, so no "
            "image findings could be correlated with the note. Check that the right image was uploaded.",
            ["image"]))

    for f in bundle.fused:
        asserted = [s for s in f.signals if s.kind == "asserted"]
        negated = [s for s in f.signals if s.kind == "negated"]
        support = sum(s.llr for s in f.signals if s.kind in ("supporting", "vital"))
        p = f.image_probability or 0.0

        if asserted and f.image_level == "unlikely":
            conflicts.append(Conflict(
                "note_asserts_image_negative", "medium", f.finding,
                f"{asserted[0].source} states '{asserted[0].term}', but the image classifier does not "
                f"detect {f.finding} (probability {p:.2f}). Possible reasons: early or subtle disease, "
                "a classifier miss, a different study, or a documentation error.",
                [asserted[0].source, "image"]))
        elif negated and f.image_positive:
            conflicts.append(Conflict(
                "image_positive_note_denies", "high", f.finding,
                f"The image classifier suggests {f.finding} (probability {p:.2f}), but "
                f"{negated[0].source} records 'no {negated[0].term}'. Review the image directly.",
                ["image", negated[0].source]))
        elif (not asserted and not negated and support >= SUSPICION_LLR and f.image_level == "unlikely"
              and f.finding in SUSPICION_FINDINGS):
            terms = ", ".join(s.term for s in f.signals if s.kind in ("supporting", "vital"))
            conflicts.append(Conflict(
                "clinical_suspicion_unconfirmed", "info", f.finding,
                f"Clinical signs ({terms}) are compatible with {f.finding}, but the image classifier "
                f"does not detect it (probability {p:.2f}). Radiographic signs can lag symptoms.",
                [s.source for s in f.signals if s.kind in ("supporting", "vital")][:2] + ["image"]))
        # Only "unmentioned" when nothing in the note points to it (symptoms count as mentioned).
        elif (not asserted and not negated and support == 0 and f.image_positive
              and p >= HIGH_CONFIDENCE and bundle.has_text):
            conflicts.append(Conflict(
                "unmentioned_image_finding", "info", f.finding,
                f"The image classifier suggests {f.finding} (probability {p:.2f}), which the note does "
                "not mention. It may be incidental or undocumented.",
                ["image"]))
    return conflicts
