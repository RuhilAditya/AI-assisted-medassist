"""Query router: decide which evidence a question needs.

Rule-based on purpose: it is instant, deterministic and easy to explain, and
misrouting is cheap because the multimodal route sees everything anyway.

Routes
  text        about the notes/history/medications          -> notes + RAG passages
  image       about the X-ray itself                       -> image findings + similar cases
  multimodal  correlation / overall assessment / ambiguous -> everything
  triage      urgency / disposition                        -> everything, triage-focused prompt
"""
from __future__ import annotations

import re
from dataclasses import dataclass

TRIAGE_WORDS = r"\b(triage|urgent|urgency|emergency|priority|prioriti[sz]e|escalat\w*|admit|admission|how sick|how serious|red flags?|news2?|disposition)\b"
IMAGE_WORDS = r"\b(x-?ray|cxr|image|scan|film|radiograph\w*|heat ?map|see on|visible|opacit\w*|effusion|cardiomegaly|nodule|lungs? look)\b"
TEXT_WORDS = r"\b(note|notes|history|medication\w*|drugs?|allerg\w*|vitals?|symptoms?|complain\w*|report(?:ed)? by|documented|labs?|plan)\b"
FUSION_WORDS = r"\b(correlat\w*|consistent|match\w*|agree\w*|together|both|overall|summar\w*|assess\w*|explain|interpret\w*|differential|diagnos\w*|impression|what is going on)\b"


@dataclass(frozen=True)
class Route:
    name: str  # text | image | multimodal | triage
    reason: str
    notice: str | None = None  # shown to the user when evidence is missing


def route_query(question: str, has_text: bool, has_image: bool) -> Route:
    q = question.lower()
    wants_triage = bool(re.search(TRIAGE_WORDS, q))
    wants_image = bool(re.search(IMAGE_WORDS, q))
    wants_text = bool(re.search(TEXT_WORDS, q))
    wants_fusion = bool(re.search(FUSION_WORDS, q))

    if wants_triage:
        name, reason = "triage", "asks about urgency or disposition"
    elif (wants_image and wants_text) or wants_fusion:
        name, reason = "multimodal", "asks to combine or interpret the evidence"
    elif wants_image:
        name, reason = "image", "asks about the image"
    elif wants_text:
        name, reason = "text", "asks about the clinical notes"
    else:
        name, reason = "multimodal", "general question; using all available evidence"

    notice = None
    if name == "image" and not has_image:
        name, notice = ("text" if has_text else "multimodal"), "No image has been uploaded for this case."
    elif name == "text" and not has_text:
        name, notice = ("image" if has_image else "multimodal"), "No clinical note has been uploaded for this case."
    elif name in ("multimodal", "triage"):
        missing = [m for m, ok in (("clinical note", has_text), ("image", has_image)) if not ok]
        if missing:
            notice = f"Answering without: {', '.join(missing)}."
    return Route(name, reason, notice)
