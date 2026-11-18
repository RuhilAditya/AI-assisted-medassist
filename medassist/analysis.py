"""Unimodal analysis services used by the API (and by Phase 3 fusion).

image:  modality gate -> CXR finding scores -> similar prior cases
text:   RAG question answering over a case's de-identified notes
"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

from PIL import Image

from medassist.image_pipeline.biomedclip import get_biomedclip
from medassist.image_pipeline.cxr import CXRClassifier, detect_modality
from medassist.text_pipeline.rag import KnowledgeBase, Passage, answer_question


@lru_cache(maxsize=1)
def get_classifier() -> CXRClassifier:
    return CXRClassifier()  # linear probe if trained, else zero-shot


def analyze_image(image: Image.Image, k_similar: int = 3) -> dict:
    modality = detect_modality(get_biomedclip(), image)
    result: dict = {"modality": asdict(modality), "findings": [], "similar_cases": [], "warnings": []}
    if not modality.is_chest_xray:
        result["warnings"].append(
            f"Image looks like '{modality.modality}' ({modality.confidence:.0%}); the chest X-ray "
            "finding classifier was not run. Only chest X-rays are supported for finding detection."
        )
        return result

    classifier = get_classifier()
    result["classifier_mode"] = classifier.mode
    result["findings"] = [asdict(s) for s in classifier.classify(image)]
    try:
        from medassist.image_pipeline.case_retrieval import get_case_retriever

        result["similar_cases"] = [asdict(c) for c in get_case_retriever().similar(image, k=k_similar)]
    except FileNotFoundError:
        result["warnings"].append("Similar-case index not built (python -m scripts.eval_retrieval).")
    return result


def heatmap_image(image: Image.Image, finding: str) -> Image.Image:
    from medassist.image_pipeline.cxr import overlay_heatmap

    classifier = get_classifier()
    if finding not in classifier.findings:
        raise ValueError(f"Unknown finding '{finding}'. Choose from: {', '.join(classifier.findings)}")
    return overlay_heatmap(image, classifier.occlusion_heatmap(image, finding))


def ask_case_notes(case: dict, question: str, k: int = 4) -> dict:
    notes = case.get("notes", [])
    if not notes:
        return {"answer": "This case has no notes yet. Upload a clinical note first.", "citations": []}
    kb = KnowledgeBase()
    kb.add_passages([Passage(id=n["note_id"], text=n["text"], source=n["note_id"]) for n in notes])
    rag = answer_question(question, kb, k=k)
    return {
        "answer": rag.answer,
        "citations": [
            {"number": i, "source": h.metadata["source"], "text": h.metadata["text"],
             "score": round(h.score, 3), "cited": i in rag.cited_numbers}
            for i, h in enumerate(rag.citations, 1)
        ],
    }
