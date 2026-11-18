"""MedAssist orchestrator: one call per clinician question.

    question --de-ID--> router
    case notes ------> structured entities + RAG passages --+
    case image ------> modality gate, CXR findings, similar --+--> evidence bundle (fused scores)
                                                               |
                         conflicts (rules) + triage (NEWS2 + red flags) + follow-ups
                                                               |
                     prompt template (route) --> Llama 3.1 (LangChain) --> SUMMARY / DETAILS
Deterministic parts are returned even if the LLM is unavailable, so the
clinician still gets findings, conflicts and triage.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict
from functools import lru_cache

from PIL import Image

from medassist.fusion import prompts
from medassist.fusion.conflicts import detect_conflicts
from medassist.fusion.evidence import build_bundle
from medassist.fusion.followup import suggest
from medassist.fusion.router import route_query
from medassist.fusion.triage import assess
from medassist.ingestion.store import CaseStore

DISCLAIMER = ("Decision support only: automated output from a research prototype. "
              "Verify against the source images and records; not a diagnosis.")
CITATION_RE = re.compile(r"\[((?:note|image|case|ref|triage|conflict)[\w-]*)\]")


def extract_citations(text: str) -> list[str]:
    """Collect evidence ids from [x] and grouped [x, y] brackets."""
    ids: set[str] = set()
    for group in re.findall(r"\[([^\]]{1,80})\]", text):
        for part in group.split(","):
            if CITATION_RE.fullmatch(f"[{part.strip()}]"):
                ids.add(part.strip())
    return sorted(ids)


def parse_sections(text: str) -> tuple[str, str]:
    m = re.search(r"(?is)summary\s*:\s*(.*?)\s*(?:\*\*)?details\s*:?\s*(?:\*\*)?\s*(.*)$", text)
    if not m:
        return text.strip(), text.strip()
    summary = m.group(1).strip().strip("*").strip()
    return summary or text.strip(), m.group(2).strip() or text.strip()


@lru_cache(maxsize=1)
def _default_chain():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama

    from medassist.config import get_settings

    s = get_settings()
    prompt = ChatPromptTemplate.from_messages([("system", prompts.SAFETY_SYSTEM_PROMPT), ("human", "{user}")])
    # The evidence block + answer can exceed 2,048 tokens, so chat uses 4,096 (slower on 6 GB GPUs).
    llm = ChatOllama(model=s.llm_model, base_url=s.ollama_url, temperature=0.1, num_ctx=4096, num_predict=500, seed=0)
    return prompt | llm | StrOutputParser()


class MedAssistant:
    def __init__(self, store: CaseStore, chain=None, deidentifier=None, image_analyzer=None, retriever=None):
        self.store = store
        self._chain = chain
        self._deid = deidentifier
        self._analyze = image_analyzer
        self._retrieve = retriever

    # ---------- dependencies (lazy, swappable in tests) ----------
    @property
    def chain(self):
        return self._chain or _default_chain()

    def _deidentify(self, text: str) -> str:
        if self._deid is None:
            from medassist.ingestion.deid import get_deidentifier
            self._deid = get_deidentifier()
        return self._deid.deidentify(text).text

    def _image_analysis(self, case_id: str, image: dict) -> dict:
        if image.get("analysis"):
            return image["analysis"]  # cached on first use
        if self._analyze is None:
            from medassist.analysis import analyze_image
            self._analyze = analyze_image
        analysis = self._analyze(Image.open(self.store.image_path(case_id, image["image_id"])))
        self.store.update_image_record(case_id, image["image_id"], analysis=analysis)
        return analysis

    def _passages(self, case: dict, question: str, k: int = 4) -> list[dict]:
        if not case.get("notes"):
            return []
        if self._retrieve is not None:
            return self._retrieve(case, question, k)
        from medassist.text_pipeline.rag import KnowledgeBase, Passage

        kb = KnowledgeBase()
        kb.add_passages([Passage(n["note_id"], n["text"], n["note_id"]) for n in case["notes"]])
        return [{"text": h.metadata["text"], "source": h.metadata["source"], "score": round(h.score, 3)}
                for h in kb.retrieve(question, k=k)]

    # ---------- main entry ----------
    def chat(self, case_id: str, question: str, image_id: str | None = None) -> dict:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal t0
            timings[name] = round(time.perf_counter() - t0, 2)
            t0 = time.perf_counter()

        case = self.store.get_case(case_id)
        safe_question = self._deidentify(question)
        lap("deidentify_question")

        image = None
        if case.get("images"):
            image = next((i for i in case["images"] if i["image_id"] == image_id), None) if image_id else case["images"][-1]
        analysis = self._image_analysis(case_id, image) if image else None
        lap("image_analysis")

        passages = self._passages(case, safe_question)
        lap("retrieval")

        bundle = build_bundle(case, analysis, image["image_id"] if image else None, passages)
        route = route_query(safe_question, bundle.has_text, bundle.has_image)
        conflicts = detect_conflicts(bundle)
        triage = assess(bundle)
        followups = suggest(bundle, conflicts, triage, route.name)
        history = self.store.get_chat(case_id)
        lap("fusion_rules")

        user_msg = prompts.USER_TEMPLATE.format(
            evidence=bundle.to_prompt(include_similar=route.name in ("image", "multimodal")),
            conflicts=prompts.format_conflicts(conflicts),
            triage=prompts.format_triage(triage),
            history=prompts.format_history(history),
            question=safe_question,
            task=prompts.ROUTE_TASKS[route.name],
        )
        llm_error = None
        try:
            raw = self.chain.invoke({"user": user_msg})
            summary, details = parse_sections(raw)
        except Exception as exc:  # LLM down: still return the deterministic analysis
            llm_error = f"{type(exc).__name__}: {exc}"
            summary = "The language model is unavailable, so no written answer was generated. The findings, conflicts and triage below were computed without it."
            details = ""
        lap("llm")

        cited = extract_citations(summary + " " + details)
        response = {
            "question": safe_question,
            "route": asdict(route),
            "summary": summary,
            "details": details,
            "citations": cited,
            "evidence": bundle.to_dict(),
            "conflicts": [asdict(c) for c in conflicts],
            "triage": asdict(triage),
            "follow_ups": asdict(followups),
            "disclaimer": DISCLAIMER,
            "llm_error": llm_error,
            "timings_s": timings,
        }
        self.store.add_chat_turn(case_id, {
            "question": safe_question, "summary": summary, "route": route.name,
            "triage_level": triage.level, "conflicts": len(conflicts),
        })
        return response
