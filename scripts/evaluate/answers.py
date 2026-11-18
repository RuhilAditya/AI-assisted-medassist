"""Answer faithfulness and latency of the full assistant (real Llama), on the 7 demo cases.

Each case gets 4 questions (one per route). Every answer is scored automatically:
  format compliance    reply has the SUMMARY / DETAILS structure
  citation coverage    share of claims (sentences / bullets with >= 6 words) carrying a citation
  citation validity    share of cited ids that exist in the evidence (note_1 present, case-2 <= #cases ...)
  ungrounded numbers   numbers in the answer that appear nowhere in the evidence prompt (hallucination proxy)
  triage consistency   triage-route answers state the rule-computed level
  conflict disclosure  when a HIGH/MEDIUM conflict exists, the answer mentions it
  unsafe certainty     phrases like "ruled out", "definitely", "confirmed diagnosis"
  latency              evidence stage (rules, retrieval) and LLM stage, per route

Runs on a temporary copy of the demo cases, so their chat history stays clean.
Usage: python -m scripts.evaluate.answers [--cases N]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from medassist.evaluation.results import save_results
from medassist.fusion.assistant import MedAssistant, _default_chain, extract_citations, parse_sections
from medassist.ingestion.store import DEFAULT_ROOT, CaseStore

QUESTIONS = {
    "multimodal": "Interpret this X-ray and correlate it with the symptoms.",
    "triage": "How urgent is this case, and why?",
    "text": "What does the clinical note say about history, medications and vital signs?",
    "image": "What does the X-ray show?",
}
LEVELS = ("EMERGENCY", "URGENT", "SOON", "ROUTINE")
UNSAFE = re.compile(r"\b(ruled out|rules out|rule[sd]? it out|definitely|certainly|confirmed diagnosis|diagnosis is confirmed|"
                    r"100%|no doubt)\b", re.IGNORECASE)
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def claims(text: str) -> list[str]:
    parts = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*•0123456789. ").strip()
        if not line or line.endswith(":"):
            continue
        parts += [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if len(s.split()) >= 6]
    return parts


def valid_ids(partial: dict) -> set[str]:
    ev = partial["evidence"]
    ids = {n["note_id"] for n in ev["notes"]} | {"triage", "conflict"}
    if ev["image_analysis"]:
        ids.add("image")
    ids |= {f"case-{i}" for i in range(1, len((ev["image_analysis"] or {}).get("similar_cases", [])) + 1)}
    ids |= {f"ref-{i}" for i in range(1, len(ev["passages"]) + 1)}
    return ids


def score(partial: dict, user_msg: str, raw: str, question: str) -> dict:
    summary, details = parse_sections(raw)
    text = f"{summary}\n{details}"
    cl = claims(text)
    cited_claims = sum(bool(re.search(r"\[[^\]]*(note_|image|case-|ref-|triage|conflict)[^\]]*\]", c)) for c in cl)
    cites = extract_citations(text)
    valid = valid_ids(partial)
    evidence_nums = set(NUM.findall(user_msg))
    answer_nums = [n for n in NUM.findall(re.sub(r"\[[^\]]*\]", "", text)) if n not in NUM.findall(question)]
    ungrounded = [n for n in answer_nums if n not in evidence_nums and n.rstrip("0").rstrip(".") not in evidence_nums]
    level = partial["triage"]["level"]
    severe = [c for c in partial["conflicts"] if c["severity"] in ("high", "medium")]
    mentions_conflict = bool(re.search(r"conflict|discrepan|disagree|inconsisten|contradict", text, re.IGNORECASE)) or any(
        (c["finding"] or "") and c["finding"] in text.lower() for c in severe)
    return {
        "format_ok": bool(re.search(r"summary", raw, re.I) and re.search(r"details", raw, re.I)),
        "claims": len(cl),
        "cited_claims": cited_claims,
        "citations": len(cites),
        "valid_citations": sum(c in valid for c in cites),
        "numbers": len(answer_nums),
        "ungrounded_numbers": ungrounded,
        "states_triage_level": level in summary.upper() or level in text.upper(),
        "other_level_in_summary": [lv for lv in LEVELS if lv != level and re.search(rf"\b{lv}\b", summary)],
        "has_severe_conflict": bool(severe),
        "discloses_conflict": mentions_conflict if severe else None,
        "unsafe_phrases": UNSAFE.findall(text),
        "words": len(text.split()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=7)
    args = parser.parse_args()

    index = json.loads((DEFAULT_ROOT / "demo_index.json").read_text(encoding="utf-8"))
    tmp = Path(tempfile.mkdtemp(prefix="medassist_eval_"))
    for cid in index.values():
        shutil.copytree(DEFAULT_ROOT / cid, tmp / cid)
    store = CaseStore(tmp)
    assistant = MedAssistant(store)
    chain = _default_chain()

    rows = []
    for name, cid in list(index.items())[: args.cases]:
        for route, question in QUESTIONS.items():
            t0 = time.perf_counter()
            partial, user_msg, _ = assistant.prepare(cid, question)
            t_evidence = time.perf_counter() - t0
            t1 = time.perf_counter()
            raw = chain.invoke({"user": user_msg})
            t_llm = time.perf_counter() - t1
            s = score(partial, user_msg, raw, question)
            s.update({"case": name, "asked_route": route, "routed_to": partial["route"]["name"],
                      "evidence_s": round(t_evidence, 2), "llm_s": round(t_llm, 1), "answer": raw})
            rows.append(s)
            print(f"{name:<32}{route:<11}-> {s['routed_to']:<10} cites {s['cited_claims']}/{s['claims']}  "
                  f"valid {s['valid_citations']}/{s['citations']}  ungrounded {len(s['ungrounded_numbers'])}  "
                  f"llm {t_llm:.0f}s", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)

    def frac(num, den):
        return round(num / den, 3) if den else None

    triage_rows = [r for r in rows if r["routed_to"] == "triage"]
    conflict_rows = [r for r in rows if r["has_severe_conflict"]]
    summary = {
        "answers": len(rows),
        "routing_accuracy": frac(sum(r["asked_route"] == r["routed_to"] for r in rows), len(rows)),
        "format_compliance": frac(sum(r["format_ok"] for r in rows), len(rows)),
        "citation_coverage": frac(sum(r["cited_claims"] for r in rows), sum(r["claims"] for r in rows)),
        "citation_validity": frac(sum(r["valid_citations"] for r in rows), sum(r["citations"] for r in rows)),
        "ungrounded_number_rate": frac(sum(len(r["ungrounded_numbers"]) for r in rows), sum(r["numbers"] for r in rows)),
        "triage_level_stated": frac(sum(r["states_triage_level"] for r in triage_rows), len(triage_rows)),
        "triage_contradictions": sum(bool(r["other_level_in_summary"]) for r in triage_rows),
        "conflict_disclosure": frac(sum(bool(r["discloses_conflict"]) for r in conflict_rows), len(conflict_rows)),
        "answers_with_unsafe_phrases": sum(bool(r["unsafe_phrases"]) for r in rows),
        "median_words": statistics.median(r["words"] for r in rows),
        "latency_s": {
            "evidence_median": round(statistics.median(r["evidence_s"] for r in rows), 2),
            "llm_median": round(statistics.median(r["llm_s"] for r in rows), 1),
            "llm_by_route": {rt: round(statistics.median(r["llm_s"] for r in rows if r["routed_to"] == rt), 1)
                             for rt in {r["routed_to"] for r in rows}},
        },
    }
    print("\n" + json.dumps(summary, indent=2))
    path = save_results("phase5_answers", {"summary": summary, "questions": QUESTIONS, "rows": rows})
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
