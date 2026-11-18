"""Phase 3 tests: router, fusion math, conflicts, NEWS2 triage, follow-ups, orchestrator.
All fast: models and the LLM are replaced by fakes."""
from __future__ import annotations

import pytest

from medassist.fusion.assistant import MedAssistant, extract_citations, parse_sections
from medassist.fusion.conflicts import detect_conflicts
from medassist.fusion.evidence import EvidenceBundle, NoteEvidence, build_bundle, fuse_findings
from medassist.fusion.followup import suggest
from medassist.fusion.router import route_query
from medassist.fusion.triage import assess, news2
from medassist.ingestion.store import CaseStore


def note(active=(), negated=(), history=(), vitals=None, note_id="note_1") -> NoteEvidence:
    return NoteEvidence(note_id, list(active), list(negated), list(history), [], [], [], vitals or {})


def image_findings(**probs) -> list[dict]:
    out = []
    for finding, p in probs.items():
        name = finding.replace("_", " ")
        level = "likely" if p >= 0.5 else "possible" if p >= 0.2 else "unlikely"
        out.append({"finding": name, "probability": p, "positive": p >= 0.5, "level": level})
    return out


def bundle_with(notes, **probs) -> EvidenceBundle:
    b = EvidenceBundle(notes=notes, image_id="image_1",
                       image_analysis={"modality": {"modality": "chest X-ray", "confidence": 0.99, "is_chest_xray": True}})
    b.fused = fuse_findings(image_findings(**probs), b)
    return b


# ---------- router ----------
@pytest.mark.parametrize("question,expected", [
    ("How urgent is this patient?", "triage"),
    ("What does the x-ray show?", "image"),
    ("What medications is the patient on?", "text"),
    ("Correlate the image with the symptoms", "multimodal"),
    ("hello", "multimodal"),
])
def test_router(question, expected):
    assert route_query(question, has_text=True, has_image=True).name == expected


def test_router_falls_back_when_modality_missing():
    r = route_query("What does the x-ray show?", has_text=True, has_image=False)
    assert r.name == "text" and "No image" in r.notice


# ---------- fusion math ----------
def test_assertion_raises_and_negation_lowers_image_probability():
    up = bundle_with([note(active=["pneumonia"])], lung_opacity=0.3).fused[0]
    down = bundle_with([note(negated=["pleural effusion"])], pleural_effusion=0.3).fused[0]
    assert up.fused_probability > 0.7 and up.text_llr == pytest.approx(2.0)
    assert down.fused_probability < 0.1


def test_support_is_capped_and_vitals_count():
    b = bundle_with([note(active=["fever", "productive cough", "cough", "chills", "crackles"],
                          vitals={"spo2": "88"})], lung_opacity=0.5)
    assert b.fused[0].text_llr == pytest.approx(1.2)  # cap


def test_chronic_history_counts_as_assertion_but_acute_history_only_supports():
    chronic = bundle_with([note(history=["congestive heart failure"])], cardiomegaly=0.3).fused[0]
    acute = bundle_with([note(history=["pneumonia"])], lung_opacity=0.3).fused[0]
    assert any(s.kind == "asserted" for s in chronic.signals)
    assert all(s.kind != "asserted" for s in acute.signals) and acute.text_llr == pytest.approx(0.4)


# ---------- conflicts ----------
def test_conflict_image_positive_note_denies_is_high():
    c = detect_conflicts(bundle_with([note(negated=["pleural effusion"])], pleural_effusion=0.95))
    assert [(x.kind, x.severity) for x in c] == [("image_positive_note_denies", "high")]


def test_conflict_note_asserts_image_negative():
    c = detect_conflicts(bundle_with([note(active=["pneumonia"])], lung_opacity=0.05))
    assert c[0].kind == "note_asserts_image_negative" and c[0].finding == "lung opacity"


def test_no_conflict_when_symptoms_explain_image_finding():
    c = detect_conflicts(bundle_with([note(active=["fever", "productive cough"])], lung_opacity=0.9))
    assert c == []


def test_internal_note_inconsistency():
    b = bundle_with([note(active=["fever"]), note(negated=["fever"], note_id="note_2")])
    assert detect_conflicts(b)[0].kind == "note_internal_inconsistency"


def test_modality_mismatch_conflict():
    b = EvidenceBundle(notes=[note(active=["cough"])],
                       image_analysis={"modality": {"modality": "brain MRI", "confidence": 0.9, "is_chest_xray": False}})
    assert detect_conflicts(b)[0].kind == "modality_mismatch"


# ---------- triage ----------
def test_news2_matches_hand_calculation():
    score, comps = news2({"respiratory_rate": "24", "spo2": "90", "blood_pressure": "124/78",
                          "heart_rate": "108", "temperature": "38.8 C"}, confused=False)
    assert comps == {"respiratory_rate": 2, "spo2": 3, "systolic_bp": 0, "heart_rate": 1, "temperature": 1}
    assert score == 7


def test_news2_needs_three_observations():
    assert news2({"spo2": "97"}, confused=False)[0] is None


@pytest.mark.parametrize("active,vitals,level", [
    ([], {"respiratory_rate": "14", "spo2": "99", "heart_rate": "70", "temperature": "36.8 C"}, "ROUTINE"),
    (["slurred speech"], {}, "EMERGENCY"),
    (["chest pain", "shortness of breath"], {}, "URGENT"),
    ([], {"respiratory_rate": "28", "spo2": "86", "heart_rate": "118", "temperature": "37.4 C"}, "EMERGENCY"),
])
def test_triage_levels(active, vitals, level):
    assert assess(bundle_with([note(active=active, vitals=vitals)])).level == level


def test_image_finding_plus_hypoxia_escalates():
    b = bundle_with([note(vitals={"spo2": "92"})], pleural_effusion=0.9)
    assert assess(b).level == "URGENT"


# ---------- follow-ups ----------
def test_followups_request_missing_vitals_and_image():
    b = EvidenceBundle(notes=[note(active=["cough"], vitals={"spo2": "95"})])
    f = suggest(b, [], assess(b), "text")
    assert any("vital signs" in r for r in f.requests)
    assert any("chest X-ray" in r for r in f.requests)


# ---------- orchestrator ----------
class FakeChain:
    def __init__(self, reply="SUMMARY: Effusion likely [image].\nDETAILS:\n- SpO2 92% [note_1, image]"):
        self.reply, self.calls = reply, []

    def invoke(self, inputs):
        self.calls.append(inputs["user"])
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class FakeDeid:
    def deidentify(self, text):
        return type("R", (), {"text": text.replace("Ravi", "[PERSON]")})()


def _case(tmp_path):
    store = CaseStore(tmp_path)
    cid = store.create_case(title="test")
    store.add_note(cid, {"text": "fever", "summary": {"active_problems": ["shortness of breath"],
                                                       "negated_findings": ["pleural effusion"],
                                                       "vitals": {"spo2": "92"}}})
    store.add_image(cid, b"png", {"source_format": "png"})
    return store, cid


def _assistant(store, chain):
    analysis = {"modality": {"modality": "chest X-ray", "confidence": 0.99, "is_chest_xray": True},
                "findings": image_findings(pleural_effusion=0.95, nodule=0.05), "similar_cases": [], "warnings": []}
    return MedAssistant(store, chain=chain, deidentifier=FakeDeid(),
                        image_analyzer=lambda img: analysis, retriever=lambda c, q, k: [])


def test_assistant_end_to_end_with_fakes(tmp_path, monkeypatch):
    store, cid = _case(tmp_path)
    monkeypatch.setattr("medassist.fusion.assistant.Image.open", lambda p: None)
    chain = FakeChain()
    r = _assistant(store, chain).chat(cid, "Ravi: correlate the x-ray with the symptoms")
    assert r["route"]["name"] == "multimodal"
    assert r["summary"] == "Effusion likely [image]."
    assert set(r["citations"]) == {"image", "note_1"}
    assert r["conflicts"][0]["kind"] == "image_positive_note_denies"
    assert r["triage"]["level"] == "URGENT"
    assert "Ravi" not in chain.calls[0]                       # question de-identified before the LLM
    assert "Ravi" not in (tmp_path / cid / "chat.jsonl").read_text()  # and before storage
    assert store.get_case(cid)["images"][0]["analysis"]       # analysis cached


def test_assistant_survives_llm_failure(tmp_path, monkeypatch):
    store, cid = _case(tmp_path)
    monkeypatch.setattr("medassist.fusion.assistant.Image.open", lambda p: None)
    r = _assistant(store, FakeChain(reply=ConnectionError("ollama down"))).chat(cid, "How urgent?")
    assert r["llm_error"] and "unavailable" in r["summary"]
    assert r["triage"]["level"] == "URGENT" and r["conflicts"]  # deterministic parts still returned


def test_parse_sections_and_citations():
    s, d = parse_sections("**SUMMARY:** a [note_1].\n**DETAILS:**\n- b [image, case-2]")
    assert s == "a [note_1]." and d.startswith("- b")
    assert extract_citations(s + d + " [not a cite]") == ["case-2", "image", "note_1"]


def test_build_bundle_from_case_record():
    case = {"notes": [{"note_id": "note_1", "summary": {"active_problems": ["pneumonia"], "vitals": {"spo2": "90"}}}]}
    b = build_bundle(case, {"modality": {"is_chest_xray": True}, "findings": image_findings(lung_opacity=0.4)}, "image_1")
    assert b.fused[0].fused_probability > 0.8 and b.vitals == {"spo2": "90"}
    assert "[note_1]" in b.to_prompt() and "[image]" in b.to_prompt()
