"""Phase 4 tests: endpoints the React UI depends on (overview, streaming chat,
case listing, note fetch) and the streaming generator. Models and LLM are faked."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from medassist.fusion.assistant import MedAssistant
from medassist.ingestion.store import CaseStore
from tests.test_fusion import FakeChain, FakeDeid, image_findings


class StreamingChain(FakeChain):
    def stream(self, inputs):
        self.calls.append(inputs["user"])
        for piece in ["SUMMARY: Effusion ", "likely [image].", "\nDETAILS:\n- SpO2 92% [note_1]"]:
            yield piece


@pytest.fixture()
def setup(tmp_path, monkeypatch):
    store = CaseStore(tmp_path)
    cid = store.create_case(title="Demo", tags={"demo": "x"})
    store.add_note(cid, {"text": "fever", "summary": {"active_problems": ["shortness of breath"],
                                                       "negated_findings": ["pleural effusion"],
                                                       "vitals": {"spo2": "92"}}})
    store.add_image(cid, b"png", {"source_format": "png"})
    analysis = {"modality": {"modality": "chest X-ray", "confidence": 0.99, "is_chest_xray": True},
                "findings": image_findings(pleural_effusion=0.95), "similar_cases": [], "warnings": []}
    chain = StreamingChain()

    def factory(_store):
        return MedAssistant(store, chain=chain, deidentifier=FakeDeid(),
                            image_analyzer=lambda img: analysis, retriever=lambda c, q, k: [])

    from api import main
    monkeypatch.setattr(main, "get_store", lambda: store)
    monkeypatch.setattr("medassist.fusion.assistant.MedAssistant", factory)
    monkeypatch.setattr("medassist.fusion.assistant.Image.open", lambda p: None)
    return TestClient(main.app), store, cid


def test_list_cases_includes_title_and_tags(setup):
    client, _, cid = setup
    cases = client.get("/cases").json()
    assert cases[0]["case_id"] == cid and cases[0]["title"] == "Demo" and cases[0]["tags"]["demo"] == "x"
    assert cases[0]["notes"] == 1 and cases[0]["images"] == 1


def test_create_case_with_title(setup):
    client, store, _ = setup
    cid = client.post("/cases", json={"title": "Chest pain"}).json()["case_id"]
    assert store.get_case(cid)["title"] == "Chest pain"
    assert client.post("/cases").status_code == 201  # body optional


def test_overview_is_deterministic_and_stores_nothing(setup):
    client, store, cid = setup
    body = client.get(f"/cases/{cid}/overview").json()
    assert body["triage"]["level"] == "URGENT"
    assert body["conflicts"][0]["kind"] == "image_positive_note_denies"
    assert body["evidence"]["fused_findings"][0]["finding"] == "pleural effusion"
    assert store.get_chat(cid) == []


def test_stream_emits_evidence_tokens_then_done(setup):
    client, store, cid = setup
    with client.stream("POST", f"/cases/{cid}/chat/stream", json={"question": "Ravi: is there an effusion?"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = []
        for line in r.iter_lines():
            if line.startswith("event:"):
                events.append([line.split(": ", 1)[1], None])
            elif line.startswith("data:"):
                events[-1][1] = json.loads(line.split(": ", 1)[1])
    names = [e[0] for e in events]
    assert names[0] == "evidence" and names[-1] == "done" and names.count("token") == 3
    assert events[0][1]["triage"]["level"] == "URGENT"       # evidence arrives before the answer
    done = events[-1][1]
    assert done["summary"] == "Effusion likely [image]." and set(done["citations"]) == {"image", "note_1"}
    assert "Ravi" not in done["question"]
    assert len(store.get_chat(cid)) == 1


def test_stream_unknown_case_is_404(setup):
    client, _, _ = setup
    assert client.post("/cases/000000000000/chat/stream", json={"question": "hi there"}).status_code == 404


def test_get_note(setup):
    client, _, cid = setup
    assert client.get(f"/cases/{cid}/notes/note_1").json()["text"] == "fever"
    assert client.get(f"/cases/{cid}/notes/note_9").status_code == 404
