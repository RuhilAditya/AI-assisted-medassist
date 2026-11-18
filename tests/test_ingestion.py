"""Phase 1 tests. De-ID / NER tests load spaCy models once per session."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pydicom
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydicom.data import get_testdata_file

from medassist.ingestion.clinical_ner import extract_vitals, normalise
from medassist.ingestion.dicom_anon import anonymize_dicom
from medassist.ingestion.loaders import DataValidationError
from medassist.ingestion.store import CaseNotFoundError, CaseStore
from medassist.ingestion.text_loader import load_text_upload


# ---------- model fixtures (slow, shared) ----------
@pytest.fixture(scope="session")
def deid():
    from medassist.ingestion.deid import get_deidentifier
    return get_deidentifier()


@pytest.fixture(scope="session")
def ner():
    from medassist.ingestion.clinical_ner import get_ner
    return get_ner()


# ---------- text de-identification ----------
def test_deid_removes_core_identifiers(deid):
    text = ("Patient John Smith, MRN: AB-123456, phone +91 98765 43210, "
            "email john.smith@example.com, seen on 12/05/2024 in Mumbai.")
    out = deid.deidentify(text)
    for phi in ("John Smith", "AB-123456", "98765", "john.smith@example.com", "12/05/2024", "Mumbai"):
        assert phi not in out.text
    assert "[PERSON]" in out.text and "[MRN]" in out.text
    assert out.counts.get("PERSON", 0) >= 1


def test_deid_keeps_eponyms_and_durations(deid):
    text = "History of Parkinson's disease. Cough for 3 days. Positive Murphy sign."
    out = deid.deidentify(text).text
    assert "Parkinson's" in out and "3 days" in out and "Murphy" in out


def test_deid_never_deletes_negation_cues_or_ages(deid):
    # Regression: Presidio tagged "No night" as DATE_TIME, flipping a negated finding.
    out = deid.deidentify("56-year-old male. No night sweats. Fever last night. On ramipril.").text
    assert "No night sweats" in out
    assert "56-year-old" in out and "ramipril" in out


def test_deid_surname_first_format(deid):
    out = deid.deidentify("Ward Round Note - Sharma, Ramesh (MRN 552-019-334)").text
    assert "Sharma" not in out and "Ramesh" not in out


def test_deid_report_contains_no_phi_values(deid):
    out = deid.deidentify("Seen by Dr. Priya Sharma.")
    assert all(not hasattr(f, "text") for f in out.findings)
    assert "Priya" not in str(out.findings)


def test_deid_handles_empty_text(deid):
    assert deid.deidentify("   ").text == ""


# ---------- DICOM de-identification ----------
def test_anonymize_dicom_strips_phi_and_private_tags():
    path = Path(get_testdata_file("CT_small.dcm"))
    original = pydicom.dcmread(path)
    assert "PatientName" in original

    cleaned_bytes, report = anonymize_dicom(path.read_bytes())
    cleaned = pydicom.dcmread(io.BytesIO(cleaned_bytes))
    assert "PatientName" not in cleaned and "PatientID" not in cleaned
    assert cleaned.PatientIdentityRemoved == "YES"
    assert cleaned.StudyInstanceUID != original.StudyInstanceUID
    assert not any(elem.tag.is_private for elem in cleaned)
    assert "PatientName" in report.removed_tags
    assert cleaned.pixel_array.shape == original.pixel_array.shape  # pixels intact


# ---------- clinical NER ----------
def test_ner_detects_negation(ner):
    result = ner.extract("Patient reports fever and cough. Denies chest pain. No hemoptysis.")
    assert "fever" in result.active() and "cough" in result.active()
    assert {"chest pain", "hemoptysis"} <= set(result.negated())


def test_ner_family_history_is_not_active(ner):
    result = ner.extract("Mother had stroke. Patient has headache.")
    active = result.active()
    assert "headache" in active
    assert all("stroke" not in a for a in active)


def test_ner_labels_medications(ner):
    result = ner.extract("Started on metformin and amlodipine for diabetes.")
    meds = result.active("MEDICATION")
    assert "metformin" in meds and "amlodipine" in meds


def test_normalise_expands_abbreviations():
    assert normalise("SOB") == "shortness of breath"
    assert normalise("HTN.") == "hypertension"
    assert normalise("Fever") == "fever"


def test_extract_vitals():
    v = extract_vitals("Temp 38.9 C, HR 112, RR 26, BP 132/84, SpO2 89% on room air.")
    assert v == {"temperature": "38.9 C", "heart_rate": "112", "respiratory_rate": "26",
                 "blood_pressure": "132/84", "spo2": "89"}


# ---------- text loader ----------
def test_text_loader_reads_txt_and_rejects_others():
    assert load_text_upload("n.txt", "hello\n".encode()) == "hello"
    with pytest.raises(DataValidationError):
        load_text_upload("n.docx", b"x")
    with pytest.raises(DataValidationError):
        load_text_upload("n.pdf", b"not a pdf")


# ---------- store ----------
def test_store_roundtrip_and_audit(tmp_path):
    store = CaseStore(tmp_path)
    cid = store.create_case()
    nid = store.add_note(cid, {"text": "deidentified", "deid": {"counts": {"PERSON": 1}}})
    case = store.get_case(cid)
    assert case["notes"][0]["note_id"] == nid
    audit = (tmp_path / "audit.log").read_text()
    assert "add_note" in audit and "deidentified" not in audit  # no content in audit log


def test_store_rejects_path_traversal(tmp_path):
    with pytest.raises(CaseNotFoundError):
        CaseStore(tmp_path).get_case("../../etc")


# ---------- API (end-to-end ingestion) ----------
@pytest.fixture()
def client(tmp_path, monkeypatch, deid, ner):
    from api import main

    main.get_store.cache_clear()
    monkeypatch.setattr(main, "get_store", lambda: CaseStore(tmp_path))
    return TestClient(main.app)


def test_api_note_ingestion_deidentifies_before_storage(client, tmp_path):
    cid = client.post("/cases").json()["case_id"]
    resp = client.post(f"/cases/{cid}/notes",
                       data={"text": "Patient Ravi Kumar has fever. Denies cough. SpO2 91%."})
    assert resp.status_code == 201
    body = resp.json()
    assert "Ravi Kumar" not in body["text"]
    assert "fever" in body["summary"]["active_problems"]
    assert "cough" in body["summary"]["negated_findings"]
    stored = "".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json"))
    assert "Ravi" not in stored


def test_api_dicom_upload_is_anonymised(client, tmp_path):
    cid = client.post("/cases").json()["case_id"]
    path = Path(get_testdata_file("CT_small.dcm"))
    resp = client.post(f"/cases/{cid}/images", files={"file": ("patient_scan.dcm", path.read_bytes())})
    assert resp.status_code == 201
    body = resp.json()
    assert "PatientName" in body["deid"]["removed_tags"]
    img = client.get(f"/cases/{cid}/images/{body['image_id']}")
    assert img.status_code == 200 and Image.open(io.BytesIO(img.content)).mode == "L"
    assert "patient_scan" not in "".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json"))


def test_api_rejects_unknown_case_and_bad_input(client):
    assert client.get("/cases/000000000000").status_code == 404
    cid = client.post("/cases").json()["case_id"]
    assert client.post(f"/cases/{cid}/notes").status_code == 422
    bad = client.post(f"/cases/{cid}/images", files={"file": ("x.png", b"nope")})
    assert bad.status_code == 422


def test_png_upload_roundtrip(client):
    cid = client.post("/cases").json()["case_id"]
    buf = io.BytesIO()
    Image.fromarray(np.full((20, 20), 128, dtype=np.uint8)).save(buf, format="PNG")
    resp = client.post(f"/cases/{cid}/images", files={"file": ("x.png", buf.getvalue())})
    assert resp.status_code == 201 and resp.json()["deid"] is None
