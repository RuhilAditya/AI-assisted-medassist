"""Phase 2 tests.

Fast tests (always run): vector index, chunking, RGB detection, citation parsing.
Model tests (marked `models`): need BiomedCLIP weights + prepared IU X-Ray data;
skipped automatically when the data is missing. LLM tests also need Ollama.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from medassist.ingestion.loaders import load_upload
from medassist.text_pipeline.rag import RagAnswer, answer_question, chunk_text
from medassist.vector_index import SearchHit, VectorIndex


def _png(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


# ---------- vector index ----------
def test_vector_index_search_and_persistence(tmp_path):
    index = VectorIndex(dim=3)
    index.add(["a", "b", "c"], np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]]), [{"n": 1}, {"n": 2}, {"n": 3}])
    hits = index.search(np.array([1, 0, 0]), k=2)
    assert [h.id for h in hits] == ["a", "c"] and hits[0].score == pytest.approx(1.0)
    assert [h.id for h in index.search(np.array([1, 0, 0]), k=1, exclude={"a"})] == ["c"]

    index.save(tmp_path / "idx")
    loaded = VectorIndex.load(tmp_path / "idx")
    assert loaded.search_many(np.array([[0, 1, 0]]), k=1) == [["b"]]
    assert loaded.search(np.array([1, 0, 0]), k=1)[0].metadata == {"n": 1}


def test_vector_index_rejects_wrong_dim():
    with pytest.raises(ValueError):
        VectorIndex(dim=4).add(["x"], np.zeros((1, 3)))


# ---------- chunking ----------
def test_chunk_text_respects_size_and_overlaps():
    text = " ".join(f"Sentence number {i} about the patient." for i in range(40))
    chunks = chunk_text(text, max_chars=200)
    assert len(chunks) > 3 and all(len(c) <= 260 for c in chunks)
    last_sentence_of_first = chunks[0].split(". ")[-1]
    assert last_sentence_of_first.rstrip(".") in chunks[1]  # one-sentence overlap


# ---------- RGB vs grayscale ----------
def test_color_image_keeps_color_but_gray_rgb_does_not():
    gray_as_rgb = np.repeat(np.random.randint(0, 255, (40, 40, 1), dtype=np.uint8), 3, axis=2)
    assert not load_upload("xray.png", _png(gray_as_rgb)).is_color
    color = np.zeros((40, 40, 3), dtype=np.uint8)
    color[..., 0], color[..., 2] = 200, 60  # pink/purple like H&E stain
    loaded = load_upload("slide.png", _png(color))
    assert loaded.is_color and loaded.color.shape == (40, 40, 3)


# ---------- RAG citation handling (LLM mocked) ----------
class _FakeKB:
    def retrieve(self, query, k=4):
        return [SearchHit("n1#0", 0.9, {"text": "Fever for 3 days.", "source": "note_1"}),
                SearchHit("n1#1", 0.7, {"text": "SpO2 89% on air.", "source": "note_1"})]


class _FakeChain:
    def __init__(self, reply):
        self.reply = reply
        self.seen = None

    def invoke(self, inputs):
        self.seen = inputs
        return self.reply


def test_answer_question_parses_valid_citations_only():
    chain = _FakeChain("Febrile for 3 days [1] and hypoxic [2]; see also [7].")
    result = answer_question("Is the patient hypoxic?", _FakeKB(), chain=chain)
    assert isinstance(result, RagAnswer)
    assert result.cited_numbers == [1, 2]  # [7] does not exist -> ignored
    assert "[1] (note_1) Fever for 3 days." in chain.seen["context"]


# ---------- model-backed tests ----------
def _iu_ready() -> bool:
    from medassist.datasets.iu_xray import STUDIES_FILE
    from medassist.image_pipeline.cxr import PROBE_FILE

    return STUDIES_FILE.exists() and PROBE_FILE.exists()


models = pytest.mark.skipif(not _iu_ready(), reason="IU X-Ray data / probe not prepared")


@pytest.fixture(scope="module")
def classifier():
    from medassist.image_pipeline.cxr import CXRClassifier
    return CXRClassifier()


@models
def test_classifier_defaults_to_ensemble_and_scores_all_findings(classifier):
    from medassist.datasets.iu_xray import FINDINGS, load_studies

    study = load_studies("test")[0]
    scores = classifier.classify(Image.open(study.image_path()))
    assert classifier.mode == "ensemble"
    assert {s.finding for s in scores} == set(FINDINGS)
    assert all(0.0 <= s.probability <= 1.0 for s in scores)


@models
def test_occlusion_heatmap_shape_and_range(classifier):
    from medassist.datasets.iu_xray import load_studies

    img = Image.open(load_studies("test")[0].image_path())
    heat = classifier.occlusion_heatmap(img, "cardiomegaly", patch=56, stride=56)  # coarse = fast
    side = max(img.size)
    assert heat.shape == (side, side) and 0.0 <= heat.min() and heat.max() <= 1.0


@models
def test_modality_gate_rejects_non_chest_images(classifier):
    from medassist.datasets.iu_xray import load_studies
    from medassist.image_pipeline.cxr import detect_modality

    cxr = Image.open(load_studies("test")[0].image_path())
    assert detect_modality(classifier.model, cxr).is_chest_xray
    chart = Image.new("RGB", (300, 300), "white")
    for x in range(30, 270, 40):  # crude bar chart
        chart.paste((40, 90, 200), (x, 300 - (x % 200) - 40, x + 25, 280))
    assert not detect_modality(classifier.model, chart).is_chest_xray


@models
def test_api_analyze_and_heatmap(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from api import main
    from medassist.datasets.iu_xray import load_studies
    from medassist.ingestion.store import CaseStore

    monkeypatch.setattr(main, "get_store", lambda: CaseStore(tmp_path))
    client = TestClient(main.app)
    cid = client.post("/cases").json()["case_id"]
    path = load_studies("test")[0].image_path()
    image_id = client.post(f"/cases/{cid}/images", files={"file": ("cxr.png", path.read_bytes())}).json()["image_id"]

    body = client.post(f"/cases/{cid}/images/{image_id}/analyze").json()
    assert body["modality"]["is_chest_xray"] and len(body["findings"]) == 9
    assert len(body["similar_cases"]) == 3

    bad = client.get(f"/cases/{cid}/images/{image_id}/heatmap", params={"finding": "broken leg"})
    assert bad.status_code == 422
