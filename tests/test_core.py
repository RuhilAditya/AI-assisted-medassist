"""Fast unit tests (no model downloads). Run: pytest -q"""
from __future__ import annotations

import io
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image
from pydicom.data import get_testdata_file

from medassist.evaluation.metrics import classification_metrics, recall_at_k
from medassist.image_pipeline.preprocess import enhance_contrast, to_model_image
from medassist.ingestion.loaders import DataValidationError, load_upload
from medassist.llm.ollama_client import LLMError, OllamaClient


def _png_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


# ---------- ingestion ----------
def test_load_png_returns_grayscale_uint8():
    rgb = np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
    img = load_upload("scan.png", _png_bytes(rgb))
    assert img.pixels.shape == (32, 48) and img.pixels.dtype == np.uint8
    assert img.source_format == "png"


def test_load_dicom_applies_windowing_and_keeps_only_safe_tags():
    path = Path(get_testdata_file("CT_small.dcm"))
    img = load_upload(path.name, path.read_bytes())
    assert img.pixels.dtype == np.uint8
    assert img.pixels.min() == 0 and img.pixels.max() == 255  # full dynamic range
    assert "PatientName" not in img.metadata and "PatientID" not in img.metadata
    assert img.metadata.get("Modality") == "CT"


@pytest.mark.parametrize(
    "name,content,message",
    [
        ("a.png", b"", "empty"),
        ("a.txt", b"hello", "Only PNG"),
        ("a.png", b"not an image", "Unable to read"),
    ],
)
def test_load_upload_rejects_bad_input(name, content, message):
    with pytest.raises(DataValidationError, match=message):
        load_upload(name, content)


# ---------- preprocessing ----------
def test_clahe_preserves_shape_and_increases_contrast():
    # Narrow-range intensity gradient, like soft tissue on an under-exposed film.
    low_contrast = np.tile(np.linspace(100, 140, 64).astype(np.uint8), (64, 1))
    enhanced = enhance_contrast(low_contrast)
    assert enhanced.shape == low_contrast.shape
    assert enhanced.std() > low_contrast.std()


def test_to_model_image_is_rgb():
    assert to_model_image(np.zeros((10, 10), dtype=np.uint8)).mode == "RGB"


# ---------- metrics ----------
def test_classification_metrics_binary_with_auroc():
    m = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], y_score=np.array([0.1, 0.6, 0.8, 0.9]))
    assert m.accuracy == 0.75 and m.auroc == 1.0


def test_classification_metrics_rejects_mismatch():
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0])


def test_recall_at_k():
    ranked = [["a", "b", "c"], ["x", "y", "z"]]
    relevant = [{"b"}, {"q"}]
    assert recall_at_k(ranked, relevant, k=1) == 0.0
    assert recall_at_k(ranked, relevant, k=2) == 0.5


# ---------- LLM client ----------
def _fake_response(json_body):
    resp = mock.Mock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = json_body
    return resp


def test_ollama_chat_returns_content():
    with mock.patch("requests.post", return_value=_fake_response({"message": {"content": " hi "}})):
        assert OllamaClient().chat([{"role": "user", "content": "x"}]) == "hi"


def test_ollama_chat_rejects_empty_reply():
    with mock.patch("requests.post", return_value=_fake_response({"message": {"content": ""}})):
        with pytest.raises(LLMError, match="empty"):
            OllamaClient().chat([{"role": "user", "content": "x"}])
