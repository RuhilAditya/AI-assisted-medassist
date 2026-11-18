"""Chest X-ray finding classifier on top of BiomedCLIP.

Two modes:
  zero-shot     no training. For each finding, compare the image with an
                ensemble of positive prompts ("chest x-ray showing pleural
                effusion") vs negative prompts ("chest x-ray with no pleural
                effusion"); a 2-way softmax gives P(finding).
  linear probe  one logistic regression per finding on frozen BiomedCLIP image
                features, trained on the IU X-Ray train split. Cheap, and
                usually a large gain over zero-shot (reported as an ablation).

Explainability: occlusion sensitivity. Grey out each patch of the image and
measure how much the finding score drops; big drops = regions the model relied
on. Model-agnostic and easy to explain, unlike attention maps.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from medassist.datasets.iu_xray import DATA_DIR, FINDINGS
from medassist.image_pipeline.biomedclip import BiomedCLIP, get_biomedclip

PROBE_FILE = DATA_DIR / "linear_probe.json"

POSITIVE_TEMPLATES = [
    "chest x-ray showing {}",
    "a chest radiograph with {}",
    "{} seen on chest x-ray",
    "frontal chest x-ray demonstrating {}",
]
NEGATIVE_TEMPLATES = [
    "chest x-ray with no {}",
    "a chest radiograph without {}",
    "normal chest x-ray, no {}",
]
# Radiology wording BiomedCLIP is more likely to have seen in PubMed captions.
PROMPT_ALIASES = {
    "lung opacity": ["lung opacity", "airspace opacity", "pulmonary consolidation"],
    "granuloma": ["calcified granuloma", "granuloma"],
    "nodule": ["pulmonary nodule", "lung nodule"],
    "emphysema": ["emphysema", "hyperinflated lungs"],
    "pulmonary edema": ["pulmonary edema", "pulmonary vascular congestion"],
    "fracture": ["rib fracture", "bone fracture"],
}


@dataclass(frozen=True)
class FindingScore:
    finding: str
    probability: float
    positive: bool


def pad_to_square(image: Image.Image, fill: int = 0) -> Image.Image:
    """Letterbox instead of centre-crop so lung edges are never cut off."""
    w, h = image.size
    side = max(w, h)
    canvas = Image.new(image.mode, (side, side), (fill,) * len(image.getbands()))
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


class CXRClassifier:
    def __init__(self, model: BiomedCLIP | None = None, mode: str = "auto") -> None:
        """mode: "auto" (ensemble if a probe exists, else zero-shot) | "zero-shot" | "probe" | "ensemble"."""
        self.model = model or get_biomedclip()
        self.findings = FINDINGS
        self.probe = self._load_probe() if mode != "zero-shot" else None
        if mode in ("probe", "ensemble") and self.probe is None:
            raise FileNotFoundError(f"No linear probe at {PROBE_FILE}. Run: python -m scripts.eval_image --train-probe")
        self.mode = ("ensemble" if self.probe else "zero-shot") if mode == "auto" else mode
        self._pos, self._neg = self._prompt_embeddings()
        key = {"ensemble": "ensemble_thresholds", "probe": "thresholds"}.get(self.mode)
        self.thresholds = (self.probe or {}).get(key, {}) if key else {}
        if not self.thresholds:
            self.thresholds = (self.probe or {}).get("zero_shot_thresholds", {f: 0.5 for f in self.findings})

    # ---------- prompts ----------
    def _prompt_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        pos, neg = [], []
        for finding in self.findings:
            names = PROMPT_ALIASES.get(finding, [finding])
            p = self.model.embed_texts([t.format(n) for t in POSITIVE_TEMPLATES for n in names]).mean(0)
            q = self.model.embed_texts([t.format(n) for t in NEGATIVE_TEMPLATES for n in names]).mean(0)
            pos.append(p / np.linalg.norm(p))
            neg.append(q / np.linalg.norm(q))
        return np.stack(pos), np.stack(neg)

    # ---------- features & scoring ----------
    def embed(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            out.append(self.model.embed_images([pad_to_square(im.convert("RGB")) for im in images[i:i + batch_size]]))
        return np.vstack(out)

    def zero_shot_logits(self, features: np.ndarray) -> np.ndarray:
        """(N, F) logit = scale * (sim_pos - sim_neg); sigmoid of it == 2-way softmax."""
        scale = float(self.model.model.logit_scale.exp())
        return scale * (features @ self._pos.T - features @ self._neg.T)

    def probe_logits(self, features: np.ndarray) -> np.ndarray:
        w = np.array(self.probe["weights"])        # (F, D)
        b = np.array(self.probe["bias"])           # (F,)
        mu, sd = np.array(self.probe["mean"]), np.array(self.probe["std"])
        return ((features - mu) / sd) @ w.T + b

    def probabilities(self, features: np.ndarray, mode: str | None = None) -> np.ndarray:
        """Ensemble = equal-weight average of zero-shot and probe logits (weight fixed
        at 0.5, not tuned; see docs/design.md for the sensitivity check)."""
        mode = mode or self.mode
        if mode == "probe":
            logits = self.probe_logits(features)
        elif mode == "ensemble":
            logits = 0.5 * self.zero_shot_logits(features) + 0.5 * self.probe_logits(features)
        else:
            logits = self.zero_shot_logits(features)
        return 1.0 / (1.0 + np.exp(-logits))

    def classify(self, image: Image.Image) -> list[FindingScore]:
        probs = self.probabilities(self.embed([image]))[0]
        scores = [
            FindingScore(f, round(float(p), 3), bool(p >= self.thresholds.get(f, 0.5)))
            for f, p in zip(self.findings, probs)
        ]
        return sorted(scores, key=lambda s: s.probability, reverse=True)

    # ---------- linear probe ----------
    @staticmethod
    def _load_probe() -> dict | None:
        return json.loads(PROBE_FILE.read_text(encoding="utf-8")) if PROBE_FILE.exists() else None

    @staticmethod
    def save_probe(probe: dict, path: Path = PROBE_FILE) -> None:
        path.write_text(json.dumps(probe), encoding="utf-8")

    # ---------- explainability ----------
    @torch.no_grad()
    def occlusion_heatmap(
        self, image: Image.Image, finding: str, patch: int = 32, stride: int = 16, batch_size: int = 64
    ) -> np.ndarray:
        """Return an (H, W) map in [0, 1] aligned with the padded-square input image."""
        idx = self.findings.index(finding)
        square = pad_to_square(image.convert("RGB"))
        base = self.model.preprocess(square).to(self.model.device)  # (3, S, S)
        size = base.shape[-1]
        direction = torch.from_numpy(self._pos[idx] - self._neg[idx]).to(self.model.device)

        def score(batch: torch.Tensor) -> torch.Tensor:
            feats = self.model.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.float() @ direction.float()

        reference = score(base.unsqueeze(0))[0]
        coords = [(y, x) for y in range(0, size - patch + 1, stride) for x in range(0, size - patch + 1, stride)]
        drops = torch.zeros(len(coords), device=self.model.device)
        for start in range(0, len(coords), batch_size):
            chunk = coords[start:start + batch_size]
            batch = base.unsqueeze(0).repeat(len(chunk), 1, 1, 1)
            for j, (y, x) in enumerate(chunk):
                batch[j, :, y:y + patch, x:x + patch] = 0.0  # 0 = dataset mean after normalisation
            drops[start:start + len(chunk)] = reference - score(batch)

        heat = np.zeros((size, size), dtype=np.float32)
        counts = np.zeros((size, size), dtype=np.float32)
        for (y, x), d in zip(coords, drops.cpu().numpy()):
            heat[y:y + patch, x:x + patch] += max(float(d), 0.0)
            counts[y:y + patch, x:x + patch] += 1
        heat = heat / np.maximum(counts, 1)
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=patch / 4)
        if heat.max() > 0:
            heat = heat / heat.max()
        return cv2.resize(heat, square.size, interpolation=cv2.INTER_CUBIC).clip(0, 1)


MODALITY_LABELS = [
    "chest X-ray", "bone X-ray", "brain MRI", "abdominal CT scan", "chest CT scan",
    "ultrasound image", "histopathology slide", "photograph of skin", "chart or diagram",
]


@dataclass(frozen=True)
class ModalityCheck:
    modality: str
    confidence: float
    is_chest_xray: bool


def detect_modality(model: BiomedCLIP, image: Image.Image, min_confidence: float = 0.5) -> ModalityCheck:
    """Gate: the CXR classifier is only meaningful on chest X-rays."""
    top = model.zero_shot(pad_to_square(image.convert("RGB")), MODALITY_LABELS)[0]
    return ModalityCheck(top.label, round(top.score, 3), top.label == "chest X-ray" and top.score >= min_confidence)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Colour-map the heatmap (JET) on top of the padded-square grayscale image."""
    square = np.asarray(pad_to_square(image.convert("RGB")))
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    mask = (heatmap[..., None] > 0.15).astype(np.float32) * alpha  # only tint salient areas
    blended = square * (1 - mask) + colored * mask
    return Image.fromarray(blended.astype(np.uint8))
