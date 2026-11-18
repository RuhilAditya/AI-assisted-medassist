"""BiomedCLIP wrapper: zero-shot image labelling and image/text embeddings.

BiomedCLIP maps images and text into one shared 512-d space. That gives us:
  * zero-shot classification: compare an image with prompts like
    "a chest X-ray showing pneumonia" and softmax the similarities;
  * cross-modal retrieval: find reports whose embedding is closest to an image.
It does NOT generate free text; the LLM does that from these outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from medassist.config import get_settings

PROMPT_TEMPLATE = "this is a photo of {}"  # template from the BiomedCLIP paper


@dataclass(frozen=True)
class LabelScore:
    label: str
    score: float


class BiomedCLIP:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        import open_clip  # imported lazily: heavy

        settings = get_settings()
        self.model_name = model_name or settings.biomedclip_model
        self.device = device or settings.device
        self.model, self.preprocess = open_clip.create_model_from_pretrained(self.model_name)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.to(self.device).eval()
        self.context_length = 256

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        features = self.model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy()

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts, context_length=self.context_length).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy()

    @torch.no_grad()
    def zero_shot(
        self,
        image: Image.Image,
        labels: list[str],
        template: str = PROMPT_TEMPLATE,
    ) -> list[LabelScore]:
        """Score each label against the image; returns labels sorted by probability."""
        image_vec = torch.from_numpy(self.embed_images([image])).to(self.device)
        text_vecs = torch.from_numpy(
            self.embed_texts([template.format(label) for label in labels])
        ).to(self.device)
        logit_scale = self.model.logit_scale.exp()
        probs = (logit_scale * image_vec @ text_vecs.T).softmax(dim=-1)[0].cpu().tolist()
        ranked = sorted(zip(labels, probs), key=lambda pair: pair[1], reverse=True)
        return [LabelScore(label, float(score)) for label, score in ranked]


@lru_cache(maxsize=1)
def get_biomedclip() -> BiomedCLIP:
    """Process-wide singleton so the model loads once."""
    return BiomedCLIP()
