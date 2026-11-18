"""OpenCV preprocessing applied before images reach BiomedCLIP."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from medassist.ingestion.loaders import LoadedImage


def enhance_contrast(pixels: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE: local contrast enhancement, standard for radiographs."""
    if pixels.dtype != np.uint8:
        pixels = cv2.normalize(pixels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(pixels)


def to_model_image(pixels: np.ndarray, apply_clahe: bool = True) -> Image.Image:
    """Grayscale uint8 array -> RGB PIL image ready for an open_clip transform.

    Resizing/cropping to 224x224 is left to the model's own preprocess
    transform so it matches what the model was trained with.
    """
    if apply_clahe:
        pixels = enhance_contrast(pixels)
    return Image.fromarray(pixels).convert("RGB")


def prepare_for_model(image: LoadedImage, apply_clahe: bool = False) -> Image.Image:
    """Pick the right path: colour images (histopathology, photos) keep their
    colour; grayscale images (radiology) optionally get CLAHE.

    CLAHE defaults to off: BiomedCLIP was trained on unenhanced figures from
    PubMed papers, and eval_image.py measures whether CLAHE helps or hurts.
    """
    if image.is_color:
        return Image.fromarray(image.color)
    return to_model_image(image.pixels, apply_clahe=apply_clahe)
