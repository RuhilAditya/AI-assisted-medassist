"""OpenCV preprocessing applied before images reach BiomedCLIP."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


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
