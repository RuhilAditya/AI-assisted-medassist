"""Load uploaded medical images (DICOM / PNG / JPEG) into a common format.

Every loader returns a `LoadedImage`: an 8-bit grayscale numpy array plus
non-identifying metadata. Patient-identifying DICOM tags are never copied out.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixels import apply_modality_lut, apply_voi_lut

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
IMAGE_SUFFIXES = {"png", "jpg", "jpeg"}
DICOM_SUFFIXES = {"dcm", "dicom"}

# Technical tags that are safe to keep (no PHI).
SAFE_DICOM_TAGS = (
    "Modality",
    "BodyPartExamined",
    "ViewPosition",
    "PhotometricInterpretation",
    "Rows",
    "Columns",
    "BitsStored",
)


class DataValidationError(ValueError):
    """Raised when an upload cannot be safely loaded."""


@dataclass(frozen=True)
class LoadedImage:
    name: str
    pixels: np.ndarray  # uint8, H x W, grayscale
    source_format: str  # "dicom" | "png" | "jpeg"
    metadata: dict[str, str] = field(default_factory=dict)


def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Min-max scale any numeric array to 0-255."""
    array = array.astype(np.float32)
    lo, hi = float(array.min()), float(array.max())
    if hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    return ((array - lo) / (hi - lo) * 255.0).astype(np.uint8)


def load_dicom(name: str, content: bytes) -> LoadedImage:
    try:
        ds = pydicom.dcmread(io.BytesIO(content))
        raw = ds.pixel_array
    except Exception as exc:  # pydicom raises many types
        raise DataValidationError(f"Unable to read DICOM file '{name}'.") from exc

    if raw.ndim == 3 and ds.get("SamplesPerPixel", 1) == 1:
        raw = raw[raw.shape[0] // 2]  # multi-frame: take middle slice
    elif raw.ndim == 3:
        raw = np.asarray(Image.fromarray(_to_uint8(raw)).convert("L"))

    # Stored values -> real units (HU for CT), then apply the scanner's
    # display window. Skipping this is what clips 12/16-bit X-rays.
    pixels = apply_modality_lut(raw, ds)
    pixels = apply_voi_lut(pixels, ds)
    pixels = _to_uint8(pixels)

    if ds.get("PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = 255 - pixels  # MONOCHROME1 means "white = low value"

    metadata = {tag: str(ds.get(tag)) for tag in SAFE_DICOM_TAGS if tag in ds}
    return LoadedImage(name=name, pixels=pixels, source_format="dicom", metadata=metadata)


def load_standard_image(name: str, content: bytes) -> LoadedImage:
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except Exception as exc:
        raise DataValidationError(f"Unable to read image '{name}'.") from exc
    pixels = np.asarray(image.convert("L"), dtype=np.uint8)
    fmt = "png" if (image.format or "").upper() == "PNG" else "jpeg"
    return LoadedImage(name=name, pixels=pixels, source_format=fmt)


def load_upload(name: str, content: bytes) -> LoadedImage:
    """Validate an upload and dispatch to the right loader."""
    if not content:
        raise DataValidationError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise DataValidationError("Uploaded files must be 20 MB or smaller.")
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if suffix in DICOM_SUFFIXES or content[128:132] == b"DICM":
        return load_dicom(name, content)
    if suffix in IMAGE_SUFFIXES:
        return load_standard_image(name, content)
    raise DataValidationError("Only PNG, JPEG, and DICOM files are supported.")
