"""Image robustness: how much does the CXR classifier degrade under realistic corruption?

Corruptions (3 severities each), applied to the IU X-Ray test images before BiomedCLIP:
  gaussian noise   sigma 5 / 15 / 30 grey levels      (sensor noise, low dose)
  gaussian blur    sigma 1 / 2 / 4 px at 512 px       (motion, defocus)
  rotation         5 / 10 / 20 degrees                (patient positioning)
  low contrast     factor 0.7 / 0.5 / 0.3             (under/over-exposure)
  JPEG             quality 50 / 20 / 5                (re-saved screenshots, messaging apps)
  low resolution   256 / 128 / 64 px, upsampled back  (thumbnails, phone photos of films)

Metrics per condition: macro AUROC and abnormal-vs-normal AUROC (calibrated ensemble,
the deployed classifier), and mean absolute change in probability vs the clean image.
Usage: python -m scripts.evaluate.robustness
"""
from __future__ import annotations

import io
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import roc_auc_score

from medassist.datasets.iu_xray import FINDINGS, load_studies
from medassist.evaluation.results import save_results
from medassist.image_pipeline.cxr import CXRClassifier

SEED = 11


def noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    a = np.asarray(img, dtype=np.float32)
    return Image.fromarray(np.clip(a + rng.normal(0, sigma, a.shape), 0, 255).astype(np.uint8))


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("L")


def lowres(img: Image.Image, side: int) -> Image.Image:
    small = img.copy()
    small.thumbnail((side, side), Image.Resampling.BILINEAR)
    return small.resize(img.size, Image.Resampling.BILINEAR)


CORRUPTIONS = {
    "gaussian noise": [(f"sigma {s}", lambda im, rng, s=s: noise(im, s, rng)) for s in (5, 15, 30)],
    "gaussian blur": [(f"sigma {s}px", lambda im, rng, s=s: im.filter(ImageFilter.GaussianBlur(s))) for s in (1, 2, 4)],
    "rotation": [(f"{d} deg", lambda im, rng, d=d: im.rotate(d, resample=Image.Resampling.BILINEAR, fillcolor=0)) for d in (5, 10, 20)],
    "low contrast": [(f"x{c}", lambda im, rng, c=c: ImageEnhance.Contrast(im).enhance(c)) for c in (0.7, 0.5, 0.3)],
    "jpeg": [(f"quality {q}", lambda im, rng, q=q: jpeg(im, q)) for q in (50, 20, 5)],
    "low resolution": [(f"{s}px", lambda im, rng, s=s: lowres(im, s)) for s in (256, 128, 64)],
}


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    cols = [j for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return {
        "macro_auroc": round(float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in cols])), 3),
        "abnormal_auroc": round(float(roc_auc_score((y.sum(1) > 0).astype(int), p.max(1))), 3),
    }


def main() -> None:
    test = load_studies("test")
    y = np.array([[f in s.labels for f in FINDINGS] for s in test], dtype=int)
    clf = CXRClassifier(mode="ensemble")
    clean_imgs = [Image.open(s.image_path()).convert("L") for s in test]

    def score(images: list[Image.Image]) -> np.ndarray:
        return clf.probabilities(clf.embed([im.convert("RGB") for im in images]))

    t0 = time.perf_counter()
    p_clean = score(clean_imgs)
    base = metrics(y, p_clean)
    print(f"clean: macro AUROC {base['macro_auroc']}  abnormal AUROC {base['abnormal_auroc']}  "
          f"({time.perf_counter() - t0:.0f}s for {len(test)} images)\n")
    print(f"{'corruption':<16}{'severity':<14}{'macro AUROC':>12}{'drop':>8}{'abn AUROC':>11}{'mean |dp|':>11}")

    results = {"clean": base, "corruptions": {}}
    for name, levels in CORRUPTIONS.items():
        results["corruptions"][name] = []
        for label, fn in levels:
            rng = np.random.default_rng(SEED)
            p = score([fn(im, rng) for im in clean_imgs])
            m = metrics(y, p)
            m.update({"severity": label,
                      "macro_auroc_drop": round(base["macro_auroc"] - m["macro_auroc"], 3),
                      "mean_abs_prob_change": round(float(np.abs(p - p_clean).mean()), 4)})
            results["corruptions"][name].append(m)
            print(f"{name:<16}{label:<14}{m['macro_auroc']:>12}{-m['macro_auroc_drop']:>+8.3f}"
                  f"{m['abnormal_auroc']:>11}{m['mean_abs_prob_change']:>11.4f}")

    path = save_results("phase5_robustness", {"dataset": {"name": "IU X-Ray test", "n": len(test)},
                                              "classifier": "ensemble (calibrated)", **results})
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
