"""Image-only baseline on IU X-Ray (test split).

Variants
  zero-shot            BiomedCLIP prompt ensembles, no training
  zero-shot + CLAHE    same, with OpenCV contrast enhancement (does it help?)
  linear probe         logistic regression on frozen features (train split)
  ensemble             equal-weight average of zero-shot and probe logits

Metrics per finding: AUROC (threshold-free, primary), F1 at a threshold chosen
on the TRAIN split only. Plus "abnormal vs normal" AUROC using max finding prob.
Per-study predictions are saved for the Phase 3 multimodal comparison.

Usage: python -m scripts.eval_image [--train-probe]
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import cross_val_predict

from medassist.datasets.iu_xray import DATA_DIR, FINDINGS, load_studies
from medassist.evaluation.results import save_results
from medassist.image_pipeline.cxr import CXRClassifier
from medassist.image_pipeline.preprocess import enhance_contrast

FEATURE_CACHE = DATA_DIR / "features"


def label_matrix(studies) -> np.ndarray:
    return np.array([[f in s.labels for f in FINDINGS] for s in studies], dtype=int)


def features_for(clf: CXRClassifier, studies, split: str, clahe: bool) -> np.ndarray:
    path = FEATURE_CACHE / f"{split}{'_clahe' if clahe else ''}.npy"
    if path.exists():
        cached = np.load(path)
        if len(cached) == len(studies):
            return cached
    images = []
    for s in studies:
        img = Image.open(s.image_path()).convert("L")
        if clahe:
            img = Image.fromarray(enhance_contrast(np.asarray(img)))
        images.append(img.convert("RGB"))
    feats = clf.embed(images)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, feats)
    return feats


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if y.sum() == 0:
        return 0.5
    candidates = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    return float(max(candidates, key=lambda t: f1_score(y, p >= t, zero_division=0)))


def evaluate(name: str, p_train: np.ndarray, p_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> dict:
    per_finding = {}
    for j, f in enumerate(FINDINGS):
        if y_test[:, j].sum() == 0 or y_test[:, j].sum() == len(y_test):
            continue
        t = best_threshold(y_train[:, j], p_train[:, j])
        per_finding[f] = {
            "auroc": round(float(roc_auc_score(y_test[:, j], p_test[:, j])), 3),
            "f1": round(float(f1_score(y_test[:, j], p_test[:, j] >= t, zero_division=0)), 3),
            "threshold": round(t, 4),
            "positives": int(y_test[:, j].sum()),
        }
    abnormal = (y_test.sum(1) > 0).astype(int)
    summary = {
        "macro_auroc": round(float(np.mean([v["auroc"] for v in per_finding.values()])), 3),
        "macro_f1": round(float(np.mean([v["f1"] for v in per_finding.values()])), 3),
        "abnormal_vs_normal_auroc": round(float(roc_auc_score(abnormal, p_test.max(1))), 3),
    }
    return {"variant": name, "summary": summary, "per_finding": per_finding}


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def calibration_report(y_test: np.ndarray, p_cal: np.ndarray, p_raw: np.ndarray) -> dict:
    """Mean ECE and Brier score across findings, before vs after Platt scaling."""
    def mean_metric(fn, p):
        return round(float(np.mean([fn(y_test[:, j], p[:, j]) for j in range(y_test.shape[1])])), 4)
    brier = lambda y, p: float(np.mean((p - y) ** 2))  # noqa: E731
    return {
        "ece_raw": mean_metric(expected_calibration_error, p_raw),
        "ece_calibrated": mean_metric(expected_calibration_error, p_cal),
        "brier_raw": mean_metric(brier, p_raw),
        "brier_calibrated": mean_metric(brier, p_cal),
    }


def train_probe(x_train: np.ndarray, y_train: np.ndarray) -> tuple[dict, np.ndarray]:
    mu, sd = x_train.mean(0), x_train.std(0) + 1e-6
    xs = (x_train - mu) / sd
    weights, bias, oof = [], [], np.zeros_like(y_train, dtype=float)
    for j in range(len(FINDINGS)):
        model = LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000)
        # Out-of-fold predictions on train -> honest threshold selection.
        oof[:, j] = cross_val_predict(model, xs, y_train[:, j], cv=5, method="predict_proba")[:, 1]
        model.fit(xs, y_train[:, j])
        weights.append(model.coef_[0].tolist())
        bias.append(float(model.intercept_[0]))
    thresholds = {f: best_threshold(y_train[:, j], oof[:, j]) for j, f in enumerate(FINDINGS)}
    probe = {"findings": FINDINGS, "weights": weights, "bias": bias,
             "mean": mu.tolist(), "std": sd.tolist(), "thresholds": thresholds}
    return probe, oof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-probe", action="store_true", help="(re)train and save the linear probe")
    args = parser.parse_args()

    train, test = load_studies("train"), load_studies("test")
    y_train, y_test = label_matrix(train), label_matrix(test)
    clf = CXRClassifier(mode="zero-shot")
    print(f"train={len(train)} test={len(test)} device={clf.model.device}")

    results = []
    for clahe in (False, True):
        x_tr, x_te = features_for(clf, train, "train", clahe), features_for(clf, test, "test", clahe)
        name = "zero-shot" + (" + CLAHE" if clahe else "")
        results.append(evaluate(name, clf.probabilities(x_tr, "zero-shot"),
                                clf.probabilities(x_te, "zero-shot"), y_train, y_test))
        if not clahe:
            x_train_plain, x_test_plain = x_tr, x_te

    probe, oof = train_probe(x_train_plain, y_train)
    clf.probe = probe
    p_test_probe = clf.probabilities(x_test_plain, "probe")
    results.append(evaluate("linear probe", oof, p_test_probe, y_train, y_test))

    # Ensemble: equal-weight logit average (weight fixed at 0.5, not tuned).
    def logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    z_train = clf.probabilities(x_train_plain, "zero-shot")
    raw_train_logit = 0.5 * clf.zero_shot_logits(x_train_plain) + 0.5 * logit(oof)  # OOF on train

    # Platt calibration per finding on train OOF logits (test data never touched).
    a, b = [], []
    for j in range(len(FINDINGS)):
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(raw_train_logit[:, j:j + 1], y_train[:, j])
        a.append(float(platt.coef_[0, 0]))
        b.append(float(platt.intercept_[0]))
    probe["ensemble_calibration"] = {"a": a, "b": b}
    p_train_ens = 1 / (1 + np.exp(-(raw_train_logit * np.array(a) + np.array(b))))
    clf.probe = probe
    p_test_ens = clf.probabilities(x_test_plain, "ensemble")
    results.append(evaluate("ensemble (ZS+probe)", p_train_ens, p_test_ens, y_train, y_test))
    results[-1]["calibration"] = calibration_report(y_test, p_test_ens,
                                                    clf.probabilities(x_test_plain, "ensemble-raw"))

    probe["zero_shot_thresholds"] = {f: best_threshold(y_train[:, j], z_train[:, j]) for j, f in enumerate(FINDINGS)}
    probe["ensemble_thresholds"] = {f: best_threshold(y_train[:, j], p_train_ens[:, j]) for j, f in enumerate(FINDINGS)}
    if args.train_probe or not (DATA_DIR / "linear_probe.json").exists():
        CXRClassifier.save_probe(probe)
        print("Saved linear probe + thresholds ->", DATA_DIR / "linear_probe.json")

    print(f"\n{'variant':<20}{'macro AUROC':>12}{'macro F1':>10}{'abn-vs-norm':>13}")
    for r in results:
        s = r["summary"]
        print(f"{r['variant']:<20}{s['macro_auroc']:>12}{s['macro_f1']:>10}{s['abnormal_vs_normal_auroc']:>13}")
    print(f"\nEnsemble calibration (test): {results[-1]['calibration']}")
    print(f"\n{'finding':<18}{'pos':>5}" + "".join(f"{r['variant'][:14]:>16}" for r in results) + "   (AUROC)")
    for f in FINDINGS:
        if f in results[0]["per_finding"]:
            row = "".join(f"{r['per_finding'][f]['auroc']:>16}" for r in results)
            print(f"{f:<18}{results[0]['per_finding'][f]['positives']:>5}{row}")

    predictions = {
        s.uid: {"zero_shot": clf.probabilities(x_test_plain[i:i + 1], "zero-shot")[0].round(4).tolist(),
                "probe": p_test_probe[i].round(4).tolist(),
                "ensemble": p_test_ens[i].round(4).tolist()}
        for i, s in enumerate(test)
    }
    path = save_results("phase2_image_only", {
        "dataset": {"name": "IU X-Ray (frontal)", "train": len(train), "test": len(test), "findings": FINDINGS},
        "results": results,
        "test_predictions": predictions,
    })
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
