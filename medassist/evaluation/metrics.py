"""Evaluation metrics (thin, tested wrappers around scikit-learn)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float | None = None


def classification_metrics(
    y_true: list[str] | list[int],
    y_pred: list[str] | list[int],
    y_score: np.ndarray | None = None,
    average: str = "macro",
) -> ClassificationMetrics:
    """Accuracy, macro P/R/F1 and (optionally) AUROC.

    For binary tasks pass `y_score` as the positive-class probability (shape N);
    for multi-class pass an N x C probability matrix (one-vs-rest AUROC).
    """
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true and y_pred must be non-empty and of equal length.")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    auroc = None
    if y_score is not None and len(set(y_true)) > 1:
        multi = np.ndim(y_score) == 2
        auroc = float(roc_auc_score(y_true, y_score, multi_class="ovr" if multi else "raise"))
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        auroc=auroc,
    )


def recall_at_k(ranked_ids: list[list[str]], relevant_ids: list[set[str]], k: int) -> float:
    """Fraction of queries with at least one relevant item in the top-k results."""
    if len(ranked_ids) != len(relevant_ids) or not ranked_ids:
        raise ValueError("ranked_ids and relevant_ids must be non-empty and aligned.")
    hits = sum(bool(set(ranked[:k]) & relevant) for ranked, relevant in zip(ranked_ids, relevant_ids))
    return hits / len(ranked_ids)
