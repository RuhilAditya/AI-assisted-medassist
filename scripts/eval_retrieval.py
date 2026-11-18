"""Cross-modal retrieval evaluation on IU X-Ray.

1. Exact pairing (test split): for each X-ray, rank all test reports by
   BiomedCLIP image-text similarity. Recall@K = own report in top K.
2. Similar-case retrieval (test X-ray -> train knowledge base): label
   precision@K = share of retrieved cases whose MeSH-derived labels overlap the
   query's. Compared with a random-retrieval baseline (the same metric for
   randomly drawn cases), so the gain is measurable.

Also (re)builds the case indexes used by the app.
Usage: python -m scripts.eval_retrieval
"""
from __future__ import annotations

import random

import numpy as np

from medassist.datasets.iu_xray import load_studies
from medassist.evaluation.metrics import recall_at_k
from medassist.evaluation.results import save_results
from medassist.image_pipeline.case_retrieval import build_indexes
from medassist.image_pipeline.cxr import CXRClassifier
from scripts.eval_image import features_for

KS = (1, 5, 10)


def label_overlap(a: list[str], b: list[str]) -> bool:
    return bool(set(a) & set(b))


def main() -> None:
    train, test = load_studies("train"), load_studies("test")
    clf = CXRClassifier(mode="zero-shot")
    model = clf.model

    print("Building knowledge-base indexes from the train split ...")
    report_index, image_index = build_indexes(clf, train)

    x_test = features_for(clf, test, "test", clahe=False)
    test_reports = np.vstack([model.embed_texts([s.report for s in test[i:i + 64]]) for i in range(0, len(test), 64)])

    # 1) exact image -> own report among test reports
    sims = x_test @ test_reports.T
    ranked = [[test[j].uid for j in np.argsort(-row)] for row in sims]
    relevant = [{s.uid} for s in test]
    exact = {f"recall@{k}": round(recall_at_k(ranked, relevant, k), 3) for k in KS}
    exact_random = {f"recall@{k}": round(k / len(test), 3) for k in KS}

    # 2) similar-case retrieval into the train knowledge base
    rng = random.Random(0)
    labels_by_uid = {s.uid: s.labels for s in train}
    similar = {}
    for name, index in (("image->report", report_index), ("image->image", image_index)):
        top = index.search_many(x_test, k=max(KS))
        similar[name] = {
            f"label_precision@{k}": round(float(np.mean([
                np.mean([label_overlap(q.labels, labels_by_uid[uid]) for uid in hits[:k]])
                for q, hits in zip(test, top)
            ])), 3) for k in KS
        }
    random_precision = round(float(np.mean([
        np.mean([label_overlap(q.labels, rng.choice(train).labels) for _ in range(50)]) for q in test
    ])), 3)

    print(f"\nExact report retrieval (test, n={len(test)} candidates)")
    for k in KS:
        print(f"  Recall@{k:<3}: {exact[f'recall@{k}']:.3f}   (random: {exact_random[f'recall@{k}']:.3f})")
    print(f"\nSimilar-case label precision (test -> train KB, n={len(train)}); random baseline = {random_precision:.3f}")
    for name, scores in similar.items():
        print(f"  {name:<14}" + "  ".join(f"P@{k}={scores[f'label_precision@{k}']:.3f}" for k in KS))

    path = save_results("phase2_retrieval", {
        "exact_report_retrieval": {"model": exact, "random": exact_random, "candidates": len(test)},
        "similar_case_label_precision": {**similar, "random": random_precision, "kb_size": len(train)},
    })
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
