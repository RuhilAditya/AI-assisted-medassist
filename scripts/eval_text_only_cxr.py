"""Text-only baseline on IU X-Ray: predict findings from the clinical indication.

The model sees ONLY the indication (e.g. "Chest pain, shortness of breath"),
never the report (labels were derived from the report: that would be leakage).
Llama rates each finding low / medium / high; mapped to 0.1 / 0.5 / 0.9 for AUROC.
This is the text arm of the Phase 3 unimodal-vs-multimodal ablation.

Usage: python -m scripts.eval_text_only_cxr [--limit N]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from medassist.datasets.iu_xray import FINDINGS, load_studies
from medassist.evaluation.results import save_results
from medassist.llm.ollama_client import LLMError, OllamaClient

LEVEL = {"low": 0.1, "medium": 0.5, "high": 0.9}
PROMPT = """A chest X-ray was ordered with this clinical indication:
"{indication}"

Without seeing the image, estimate how likely each finding is to be present on the X-ray.
Return JSON mapping each finding to "low", "medium" or "high":
{findings}"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = all test studies")
    args = parser.parse_args()

    test = [s for s in load_studies("test") if s.indication]
    if args.limit:
        test = test[: args.limit]
    client = OllamaClient()
    print(f"Text-only (indication) baseline: {len(test)} test studies, model={client.settings.llm_model}")

    probs, failures = {}, 0
    for n, s in enumerate(test, 1):
        try:
            reply = client.chat([{"role": "user", "content": PROMPT.format(
                indication=s.indication, findings=json.dumps(FINDINGS))}], temperature=0.0, json_mode=True)
            data = {k.lower().strip(): str(v).lower().strip() for k, v in json.loads(reply).items()}
        except (LLMError, json.JSONDecodeError, AttributeError):
            data, failures = {}, failures + 1
        probs[s.uid] = [LEVEL.get(data.get(f, "low"), 0.1) for f in FINDINGS]
        if n % 50 == 0:
            print(f"  {n}/{len(test)}", flush=True)

    y = np.array([[f in s.labels for f in FINDINGS] for s in test], dtype=int)
    p = np.array([probs[s.uid] for s in test])
    per_finding = {}
    for j, f in enumerate(FINDINGS):
        if 0 < y[:, j].sum() < len(y):
            per_finding[f] = {"auroc": round(float(roc_auc_score(y[:, j], p[:, j])), 3), "positives": int(y[:, j].sum())}
    abnormal = (y.sum(1) > 0).astype(int)
    summary = {
        "macro_auroc": round(float(np.mean([v["auroc"] for v in per_finding.values()])), 3),
        "abnormal_vs_normal_auroc": round(float(roc_auc_score(abnormal, p.max(1))), 3),
        "parse_failures": failures,
    }
    print(f"\nmacro AUROC={summary['macro_auroc']}  abnormal-vs-normal AUROC={summary['abnormal_vs_normal_auroc']}  "
          f"parse failures={failures}")
    for f, v in per_finding.items():
        print(f"  {f:<18} AUROC={v['auroc']}  (pos={v['positives']})")

    path = save_results("phase2_text_only_cxr", {
        "dataset": {"name": "IU X-Ray test split (indication only)", "n": len(test)},
        "summary": summary, "per_finding": per_finding, "test_predictions": probs,
    })
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
