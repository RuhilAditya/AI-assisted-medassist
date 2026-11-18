"""Phase 1 evaluation: de-identification and clinical NER on synthetic gold notes.

Metrics
  PHI recall        share of planted PHI strings fully removed (higher = safer)
  Token leak rate   share of PHI tokens (len>2) still visible anywhere (lower = safer)
  Clinical utility  share of clinical 'keep' strings preserved after de-ID
  Active recall     expected active findings detected as active
  Negation P/R/F1   for findings the NER marks as negated

Usage:  python -m scripts.eval_ingestion [--verbose]
"""
from __future__ import annotations

import argparse
import json
import re

from medassist.config import PROJECT_ROOT
from medassist.ingestion.clinical_ner import get_ner
from medassist.ingestion.deid import get_deidentifier

SAMPLES = PROJECT_ROOT / "data" / "samples"


def _matches(expected: str, found: set[str]) -> bool:
    """Lenient match: 'seizure' matches 'seizure activity', etc."""
    return any(expected in f or f in expected for f in found)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    gold = json.loads((SAMPLES / "ingestion_gold.json").read_text(encoding="utf-8"))
    deid, ner = get_deidentifier(), get_ner()

    phi_total = phi_removed = token_total = token_leaked = 0
    keep_total = keep_kept = 0
    active_total = active_hit = 0
    neg_tp = neg_fp = neg_fn = 0
    rows = []

    for name, labels in gold.items():
        if name.startswith("_"):
            continue
        raw = (SAMPLES / "notes" / name).read_text(encoding="utf-8")
        result = deid.deidentify(raw)
        out = result.text
        out_lower = out.lower()

        leaked = [p for p in labels["phi"] if p.lower() in out_lower]
        phi_total += len(labels["phi"])
        phi_removed += len(labels["phi"]) - len(leaked)
        for phrase in labels["phi"]:
            for tok in re.findall(r"[A-Za-z0-9]{3,}", phrase):
                token_total += 1
                token_leaked += bool(re.search(rf"\b{re.escape(tok)}\b", out, re.IGNORECASE))

        lost = [k for k in labels["keep"] if k.lower() not in out_lower]
        keep_total += len(labels["keep"])
        keep_kept += len(labels["keep"]) - len(lost)

        extraction = ner.extract(out)
        active = set(extraction.active())
        negated = set(extraction.negated())
        missed_active = [a for a in labels["active"] if not _matches(a, active)]
        active_total += len(labels["active"])
        active_hit += len(labels["active"]) - len(missed_active)

        gold_neg = set(labels["negated"])
        tp = {g for g in gold_neg if _matches(g, negated)}
        fp = {n for n in negated if not _matches(n, gold_neg)}
        neg_tp += len(tp)
        neg_fn += len(gold_neg - tp)
        neg_fp += len(fp)

        rows.append((name, leaked, lost, missed_active, sorted(gold_neg - tp), sorted(fp)))
        if args.verbose:
            print(f"\n----- {name} (de-identified) -----\n{out}")
            print(f"PHI types removed: {result.counts}")
            print(f"Active: {sorted(active)}\nNegated: {sorted(negated)}\nVitals: {extraction.vitals}")

    def pct(a: int, b: int) -> str:
        return f"{100 * a / b:5.1f}%" if b else "  n/a"

    neg_p = neg_tp / (neg_tp + neg_fp) if neg_tp + neg_fp else 0.0
    neg_r = neg_tp / (neg_tp + neg_fn) if neg_tp + neg_fn else 0.0
    neg_f1 = 2 * neg_p * neg_r / (neg_p + neg_r) if neg_p + neg_r else 0.0

    print("\n================ INGESTION EVALUATION ================")
    print(f"Notes evaluated        : {len(rows)}")
    print(f"PHI recall (strings)   : {pct(phi_removed, phi_total)}  ({phi_removed}/{phi_total})")
    print(f"PHI token leak rate    : {pct(token_leaked, token_total)}  ({token_leaked}/{token_total})")
    print(f"Clinical utility kept  : {pct(keep_kept, keep_total)}  ({keep_kept}/{keep_total})")
    print(f"Active finding recall  : {pct(active_hit, active_total)}  ({active_hit}/{active_total})")
    print(f"Negation P / R / F1    : {neg_p:.2f} / {neg_r:.2f} / {neg_f1:.2f}")
    print("\nErrors per note:")
    for name, leaked, lost, missed, neg_missed, neg_wrong in rows:
        issues = {
            "PHI leaked": leaked, "clinical text lost": lost, "active missed": missed,
            "negation missed": neg_missed, "false negation": neg_wrong,
        }
        issues = {k: v for k, v in issues.items() if v}
        print(f"  {name}: {issues or 'OK'}")


if __name__ == "__main__":
    main()
