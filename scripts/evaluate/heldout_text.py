"""Held-out text evaluation: de-identification and clinical NER on 120 generated notes
(data/samples/heldout_notes.json, never used for tuning), plus typo robustness.

Metrics
  PHI recall per type      planted identifier fully removed (name, id, date, phone, email, location, facility)
  PHI token leak rate      any token (>=3 chars) of a planted identifier still visible
  Clinical text kept       clinical strings that must survive de-ID
  Active finding recall    present symptoms reported as present by the NER
  Negation P / R / F1      symptoms reported as absent
Typo robustness: the same metrics after swapping adjacent letters in 2% / 5% / 10% of
the words (identifier strings are perturbed too, as in real notes).
Usage: python -m scripts.evaluate.heldout_text
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict

from medassist.config import PROJECT_ROOT
from medassist.evaluation.results import save_results
from medassist.ingestion.clinical_ner import get_ner
from medassist.ingestion.deid import get_deidentifier

DATA = PROJECT_ROOT / "data" / "samples" / "heldout_notes.json"
TYPO_RATES = (0.0, 0.02, 0.05, 0.10)


def typo(text: str, rate: float, rng: random.Random) -> tuple[str, dict[str, str]]:
    """Swap two adjacent letters inside a fraction of words. Returns new text and a map of
    changed words so gold strings can be perturbed consistently."""
    changed: dict[str, str] = {}

    def swap(m: re.Match) -> str:
        w = m.group(0)
        if len(w) < 4 or rng.random() >= rate:
            return w
        i = rng.randint(1, len(w) - 3)
        new = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        changed[w] = new
        return new

    return re.sub(r"[A-Za-z]+", swap, text), changed


def perturb(s: str, changed: dict[str, str]) -> str:
    return re.sub(r"[A-Za-z]+", lambda m: changed.get(m.group(0), m.group(0)), s)


def lenient(expected: str, found: set[str]) -> bool:
    return any(expected in f or f in expected for f in found)


def evaluate(notes: list[dict], rate: float, deid, ner) -> dict:
    rng = random.Random(1000 + int(rate * 100))
    phi_by_type = defaultdict(lambda: [0, 0])
    tok_total = tok_leak = keep_total = keep_kept = 0
    act_total = act_hit = neg_tp = neg_fp = neg_fn = 0
    for n in notes:
        text, changed = typo(n["text"], rate, rng)
        out = deid.deidentify(text).text
        low = out.lower()
        for item in n["phi"]:
            value = perturb(item["value"], changed)
            phi_by_type[item["type"]][0] += 1
            phi_by_type[item["type"]][1] += value.lower() not in low
            for tok in re.findall(r"[A-Za-z0-9]{3,}", value):
                tok_total += 1
                tok_leak += bool(re.search(rf"\b{re.escape(tok)}\b", out, re.IGNORECASE))
        for k in n["keep"]:
            keep_total += 1
            keep_kept += perturb(k, changed).lower() in low

        ex = ner.extract(out)
        active, negated = set(ex.active()), set(ex.negated())
        act_total += len(n["active"])
        act_hit += sum(lenient(a, active) for a in n["active"])
        gold_neg = set(n["negated"])
        tp = {g for g in gold_neg if lenient(g, negated)}
        neg_tp += len(tp)
        neg_fn += len(gold_neg - tp)
        neg_fp += len({x for x in negated if not lenient(x, gold_neg)})

    p = neg_tp / max(neg_tp + neg_fp, 1)
    r = neg_tp / max(neg_tp + neg_fn, 1)
    total = sum(v[0] for v in phi_by_type.values())
    removed = sum(v[1] for v in phi_by_type.values())
    return {
        "typo_rate": rate,
        "phi_recall": round(removed / total, 3),
        "phi_recall_by_type": {k: {"n": v[0], "recall": round(v[1] / v[0], 3)} for k, v in sorted(phi_by_type.items())},
        "phi_token_leak_rate": round(tok_leak / max(tok_total, 1), 3),
        "clinical_text_kept": round(keep_kept / max(keep_total, 1), 3),
        "active_recall": round(act_hit / max(act_total, 1), 3),
        "negation_precision": round(p, 3),
        "negation_recall": round(r, 3),
        "negation_f1": round(2 * p * r / max(p + r, 1e-9), 3),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--name", default="phase5_heldout_text", help="results file name")
    args = parser.parse_args()
    raw = json.loads(open(args.data, encoding="utf-8").read())
    notes = raw["notes"]
    deid, ner = get_deidentifier(), get_ner()
    print(f"Held-out notes: {len(notes)}  (PHI items: {sum(len(n['phi']) for n in notes)})\n")
    runs = []
    for rate in TYPO_RATES:
        m = evaluate(notes, rate, deid, ner)
        runs.append(m)
        print(f"typos {rate:>4.0%}: PHI recall {m['phi_recall']:.3f} | token leak {m['phi_token_leak_rate']:.3f} | "
              f"clinical kept {m['clinical_text_kept']:.3f} | active recall {m['active_recall']:.3f} | "
              f"negation F1 {m['negation_f1']:.3f}")
    print("\nPHI recall by type (no typos):")
    for k, v in runs[0]["phi_recall_by_type"].items():
        print(f"  {k:<9} {v['recall']:.3f}  (n={v['n']})")
    path = save_results(args.name, {"dataset": {"name": "generated held-out notes", "n": len(notes), "seed": raw.get("seed")},
                                                "runs": runs})
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
