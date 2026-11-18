"""Score the MedAssist usability study (docs/user_study/).

SUS (Brooke 1996): odd items contribute (score - 1), even items (5 - score); the sum x 2.5
gives 0-100. Adjective bands after Bangor, Kortum & Miller (2009).
Usage: python -m scripts.evaluate.user_study [--csv docs/user_study/responses.csv]
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

from medassist.config import PROJECT_ROOT
from medassist.evaluation.results import save_results

DEFAULT_CSV = PROJECT_ROOT / "docs" / "user_study" / "responses.csv"
SUS_ITEMS = [f"S{i}" for i in range(1, 11)]
EXPLAIN_ITEMS = {
    "E1": "urgency and reasons clear",
    "E2": "citations helped verify answers",
    "E3": "heatmap helped understand the image model",
    "E4": "noticed note/image disagreement",
    "E5": "honest about uncertainty",
    "E6": "note + image more useful than either alone",
}
TASKS = [f"T{i}" for i in range(1, 6)]
# t critical values (two-sided 95%) for small samples; df -> t
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
       11: 2.20, 12: 2.18, 13: 2.16, 14: 2.14, 15: 2.13, 20: 2.09, 30: 2.04}


def sus_score(row: dict) -> float:
    total = 0
    for i, item in enumerate(SUS_ITEMS, start=1):
        v = int(row[item])
        if not 1 <= v <= 5:
            raise ValueError(f"{row['participant']} {item}={v}: SUS answers must be 1-5")
        total += (v - 1) if i % 2 else (5 - v)
    return total * 2.5


def adjective(score: float) -> str:
    for limit, name in ((85.6, "Excellent"), (72.6, "Good"), (52.1, "OK"), (39.2, "Poor")):
        if score >= limit:
            return name
    return "Awful"


def describe(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    t = T95.get(n - 1, T95[min((k for k in T95 if k >= n - 1), default=30)]) if n > 1 else float("nan")
    half = t * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": round(mean, 1), "sd": round(sd, 1),
            "ci95": [round(mean - half, 1), round(mean + half, 1)] if n > 1 else None,
            "median": round(statistics.median(values), 1)}


def summarise(rows: list[dict]) -> dict:
    sus = [sus_score(r) for r in rows]
    out = {"sus": describe(sus), "sus_per_participant": {r["participant"]: s for r, s in zip(rows, sus)}}
    if sus:
        out["sus"]["adjective"] = adjective(statistics.fmean(sus))
    out["explanation_items"] = {
        k: {"label": label, **describe([float(r[k]) for r in rows if r.get(k)])} for k, label in EXPLAIN_ITEMS.items()
    }
    tasks = {}
    for t in TASKS:
        succ = [float(r[f"{t}_success"]) for r in rows if r.get(f"{t}_success")]
        secs = [float(r[f"{t}_seconds"]) for r in rows if r.get(f"{t}_seconds")]
        tasks[t] = {"success_rate": round(statistics.fmean(succ), 2) if succ else None,
                    "median_seconds": round(statistics.median(secs)) if secs else None, "n": len(succ)}
    out["tasks"] = tasks
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    args = parser.parse_args()
    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"{path} not found. Copy docs/user_study/responses_template.csv to responses.csv and fill it in.")
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r.get("S1")]
    if not rows:
        raise SystemExit("No completed rows (S1 empty).")

    result = {"all": summarise(rows)}
    for group in sorted({r["background"] for r in rows}):
        result[group] = summarise([r for r in rows if r["background"] == group])

    a = result["all"]
    print(f"Participants: {a['sus']['n']}")
    ci = a["sus"]["ci95"]
    print(f"SUS: mean {a['sus']['mean']} (SD {a['sus']['sd']}"
          + (f", 95% CI {ci[0]}-{ci[1]}" if ci else "") + f") -> {a['sus']['adjective']}")
    for group in result:
        if group != "all":
            print(f"  {group}: n={result[group]['sus']['n']}, mean {result[group]['sus']['mean']}")
    print("Explanation items (1-5):")
    for k, v in a["explanation_items"].items():
        if v.get("n"):
            print(f"  {k} {v['label']:<45} {v['mean']}")
    print("Tasks:")
    for t, v in a["tasks"].items():
        print(f"  {t}: success {v['success_rate']}, median {v['median_seconds']} s (n={v['n']})")
    print(f"\nSaved -> {save_results('phase5_user_study', result)}")


if __name__ == "__main__":
    main()
