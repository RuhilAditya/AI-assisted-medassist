"""Phase 3 evaluation: does fusing text with images help, and does conflict detection work?

A. REAL DATA (IU X-Ray test split): image-only vs text-only vs fused, where the
   text is the study's real *indication* run through our NER, and fusion is the
   deployed knowledge-based log-odds update (evidence.fuse_findings). No LLM.
B. CONTROLLED SIMULATION: the same real images and labels, but each study gets a
   synthetic note generated from its labels by a documented, noisy symptom model
   (SIM_* constants below). Shows how fusion behaves when the text IS informative.
   It is a sanity check of the mechanism, not clinical evidence: the text signal
   strength is set by us.
C. CONFLICT DETECTION: discordances injected into the simulated notes ("no X"
   where the image shows X; "known X" where it does not). Reports how many are
   detected and the alert rate on notes without injections.

Paired bootstrap (1,000 resamples) gives 95% CIs for fused minus image-only AUROC.
Usage: python -m scripts.eval_fusion
"""
from __future__ import annotations

import json
import math
import random

import numpy as np
from sklearn.metrics import roc_auc_score

from medassist.datasets.iu_xray import FINDINGS, load_studies
from medassist.evaluation.results import load_results, save_results
from medassist.fusion.conflicts import detect_conflicts
from medassist.fusion.evidence import EvidenceBundle, NoteEvidence, fuse_findings, text_signals
from medassist.ingestion.clinical_ner import get_ner

SEED = 7
N_BOOT = 1000

# ---- simulation model (documented; probabilities are ours, not clinical estimates) ----
SIM_SYMPTOMS = {
    "cardiomegaly": ["orthopnea", "bilateral leg swelling"],
    "lung opacity": ["fever", "productive cough"],
    "atelectasis": ["shortness of breath"],
    "pleural effusion": ["pleuritic chest pain", "shortness of breath"],
    "granuloma": ["history of tuberculosis"],
    "nodule": ["weight loss", "hemoptysis"],
    "emphysema": ["wheezing", "chronic cough"],
    "pulmonary edema": ["orthopnea", "crackles"],
    "fracture": ["a recent fall", "rib pain"],
}
SIM_ASSERTION = {
    "cardiomegaly": "known congestive heart failure", "lung opacity": "clinically suspected pneumonia",
    "atelectasis": "suspected atelectasis", "pleural effusion": "pleural effusion on prior ultrasound",
    "granuloma": "old granulomatous disease", "nodule": "known pulmonary nodule",
    "emphysema": "known COPD", "pulmonary edema": "fluid overload", "fracture": "rib fracture",
}
SIM_FINDING_NAME = {f: f for f in FINDINGS}
P_SYMPTOM, P_ASSERT, P_DISTRACTOR, P_NEGATE, P_INJECT = 0.5, 0.25, 0.08, 0.15, 0.15


def simulate_note(labels: list[str], rng: random.Random) -> str:
    present = [f for f in labels if f in FINDINGS]
    parts = ["Adult patient referred for a chest X-ray."]
    symptoms: list[str] = []
    for f in present:
        symptoms += [s for s in SIM_SYMPTOMS[f] if rng.random() < P_SYMPTOM]
        if rng.random() < P_ASSERT:
            parts.append(f"History notable for {SIM_ASSERTION[f]}.")
    for f in FINDINGS:
        if f not in present and rng.random() < P_DISTRACTOR:
            symptoms.append(rng.choice(SIM_SYMPTOMS[f]))
    if symptoms:
        parts.append("Presents with " + ", ".join(dict.fromkeys(symptoms)) + ".")
    for f in FINDINGS:
        if f not in present and rng.random() < P_NEGATE:
            parts.append(f"No {SIM_SYMPTOMS[f][0]}.")
    hypoxic = any(f in present for f in ("lung opacity", "pulmonary edema", "pleural effusion")) and rng.random() < 0.5
    parts.append(f"SpO2 {rng.randint(88, 93) if hypoxic else rng.randint(96, 99)}%.")
    return " ".join(parts)


def note_evidence(text: str, ner, note_id: str = "note_1") -> NoteEvidence:
    ex = ner.extract(text)
    return NoteEvidence(note_id, ex.active(), ex.negated(), ex.past_history(), ex.family_history(),
                        ex.active("MEDICATION"), ex.active("TEST"), ex.vitals)


def score_matrices(studies, notes: dict[str, NoteEvidence], img: dict[str, list[float]], prevalence: np.ndarray):
    p_img, p_txt, p_fus = [], [], []
    for s in studies:
        bundle = EvidenceBundle(notes=[notes[s.uid]])
        findings = [{"finding": f, "probability": img[s.uid][j], "positive": img[s.uid][j] >= 0.5}
                    for j, f in enumerate(FINDINGS)]
        fused = fuse_findings(findings, bundle)
        p_img.append(img[s.uid])
        p_fus.append([f.fused_probability for f in fused])
        p_txt.append([1 / (1 + math.exp(-(math.log(prevalence[j] / (1 - prevalence[j]))
                                         + sum(x.llr for x in text_signals(f, bundle)))))
                      for j, f in enumerate(FINDINGS)])
    return np.array(p_img), np.array(p_txt), np.array(p_fus)


def macro_auroc(y: np.ndarray, p: np.ndarray) -> float:
    cols = [j for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in cols]))


def compare(name: str, y, p_img, p_txt, p_fus, rng: np.random.Generator) -> dict:
    base = {k: round(macro_auroc(y, p), 3) for k, p in (("image_only", p_img), ("text_only", p_txt), ("fused", p_fus))}
    diffs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        try:
            diffs.append(macro_auroc(y[idx], p_fus[idx]) - macro_auroc(y[idx], p_img[idx]))
        except ValueError:
            continue
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    per_finding = {}
    for j, f in enumerate(FINDINGS):
        if 0 < y[:, j].sum() < len(y):
            per_finding[f] = {k: round(float(roc_auc_score(y[:, j], p[:, j])), 3)
                              for k, p in (("image_only", p_img), ("text_only", p_txt), ("fused", p_fus))}
    print(f"\n{name}")
    print(f"  macro AUROC  image-only {base['image_only']:.3f} | text-only {base['text_only']:.3f} | "
          f"fused {base['fused']:.3f}   (fused - image: {base['fused'] - base['image_only']:+.3f}, "
          f"95% CI {lo:+.3f} to {hi:+.3f})")
    return {"macro_auroc": base, "fused_minus_image_ci95": [round(lo, 3), round(hi, 3)], "per_finding": per_finding}


def conflict_eval(test, img, ner, rng: random.Random) -> dict:
    injected_total = injected_detected = clean_total = clean_alerts = 0
    detectable_total = detectable_detected = 0
    by_kind = {"denies_present_finding": [0, 0], "asserts_absent_finding": [0, 0]}
    for s in test:
        note = simulate_note(s.labels, rng)
        inject = rng.random() < P_INJECT
        kind = None
        if inject:
            present = [f for f in s.labels if f in FINDINGS]
            if present:
                target, kind = rng.choice(present), "denies_present_finding"
                note += f" No {SIM_FINDING_NAME[target]}."
            else:
                target, kind = rng.choice(FINDINGS), "asserts_absent_finding"
                note += f" Known {SIM_FINDING_NAME[target]}."
        bundle = EvidenceBundle(notes=[note_evidence(note, ner)])
        findings = [{"finding": f, "probability": img[s.uid][j], "positive": img[s.uid][j] >= 0.5,
                     "level": "likely" if img[s.uid][j] >= 0.5 else "possible" if img[s.uid][j] >= 0.2 else "unlikely"}
                    for j, f in enumerate(FINDINGS)]
        bundle.fused = fuse_findings(findings, bundle)
        hits = [c for c in detect_conflicts(bundle) if c.severity in ("high", "medium")]
        if inject:
            injected_total += 1
            wanted = "image_positive_note_denies" if kind == "denies_present_finding" else "note_asserts_image_negative"
            found = any(c.kind == wanted and c.finding == target for c in hits)
            injected_detected += found
            # Detectable = the image classifier actually agrees with the ground truth for the target
            # (a conflict with a finding the classifier cannot see is impossible to detect).
            level = findings[FINDINGS.index(target)]["level"]
            if (kind == "denies_present_finding" and level == "likely") or                (kind == "asserts_absent_finding" and level == "unlikely"):
                detectable_total += 1
                detectable_detected += found
            by_kind[kind][0] += 1
            by_kind[kind][1] += found
        else:
            clean_total += 1
            clean_alerts += bool(hits)
    result = {
        "injected": injected_total,
        "detected": injected_detected,
        "detection_rate": round(injected_detected / max(injected_total, 1), 3),
        "detectable": detectable_total,
        "detection_rate_when_image_agrees_with_truth": round(detectable_detected / max(detectable_total, 1), 3),
        "by_kind": {k: {"injected": v[0], "detected": v[1], "rate": round(v[1] / max(v[0], 1), 3)}
                    for k, v in by_kind.items()},
        "alert_rate_without_injection": round(clean_alerts / max(clean_total, 1), 3),
    }
    print(f"\nC. Conflict detection on injected discordances: {injected_detected}/{injected_total} "
          f"({result['detection_rate']:.0%})")
    for k, v in result["by_kind"].items():
        print(f"  {k:<26} {v['detected']}/{v['injected']} ({v['rate']:.0%})")
    print(f"  when the image classifier agrees with ground truth: {detectable_detected}/{detectable_total} "
          f"({result['detection_rate_when_image_agrees_with_truth']:.0%})  <- conflict logic itself")
    print(f"  alert rate on notes without injection: {result['alert_rate_without_injection']:.0%}")
    return result


def main() -> None:
    train, test = load_studies("train"), load_studies("test")
    img = {uid: v["ensemble"] for uid, v in load_results("phase2_image_only")["test_predictions"].items()}
    y = np.array([[f in s.labels for f in FINDINGS] for s in test], dtype=int)
    prevalence = np.clip(np.array([[f in s.labels for f in FINDINGS] for s in train]).mean(0), 0.01, 0.99)
    ner = get_ner()
    boot = np.random.default_rng(SEED)

    print(f"Test studies: {len(test)}  (image = calibrated ensemble from Phase 2)")
    real_notes = {s.uid: note_evidence(s.indication or "", ner) for s in test}
    a = compare("A. Real indication text (IU X-Ray)", y, *score_matrices(test, real_notes, img, prevalence), boot)
    with_terms = sum(bool(n.active or n.negated) for n in real_notes.values())
    a["indications_with_any_entity"] = with_terms
    print(f"  ({with_terms}/{len(test)} indications contain any clinical entity the NER recognises)")

    sim_rng = random.Random(SEED)
    sim_notes = {s.uid: note_evidence(simulate_note(s.labels, sim_rng), ner) for s in test}
    b = compare("B. Controlled simulation (synthetic notes from labels, noisy)", y,
                *score_matrices(test, sim_notes, img, prevalence), boot)
    b["simulation"] = {"p_symptom": P_SYMPTOM, "p_assert": P_ASSERT, "p_distractor": P_DISTRACTOR,
                       "p_negate": P_NEGATE, "seed": SEED}

    c = conflict_eval(test, img, ner, random.Random(SEED + 1))
    try:
        llm_text = load_results("phase2_text_only_cxr")["summary"]["macro_auroc"]
        a["llm_text_only_reference"] = llm_text
    except FileNotFoundError:
        pass

    path = save_results("phase3_fusion", {"A_real_indication": a, "B_simulation": b, "C_conflicts": c,
                                          "bootstrap_resamples": N_BOOT})
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
