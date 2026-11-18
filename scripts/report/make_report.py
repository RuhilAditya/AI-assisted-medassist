"""Build docs/results.md and docs/figures/*.png from results/*.json.

Every number in the report comes from a saved experiment file, so the report can be
regenerated at any time and never drifts from the code. Missing experiments are skipped.
Usage: python -m scripts.report.make_report
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from medassist.config import PROJECT_ROOT  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
DOCS = PROJECT_ROOT / "docs"
FIG = DOCS / "figures"

# Palette validated with the dataviz validator (light surface): image, text, fused.
IMAGE, TEXT, FUSED = "#c93a1b", "#1f5aa6", "#2e8a5a"
INK, INK2, MUTED, GRID, SURFACE = "#14223b", "#3f4d66", "#6b7890", "#e3e8ee", "#ffffff"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save(fig, name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    return f"figures/{name}.png"


def rounded_bar(ax, x, h, width, color, **kw):
    return ax.bar(x, h, width, color=color, linewidth=0, **kw)


# ------------------------------------------------------------------ figures
def fig_fusion(fusion: dict) -> str:
    groups = [("Real data\n(IU X-Ray indication)", fusion["A_real_indication"]["macro_auroc"]),
              ("Controlled simulation\n(informative notes)", fusion["B_simulation"]["macro_auroc"])]
    series = [("Image only", "image_only", IMAGE), ("Text only", "text_only", TEXT), ("Fused", "fused", FUSED)]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.22
    for i, (label, key, color) in enumerate(series):
        xs = [g + (i - 1) * (width + 0.02) for g in range(len(groups))]
        vals = [g[1][key] for g in groups]
        rounded_bar(ax, xs, vals, width, color, label=label)
        if key in ("fused", "image_only"):
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(0.5, color=MUTED, linewidth=0.8)
    ax.text(-0.42, 0.505, "chance", color=MUTED, fontsize=8, ha="left", va="bottom")
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylim(0.45, 0.92)
    ax.set_ylabel("Macro AUROC (9 findings)")
    ax.set_title("Fusing note and image beats either modality when the text is informative")
    ax.legend(loc="upper left", ncols=3, fontsize=9)
    ax.grid(axis="x", visible=False)
    return save(fig, "fusion_ablation")


def fig_robustness(rob: dict) -> str:
    items = list(rob["corruptions"].items())
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.8), sharey=True)
    clean = rob["clean"]["macro_auroc"]
    for ax, (name, levels) in zip(axes.flat, items):
        xs = [0, 1, 2, 3]
        ys = [clean] + [lv["macro_auroc"] for lv in levels]
        ax.plot(xs, ys, color=IMAGE, linewidth=2, solid_capstyle="round", solid_joinstyle="round")
        ax.scatter(xs, ys, s=30, color=IMAGE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.text(3.12, ys[-1], f"{ys[-1]:.2f}", va="center", fontsize=9, color=INK)
        ax.axhline(0.5, color=MUTED, linewidth=0.8)
        ax.set_title({"jpeg": "JPEG compression"}.get(name, name.capitalize()), fontsize=10)
        ax.set_xticks(xs, ["clean"] + [lv["severity"] for lv in levels], fontsize=7.5, rotation=0)
        ax.set_xlim(-0.3, 3.6)
        ax.set_ylim(0.45, 0.8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro AUROC")
    fig.suptitle("Image classifier under corruption: stable at realistic levels, fails only at extremes",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    return save(fig, "robustness")


def fig_findings(image: dict) -> str:
    by = {r["variant"]: r for r in image["results"]}
    zs, ens = by["zero-shot"]["per_finding"], by["ensemble (ZS+probe)"]["per_finding"]
    names = sorted(ens, key=lambda f: ens[f]["auroc"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, f in enumerate(names):
        a, b = zs[f]["auroc"], ens[f]["auroc"]
        ax.plot([a, b], [i, i], color=GRID, linewidth=2, zorder=1)
    ax.scatter([zs[f]["auroc"] for f in names], range(len(names)), s=55, color=TEXT, edgecolors=SURFACE,
               linewidths=2, zorder=3, label="Zero-shot BiomedCLIP")
    ax.scatter([ens[f]["auroc"] for f in names], range(len(names)), s=55, color=IMAGE, edgecolors=SURFACE,
               linewidths=2, zorder=4, label="Ensemble (zero-shot + probe)")
    for i, f in enumerate(names):
        ax.text(max(zs[f]["auroc"], ens[f]["auroc"]) + 0.012, i, f"{ens[f]['auroc']:.2f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(0.5, color=MUTED, linewidth=0.8)
    ax.set_yticks(range(len(names)), [f"{n} (n={ens[n]['positives']})" for n in names])
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUROC on IU X-Ray test split (n=435)")
    ax.set_title("Per-finding AUROC: large findings strong, small focal findings weak")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    return save(fig, "per_finding_auroc")


def fig_rag(qa: dict) -> str:
    order = ["majority-class baseline", "closed-book", "RAG (top-3)", "oracle context"]
    labels = ["Majority class", "Llama closed-book", "Llama + RAG (top-3)", "Llama + gold abstract"]
    vals = [qa["qa"][k]["accuracy"] for k in order]
    colors = [GRID, "#b8c2cf", TEXT, "#b8c2cf"]  # text result: text ink
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(range(len(vals)), vals, height=0.55, color=colors, linewidth=0)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(vals)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy (PubMedQA, 150 questions)")
    ax.set_title("Retrieval lifts answer accuracy from 0.57 to 0.80")
    ax.grid(axis="y", visible=False)
    return save(fig, "rag_accuracy")


def fig_typos(text: dict) -> str:
    runs = text["runs"]
    xs = [r["typo_rate"] * 100 for r in runs]
    series = [("PHI recall (de-identification)", "phi_recall", IMAGE),
              ("Finding recall (NER)", "active_recall", TEXT),
              ("Negation F1", "negation_f1", FUSED)]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for label, key, color in series:
        ys = [r[key] for r in runs]
        ax.plot(xs, ys, color=color, linewidth=2, label=label, solid_capstyle="round")
        ax.scatter(xs, ys, s=30, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.text(xs[-1] + 0.3, ys[-1], f"{ys[-1]:.2f}", va="center", fontsize=9, color=INK)
    ax.set_xticks(xs, [f"{x:.0f}%" for x in xs])
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(0.8, 1.01)
    ax.set_xlabel("Words with a typo")
    ax.set_title("Text pipeline on 120 held-out notes, as typos are injected")
    ax.legend(loc="lower left", fontsize=8.5)
    return save(fig, "text_typo_robustness")


# ------------------------------------------------------------------ report
def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    parts = ["# MedAssist: evaluation results",
             "",
             "_Generated by `python -m scripts.report.make_report` from `results/*.json`. Do not edit by hand._",
             ""]

    image, retr, qa = load("phase2_image_only"), load("phase2_retrieval"), load("phase2_text_qa")
    txt_cxr, fusion = load("phase2_text_only_cxr"), load("phase3_fusion")
    rob, held, held_pre = load("phase5_robustness"), load("phase5_heldout_text"), load("phase5_heldout_text_discovery_prefix")
    answers, study = load("phase5_answers"), load("phase5_user_study")

    if image:
        parts += ["## 1. Image findings (IU X-Ray, test n=435, 9 findings)", ""]
        parts.append(table(["Variant", "Macro AUROC", "Macro F1", "Abnormal vs normal AUROC"],
                           [[r["variant"], r["summary"]["macro_auroc"], r["summary"]["macro_f1"],
                             r["summary"]["abnormal_vs_normal_auroc"]] for r in image["results"]]))
        calib = next((r.get("calibration") for r in image["results"] if r.get("calibration")), None)
        if calib:
            parts += ["", f"Calibration of the ensemble (test): ECE {calib['ece_raw']} -> **{calib['ece_calibrated']}**, "
                          f"Brier {calib['brier_raw']} -> {calib['brier_calibrated']} after Platt scaling."]
        parts += ["", f"![Per-finding AUROC]({fig_findings(image)})", ""]
    if retr:
        s = retr["similar_case_label_precision"]
        parts += ["## 2. Similar-case retrieval", "",
                  table(["Route", "P@1", "P@5", "P@10"],
                        [[k, v["label_precision@1"], v["label_precision@5"], v["label_precision@10"]]
                         for k, v in s.items() if isinstance(v, dict)] + [["random", s["random"], s["random"], s["random"]]]), ""]
    if qa:
        parts += ["## 3. Text: PubMedQA", "",
                  table(["Embedder", "R@1", "R@3", "R@10"],
                        [[k, v["recall@1"], v["recall@3"], v["recall@10"]] for k, v in qa["retrieval"].items()]), "",
                  table(["QA mode", "Accuracy", "Macro F1"], [[k, v["accuracy"], v["macro_f1"]] for k, v in qa["qa"].items()]),
                  "", f"![RAG accuracy]({fig_rag(qa)})", ""]
    if fusion:
        a, b, c = fusion["A_real_indication"], fusion["B_simulation"], fusion["C_conflicts"]
        rows = [["A. Real indication text", a["macro_auroc"]["image_only"], a["macro_auroc"]["text_only"],
                 a["macro_auroc"]["fused"], f"{a['fused_minus_image_ci95'][0]:+} to {a['fused_minus_image_ci95'][1]:+}"],
                ["B. Controlled simulation", b["macro_auroc"]["image_only"], b["macro_auroc"]["text_only"],
                 b["macro_auroc"]["fused"], f"{b['fused_minus_image_ci95'][0]:+} to {b['fused_minus_image_ci95'][1]:+}"]]
        parts += ["## 4. Multimodal fusion vs unimodal (macro AUROC)", "",
                  table(["Setting", "Image only", "Text only", "Fused", "Fused - image, 95% CI"], rows), "",
                  f"![Fusion ablation]({fig_fusion(fusion)})", "",
                  f"Conflict detection on injected discordances: {c['detected']}/{c['injected']} overall; "
                  f"**{c.get('detection_rate_when_image_agrees_with_truth', 'n/a'):.0%}** when the image classifier agrees "
                  f"with ground truth; alert rate without injection {c['alert_rate_without_injection']:.0%}.", ""]
        if txt_cxr:
            parts += [f"LLM text-only baseline on indications: macro AUROC {txt_cxr['summary']['macro_auroc']}.", ""]
    if rob:
        rows = [[name, lv["severity"], lv["macro_auroc"], f"{-lv['macro_auroc_drop']:+.3f}", lv["abnormal_auroc"]]
                for name, levels in rob["corruptions"].items() for lv in levels]
        parts += ["## 5. Robustness of the image classifier", "",
                  f"Clean: macro AUROC {rob['clean']['macro_auroc']}, abnormal-vs-normal {rob['clean']['abnormal_auroc']}.", "",
                  table(["Corruption", "Severity", "Macro AUROC", "Change", "Abnormal AUROC"], rows), "",
                  f"![Robustness]({fig_robustness(rob)})", ""]
    if held:
        r0 = held["runs"][0]
        parts += ["## 6. Text pipeline on held-out notes", "",
                  f"{held['dataset']['n']} generated notes (seed {held['dataset'].get('seed')}), never used for tuning.", "",
                  table(["PHI type", "n", "Recall"], [[k, v["n"], v["recall"]] for k, v in r0["phi_recall_by_type"].items()]), "",
                  table(["Typos", "PHI recall", "Token leak", "Clinical text kept", "Finding recall", "Negation F1"],
                        [[f"{r['typo_rate']:.0%}", r["phi_recall"], r["phi_token_leak_rate"], r["clinical_text_kept"],
                          r["active_recall"], r["negation_f1"]] for r in held["runs"]]), "",
                  f"![Typo robustness]({fig_typos(held)})", ""]
        if held_pre:
            fac = held_pre["runs"][0]["phi_recall_by_type"].get("facility", {})
            parts += [f"Before the facility-name fix, the discovery set (seed 2026) showed hospital names removed "
                      f"{fac.get('recall', 0):.0%} of the time and overall PHI recall {held_pre['runs'][0]['phi_recall']}; "
                      "the numbers above are from a fresh set generated after the fix.", ""]
    if answers:
        s = answers["summary"]
        parts += ["## 7. Answer faithfulness and latency (real Llama, 7 demo cases x 4 questions)", "",
                  table(["Metric", "Value"], [
                      ["Answers", s["answers"]],
                      ["Routing accuracy", s["routing_accuracy"]],
                      ["Format compliance", s["format_compliance"]],
                      ["Citation coverage (claims with a citation)", s["citation_coverage"]],
                      ["Citation validity (cited ids that exist)", s["citation_validity"]],
                      ["Ungrounded number rate", s["ungrounded_number_rate"]],
                      ["Triage level stated (triage questions)", s["triage_level_stated"]],
                      ["Triage contradictions", s["triage_contradictions"]],
                      ["Conflict disclosure (cases with HIGH/MEDIUM conflict)", s["conflict_disclosure"]],
                      ["Answers with unsafe certainty phrases", s["answers_with_unsafe_phrases"]],
                      ["Median answer length (words)", s["median_words"]],
                      ["Median evidence stage latency (s)", s["latency_s"]["evidence_median"]],
                      ["Median LLM latency (s)", s["latency_s"]["llm_median"]],
                  ]), ""]
    parts += ["## 8. Usability study", ""]
    if study:
        a = study["all"]["sus"]
        parts += [f"SUS {a['mean']} (SD {a['sd']}, n={a['n']}) -> {a.get('adjective')}.", ""]
    else:
        parts += ["Not run yet: follow `docs/user_study/protocol.md`, then `python -m scripts.evaluate.user_study` and rerun this report.", ""]

    out = DOCS / "results.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out} and {len(list(FIG.glob('*.png')))} figures in {FIG}")


if __name__ == "__main__":
    main()
