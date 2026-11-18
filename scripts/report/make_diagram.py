"""Draw the system architecture diagram (docs/figures/architecture.png) for the report and slides.
Colours follow the UI: vermilion = image path, blue = text path, green = fusion, navy = shared.
Usage: python -m scripts.report.make_diagram
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from medassist.config import PROJECT_ROOT  # noqa: E402

IMAGE, TEXT, FUSED, INK, INK2, RULE = "#c93a1b", "#1f5aa6", "#2e8a5a", "#14223b", "#3f4d66", "#c3ccd7"
WASH = {IMAGE: "#fbe9e4", TEXT: "#e5eef9", FUSED: "#e3f2ea", INK: "#edf1f5"}
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.12",
                                linewidth=1.4, edgecolor=color, facecolor=WASH[color]))
    ax.text(x + 0.15, y + h - 0.2, title, fontsize=10.5, fontweight="bold", color=INK, va="top")
    for i, line in enumerate(lines):
        ax.text(x + 0.15, y + h - 0.58 - i * 0.3, line, fontsize=8.6, color=INK2, va="top")


def arrow(ax, a, b, color=INK2, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.6, 7.2))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    box(ax, 0.2, 4.6, 2.5, 1.6, "Clinical note", ["TXT / PDF / pasted text", "free-text EHR, discharge", "summaries, referrals"], TEXT)
    box(ax, 0.2, 1.4, 2.5, 1.6, "Medical image", ["DICOM / PNG / JPEG", "chest X-ray (benchmarked)", "other modalities gated"], IMAGE)

    box(ax, 3.2, 4.6, 2.9, 1.6, "De-identification", ["Presidio + custom recognisers", "(MRN, Aadhaar, +91, facility)", "calendar-date-only redaction"], TEXT)
    box(ax, 3.2, 1.4, 2.9, 1.6, "DICOM anonymiser + I/O", ["PS3.15 tag removal, UID swap", "modality + VOI LUT windowing", "burned-in text warning"], IMAGE)

    box(ax, 6.6, 4.6, 3.0, 2.1, "Text pipeline", ["scispaCy NER + radiology rules", "medspaCy ConText: negation,", "history, family; vitals regex", "PubMedBERT embeddings ->", "vector index -> RAG passages"], TEXT)
    box(ax, 6.6, 0.9, 3.0, 2.4, "Image pipeline (BiomedCLIP)", ["modality gate", "9 CXR findings: zero-shot +", "linear-probe ensemble,", "Platt-calibrated", "occlusion heatmaps", "similar-case retrieval"], IMAGE)

    box(ax, 10.1, 2.3, 2.7, 3.0, "Fusion layer", ["evidence bundle [note_1] [image]", "log-odds fusion of image", "scores with note evidence", "conflict detection", "NEWS2 + red-flag triage", "query router, follow-ups"], FUSED)

    box(ax, 10.1, 5.6, 2.7, 1.35, "Llama 3.1 8B (Ollama)", ["LangChain prompt per route", "explains rules' output,", "cites every claim"], INK)
    box(ax, 10.1, 0.2, 2.7, 1.75, "FastAPI + React UI", ["REST + streaming (SSE)", "case workspace, heatmaps,", "leader-line evidence tracing", "runs fully local"], INK)

    arrow(ax, (2.7, 5.4), (3.2, 5.4), TEXT)
    arrow(ax, (2.7, 2.2), (3.2, 2.2), IMAGE)
    arrow(ax, (6.1, 5.4), (6.6, 5.4), TEXT)
    arrow(ax, (6.1, 2.2), (6.6, 2.2), IMAGE)
    arrow(ax, (9.6, 5.3), (10.1, 4.6), TEXT)
    arrow(ax, (9.6, 2.2), (10.1, 3.0), IMAGE)
    arrow(ax, (11.45, 5.3), (11.45, 5.6), FUSED)
    arrow(ax, (11.45, 2.3), (11.45, 1.95), FUSED)
    arrow(ax, (12.8, 6.25), (13.15, 6.25), INK2)
    ax.plot([13.15, 13.15], [6.25, 1.1], color=INK2, linewidth=1.4)
    arrow(ax, (13.15, 1.1), (12.8, 1.1), INK2)
    ax.text(13.25, 3.7, "streamed answer", rotation=90, fontsize=8.5, color=INK2, va="center")

    ax.text(0.2, 7.0, "MedAssist architecture: late fusion of specialist encoders, rules decide, the LLM explains",
            fontsize=13, fontweight="bold", color=INK)
    ax.text(0.2, 0.45, "Everything runs locally on one 6 GB GPU; no patient data leaves the machine.",
            fontsize=9, color=INK2, style="italic")
    out = PROJECT_ROOT / "docs" / "figures" / "architecture.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
