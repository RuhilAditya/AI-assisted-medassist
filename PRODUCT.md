# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React + Vite + TypeScript frontend in `ui/`, talking to the existing FastAPI backend (`api/main.py`); the production build is served by FastAPI so the demo runs from one command. Chosen by the user in the Phase 2 discussion (Streamlit was dropped because the interface is the most visible part of the evaluation).

## Users

Primary: qualified clinicians (emergency, ward and outpatient doctors) reviewing one patient case at a time: a de-identified clinical note plus a chest X-ray, under time pressure, asking questions in plain language and deciding what to do next.

Also, and in practice first: B.Tech capstone examiners watching a live demo on a projector in a lit room or on a laptop up close. They judge it by whether it reads as a credible clinical tool. Realism is what impresses them, so the interface is designed for the clinician, not as a showcase.

## Product Purpose

MedAssist is a multimodal clinical decision-support assistant. For one case it combines the clinical note and the image, answers questions with evidence-cited explanations, flags where the note and the image disagree, and gives rule-based triage guidance (NEWS2 + red flags). Success: a clinician can see in seconds how urgent the case is, what the image suggests, whether text and image agree, and why the assistant says what it says, and can verify every claim against its source.

## Positioning

Unlike a chatbot or a single-modality classifier, MedAssist shows its reasoning: every claim is traced to a specific note line, image finding (with calibrated probability and a heatmap), similar prior case or retrieved passage, and cross-modal conflicts are surfaced instead of smoothed over. Safety-relevant outputs (triage level, conflicts) are computed by transparent rules and only explained by the LLM.

## Operating Context

- One case at a time: upload or pick a case, read the triage and findings, ask questions, follow suggested next questions, look at the heatmap for a finding, check conflicts.
- Answers take ~30 s from the local 8B LLM on a 6 GB GPU; findings, conflicts and triage are available in ~1 s. The interface must show the deterministic results immediately and stream or wait for the written answer.
- Demo cases (7, synthetic notes on real IU X-Ray images) are the main presentation path: `data/samples/demo_cases.json`.
- Everything runs locally; no patient data leaves the machine.

## Capabilities and Constraints

- Backend endpoints: cases list/create, note upload (TXT/PDF/paste), image upload (DICOM/PNG/JPEG), image analysis (modality gate, 9 chest X-ray findings with likely/possible/unlikely bands, similar cases), occlusion heatmap per finding, multimodal chat (route, summary, details, citations, fused findings, conflicts, NEWS2 triage, follow-ups, timings), chat history.
- Triage levels: EMERGENCY, URGENT, SOON, ROUTINE. Conflict severities: high, medium, info.
- Only chest X-rays get finding analysis; other modalities are flagged.
- Research prototype, not a medical device: every answer carries a decision-support disclaimer; no confirmed diagnoses.
- Notes are shown de-identified ([PERSON], [MRN], [DATE] placeholders); raw uploads are never stored.

## Brand Commitments

Name: MedAssist. No logo or brand assets exist yet. Voice: clinical, concise, honest about uncertainty ("not detected by the classifier", never "ruled out").

The user ruled out (binding): a generic AI dashboard look (purple gradients, glass cards, big-number tiles, a ChatGPT-clone chat), anything toy-like or flashy, cold grey 2005-era hospital software, and any design that hides the AI's reasoning (evidence, scores, heatmaps and conflicts must stay visible).

## Evidence on Hand

- Real evaluation results in `results/*.json` and `docs/design.md` (e.g. CXR macro AUROC 0.752, RAG 0.573 -> 0.800 on PubMedQA, fusion and conflict-detection results). These may be shown, with their caveats.
- Demo cases: synthetic notes (fictional names, labelled synthetic) on real de-identified IU X-Ray images.
- No real users, testimonials, deployments or clinical validation exist; never imply them.

## Product Principles

1. Evidence before answers: findings, conflicts and triage are visible before and beside the written answer.
2. Every claim is traceable to its source in one interaction.
3. Disagreement is information: conflicts are prominent, never buried.
4. Honest uncertainty: probabilities are calibrated and shown as bands, with limitations stated.
5. Calm under pressure: urgency is unmistakable without alarm theatre.

## Accessibility & Inclusion

Must stay legible on a projector in a lit room (high contrast, no meaning carried by colour alone, generous type for key values) and work on a laptop and on a phone. WCAG 2.1 AA contrast and full keyboard operation.
