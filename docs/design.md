# MedAssist: Design Document (v0.1, Phase 0)

> Research prototype for academic evaluation. Not a medical device; outputs are
> decision *support* for qualified clinicians and are never a diagnosis.

## 1. Aim

Build an assistant that accepts **clinical text** (notes, reports) and **medical
images** (DICOM/PNG/JPEG), reasons over both, and returns evidence-cited answers,
triage guidance and follow-up questions, and to show **measurably** that
combining modalities beats either modality alone.

## 2. Scope decisions

| Decision | Choice | Reason |
|---|---|---|
| Primary image domain | **Chest X-ray** | Best public paired image+report data (IU X-Ray); clinically common; keeps evaluation deep rather than wide. Other modalities are handled zero-shot but not formally benchmarked. |
| Fusion strategy | **Late fusion** | Each modality is encoded by a specialist model; outputs are combined into a structured evidence bundle that the LLM reasons over. Explainable and testable per stage. |
| LLM | Llama 3.1 8B (Ollama, 4-bit) | Runs locally on a 6 GB GPU; no patient data leaves the machine. Llama 3.2 3B as a fast fallback. |
| Image encoder | BiomedCLIP (open_clip) | Trained on 15M biomedical image-caption pairs (PMC-15M); shared image-text space enables zero-shot labels **and** image-to-report retrieval. |
| Text embeddings | BioBERT sentence model (S-T) | Domain-specific semantic search; ClinicalBERT compared as an ablation. |
| Orchestration | LangChain (core + Ollama) | Prompt templates, retrieval chains, routing. |
| UI | Streamlit | Fast to build; effort goes to the AI pipeline. |
| Privacy | Local-only inference, de-identification on ingest, PHI-free DICOM metadata | Compliance story for evaluation. |

## 3. Architecture

```
 Upload (UI) ──► FastAPI ──► De-identification ──┬──► TEXT PIPELINE
                                                 │     NER + negation (scispaCy/medspaCy)
                                                 │     BioBERT embeddings ─► vector store
                                                 │
                                                 └──► IMAGE PIPELINE
                                                       pydicom (modality+VOI LUT) ─► OpenCV CLAHE
                                                       BiomedCLIP: zero-shot labels, similar-case
                                                       retrieval, saliency heatmap
                                                              │
                         FUSION / ORCHESTRATOR (LangChain) ◄──┘
                         router → evidence bundle → prompt template
                                         │
                         Llama 3.1 8B (Ollama) → SUMMARY + DETAILS (cited)
                                         │
                         triage rules (red flags) + follow-up questions
```

## 4. Module map

| Package | Responsibility | Phase |
|---|---|---|
| `medassist/ingestion` | Upload validation, DICOM/PNG loading, de-identification, NER | 0-1 |
| `medassist/image_pipeline` | Preprocessing, BiomedCLIP inference, heatmaps | 0-2 |
| `medassist/text_pipeline` | Embeddings, vector store, RAG | 0-2 |
| `medassist/fusion` | Router, prompts, evidence bundle, triage | 3 |
| `medassist/llm` | Ollama client | 0 |
| `medassist/evaluation` | Metrics + benchmark runners | 0, 5 |
| `api/` | FastAPI endpoints | 0-4 |
| `ui/` | Streamlit interface | 4 |

## 5. Datasets (planned)

| Dataset | Use |
|---|---|
| IU X-Ray (Open-i) | Paired CXR + reports: retrieval, cross-modal agreement, demo cases |
| NIH ChestX-ray14 / CheXpert (subset) | Zero-shot classification benchmark (AUROC/F1) |
| VQA-RAD | Medical VQA: pipeline vs. end-to-end VLM (LLaVA) |
| PubMedQA (subset) | Text QA, RAG vs. no-RAG |
| MTSamples, Synthea | Clinical notes / synthetic EHRs for demos and de-ID tests |
| pydicom test data | DICOM handling tests |

## 6. Evaluation plan (summary)

Zero-shot AUROC/F1 · Recall@K for image→report retrieval · QA accuracy ·
**unimodal vs. multimodal ablation** (headline result) · robustness under
noise/blur/rotation/compression · RAG faithfulness · de-ID PHI recall ·
latency · user study (SUS + Likert).

**Integrity rule:** no baseline may read ground-truth labels. Every number in
the report must be reproducible with one command from `medassist/evaluation`.

## 7. Phase plan

| Phase | Weeks | Deliverable |
|---|---|---|
| 0 Setup | 1-2 | Repo, env (CUDA), models load, smoke test passes ✅ |
| 1 Ingestion & privacy | 3-4 | Upload API, DICOM pipeline, de-ID, NER + negation ✅ |
| 2 Unimodal modules | 5-8 | RAG over notes; BiomedCLIP labels + retrieval + heatmaps; unimodal baselines |
| 3 Fusion | 9-11 | Router, evidence bundle, prompt templates, triage, conflict detection |
| 4 Interface | 12-13 | Streamlit app with chat, uploads, heatmap viewer, evidence panel |
| 5 Evaluation | 14-16 | Full benchmark suite + user study |
| 6 Docs & demo | 17-20 | Report, slides, demo video |

## 8. Hardware

Developed on an RTX 4050 Laptop (6 GB VRAM), 16 GB RAM. Llama 3.1 8B (Q4,
~4.9 GB) and BiomedCLIP (~0.8 GB) share the GPU; Ollama offloads layers to
CPU if VRAM is short. Large benchmark runs can move to Kaggle/Colab.

## 9. Phase 1 design notes (for the report)

- **De-ID = Presidio + domain rules.** Custom recognisers for MRN/UHID,
  Aadhaar, Indian mobile numbers and ages >89; an allow-list of medical
  eponyms (Parkinson, Murphy sign, Kerley lines…).
- **Finding: generic de-ID harms clinical meaning.** Presidio's DATE_TIME
  detector redacted "No night" in "No night sweats", deleting the negation cue
  and flipping the finding to positive. It also removed ages ("56-year-old")
  and a drug name. Fix: only calendar dates are redacted (Safe Harbor's actual
  requirement); regression-tested. A good discussion point: privacy tools must
  be validated for clinical *utility*, not just recall.
- **NER = scispaCy BC5CDR + medspaCy.** The BC5CDR model sometimes absorbs
  cue words into entities ("Denies chest pain" as one DISEASE span), which hides
  them from ConText. A custom pipeline component trims leading cue words
  before ConText runs.
- **Historical vs active.** "History of hypertension" is tagged historical by
  ConText; such conditions are reported separately as `past_history` so chronic
  comorbidities still reach the reasoning step.
- **Images are converted to grayscale** in the loader. Correct for radiology,
  but **histopathology needs colour**. Phase 2 must add an RGB path.

## 10. Known limitations / risks

- Zero-shot BiomedCLIP accuracy on CXR findings is moderate; mitigation: a
  linear probe on frozen features as an improved variant.
- The LLM can hallucinate; mitigated by evidence-only prompting, citations,
  and conflict flags, and measured with faithfulness metrics.
- Public datasets are de-identified already; de-ID is tested on synthetic PHI.
