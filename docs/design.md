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
| Text embeddings | PubMedBERT fine-tuned for retrieval (S-T) | Won the Phase 2 comparison against BioBERT, ClinicalBERT and a general model (see section 10). |
| Orchestration | LangChain (core + Ollama) | Prompt templates, retrieval chains, routing. |
| UI | React + Vite + TypeScript, served by FastAPI | The interface is the most visible part of the evaluation, so a custom frontend replaces Streamlit (which limits layout and styling). Built with the Impeccable design skill (`.claude/skills/impeccable`, hooks not installed). The production build is served by FastAPI, so the demo runs from one command. |
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
| `ui/` | React + Vite frontend | 4 |

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
| 2 Unimodal modules | 5-8 | RAG over notes; BiomedCLIP labels + retrieval + heatmaps; unimodal baselines ✅ |
| 3 Fusion | 9-11 | Router, evidence bundle, prompt templates, triage, conflict detection ✅ |
| 4 Interface | 12-13 | React app: case workspace with uploads, chat, image viewer with heatmap overlay, findings + evidence panel, triage banner |
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

## 10. Phase 2 design notes and results

**Data.** IU X-Ray via a Hugging Face mirror of the Kaggle/Open-i release (the
NIH bulk endpoint returns an error page). One frontal image per study; labels
from NLM-assigned MeSH terms grouped into 9 findings + normal; studies tagged
"Technical Quality Unsatisfactory"/"No Indexing" excluded. Stratified subset of
1,500 studies (rare findings guaranteed), fixed 70/30 train/test split (seed
42): train = 1,065 (retrieval knowledge base + probe training), test = 435.
Images resized to 512 px (147 MB). `python -m scripts.prepare_iu_xray --n N`
scales it up.

**Leakage rule.** Model inputs may use the image and the *indication* only.
Findings/impression are the label source, so they are used only as the
retrieval corpus (other patients' reports), never as input for the same study.

**Image classifier (BiomedCLIP).** Zero-shot = prompt ensembles (4 positive x 3
negative templates, radiology synonyms), 2-way softmax per finding. Linear probe
= logistic regression per finding on frozen features, C=0.05, balanced classes;
thresholds from 5-fold out-of-fold predictions on train. Ensemble = equal-weight
logit average. *Honesty note:* the 0.5 weight is the untuned default; we looked
at 0.3/0.7 on test afterwards as a sensitivity check (macro AUROC 0.747 / 0.746
vs 0.751), so the gain does not depend on tuning.

| Variant (test, n=435) | Macro AUROC | Macro F1 | Abnormal-vs-normal AUROC |
|---|---|---|---|
| Zero-shot | 0.710 | 0.327 | 0.692 |
| Zero-shot + CLAHE | 0.704 | 0.307 | 0.696 |
| Linear probe | 0.728 | 0.349 | 0.754 |
| **Ensemble (default, Platt-calibrated)** | **0.752** | **0.363** | **0.773** |

Per finding (ensemble AUROC): effusion 0.91, cardiomegaly 0.88, edema 0.87,
emphysema 0.86, opacity 0.80, atelectasis 0.70, fracture 0.62, granuloma 0.57,
nodule 0.56. Small focal findings are weak at 224 px input resolution; this
is a limitation to discuss rather than hide.

Findings: (1) **CLAHE slightly hurts** zero-shot BiomedCLIP, which was trained
on unenhanced PubMed figures, so it is off by default. (2) Zero-shot and probe
have **complementary strengths** (zero-shot wins on effusion/edema/fracture,
where few training positives make the probe overfit; the probe wins on
emphysema/granuloma). That is why the ensemble beats both.

**Explainability.** Occlusion sensitivity (32 px patches, stride 16, ~170
masked forward passes batched on GPU, about 1 s). Model-agnostic, so it
explains the same score the user sees.

**Modality gate.** BiomedCLIP zero-shot over 9 modality labels; CXR findings
are only produced when the top label is "chest X-ray" with p >= 0.5.

**Retrieval (BiomedCLIP shared space).**

| Task | Metric | Model | Random |
|---|---|---|---|
| Similar case, image->image (test -> 1,065 KB) | label precision@1 / @5 / @10 | 0.49 / 0.44 / 0.43 | 0.28 |
| Similar case, image->report | label precision@1 / @5 / @10 | 0.38 / 0.41 / 0.41 | 0.28 |
| Exact own-report retrieval (435 candidates) | Recall@1 / @5 / @10 | 0.002 / 0.039 / 0.062 | 0.002 / 0.011 / 0.023 |

Exact report retrieval is near chance because IU reports are heavily
templated (~40% normal studies have near-identical text), so the task is
ill-posed. Label precision is the meaningful metric and shows a clear gain.
The app uses image->image retrieval to surface similar prior cases.

**Text RAG.** Sentence-aware chunking (600 chars, 1-sentence overlap) ->
sentence embeddings -> `VectorIndex` -> LangChain `prompt | ChatOllama |
StrOutputParser`, numbered passages and enforced [n] citations; citation numbers
outside the retrieved set are discarded.

**Text evaluation: PubMedQA** (pqa_labeled, 1,000 expert-labelled questions).

Retrieval (each question's own abstract among all 1,000):

| Embedder | R@1 | R@3 | R@10 |
|---|---|---|---|
| BioBERT (S-T, NLI/STS-tuned) | 0.899 | 0.959 | 0.986 |
| ClinicalBERT (raw, mean-pooled) | 0.159 | 0.251 | 0.373 |
| **PubMedBERT (MS-MARCO retrieval-tuned): new default** | **0.975** | **0.989** | 0.994 |
| MiniLM (general domain) | 0.971 | 0.989 | 0.997 |

Takeaways for the report: (1) *Domain pre-training is not retrieval ability.*
Raw ClinicalBERT, a strong clinical encoder, is nearly useless for search
without sentence-level fine-tuning. (2) *The training objective matters more
than the domain*: a small general model trained for retrieval matches the
biomedical one, and both beat BioBERT tuned for NLI/similarity. So the default
moved to PubMedBERT-MARCO (biomedical *and* retrieval-tuned); the brief's
BioBERT/ClinicalBERT are kept as measured baselines. Caveat: PubMedQA questions
are derived from paper titles, so lexical overlap makes retrieval easier than
for real clinical queries.

QA with Llama 3.1 8B (150-question random sample, seed 42, temperature 0):

| Mode | Accuracy | Macro F1 |
|---|---|---|
| Majority class ("yes") | 0.647 | 0.262 |
| Closed-book | 0.573 | 0.361 |
| **RAG, top-3 retrieved** | **0.800** | **0.546** |
| Oracle (gold abstract) | 0.820 | 0.562 |

RAG adds +22.7 accuracy points and recovers 92% of the gap to the oracle.
Closed-book is *below* the majority baseline: without evidence the 8B model
answers confidently but often wrongly, a direct argument for evidence-grounded
answers. Macro F1 stays low because "maybe" (11% of labels) is rarely predicted.

**Text-only CXR baseline** (Llama reads only the indication, rates each finding
low/medium/high -> 0.1/0.5/0.9; 408 test studies with an indication, 0 parse
failures): **macro AUROC 0.552, abnormal-vs-normal 0.485, i.e. near chance.**
Best findings: edema 0.65, fracture 0.62. IU X-Ray indications are very short
("chest pain", "preop") and carry little information about the radiograph,
so on this dataset the image dominates. **Implication for Phase 3:** fusion
cannot gain much AUROC from such thin text; the multimodal value must be shown
via richer clinical context (synthetic case vignettes with symptoms/vitals)
and qualitative capabilities (conflict detection, triage, cited reasoning),
not by overstating a small AUROC difference. The 3-level rating also coarsens
the ranking that AUROC measures.

**Hardware note.** On the 6 GB RTX 4050, llama3.1:8b runs ~20% on CPU even with
a 2,048-token context (weights ~4.9 GB + Windows display memory): ~5 s per short
answer, longer with abstracts in the prompt.

## 11. Phase 3 design notes and results (fusion)

**Pipeline per question** (`medassist/fusion/assistant.py`): de-identify the
question -> route -> image analysis (cached per image) -> RAG passages from the
notes -> evidence bundle with fused scores -> conflicts + triage + follow-ups
(all rule-based) -> route-specific prompt -> Llama 3.1 -> SUMMARY/DETAILS with
citations. If the LLM is down, the deterministic parts are still returned.

**Design principle: the LLM explains; rules decide.** Triage levels and
conflicts are computed deterministically and handed to the LLM as facts. This
keeps safety-relevant outputs reproducible and testable, and limits the LLM to
what it is good at (synthesis and plain-language explanation).

**Calibration (added in Phase 3).** The ensemble's raw probabilities were badly
calibrated (zero-shot logits have a huge scale: "fracture 0.90" on films with
no fracture). Per-finding Platt scaling fitted on train out-of-fold predictions:
mean ECE 0.109 -> **0.022**, Brier 0.114 -> 0.081 (test), AUROC unchanged.
Clinician-facing bands: *likely* (p >= 0.5), *possible* (p >= max(F1 threshold,
0.2)), *unlikely*. F1-optimal thresholds are kept for benchmarking only; for rare
findings they are ~0.05, far too alarming to show as "positive".

**Numeric fusion** (`evidence.py`, `knowledge.py`): fused = sigmoid(logit(p_image)
+ LLR_text). Hand-set, documented LLRs: assertion +2.0, negation -1.5,
supporting symptom/vital +0.4 each (cap +1.2); a documented *history* of a
chronic cause (CHF, COPD, TB) counts as an assertion for chronic findings
(cardiomegaly, emphysema, granuloma, nodule) but only as support for acute ones.

**Conflict kinds:** note asserts / image negative; image positive / note denies
(HIGH); clinical suspicion unconfirmed (info; acute findings only, strong support
needed); unmentioned high-confidence image finding (info; only when nothing in
the note points to it); inconsistency between notes; modality mismatch.

**Triage:** NEWS2 (RCP 2017; room-air scale, needs >= 3 vitals) + red flags (stroke
signs, reduced consciousness, haemoptysis, syncope, confusion, chest pain with
breathlessness) + image escalation (acute CXR finding with SpO2 < 94%).
EMERGENCY > URGENT > SOON > ROUTINE. Unit-tested against hand-calculated NEWS2.

**Router:** keyword rules -> text / image / multimodal / triage, with fallback
when a modality is missing. Deterministic and explainable; misrouting is cheap
because the multimodal route sees all evidence.

**NER fixes found by the fusion evaluation:** scispaCy BC5CDR does not tag
radiology vocabulary ("lung opacity", "nodule", "consolidation"), so target
rules for 30 radiology terms were added (Phase 1 metrics unchanged).

### Results (`python -m scripts.eval_fusion`, IU X-Ray test n=435, 1,000-sample paired bootstrap)

| Setting | Image-only | Text-only | **Fused** | Fused - image (95% CI) |
|---|---|---|---|---|
| A. Real indication text (NER + knowledge fusion) | 0.752 | 0.546 | **0.759** | +0.007 (+0.001 to +0.014) |
| B. Controlled simulation (synthetic notes from labels) | 0.752 | 0.843 | **0.867** | +0.115 (+0.097 to +0.133) |

(macro AUROC over 9 findings; LLM text-only on indications, Phase 2: 0.552.)

- **A** is the honest real-data result: a small but statistically significant
  gain. IU indications are a few words (302/435 contain any recognisable clinical
  entity), so text has little to add.
- **B** checks the mechanism with informative text (symptom model: each true
  finding's typical symptoms p=0.5, assertion p=0.25, distractors p=0.08,
  negations p=0.15). Fused beats *both* unimodal arms, i.e. the modalities are
  complementary when both carry signal. The text strength is set by us, so this
  is a sanity check of the fusion method, not clinical evidence.

**Conflict detection** (discordances injected into 15% of simulated notes):

| | Detected |
|---|---|
| All injected conflicts | 29/61 (48%) |
| "Note denies X, image shows X" | 12/42 (29%) |
| "Note asserts X, image normal" | 17/19 (90%) |
| **When the image classifier agrees with ground truth** | **29/29 (100%)** |
| Alert rate on notes without injection | 17% |

The rule logic catches every conflict it can see; all misses come from the
image classifier not detecting the finding (mostly subtle ones). The 17% alert
rate comes from true findings documented in the note that the classifier
misses: real text/image discordances, but false alarms relative to ground truth.

**End-to-end latency** (RTX 4050, warm models): rules + retrieval < 1 s; image
analysis ~1 s (cached after the first question); Llama answer ~28 s (8B model
partly on CPU; the main bottleneck). Cold start adds ~15 s of model loading.

**Demo cases** (`data/samples/demo_cases.json`, `python -m scripts.load_demo_cases
--analyze --reset`): 7 synthetic notes on real test-split X-rays: concordant
pneumonia (EMERGENCY, NEWS2 7), heart failure (URGENT), missed effusion (HIGH
conflict), unconfirmed pneumonia (conflict), COPD (EMERGENCY, NEWS2 8), routine
screening (ROUTINE), wrong image (modality mismatch). Curated for demonstration
and never used in evaluation.

## 12. Known limitations / risks

- Zero-shot BiomedCLIP accuracy on CXR findings is moderate; mitigation: a
  linear probe on frozen features as an improved variant.
- The LLM can hallucinate; mitigated by evidence-only prompting, citations,
  and conflict flags, and measured with faithfulness metrics.
- Public datasets are de-identified already; de-ID is tested on synthetic PHI.
