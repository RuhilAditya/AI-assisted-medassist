# MedAssist: Multimodal Medical Assistant

AI assistant that combines **clinical text** and **medical images** to support
clinical decision-making, triage, and information retrieval. B.Tech final-year
capstone project.

> ⚠️ Research prototype. Not a medical device. Outputs must be verified by a
> qualified clinician.

**Stack:** Llama 3.1 (Ollama) · BiomedCLIP · Sentence Transformers (PubMedBERT,
with BioBERT/ClinicalBERT as measured baselines) · LangChain · OpenCV · pydicom ·
Presidio · scispaCy/medspaCy · FastAPI · React (Vite)

See [docs/design.md](docs/design.md) for architecture, scope and the phase plan.

## Deliverables

| What | Where |
|---|---|
| Project report (Word, A4, 36 pages) | [docs/deliverables/MedAssist_Project_Report.docx](docs/deliverables/MedAssist_Project_Report.docx) |
| Presentation (PowerPoint, 16 slides, speaker notes) | [docs/deliverables/MedAssist_Presentation.pptx](docs/deliverables/MedAssist_Presentation.pptx) |
| All results tables and figures | [docs/results.md](docs/results.md), [docs/figures/](docs/figures/) |
| Design document (decisions, per-phase notes) | [docs/design.md](docs/design.md) |
| Interface design system | [DESIGN.md](DESIGN.md), [PRODUCT.md](PRODUCT.md) |
| Usability study kit | [docs/user_study/](docs/user_study/) |

Placeholders in the report and slides (`[Student Name]`, `[Roll Number]`, `[Guide Name]`,
`[College]` …) must be filled in. Both files are generated from `results/*.json`, so after new
results (e.g. the user study) run `python -m scripts.report.make_report`, then
`node docs/deliverables/build_report.js` and `node docs/deliverables/build_slides.js` (once:
`npm --prefix docs/deliverables install`).
Open the regenerated report in Word and update the table of contents (right-click > Update Field).
Demo deep links: `http://127.0.0.1:8000/?case=conflict-missed-effusion&trace=pleural%20effusion`.

## Status

| Phase | Status |
|---|---|
| 0 Setup | ✅ models load on GPU; smoke test 5/5 |
| 1 Ingestion & privacy | ✅ de-ID (text + DICOM), clinical NER with negation, upload API |
| 2 Unimodal modules | ✅ CXR findings (BiomedCLIP), heatmaps, similar cases, RAG; unimodal baselines |
| 3 Fusion | ✅ multimodal assistant: routing, fused scores, conflicts, NEWS2 triage, cited answers |
| 4 Interface | ✅ React app: case workspace, streaming cited answers, heatmaps, leader-line evidence tracing |
| 5 Evaluation | ✅ robustness, held-out text set, answer faithfulness, latency, user-study kit, report + figures |
| 6 Docs & demo | ✅ project report (.docx), presentation (.pptx), results report, user-study kit |

### Phase 2 results (details and caveats in [docs/design.md](docs/design.md) §10)

| Task | Data | Result |
|---|---|---|
| CXR findings, ensemble (zero-shot + probe) | IU X-Ray test, n=435, 9 findings | macro AUROC **0.752** (zero-shot alone 0.710) |
| Abnormal vs normal | same | AUROC **0.773** (calibrated: ECE 0.022) |
| Similar-case retrieval (image->image) | test -> 1,065-study KB | label precision@1 **0.49** (random 0.28) |
| Abstract retrieval (PubMedBERT-MARCO) | PubMedQA, 1,000 abstracts | Recall@1 **0.975** |
| QA, Llama 3.1 8B | PubMedQA, 150 questions | closed-book 0.573 -> RAG **0.800** (oracle 0.820) |
| CXR findings from indication text only (Llama) | IU X-Ray test, n=408 | macro AUROC 0.552 (near chance: indications are very short) |

### Phase 3 results (details in [docs/design.md](docs/design.md) §11)

| Task | Result |
|---|---|
| Fusion on real data (image + real indication text) | macro AUROC 0.752 -> **0.759** (+0.007, 95% CI +0.001 to +0.014) |
| Fusion, controlled simulation (informative notes) | image 0.752 / text 0.843 -> fused **0.867** |
| Conflict detection when the image classifier is right | **29/29 (100%)** of injected conflicts |
| End-to-end answer time | ~30 s (LLM-bound on a 6 GB GPU) |

### Phase 5 results (all tables and figures: [docs/results.md](docs/results.md))

| Evaluation | Result |
|---|---|
| Image robustness (6 corruptions x 3 levels) | <= 0.017 AUROC drop at realistic levels; fails only at extremes (noise sigma 30, JPEG q5: -0.10) |
| De-ID on 120 fresh held-out notes | PHI recall **99.8%** (facility names fixed after a discovery set showed 0%), token leak 2.1% |
| NER on held-out notes | finding recall 99.7%, negation F1 1.00; at 10% typos 0.91 / 0.94 |
| Answer faithfulness (28 real answers) | routing 100%, citation validity 97.8%, ungrounded numbers 0.4%, triage stated 100%, conflicts disclosed 91.7%, unsafe phrases 0 |
| Weak spot | citation coverage 50% of claims |
| Latency | evidence 0.07 s; LLM median 15.8 s (text 11 s, multimodal 27 s) |
| Usability (SUS) | kit ready in `docs/user_study/`; to be run with participants |

Reproduce: `python -m scripts.evaluate.image`, `evaluate.retrieval`, `evaluate.text_qa`,
`evaluate.text_only_cxr`, `evaluate.fusion`, `evaluate.robustness`, `evaluate.heldout_text --data data/samples/heldout_notes_test.json`,
`evaluate.answers`; then `python -m scripts.report.make_report` rebuilds `docs/results.md` and `docs/figures/`.
Each experiment writes `results/<experiment>.json`.

### Phase 1 results (synthetic gold set, `python -m scripts.evaluate.ingestion`)

| Metric | Score |
|---|---|
| PHI recall (planted identifiers removed) | 100% (33/33) |
| PHI token leak rate | 0% (0/66) |
| Clinical text preserved after de-ID | 100% (36/36) |
| Active finding recall | 94.7% (18/19) |
| Negation precision / recall / F1 | 1.00 / 1.00 / 1.00 |

*Caveat: 6 notes, written by the developers, and used while tuning the rules. Use the
Phase 5 held-out numbers (120 generated notes, never used for tuning) in the report.*

## Requirements

- Windows/Linux/macOS, Python **3.11**
- NVIDIA GPU recommended (developed on RTX 4050, 6 GB); CPU works but is slower
- [Ollama](https://ollama.com) installed and running
- ~15 GB free disk (models)

## Setup

```bash
# 1. Virtual environment
py -3.11 -m venv .venv          # Linux/macOS: python3.11 -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate

# 2. PyTorch with CUDA (pick the index URL matching `nvidia-smi` CUDA version;
#    use the plain `pip install torch torchvision` for CPU-only)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Everything else (includes spaCy / scispaCy models, ~800 MB)
pip install -r requirements.txt

# 4. LLM
ollama pull llama3.1:8b

# 5. Optional: override defaults
copy .env.example .env

# 6. Datasets for Phase 2 (IU X-Ray subset: ~2.7 GB download, 150 MB kept)
python -m scripts.data.prepare_iu_xray --n 1500
python -m scripts.evaluate.image --train-probe   # also trains the linear probe
python -m scripts.evaluate.retrieval              # also builds the similar-case index

# 7. Demo cases (7 synthetic notes on real X-rays)
python -m scripts.data.load_demo_cases --analyze --reset

# 8. Build the web interface (Node 18+)
npm --prefix ui install
npm --prefix ui run build
```

## Run the app

```bash
uvicorn api.main:app --port 8000   # serves the API and the built UI
```

Open http://127.0.0.1:8000. For UI development with hot reload, run the backend as above
and `npm --prefix ui run dev` (http://127.0.0.1:5173; API calls are proxied to :8000).

The interface: a case rail; a triage band (NEWS2 + red flags); the X-ray with per-finding
heatmaps and the de-identified note with entity highlighting; an evidence column (triage
reasons, note-vs-image conflicts, calibrated findings as reference bands, similar prior
cases); and an answer log that streams cited answers. Hover any citation, or any finding,
to draw leader lines to the exact evidence. Design system: [DESIGN.md](DESIGN.md).

## Verify

```bash
pytest -q                          # fast unit tests, no downloads
python -m scripts.smoke_test       # loads every model end-to-end (first run downloads ~1 GB)
python -m scripts.evaluate.ingestion   # de-ID + NER metrics (add --verbose to see outputs)
uvicorn api.main:app --reload      # then open http://127.0.0.1:8000/docs
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/status` | Liveness; which models/services are configured and reachable |
| POST | `/cases` | Create a case, returns `case_id` |
| POST | `/cases/{id}/notes` | Upload TXT/PDF (`file`) or paste `text`; de-identified, then entities extracted |
| POST | `/cases/{id}/images` | Upload DICOM/PNG/JPEG; DICOM tags anonymised |
| GET | `/cases/{id}` | Case with de-identified notes, entity summaries, image metadata |
| GET | `/cases/{id}/images/{image_id}` | Processed image (PNG) |
| POST | `/cases/{id}/images/{image_id}/analyze` | Modality check, 9 CXR finding probabilities, 3 similar prior cases |
| GET | `/cases/{id}/images/{image_id}/heatmap?finding=...` | Occlusion heatmap overlay (PNG) |
| POST | `/cases/{id}/ask` | `{"question": ...}`: RAG answer over the case's notes, with citations |
| GET | `/cases` | List cases (title, counts) |
| POST | `/cases/{id}/chat` | `{"question", "image_id"?}`: **full multimodal answer**: route, cited summary/details, fused findings, conflicts, NEWS2 triage, follow-ups, timings |
| GET | `/cases/{id}/chat` | Conversation history (questions stored de-identified) |
| GET | `/cases/{id}/overview` | Findings, conflicts, triage and follow-ups without a question (~1 s) |
| POST | `/cases/{id}/chat/stream` | Same as `/chat` as server-sent events: `evidence` first, then `token`s, then `done` |
| GET | `/cases/{id}/notes/{note_id}` | One de-identified note with entities |

Privacy guarantees: raw uploads are never written to disk; original filenames
are discarded; the audit log (`data/store/audit.log`) records actions and PHI
*types* removed, never content.

## Project layout

```
medassist/                Python package: all the AI logic
  config.py               settings from env/.env
  ingestion/              upload loaders, text + DICOM de-identification, clinical NER, case store
  image_pipeline/         BiomedCLIP, chest X-ray classifier + heatmaps, similar-case retrieval
  text_pipeline/          sentence embeddings, RAG (LangChain)
  fusion/                 evidence bundle, fused scores, conflicts, NEWS2 triage, router,
                          prompts, and the orchestrator (assistant.py)
  llm/                    Ollama client
  datasets/               IU X-Ray loader (labels from MeSH)
  evaluation/             metrics and result saving
  analysis.py             image / text analysis services used by the API
  vector_index.py         cosine-similarity index
api/main.py               FastAPI backend (also serves the built UI)
ui/                       React + Vite + TypeScript web interface
scripts/
  smoke_test.py           checks that every model loads and runs
  data/                   prepare IU X-Ray, generate held-out notes, load demo cases
  evaluate/               one script per experiment; each writes results/<name>.json
  report/                 builds docs/results.md and docs/figures/ from results/
tests/                    pytest suite, one file per area (core, ingestion, image_and_rag,
                          fusion, api, evaluation_tools)
results/                  experiment outputs (JSON), committed for reproducibility
data/samples/             small synthetic notes and gold labels (large data is not committed)
docs/
  deliverables/           project report (.docx), presentation (.pptx) and their generators
  design.md               design decisions and per-phase notes
  results.md, figures/    all result tables and charts
  user_study/             usability-study protocol, questionnaire, response template
DESIGN.md, PRODUCT.md     interface design system and product record (Impeccable skill)
```

Created locally and git-ignored: `.venv/`, `models_cache/` (model weights),
`data/processed/` (IU X-Ray subset), `data/store/` (cases), `logs/`, `node_modules/`, `ui/dist/`.
