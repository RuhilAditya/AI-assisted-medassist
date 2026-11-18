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

## Status

| Phase | Status |
|---|---|
| 0 Setup | ✅ models load on GPU; smoke test 5/5 |
| 1 Ingestion & privacy | ✅ de-ID (text + DICOM), clinical NER with negation, upload API |
| 2 Unimodal modules | ✅ CXR findings (BiomedCLIP), heatmaps, similar cases, RAG; unimodal baselines |
| 3 Fusion | ✅ multimodal assistant: routing, fused scores, conflicts, NEWS2 triage, cited answers |
| 4 Interface | ⏳ next (React) |
| 5-6 | planned |

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

Reproduce: `python -m scripts.eval_image`, `eval_retrieval`, `eval_text_qa`,
`eval_text_only_cxr`, `eval_fusion`. Each writes `results/<experiment>.json`.

### Phase 1 results (synthetic gold set, `python -m scripts.eval_ingestion`)

| Metric | Score |
|---|---|
| PHI recall (planted identifiers removed) | 100% (33/33) |
| PHI token leak rate | 0% (0/66) |
| Clinical text preserved after de-ID | 100% (36/36) |
| Active finding recall | 94.7% (18/19) |
| Negation precision / recall / F1 | 1.00 / 1.00 / 1.00 |

*Caveat: 6 notes, written by the developers, and used while tuning the rules. A
larger held-out set is planned for Phase 5 before these numbers go in the report.*

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
python -m scripts.prepare_iu_xray --n 1500
python -m scripts.eval_image --train-probe   # also trains the linear probe
python -m scripts.eval_retrieval              # also builds the similar-case index

# 7. Demo cases (7 synthetic notes on real X-rays)
python -m scripts.load_demo_cases --analyze --reset
```

## Verify

```bash
pytest -q                          # fast unit tests, no downloads
python -m scripts.smoke_test       # loads every model end-to-end (first run downloads ~1 GB)
python -m scripts.eval_ingestion   # de-ID + NER metrics (add --verbose to see outputs)
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

Privacy guarantees: raw uploads are never written to disk; original filenames
are discarded; the audit log (`data/store/audit.log`) records actions and PHI
*types* removed, never content.

## Project layout

```
medassist/
  config.py           settings from env/.env
  ingestion/          loaders, text + DICOM de-ID, clinical NER, case store, pipeline
  image_pipeline/     preprocessing, BiomedCLIP, CXR classifier + heatmaps, case retrieval
  text_pipeline/      sentence embeddings, RAG (LangChain)
  datasets/           IU X-Ray loader (labels from MeSH)
  vector_index.py     persistent cosine-similarity index
  analysis.py         image / text analysis services used by the API
  fusion/             router, evidence bundle + fused scores, conflicts, NEWS2 triage,
                      follow-ups, prompts, orchestrator (assistant.py)
  llm/                Ollama client
  evaluation/         metrics, result saving
api/                  FastAPI app
ui/                   React + Vite frontend (Phase 4)
scripts/              smoke test, dataset preparation, evaluation scripts
results/              evaluation outputs (JSON), committed for reproducibility
tests/                pytest suite
data/samples/         small synthetic samples (large datasets are not committed)
docs/                 design document, report material
```
