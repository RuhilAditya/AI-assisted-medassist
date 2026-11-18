# MedAssist: Multimodal Medical Assistant

AI assistant that combines **clinical text** and **medical images** to support
clinical decision-making, triage, and information retrieval. B.Tech final-year
capstone project.

> ⚠️ Research prototype. Not a medical device. Outputs must be verified by a
> qualified clinician.

**Stack:** Llama 3.1 (Ollama) · BiomedCLIP · BioBERT/ClinicalBERT (Sentence
Transformers) · LangChain · OpenCV · pydicom · FastAPI · Streamlit

See [docs/design.md](docs/design.md) for architecture, scope and the phase plan.

## Status

| Phase | Status |
|---|---|
| 0 Setup | ✅ models load on GPU; smoke test 5/5 |
| 1 Ingestion & privacy | ✅ de-ID (text + DICOM), clinical NER with negation, upload API |
| 2 Unimodal modules | ⏳ next |
| 3-6 | planned |

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
```

## Verify

```bash
pytest -q                          # fast unit tests, no downloads
python -m scripts.smoke_test       # loads every model end-to-end (first run downloads ~1 GB)
python -m scripts.eval_ingestion   # de-ID + NER metrics (add --verbose to see outputs)
uvicorn api.main:app --reload      # then open http://127.0.0.1:8000/docs
```

## API (so far)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/status` | Liveness; which models/services are configured and reachable |
| POST | `/cases` | Create a case, returns `case_id` |
| POST | `/cases/{id}/notes` | Upload TXT/PDF (`file`) or paste `text`; de-identified, then entities extracted |
| POST | `/cases/{id}/images` | Upload DICOM/PNG/JPEG; DICOM tags anonymised |
| GET | `/cases/{id}` | Case with de-identified notes, entity summaries, image metadata |
| GET | `/cases/{id}/images/{image_id}` | Processed image (PNG) |

Privacy guarantees: raw uploads are never written to disk; original filenames
are discarded; the audit log (`data/store/audit.log`) records actions and PHI
*types* removed, never content.

## Project layout

```
medassist/
  config.py           settings from env/.env
  ingestion/          loaders, text + DICOM de-ID, clinical NER, case store, pipeline
  image_pipeline/     OpenCV preprocessing, BiomedCLIP
  text_pipeline/      clinical embeddings (+ vector store/RAG in Phase 2)
  fusion/             prompts (+ router, evidence bundle, triage in Phase 3)
  llm/                Ollama client
  evaluation/         metrics (+ benchmark runners in Phase 5)
api/                  FastAPI app
ui/                   Streamlit app (Phase 4)
scripts/              smoke test, data download helpers
tests/                pytest suite
data/samples/         small synthetic samples (large datasets are not committed)
docs/                 design document, report material
```
