"""Phase 0 smoke test: verifies every model and utility actually runs.

Usage:  python -m scripts.smoke_test            (all checks)
        python -m scripts.smoke_test --skip-llm  (no Ollama needed)
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from medassist.config import PROJECT_ROOT, get_settings

RESULTS: list[tuple[str, bool, str]] = []
SAMPLES = PROJECT_ROOT / "data" / "samples"


def check(name: str):
    def wrap(fn):
        def run(*args, **kwargs):
            start = time.perf_counter()
            try:
                detail = fn(*args, **kwargs)
                RESULTS.append((name, True, f"{detail}  ({time.perf_counter() - start:.1f}s)"))
            except Exception as exc:  # report and keep going
                traceback.print_exc()
                RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        return run
    return wrap


@check("PyTorch / GPU")
def check_torch() -> str:
    import torch

    if not torch.cuda.is_available():
        return f"torch {torch.__version__}, CPU only (GPU not detected)"
    props = torch.cuda.get_device_properties(0)
    return f"torch {torch.__version__}, {props.name}, {props.total_memory / 2**30:.1f} GB"


@check("DICOM loading (pydicom + VOI LUT)")
def check_dicom() -> str:
    from pydicom.data import get_testdata_file

    from medassist.ingestion.loaders import load_upload

    path = Path(get_testdata_file("CT_small.dcm"))
    img = load_upload(path.name, path.read_bytes())
    assert img.pixels.dtype.name == "uint8" and img.pixels.max() > img.pixels.min()
    return f"{img.pixels.shape}, metadata={img.metadata}"


def fetch_biomedclip_examples() -> list[Path]:
    """Download the example images shipped with the BiomedCLIP model repo."""
    from huggingface_hub import hf_hub_download, list_repo_files

    repo = get_settings().biomedclip_model.removeprefix("hf-hub:")
    wanted = [
        f for f in list_repo_files(repo)
        if f.startswith("example_data/") and f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    out_dir = SAMPLES / "biomedclip_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for remote in wanted:
        local = out_dir / Path(remote).name
        if not local.exists():
            local.write_bytes(Path(hf_hub_download(repo, remote)).read_bytes())
        paths.append(local)
    return paths


@check("BiomedCLIP zero-shot (open_clip)")
def check_biomedclip() -> str:
    from medassist.image_pipeline.biomedclip import BiomedCLIP
    from medassist.image_pipeline.preprocess import to_model_image
    from medassist.ingestion.loaders import load_upload

    images = fetch_biomedclip_examples()
    assert images, "no example images found in the BiomedCLIP repo"
    model = BiomedCLIP()
    labels = [
        "chest X-ray", "brain MRI", "bone X-ray", "hematoxylin and eosin histopathology",
        "squamous cell carcinoma histopathology", "adenocarcinoma histopathology",
        "covid line chart", "immunohistochemistry histopathology", "pie chart",
    ]
    lines = []
    for path in images:
        loaded = load_upload(path.name, path.read_bytes())
        top = model.zero_shot(to_model_image(loaded.pixels, apply_clahe=False), labels)[0]
        lines.append(f"\n      {path.name:<45} -> {top.label} ({top.score:.2f})")
    return f"device={model.device}, {len(images)} images:" + "".join(lines)


@check("Clinical text embeddings (Sentence Transformers)")
def check_embeddings() -> str:
    from medassist.text_pipeline.embeddings import ClinicalEmbedder

    emb = ClinicalEmbedder()
    query, related, unrelated = emb.embed([
        "Patient has fever, productive cough and right lower lobe consolidation.",
        "Findings consistent with community-acquired pneumonia.",
        "Healed fracture of the distal radius, no acute bony injury.",
    ])
    sim_related, sim_unrelated = float(query @ related), float(query @ unrelated)
    assert sim_related > sim_unrelated, "embedding similarity ordering is wrong"
    return f"dim={emb.dimension}, related={sim_related:.2f} > unrelated={sim_unrelated:.2f}"


@check("Llama via Ollama")
def check_llm() -> str:
    from medassist.llm.ollama_client import OllamaClient

    client = OllamaClient()
    settings = client.settings
    models = client.list_models()
    assert any(m.startswith(settings.llm_model) for m in models), (
        f"'{settings.llm_model}' not pulled. Run: ollama pull {settings.llm_model}"
    )
    reply = client.chat([{"role": "user", "content": "In one short sentence: what is a chest X-ray used for?"}])
    return f"{settings.llm_model}: {reply[:120]!r}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    check_torch()
    check_dicom()
    check_biomedclip()
    check_embeddings()
    if not args.skip_llm:
        check_llm()

    print("\n================ SMOKE TEST ================")
    for name, ok, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    failed = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
