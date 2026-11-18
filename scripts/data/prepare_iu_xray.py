"""Download and prepare a stratified IU X-Ray subset.

Source: Hugging Face mirror of the Kaggle/Open-i release
(sasi2004/chest-xrays-indiana-university), because the NIH Open-i bulk
download endpoint currently returns an error page.

Steps: read reports + projections CSVs -> keep one frontal image per study ->
derive labels from MeSH -> stratified sample -> download, resize to 512 px ->
write data/processed/iu_xray/studies.jsonl with a fixed train/test split.

Usage: python -m scripts.data.prepare_iu_xray [--n 1500] [--workers 8]
"""
from __future__ import annotations

import argparse
import io
import random
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

from medassist.datasets.iu_xray import (
    DATA_DIR, FINDINGS, IMAGES_DIR, Study, clean_text, labels_from_mesh, mesh_terms, save_studies,
)

REPO = "sasi2004/chest-xrays-indiana-university"
RARE_FIRST = ["pleural effusion", "nodule", "emphysema", "pulmonary edema", "fracture"]
SEED = 42
TEST_FRACTION = 0.3
MAX_SIDE = 512


def fetch_csv(name: str) -> pd.DataFrame:
    target = DATA_DIR / name
    if not target.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download(REPO, name, repo_type="dataset"), target)
    return pd.read_csv(target)


def build_candidates() -> list[Study]:
    reports = fetch_csv("indiana_reports.csv")
    projections = fetch_csv("indiana_projections.csv")
    frontal = projections[projections.projection == "Frontal"].drop_duplicates("uid").set_index("uid")

    studies = []
    for row in reports.itertuples():
        if row.uid not in frontal.index:
            continue
        terms = mesh_terms(row.MeSH)
        labels = labels_from_mesh(terms)
        findings, impression = clean_text(row.findings), clean_text(row.impression)
        if labels is None or labels == [] or not (findings or impression):
            continue  # excluded, off-target abnormal, or no report text
        studies.append(Study(
            uid=str(row.uid),
            image=f"images/{row.uid}.png",
            indication=clean_text(row.indication),
            findings=findings,
            impression=impression,
            mesh=terms,
            labels=labels,
            split="",
            extra={"source_file": frontal.loc[row.uid, "filename"]},
        ))
    return studies


def stratified_sample(studies: list[Study], n: int, normal_fraction: float = 0.4) -> list[Study]:
    rng = random.Random(SEED)
    normals = [s for s in studies if s.is_normal]
    abnormal = [s for s in studies if not s.is_normal]
    rng.shuffle(normals)
    rng.shuffle(abnormal)

    n_abnormal = min(len(abnormal), n - int(n * normal_fraction))
    chosen: dict[str, Study] = {}
    for label in RARE_FIRST:  # guarantee rare findings are represented
        for s in abnormal:
            if label in s.labels:
                chosen[s.uid] = s
    for s in abnormal:
        if len(chosen) >= n_abnormal:
            break
        chosen.setdefault(s.uid, s)
    for s in normals[: max(0, n - len(chosen))]:
        chosen[s.uid] = s

    sample = sorted(chosen.values(), key=lambda s: int(s.uid))
    for s in sample:  # deterministic split by hash-free shuffle
        s.split = "test" if rng.random() < TEST_FRACTION else "train"
    return sample


def download_one(study: Study, tmp: Path) -> str | None:
    out = DATA_DIR / study.image
    if out.exists():
        return None
    remote = f"images/images_normalized/{study.extra['source_file']}"
    local = Path(hf_hub_download(REPO, remote, repo_type="dataset", local_dir=tmp))
    img = Image.open(io.BytesIO(local.read_bytes())).convert("L")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="PNG", optimize=True)
    local.unlink(missing_ok=True)  # keep only the resized copy
    return study.uid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1500, help="number of studies to prepare")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    candidates = build_candidates()
    sample = stratified_sample(candidates, args.n)
    print(f"Eligible studies: {len(candidates)}; selected: {len(sample)}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    with tempfile.TemporaryDirectory(dir=DATA_DIR) as tmp, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(download_one, s, Path(tmp)): s for s in sample}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as exc:  # network hiccups: report, keep going
                failures.append((futures[fut].uid, repr(exc)))
            if i % 100 == 0 or i == len(sample):
                print(f"  images: {i}/{len(sample)}", flush=True)

    failed = {uid for uid, _ in failures}
    sample = [s for s in sample if s.uid not in failed and (DATA_DIR / s.image).exists()]
    save_studies(sample)

    print(f"\nSaved {len(sample)} studies ({len(failures)} download failures) -> {DATA_DIR / 'studies.jsonl'}")
    for split in ("train", "test"):
        subset = [s for s in sample if s.split == split]
        counts = {f: sum(f in s.labels for s in subset) for f in ["normal", *FINDINGS]}
        print(f"  {split:5s} n={len(subset):4d}  {counts}")


if __name__ == "__main__":
    main()
