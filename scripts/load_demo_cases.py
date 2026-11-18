"""Load the demo cases into the case store through the real ingestion pipeline
(de-identification, NER, DICOM/PNG handling), so the demo exercises everything.

Usage: python -m scripts.load_demo_cases [--analyze] [--reset]
  --analyze  also pre-compute image analysis so the first question is fast
  --reset    delete the demo cases a previous run created (other cases untouched)
"""
from __future__ import annotations

import argparse
import json
import re
import shutil

from medassist.config import PROJECT_ROOT
from medassist.datasets.iu_xray import DATA_DIR
from medassist.ingestion.pipeline import ingest_image, ingest_note
from medassist.ingestion.store import CaseStore

DEMO_FILE = PROJECT_ROOT / "data" / "samples" / "demo_cases.json"
EXAMPLES_DIR = PROJECT_ROOT / "data" / "samples" / "biomedclip_examples"


def image_path(spec: dict):
    if spec["source"] == "iu_xray":
        return DATA_DIR / "images" / f"{spec['uid']}.png"
    return EXAMPLES_DIR / spec["file"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--reset", action="store_true", help="delete demo cases created by a previous run")
    args = parser.parse_args()

    store = CaseStore()
    previous = store.root / "demo_index.json"
    if args.reset and previous.exists():  # remove only cases this script created earlier
        for cid in json.loads(previous.read_text(encoding="utf-8")).values():
            if re.fullmatch(r"[0-9a-f]{12}", cid) and (store.root / cid).is_dir():
                shutil.rmtree(store.root / cid)
        print("Removed previous demo cases.")
    cases = json.loads(DEMO_FILE.read_text(encoding="utf-8"))["cases"]
    index = {}
    for case in cases:
        path = image_path(case["image"])
        if not path.exists():
            print(f"  skip {case['id']}: image missing ({path}); run scripts.prepare_iu_xray / smoke_test first")
            continue
        cid = store.create_case(title=case["title"], tags={"demo": case["id"], "shows": case["shows"]})
        note = ingest_note(store, cid, text=case["note"])
        image = ingest_image(store, cid, path.name, path.read_bytes())
        if args.analyze:
            from medassist.analysis import analyze_image
            from PIL import Image

            store.update_image_record(cid, image["image_id"],
                                      analysis=analyze_image(Image.open(store.image_path(cid, image["image_id"]))))
        index[case["id"]] = cid
        removed = sum(note["deid"]["counts"].values())
        print(f"  {case['id']:<32} case={cid}  PHI removed={removed}  active={note['summary']['active_problems'][:4]}")
    (store.root / "demo_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nLoaded {len(index)} demo cases -> {store.root / 'demo_index.json'}")


if __name__ == "__main__":
    main()
