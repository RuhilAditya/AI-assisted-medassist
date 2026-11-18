"""Similar-case retrieval over the IU X-Ray knowledge base (train split).

Two routes, both through BiomedCLIP's shared space:
  image -> report   cross-modal: compare the query X-ray with report *text*
                    embeddings (uses the image-text alignment directly)
  image -> image    find visually similar X-rays, return their reports
The retrieved reports are evidence for the LLM ("similar prior cases were
described as ..."), never presented as the patient's own report.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from PIL import Image

from medassist.datasets.iu_xray import DATA_DIR, Study, load_studies
from medassist.image_pipeline.cxr import CXRClassifier, pad_to_square
from medassist.vector_index import VectorIndex

REPORT_INDEX = DATA_DIR / "index" / "reports"
IMAGE_INDEX = DATA_DIR / "index" / "images"


@dataclass(frozen=True)
class SimilarCase:
    uid: str
    score: float
    report: str
    labels: list[str]
    route: str  # "image->report" | "image->image"


def build_indexes(classifier: CXRClassifier, studies: list[Study] | None = None) -> tuple[VectorIndex, VectorIndex]:
    studies = studies if studies is not None else load_studies("train")
    model = classifier.model
    meta = [{"report": s.report, "labels": s.labels} for s in studies]
    ids = [s.uid for s in studies]

    report_vecs = []
    for i in range(0, len(studies), 64):
        report_vecs.append(model.embed_texts([s.report for s in studies[i:i + 64]]))
    import numpy as np

    reports = VectorIndex(dim=report_vecs[0].shape[1])
    reports.add(ids, np.vstack(report_vecs), meta)

    images = [Image.open(s.image_path()) for s in studies]
    image_vecs = classifier.embed(images)
    image_index = VectorIndex(dim=image_vecs.shape[1])
    image_index.add(ids, image_vecs, meta)

    reports.save(REPORT_INDEX)
    image_index.save(IMAGE_INDEX)
    return reports, image_index


class CaseRetriever:
    def __init__(self, classifier: CXRClassifier) -> None:
        if not REPORT_INDEX.with_suffix(".npy").exists():
            raise FileNotFoundError("Case indexes missing. Run: python -m scripts.build_case_index")
        self.classifier = classifier
        self.reports = VectorIndex.load(REPORT_INDEX)
        self.images = VectorIndex.load(IMAGE_INDEX)

    def similar(self, image: Image.Image, k: int = 3, route: str = "image->image") -> list[SimilarCase]:
        query = self.classifier.embed([pad_to_square(image)])[0]
        index = self.images if route == "image->image" else self.reports
        return [
            SimilarCase(h.id, round(h.score, 3), h.metadata["report"], h.metadata["labels"], route)
            for h in index.search(query, k=k)
        ]


@lru_cache(maxsize=1)
def get_case_retriever() -> CaseRetriever:
    from medassist.image_pipeline.cxr import CXRClassifier

    return CaseRetriever(CXRClassifier())
