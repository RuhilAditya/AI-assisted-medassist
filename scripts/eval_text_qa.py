"""Text-only evaluation on PubMedQA (pqa_labeled, 1000 expert-annotated questions).

Part A: retrieval. All 1000 abstracts are the knowledge base; for each question,
        is its own abstract retrieved? Recall@K for several embedding models
        (BioBERT vs ClinicalBERT vs PubMedBERT-MARCO vs a general-domain model).
Part B: question answering with Llama (yes / no / maybe):
          closed-book  no context
          RAG          top-3 retrieved passages (best embedder from Part A)
          oracle       the gold abstract (upper bound)

Usage: python -m scripts.eval_text_qa [--limit 150] [--skip-qa]
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, f1_score

from medassist.evaluation.results import save_results
from medassist.llm.ollama_client import OllamaClient
from medassist.text_pipeline.embeddings import ClinicalEmbedder
from medassist.vector_index import VectorIndex

EMBEDDERS = {
    "BioBERT (S-T, NLI/STS)": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "ClinicalBERT (mean-pooled)": "emilyalsentzer/Bio_ClinicalBERT",
    "PubMedBERT (MS-MARCO)": "pritamdeka/S-PubMedBert-MS-MARCO",
    "MiniLM (general domain)": "sentence-transformers/all-MiniLM-L6-v2",
}
KS = (1, 3, 5, 10)
LABELS = ["yes", "no", "maybe"]

QA_SYSTEM = ("You answer biomedical research questions. Reply with a one-sentence justification, "
             "then a final line exactly of the form 'Answer: yes', 'Answer: no' or 'Answer: maybe'.")


def load_pubmedqa() -> pd.DataFrame:
    path = hf_hub_download("qiaojin/PubMedQA", "pqa_labeled/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df["abstract"] = df["context"].apply(lambda c: " ".join(c["contexts"]))
    return df.reset_index(drop=True)


def retrieval_eval(df: pd.DataFrame) -> tuple[dict, str, dict]:
    results, rankings = {}, {}
    for name, model_id in EMBEDDERS.items():
        start = time.perf_counter()
        emb = ClinicalEmbedder(model_id)
        index = VectorIndex(emb.dimension)
        index.add([str(p) for p in df.pubid], emb.embed(df.abstract.tolist()))
        top = index.search_many(emb.embed(df.question.tolist()), k=max(KS))
        gold = [str(p) for p in df.pubid]
        results[name] = {f"recall@{k}": round(float(np.mean([g in t[:k] for g, t in zip(gold, top)])), 3) for k in KS}
        results[name]["seconds"] = round(time.perf_counter() - start, 1)
        rankings[name] = top
        print(f"  {name:<28}" + "  ".join(f"R@{k}={results[name][f'recall@{k}']:.3f}" for k in KS))
        del emb
    best = max(results, key=lambda n: results[n]["recall@3"])
    return results, best, rankings[best]


def parse_answer(text: str) -> str:
    match = re.search(r"answer\s*:\s*(yes|no|maybe)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    tail = text.lower()[-40:]
    return next((label for label in LABELS if label in tail), "maybe")


def ask(client: OllamaClient, question: str, context: str | None) -> str:
    user = f"Question: {question}" if context is None else f"Context:\n{context}\n\nQuestion: {question}"
    reply = client.chat([{"role": "system", "content": QA_SYSTEM}, {"role": "user", "content": user}],
                        temperature=0.0)
    return parse_answer(reply)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=150, help="questions used for the (slow) LLM part")
    parser.add_argument("--skip-qa", action="store_true")
    args = parser.parse_args()

    df = load_pubmedqa()
    print(f"PubMedQA labeled: {len(df)} questions; label mix {df.final_decision.value_counts().to_dict()}")
    print("\nA) Retrieval over 1000 abstracts")
    retrieval, best, best_ranking = retrieval_eval(df)
    print(f"Best embedder for RAG: {best}")

    qa = {}
    if not args.skip_qa:
        client = OllamaClient()
        sample = df.sample(n=min(args.limit, len(df)), random_state=42)
        abstracts = dict(zip(df.pubid.astype(str), df.abstract))
        preds = {"closed-book": [], "RAG (top-3)": [], "oracle context": []}
        print(f"\nB) QA with {client.settings.llm_model} on {len(sample)} questions")
        for n, (i, row) in enumerate(sample.iterrows(), 1):
            retrieved = "\n\n".join(f"[{j}] {abstracts[p]}" for j, p in enumerate(best_ranking[i][:3], 1))
            preds["closed-book"].append(ask(client, row.question, None))
            preds["RAG (top-3)"].append(ask(client, row.question, retrieved))
            preds["oracle context"].append(ask(client, row.question, row.abstract))
            if n % 25 == 0:
                print(f"  {n}/{len(sample)}", flush=True)
        gold = sample.final_decision.tolist()
        for mode, p in preds.items():
            qa[mode] = {"accuracy": round(accuracy_score(gold, p), 3),
                        "macro_f1": round(f1_score(gold, p, labels=LABELS, average="macro", zero_division=0), 3)}
        majority = max(set(gold), key=gold.count)
        qa["majority-class baseline"] = {"accuracy": round(gold.count(majority) / len(gold), 3),
                                         "macro_f1": round(f1_score(gold, [majority] * len(gold), labels=LABELS,
                                                                    average="macro", zero_division=0), 3)}
        print(f"\n{'mode':<26}{'accuracy':>10}{'macro F1':>10}")
        for mode, m in qa.items():
            print(f"{mode:<26}{m['accuracy']:>10}{m['macro_f1']:>10}")

    path = save_results("phase2_text_qa", {
        "dataset": {"name": "PubMedQA pqa_labeled", "questions": len(df), "qa_sample": 0 if args.skip_qa else args.limit},
        "retrieval": retrieval, "rag_embedder": best, "qa": qa,
    })
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
