"""Retrieval-augmented question answering over clinical text (LangChain + Ollama).

    passages --chunk--> BioBERT embeddings --> VectorIndex
    question --embed--> top-k chunks --> numbered context --> Llama --> cited answer

Answers must cite chunks as [1], [2] ...; the citation list maps those numbers
back to sources so the UI can show the evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from medassist.config import get_settings
from medassist.text_pipeline.embeddings import ClinicalEmbedder, get_embedder
from medassist.vector_index import SearchHit, VectorIndex

RAG_SYSTEM = """You are a clinical information assistant. Answer the question using ONLY
the numbered context passages. Cite every claim with the passage number in square
brackets, e.g. [2]. If the passages do not contain the answer, say exactly what is
missing instead of guessing. Be concise. This is decision support, not a diagnosis."""

RAG_HUMAN = """Context passages:
{context}

Question: {question}"""


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    citations: list[SearchHit]
    cited_numbers: list[int]


def chunk_text(text: str, max_chars: int = 600, overlap_sentences: int = 1) -> list[str]:
    """Sentence-aware chunking with a one-sentence overlap between chunks."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]
    chunks, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > max_chars:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


class KnowledgeBase:
    def __init__(self, embedder: ClinicalEmbedder | None = None) -> None:
        self.embedder = embedder or get_embedder()
        self.index = VectorIndex(self.embedder.dimension)

    def add_passages(self, passages: list[Passage], chunk: bool = True) -> int:
        ids, texts, meta = [], [], []
        for p in passages:
            pieces = chunk_text(p.text) if chunk else [p.text]
            for n, piece in enumerate(pieces):
                ids.append(f"{p.id}#{n}")
                texts.append(piece)
                meta.append({"text": piece, "source": p.source, "passage_id": p.id, **p.metadata})
        if texts:
            self.index.add(ids, self.embedder.embed(texts), meta)
        return len(texts)

    def retrieve(self, query: str, k: int = 4) -> list[SearchHit]:
        return self.index.search(self.embedder.embed([query])[0], k=k)


def format_context(hits: list[SearchHit]) -> str:
    return "\n\n".join(f"[{i}] ({h.metadata.get('source', 'unknown')}) {h.metadata['text']}"
                       for i, h in enumerate(hits, 1))


def build_chain(temperature: float = 0.1):
    """LangChain pipeline: prompt template | ChatOllama | string parser."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama

    settings = get_settings()
    prompt = ChatPromptTemplate.from_messages([("system", RAG_SYSTEM), ("human", RAG_HUMAN)])
    llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_url,
                     temperature=temperature, num_ctx=settings.llm_num_ctx)
    return prompt | llm | StrOutputParser()


def answer_question(question: str, kb: KnowledgeBase, k: int = 4, chain=None) -> RagAnswer:
    hits = kb.retrieve(question, k=k)
    if not hits:
        return RagAnswer("No documents are available to answer this question.", [], [])
    text = (chain or build_chain()).invoke({"context": format_context(hits), "question": question})
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
    return RagAnswer(answer=text.strip(), citations=hits, cited_numbers=cited)
