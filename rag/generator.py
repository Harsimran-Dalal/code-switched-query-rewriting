from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List

from utils import Settings, get_settings


@dataclass
class GeneratedAnswer:
    answer: str
    summary: str
    citations: List[str]
    mode: str


def _snip(text: str, max_chars: int = 650) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3].rstrip() + "..."


def _ctx_get(ctx: Any, key: str, default: Any = "") -> Any:
    if isinstance(ctx, dict):
        return ctx.get(key, default)
    return getattr(ctx, key, default)


def _build_citations(contexts: List[Any], max_citations: int = 3) -> List[str]:
    out: list[str] = []
    seen: set[str] = set()
    for ctx in contexts:
        doc_id = str(_ctx_get(ctx, "doc_id", "unknown"))
        chunk_id = str(_ctx_get(ctx, "chunk_id", "n/a"))
        key = f"{doc_id}::{chunk_id}"
        if key in seen:
            continue
        out.append(f"{doc_id} ({chunk_id})")
        seen.add(key)
        if len(out) >= max_citations:
            break
    return out


class ExtractiveAnswerGenerator:
    """CPU-friendly fallback answer generator based only on retrieved context."""

    def generate(self, query: str, contexts: List[Any]) -> GeneratedAnswer:
        if not contexts:
            return GeneratedAnswer(
                answer="No relevant context retrieved. Try a clearer query or increase TOP_K.",
                summary="No answer could be produced because no relevant context was retrieved.",
                citations=[],
                mode="extractive",
            )

        best = contexts[0]
        best_text = str(_ctx_get(best, "text", ""))
        best_snip = _snip(best_text, max_chars=650)
        answer = (
            "Based on the top retrieved context, here is the most relevant information:\n\n"
            f"{best_snip}"
        )
        summary = _snip(best_snip, max_chars=170)
        return GeneratedAnswer(
            answer=answer,
            summary=summary,
            citations=_build_citations(contexts),
            mode="extractive",
        )


class RAGGenerator:
    """Baseline answer generator for Phase 1 (extractive only)."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.extractive = ExtractiveAnswerGenerator()

    def generate(self, query: str, contexts: List[Any]) -> GeneratedAnswer:
        return self.extractive.generate(query, contexts)


if __name__ == "__main__":
    from retrieval.dense_retriever import DenseRetriever
    from utils.logger import setup_logging

    setup_logging()
    sample_query = "scholarship eligibility for undergrad admission"

    retriever = DenseRetriever()
    retriever.ensure_loaded(build_if_missing=True)
    hits = retriever.search(sample_query, top_k=3)

    generator = RAGGenerator()
    out = generator.generate(sample_query, hits)

    print("\n=== Generator Output ===")
    print("Mode   :", out.mode)
    print("Summary:", out.summary)
    print("Answer :")
    print(out.answer)
    print("Cites  :", ", ".join(out.citations))

