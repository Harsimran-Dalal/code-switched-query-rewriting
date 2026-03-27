from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from retrieval.dense_retriever import DenseRetriever
from rag.generator import RAGGenerator
from rewriting.rule_based import RuleBasedRewriter
from utils import Settings, get_settings
from utils.logger import setup_logging


@dataclass
class PipelineResult:
    query_original: str
    query_used: str
    rewrite_details: Optional[dict[str, str]]
    retrieved: list[dict[str, Any]]
    answer: str
    summary: str
    generation_mode: str
    citations: list[str]


@dataclass
class PipelineComparison:
    baseline: PipelineResult
    rewritten: PipelineResult


class RAGPipeline:
    def __init__(
        self,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.retriever = DenseRetriever(self.settings)
        self.generator = RAGGenerator(self.settings)
        self.rewriter = RuleBasedRewriter(self.settings)

    @staticmethod
    def _serialize_retrieved(retrieved) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for i, r in enumerate(retrieved):
            payload.append(
                {
                    "rank": getattr(r, "rank", i + 1),
                    "score": r.score,
                    "doc_id": r.doc_id,
                    "doc_title": r.doc_title,
                    "chunk_id": r.chunk_id,
                    "source_path": r.source_path,
                    "text": r.text,
                }
            )
        return payload

    def run(self, query: str, *, rewrite: bool = False) -> PipelineResult:
        query_original = query.strip()
        query_used = query_original
        rewrite_details: Optional[dict[str, str]] = None

        if rewrite:
            rw = self.rewriter.rewrite_result(query_original)
            query_used = rw.rewritten_query or rw.keyword_query or rw.cleaned_query or query_original
            rewrite_details = {
                "cleaned_query": rw.cleaned_query,
                "keyword_query": rw.keyword_query,
                "rewritten_query": rw.rewritten_query,
            }

        retrieved = self.retriever.search(query_used, top_k=self.settings.top_k)
        gen = self.generator.generate(query_used, retrieved)
        retrieved_payload = self._serialize_retrieved(retrieved)

        return PipelineResult(
            query_original=query_original,
            query_used=query_used,
            rewrite_details=rewrite_details,
            retrieved=retrieved_payload,
            answer=gen.answer,
            summary=gen.summary,
            generation_mode=gen.mode,
            citations=gen.citations,
        )

    def compare_modes(self, query: str) -> PipelineComparison:
        baseline = self.run(query, rewrite=False)
        rewritten = self.run(query, rewrite=True)
        return PipelineComparison(baseline=baseline, rewritten=rewritten)

    def run_to_json(self, query: str, out_path: str | Path, *, rewrite: bool = False) -> None:
        out_path = Path(out_path)
        res = self.run(query, rewrite=rewrite)
        out_path.write_text(json.dumps(asdict(res), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG pipeline with optional explainable query rewriting")
    parser.add_argument("--query", required=True, help="Input ASR transcript or text query")
    parser.add_argument("--rewrite", action="store_true", help="Enable retrieval-aware query rewriting")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both baseline and rewritten modes and print side-by-side JSON",
    )
    args = parser.parse_args()

    setup_logging()
    pipeline = RAGPipeline()
    if args.compare:
        result = pipeline.compare_modes(args.query)
    else:
        result = pipeline.run(args.query, rewrite=args.rewrite)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

