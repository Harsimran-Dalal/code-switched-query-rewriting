from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from retrieval.dense_retriever import DenseRetriever
from rag.generator import RAGGenerator
from rewriting.rule_based import RuleBasedRewriter
from utils import Settings, get_settings
from utils.logger import setup_logging

logger = logging.getLogger(__name__)

SUPPORTED_TOPIC_HINT = (
    "This query appears to be outside the supported academic/admission domain. "
    "Please ask about admission, scholarship, counselling, fees, merit, cutoff, eligibility, "
    "or required documents."
)

_DOMAIN_KEYWORDS = {
    "admission",
    "admissions",
    "college",
    "scholarship",
    "scholarships",
    "fees",
    "fee",
    "counselling",
    "counseling",
    "cutoff",
    "merit",
    "document",
    "documents",
    "eligibility",
    "eligible",
    "registration",
    "register",
    "seats",
    "seat",
    "certificate",
    "certificates",
    "hostel",
    "branch",
    "course",
    "courses",
    "undergrad",
    "ug",
    "pg",
    "process",
    "admit",
    "enrollment",
    "income",
    "minority",
    "reservation",
    "quota",
    "counselling",
    "counselling",
    "form",
    "apply",
    "application",
    "kaunseling",
    "kaunsling",
    "admision",
    "admishan",
    "scholarship",
    "fee",
    "fees",
    "cut off",
    "cutt off",
    "cutof",
    "meritlist",
    "hostel",
    "branch",
    "course",
    "dakhla",
    "dakhlaa",
    "dakhila",
    "dakhla",
    "daakhla",
    "daakhila",
    "college",
    "collage",
    "seat",
    "seatan",
    "documents",
    "doc",
    "certificate",
    "income certificate",
    "migration certificate",
    "character certificate",
    "adhaar",
    "aadhar",
    "bonafide",
    "counselling",
    "counseling",
    "merit",
    "registration",
    "scholarship",
    "eligibility",
    "form bharna",
    "form bharna",
    "fees kitni",
    "seat kitni",
    "required documents",
    "admission process",
}

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_#+.-]*")


def is_low_confidence(score: float, threshold: float = 0.30) -> bool:
    return float(score) < float(threshold)


def is_domain_relevant(query: str, min_hits: int = 1) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False

    tokens = set(_TOKEN_RE.findall(q))
    hits = sum(1 for kw in _DOMAIN_KEYWORDS if (kw in q) or (kw in tokens))
    return hits >= int(min_hits)


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
    status: str = "ok"
    reason: Optional[str] = None
    top_score: float = 0.0
    threshold: float = 0.30
    low_confidence: bool = False
    domain_relevant: bool = True
    message: Optional[str] = None


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
        retrieved_payload = self._serialize_retrieved(retrieved)
        top_score = float(retrieved_payload[0].get("score", 0.0) if retrieved_payload else 0.0)
        threshold = float(self.settings.low_confidence_threshold)
        domain_relevant = is_domain_relevant(query_original, min_hits=self.settings.domain_min_hits)
        low_confidence = is_low_confidence(top_score, threshold=threshold)

        status = "ok"
        reason: Optional[str] = None
        message: Optional[str] = None
        if not domain_relevant:
            status = "rejected"
            reason = "out_of_domain"
            message = SUPPORTED_TOPIC_HINT
        elif low_confidence:
            status = "rejected"
            reason = "low_confidence"
            message = (
                "The retriever confidence is too low for a trustworthy answer. "
                "Please rephrase your question or ask about admission, scholarship, counselling, fees, merit, cutoff, eligibility, or required documents."
            )

        if status == "rejected":
            logger.info(
                "Rejecting query. reason=%s top_score=%.3f threshold=%.3f domain_relevant=%s",
                reason,
                top_score,
                threshold,
                domain_relevant,
            )
            return PipelineResult(
                query_original=query_original,
                query_used=query_used,
                rewrite_details=rewrite_details,
                retrieved=retrieved_payload,
                answer=message or "",
                summary=message or "",
                generation_mode="rejected",
                citations=[],
                status=status,
                reason=reason,
                top_score=top_score,
                threshold=threshold,
                low_confidence=low_confidence,
                domain_relevant=domain_relevant,
                message=message,
            )

        gen = self.generator.generate(query_used, retrieved)

        return PipelineResult(
            query_original=query_original,
            query_used=query_used,
            rewrite_details=rewrite_details,
            retrieved=retrieved_payload,
            answer=gen.answer,
            summary=gen.summary,
            generation_mode=gen.mode,
            citations=gen.citations,
            status="ok",
            reason=None,
            top_score=top_score,
            threshold=threshold,
            low_confidence=low_confidence,
            domain_relevant=domain_relevant,
            message=None,
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

