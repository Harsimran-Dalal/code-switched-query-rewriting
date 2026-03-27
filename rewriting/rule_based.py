from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from utils import Settings, get_settings
from .keyword_extractor import KeywordExtractor
from .transliteration_normalizer import TransliterationNormalizer

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_#+.-]*")


@dataclass
class RewriteResult:
    cleaned_query: str
    keyword_query: str
    rewritten_query: str


class RuleBasedRewriter:
    """Explainable retrieval-aware query rewriter for noisy code-switched queries."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.normalizer = TransliterationNormalizer()
        self.extractor = KeywordExtractor()

    def _clean_tokens(self, text: str) -> list[str]:
        normalized = self.normalizer.normalize(text)
        keywords = self.extractor.extract(normalized)
        drop_tokens = {"and", "at", "with", "bring", "which", "all"}

        compact_map = {
            "documents": "docs",
            "counseling": "counselling",
        }
        cleaned: list[str] = []
        seen: set[str] = set()
        for tok in keywords:
            if tok in drop_tokens:
                continue
            t = compact_map.get(tok, tok)
            if t in seen:
                continue
            cleaned.append(t)
            seen.add(t)
        return cleaned

    def _keyword_tokens(self, cleaned_tokens: list[str], original_text: str) -> list[str]:
        expanded_map = {
            "docs": "documents",
        }
        keyword_tokens = [expanded_map.get(t, t) for t in cleaned_tokens]

        lowered = original_text.lower()
        has_admission = "admission" in keyword_tokens
        has_form = "form" in keyword_tokens
        has_documents = "documents" in keyword_tokens
        has_upload = "upload" in keyword_tokens
        has_fees = "fees" in keyword_tokens
        has_seats = "seats" in keyword_tokens
        has_course = "course" in keyword_tokens
        has_counselling = "counselling" in keyword_tokens or "counseling" in lowered

        if has_admission and has_documents and has_upload:
            keyword_tokens = [t for t in keyword_tokens if t != "form"]
            if "required" not in keyword_tokens:
                keyword_tokens.append("required")
        if has_fees and has_seats and has_course:
            if "details" not in keyword_tokens:
                keyword_tokens.append("details")
        if has_counselling and has_documents:
            keyword_tokens = [t for t in keyword_tokens if t not in {"details", "time", "bring", "with", "and"}]
            if "required" not in keyword_tokens:
                keyword_tokens.insert(1 if len(keyword_tokens) > 0 else 0, "required")

        out: list[str] = []
        seen: set[str] = set()
        for t in keyword_tokens:
            if t in seen:
                continue
            out.append(t)
            seen.add(t)
        return out

    def _rewrite_natural(self, keyword_tokens: list[str], cleaned_tokens: list[str]) -> str:
        tok_set = set(keyword_tokens)

        if {"admission", "documents", "upload"}.issubset(tok_set):
            return "required documents for admission form upload"

        if {"fees", "seats", "course"}.issubset(tok_set):
            return "fees structure and seat availability for this course"

        if "counselling" in tok_set and "documents" in tok_set:
            return "required documents for counselling"

        if keyword_tokens:
            return " ".join(keyword_tokens)
        return " ".join(cleaned_tokens)

    def rewrite_result(self, original_query: str) -> RewriteResult:
        cleaned_tokens = self._clean_tokens(original_query)
        cleaned_query = " ".join(cleaned_tokens).strip()

        keyword_tokens = self._keyword_tokens(cleaned_tokens, original_query)
        keyword_query = " ".join(keyword_tokens).strip()

        rewritten_query = self._rewrite_natural(keyword_tokens, cleaned_tokens).strip()

        if not cleaned_query:
            fallback = " ".join(_TOKEN_RE.findall(original_query.lower()))
            cleaned_query = fallback
            keyword_query = fallback
            rewritten_query = fallback

        return RewriteResult(
            cleaned_query=cleaned_query,
            keyword_query=keyword_query,
            rewritten_query=rewritten_query,
        )

    def rewrite(self, original_query: str) -> str:
        """Backwards-compatible short output for retrieval query usage."""
        result = self.rewrite_result(original_query)
        return result.rewritten_query or result.keyword_query or result.cleaned_query


if __name__ == "__main__":
    samples = [
        "Mera admission form bharne ke liye kaun kaun se docs upload karne padenge?",
        "Fees te seats da ki scene aa is course ch?",
        "Counselling time te kehde documents naal leke aane hunde ne?",
    ]

    rw = RuleBasedRewriter()
    for q in samples:
        out = rw.rewrite_result(q)
        print("INPUT         :", q)
        print("CLEANED QUERY :", out.cleaned_query)
        print("KEYWORD QUERY :", out.keyword_query)
        print("REWRITTEN     :", out.rewritten_query)
        print("-" * 80)
