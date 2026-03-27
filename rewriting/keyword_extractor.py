from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_#+.-]*")

DEFAULT_FILLERS = {
    "um",
    "uh",
    "hmm",
    "like",
    "actually",
    "basically",
    "please",
    "plz",
    "pls",
    "yaar",
    "yr",
    "bro",
    "bhai",
    "mera",
    "meri",
    "mere",
    "matlab",
    "waise",
    "accha",
    "scene",
    "okay",
    "ok",
    "haan",
    "ji",
}

DEFAULT_STOPWORDS = {
    # English function words
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "of",
    "to",
    "for",
    "in",
    "on",
    "at",
    "by",
    "from",
    "with",
    "as",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "should",
    "what",
    "which",
    "how",
    "when",
    "where",
    "why",
    # Hindi/Punjabi function words in Roman script
    "kya",
    "ka",
    "ki",
    "ke",
    "ko",
    "se",
    "mein",
    "me",
    "pe",
    "par",
    "aur",
    "ya",
    "hai",
    "h",
    "hain",
    "tha",
    "thi",
    "the",
    "hoga",
    "hogi",
    "honge",
    "kar",
    "karna",
    "karni",
    "karne",
    "liye",
    "liya",
    "padenge",
    "hunde",
    "ne",
    "all",
    "details",
    "batao",
    "batado",
    "daso",
    "daso",
    "de",
    "da",
    "di",
}

DEFAULT_PRESERVE_KEYWORDS = {
    # Education/admission domain
    "admission",
    "counselling",
    "counseling",
    "eligibility",
    "fees",
    "scholarship",
    "documents",
    "upload",
    "required",
    "course",
    "form",
    "cutoff",
    "merit",
    "application",
    "deadline",
    "seat",
    "seats",
    # Finance domain examples and common terms
    "inflation",
    "bond",
    "bonds",
    "yield",
    "yields",
    "interest",
    "rate",
    "rates",
    "impact",
}


@dataclass
class KeywordExtractorConfig:
    min_len: int = 3
    stopwords: set[str] = field(default_factory=lambda: set(DEFAULT_STOPWORDS))
    fillers: set[str] = field(default_factory=lambda: set(DEFAULT_FILLERS))
    preserve_keywords: set[str] = field(default_factory=lambda: set(DEFAULT_PRESERVE_KEYWORDS))


class KeywordExtractor:
    """Configurable keyword extraction tuned for code-switched retrieval queries."""

    def __init__(self, config: KeywordExtractorConfig | None = None):
        self.config = config or KeywordExtractorConfig()

    def extract(self, text: str) -> List[str]:
        raw = text.lower()
        raw = raw.replace("which all", "which")
        raw = raw.replace("required documents", "required documents")
        toks = [t.lower() for t in _TOKEN_RE.findall(raw)]
        out: list[str] = []
        seen: set[str] = set()
        blocked = self.config.stopwords | self.config.fillers
        canonical = {
            "docs": "documents",
            "doc": "documents",
            "counseling": "counselling",
            "seat": "seats",
        }

        for tok in toks:
            tok = canonical.get(tok, tok)
            if tok in seen:
                continue
            if tok in self.config.preserve_keywords:
                out.append(tok)
                seen.add(tok)
                continue
            if tok in blocked:
                continue
            if len(tok) < self.config.min_len:
                continue
            out.append(tok)
            seen.add(tok)
        return out

    def extract_query(self, text: str) -> str:
        return " ".join(self.extract(text))


def simple_keywords(text: str, *, min_len: int = 3) -> List[str]:
    """Backward-compatible helper for simple keyword extraction."""
    config = KeywordExtractorConfig(min_len=min_len)
    return KeywordExtractor(config).extract(text)


def merge_keywords(*keyword_lists: Sequence[str]) -> List[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for words in keyword_lists:
        for w in words:
            lw = w.lower().strip()
            if not lw or lw in seen:
                continue
            merged.append(lw)
            seen.add(lw)
    return merged


if __name__ == "__main__":
    extractor = KeywordExtractor()
    samples = [
        "Inflation ka kya impact hoga on bond yields",
        "scholarship ke liye kya documents chahiye please",
        "cutoff aur merit list kab aayegi",
    ]
    for s in samples:
        print(f"IN : {s}")
        print(f"KW : {extractor.extract_query(s)}")
        print("-" * 40)

