from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_#+.-]+")


@dataclass
class TransliterationConfig:
    token_map: Dict[str, str] = field(
        default_factory=lambda: {
            "kehde": "which",
            "kaun": "which",
            "docs": "documents",
            "doc": "documents",
            "admission": "admission",
            "admision": "admission",
            "counselling": "counselling",
            "counseling": "counselling",
            "fees": "fees",
            "fee": "fees",
            "seats": "seats",
            "seat": "seats",
            "eligiblity": "eligibility",
            "scholership": "scholarship",
            "naal": "with",
            "ch": "in",
        }
    )
    phrase_map: Dict[str, str] = field(
        default_factory=lambda: {
            "kaun kaun": "which all",
            "kinne tak": "range",
            "scene aa": "details",
            "leke aane": "bring",
            "hunde ne": "",
            "upload karne": "upload",
            "bharne ke liye": "for",
            "de ki": "about",
        }
    )
    te_as_at_hints: set[str] = field(
        default_factory=lambda: {
            "time",
            "counselling",
            "admission",
            "deadline",
            "date",
            "portal",
            "course",
            "form",
        }
    )


class TransliterationNormalizer:
    """Normalizes common Romanized Hindi/Punjabi variants in code-switched ASR text."""

    def __init__(self, config: TransliterationConfig | None = None):
        self.config = config or TransliterationConfig()

    def normalize(self, text: str) -> str:
        cleaned = _WS_RE.sub(" ", text.strip().lower())
        if not cleaned:
            return ""

        for src, dst in self.config.phrase_map.items():
            cleaned = cleaned.replace(src, dst)

        tokens = _TOKEN_RE.findall(cleaned)
        normalized_tokens: list[str] = []
        for i, tok in enumerate(tokens):
            if tok == "te":
                left = tokens[i - 1] if i > 0 else ""
                right = tokens[i + 1] if i + 1 < len(tokens) else ""
                if left in self.config.te_as_at_hints or right in self.config.te_as_at_hints:
                    normalized_tokens.append("at")
                else:
                    normalized_tokens.append("and")
                continue
            mapped = self.config.token_map.get(tok, tok)
            if mapped:
                normalized_tokens.append(mapped)
        return " ".join(normalized_tokens)


DEFAULT_NORMALIZATION: Dict[str, str] = TransliterationConfig().token_map


def normalize_romanized(text: str, mapping: Dict[str, str] | None = None) -> str:
    """Backwards-compatible helper for token-level transliteration normalization."""
    if mapping is None:
        return TransliterationNormalizer().normalize(text)

    custom = TransliterationConfig(token_map=mapping)
    return TransliterationNormalizer(custom).normalize(text)


if __name__ == "__main__":
    samples = [
        "Counselling time te kehde documents naal leke aane hunde ne",
        "Fees te seats da ki scene aa is course ch",
        "Mera admission form bharne ke liye kaun kaun se docs upload karne padenge",
    ]
    normalizer = TransliterationNormalizer()
    for s in samples:
        print(f"IN : {s}")
        print(f"OUT: {normalizer.normalize(s)}")
        print("-" * 40)

