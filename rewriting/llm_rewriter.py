from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils import Settings, get_settings


@dataclass
class LLMRewriteResult:
    rewritten_query: str
    model: str


class LLMRewriter:
    """
    Optional future module for LLM-based rewriting.

    TODO: Integrate an API/local model when you want a stronger rewriter baseline.
    This file exists to keep the architecture modular without forcing API keys.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def rewrite(self, asr_text: str) -> LLMRewriteResult:
        raise NotImplementedError(
            "TODO: LLM-based rewriting not implemented yet. Use RuleBasedRewriter for the working baseline."
        )

