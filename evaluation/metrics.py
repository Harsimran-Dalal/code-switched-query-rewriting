from __future__ import annotations

from typing import Iterable, Sequence, Set


def hit_at_k(retrieved_doc_ids: Sequence[str], expected_doc_ids: Iterable[str], k: int) -> float:
    """Binary hit metric: 1 if any relevant doc appears in top-k, else 0."""
    expected: Set[str] = set(expected_doc_ids)
    if not expected:
        return 0.0
    topk = set(retrieved_doc_ids[:k])
    return 1.0 if (topk & expected) else 0.0


def recall_at_k(retrieved_doc_ids: Sequence[str], expected_doc_ids: Iterable[str], k: int) -> float:
    """Set recall in top-k: |retrieved_relevant| / |relevant|."""
    expected: Set[str] = set(expected_doc_ids)
    if not expected:
        return 0.0
    topk = set(retrieved_doc_ids[:k])
    return float(len(topk & expected)) / float(len(expected))


def mrr_at_k(retrieved_doc_ids: Sequence[str], expected_doc_ids: Iterable[str], k: int) -> float:
    """Reciprocal rank of first relevant document in top-k."""
    expected: Set[str] = set(expected_doc_ids)
    if not expected:
        return 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


def mrr(retrieved_doc_ids: Sequence[str], expected_doc_ids: Iterable[str]) -> float:
    """MRR without truncation (equivalent to mrr_at_k with k=len(retrieved_doc_ids))."""
    return mrr_at_k(retrieved_doc_ids, expected_doc_ids, k=max(1, len(retrieved_doc_ids)))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))

