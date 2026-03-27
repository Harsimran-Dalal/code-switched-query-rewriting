from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from utils import Settings, get_settings
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class HybridRetrievedChunk:
	rank: int
	score: float
	bm25_score: float
	dense_score: float
	chunk_id: str
	doc_id: str
	doc_title: str
	source_path: str
	text: str

	def to_dict(self) -> Dict[str, Any]:
		return {
			"rank": self.rank,
			"score": self.score,
			"bm25_score": self.bm25_score,
			"dense_score": self.dense_score,
			"chunk_id": self.chunk_id,
			"doc_id": self.doc_id,
			"doc_title": self.doc_title,
			"source_path": self.source_path,
			"text": self.text,
		}


def _min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
	if not scores:
		return {}
	vals = list(scores.values())
	mn = min(vals)
	mx = max(vals)
	if abs(mx - mn) < 1e-9:
		return {k: 1.0 for k in scores}
	return {k: (v - mn) / (mx - mn) for k, v in scores.items()}


class HybridRetriever:
	"""Weighted hybrid retriever combining lexical BM25 and dense semantic scores."""

	def __init__(
		self,
		settings: Optional[Settings] = None,
		bm25_weight: float = 0.4,
		dense_weight: float = 0.6,
	):
		self.settings = settings or get_settings()
		self.bm25_weight = float(bm25_weight)
		self.dense_weight = float(dense_weight)

		total = self.bm25_weight + self.dense_weight
		if total <= 0:
			raise ValueError("bm25_weight + dense_weight must be > 0")
		self.bm25_weight /= total
		self.dense_weight /= total

		self.bm25 = BM25Retriever(self.settings)
		self.dense = DenseRetriever(self.settings)

	def ensure_loaded(self, *, build_if_missing: bool = False) -> None:
		self.bm25.ensure_loaded(build_if_missing=build_if_missing)
		self.dense.ensure_loaded(build_if_missing=build_if_missing)

	def search(
		self,
		query: str,
		top_k: Optional[int] = None,
		candidate_pool: Optional[int] = None,
	) -> List[HybridRetrievedChunk]:
		if not query or not query.strip():
			return []

		k = max(1, int(top_k or self.settings.top_k))
		pool = max(k, int(candidate_pool or (k * 3)))

		dense_hits = self.dense.search(query, top_k=pool)
		bm25_hits = self.bm25.search(query, top_k=pool)

		dense_scores = {h.chunk_id: h.score for h in dense_hits}
		bm25_scores = {h.chunk_id: h.score for h in bm25_hits}

		dense_norm = _min_max_normalize(dense_scores)
		bm25_norm = _min_max_normalize(bm25_scores)

		by_chunk: Dict[str, Dict[str, Any]] = {}
		for h in dense_hits:
			by_chunk[h.chunk_id] = {
				"chunk_id": h.chunk_id,
				"doc_id": h.doc_id,
				"doc_title": h.doc_title,
				"source_path": h.source_path,
				"text": h.text,
				"dense_score": h.score,
				"bm25_score": 0.0,
			}
		for h in bm25_hits:
			if h.chunk_id not in by_chunk:
				by_chunk[h.chunk_id] = {
					"chunk_id": h.chunk_id,
					"doc_id": h.doc_id,
					"doc_title": h.doc_title,
					"source_path": h.source_path,
					"text": h.text,
					"dense_score": 0.0,
					"bm25_score": h.score,
				}
			else:
				by_chunk[h.chunk_id]["bm25_score"] = h.score

		fused: list[tuple[float, Dict[str, Any]]] = []
		for cid, item in by_chunk.items():
			fused_score = (
				self.bm25_weight * bm25_norm.get(cid, 0.0)
				+ self.dense_weight * dense_norm.get(cid, 0.0)
			)
			fused.append((fused_score, item))

		fused.sort(key=lambda x: x[0], reverse=True)
		fused = fused[:k]

		out: list[HybridRetrievedChunk] = []
		for rank, (score, item) in enumerate(fused, start=1):
			out.append(
				HybridRetrievedChunk(
					rank=rank,
					score=float(score),
					bm25_score=float(item["bm25_score"]),
					dense_score=float(item["dense_score"]),
					chunk_id=item["chunk_id"],
					doc_id=item["doc_id"],
					doc_title=item["doc_title"],
					source_path=item["source_path"],
					text=item["text"],
				)
			)
		return out


def main() -> None:
	parser = argparse.ArgumentParser(description="Run weighted hybrid retrieval (BM25 + dense)")
	parser.add_argument("--query", required=True, help="Query text for hybrid retrieval")
	parser.add_argument("--top-k", type=int, default=None, help="Number of final results")
	parser.add_argument("--pool", type=int, default=None, help="Candidate pool before fusion")
	parser.add_argument("--bm25-weight", type=float, default=0.4, help="Lexical score weight")
	parser.add_argument("--dense-weight", type=float, default=0.6, help="Semantic score weight")
	parser.add_argument(
		"--build-if-missing",
		action="store_true",
		help="Build index artifacts automatically if missing",
	)
	args = parser.parse_args()

	setup_logging()
	retriever = HybridRetriever(
		bm25_weight=args.bm25_weight,
		dense_weight=args.dense_weight,
	)
	retriever.ensure_loaded(build_if_missing=args.build_if_missing)
	results = retriever.search(args.query, top_k=args.top_k, candidate_pool=args.pool)

	print("\n=== Hybrid Retrieval Results ===")
	print(f"Query: {args.query}")
	print(f"Weights: bm25={retriever.bm25_weight:.2f}, dense={retriever.dense_weight:.2f}")
	print(f"Total results: {len(results)}\n")
	for r in results:
		preview = (r.text[:180] + "...") if len(r.text) > 180 else r.text
		print(
			f"[{r.rank}] hybrid={r.score:.4f} | bm25={r.bm25_score:.4f} | dense={r.dense_score:.4f} | doc={r.doc_id}"
		)
		print(f"    chunk: {r.chunk_id}")
		print(f"    title: {r.doc_title}")
		print(f"    src  : {r.source_path}")
		print(f"    text : {preview}")
		print()


if __name__ == "__main__":
	main()

