from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from retrieval.index_builder import build_index
from utils import Settings, get_settings
from utils.logger import setup_logging

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_#+.-]*")


def _tokenize(text: str) -> List[str]:
	return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class BM25RetrievedChunk:
	rank: int
	score: float
	chunk_id: str
	doc_id: str
	doc_title: str
	source_path: str
	text: str

	def to_dict(self) -> Dict[str, Any]:
		return {
			"rank": self.rank,
			"score": self.score,
			"chunk_id": self.chunk_id,
			"doc_id": self.doc_id,
			"doc_title": self.doc_title,
			"source_path": self.source_path,
			"text": self.text,
		}


class BM25Retriever:
	"""Lexical retriever baseline using rank-bm25 over indexed chunks."""

	def __init__(self, settings: Optional[Settings] = None):
		self.settings = settings or get_settings()
		self._chunks: List[Dict[str, Any]] = []
		self._bm25: Optional[BM25Okapi] = None

	def load(self) -> None:
		meta_path = self.settings.index_dir / "metadata.json"
		if not meta_path.exists():
			raise FileNotFoundError(
				f"Missing metadata artifact in {self.settings.index_dir}. Run: python -m retrieval.index_builder"
			)

		payload = json.loads(meta_path.read_text(encoding="utf-8"))
		chunks = payload.get("chunks") or []
		if not chunks:
			raise RuntimeError("metadata.json has no chunks")

		tokenized_corpus: list[list[str]] = []
		for chunk in chunks:
			toks = _tokenize(str(chunk.get("text", "")))
			tokenized_corpus.append(toks if toks else [""])

		self._chunks = chunks
		self._bm25 = BM25Okapi(tokenized_corpus)
		logger.info("Loaded BM25 corpus with %d chunks", len(self._chunks))

	def ensure_loaded(self, *, build_if_missing: bool = False) -> None:
		try:
			self.load()
			return
		except FileNotFoundError:
			if not build_if_missing:
				raise
			logger.info("Metadata missing. Building index artifacts now...")
			build_index(self.settings)
			self.load()

	def search(self, query: str, top_k: Optional[int] = None) -> List[BM25RetrievedChunk]:
		if not query or not query.strip():
			return []
		if self._bm25 is None:
			self.ensure_loaded(build_if_missing=False)

		assert self._bm25 is not None
		tokens = _tokenize(query)
		if not tokens:
			return []

		k = max(1, int(top_k or self.settings.top_k))
		scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float32)
		if scores.size == 0:
			return []

		k = min(k, scores.size)
		top_indices = np.argpartition(-scores, kth=k - 1)[:k]
		ranked_indices = top_indices[np.argsort(-scores[top_indices])]

		out: list[BM25RetrievedChunk] = []
		for rank, idx in enumerate(ranked_indices.tolist(), start=1):
			chunk = self._chunks[idx]
			out.append(
				BM25RetrievedChunk(
					rank=rank,
					score=float(scores[idx]),
					chunk_id=chunk["chunk_id"],
					doc_id=chunk["doc_id"],
					doc_title=chunk["doc_title"],
					source_path=chunk["source_path"],
					text=chunk["text"],
				)
			)
		return out


def main() -> None:
	parser = argparse.ArgumentParser(description="Run top-k BM25 lexical search over indexed chunks")
	parser.add_argument("--query", required=True, help="Query text for lexical retrieval")
	parser.add_argument("--top-k", type=int, default=None, help="Number of results to return")
	parser.add_argument(
		"--build-if-missing",
		action="store_true",
		help="Build index artifacts automatically if missing",
	)
	args = parser.parse_args()

	setup_logging()
	retriever = BM25Retriever()
	retriever.ensure_loaded(build_if_missing=args.build_if_missing)
	results = retriever.search(args.query, top_k=args.top_k)

	print("\n=== BM25 Retrieval Results ===")
	print(f"Query: {args.query}")
	print(f"Total results: {len(results)}\n")
	for r in results:
		preview = (r.text[:180] + "...") if len(r.text) > 180 else r.text
		print(f"[{r.rank}] score={r.score:.4f} | doc={r.doc_id} | chunk={r.chunk_id}")
		print(f"    title: {r.doc_title}")
		print(f"    src  : {r.source_path}")
		print(f"    text : {preview}")
		print()


if __name__ == "__main__":
	main()

