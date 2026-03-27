from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from retrieval.index_builder import build_index
from utils import Settings, get_settings
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
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


class DenseRetriever:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._model = None
        self._index = None
        self._chunks: List[Dict[str, Any]] = []

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover
                raise ImportError(
                    "sentence-transformers could not be imported. Check torch/sentence-transformers installation."
                ) from exc
            logger.info("Loading embedding model: %s", self.settings.embedding_model_name)
            self._model = SentenceTransformer(self.settings.embedding_model_name)
        return self._model

    def load(self) -> None:
        logger.info("Loading retriever index artifacts from %s", self.settings.index_dir)
        try:
            import faiss
        except Exception as e:  # pragma: no cover
            raise ImportError("faiss-cpu is required. Install requirements.txt.") from e

        index_path = self.settings.index_dir / "faiss.index"
        meta_path = self.settings.index_dir / "metadata.json"

        if not index_path.exists() or not meta_path.exists():
            logger.warning("Index missing at %s", self.settings.index_dir)
            print(f"[retriever] index missing at {self.settings.index_dir}")
            raise FileNotFoundError(
                f"Missing index artifacts in {self.settings.index_dir}. Run: python -m retrieval.index_builder"
            )

        self._index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._chunks = meta.get("chunks") or []
        if not self._chunks:
            raise RuntimeError("metadata.json has no chunks")

        logger.info("Loaded FAISS index with %d chunks", len(self._chunks))

    def ensure_loaded(self, *, build_if_missing: bool = True) -> None:
        try:
            self.load()
            return
        except FileNotFoundError:
            if not build_if_missing:
                raise
            logger.info("Building index artifacts via retrieval.index_builder...")
            print("[retriever] building index via retrieval.index_builder")
            build_index(self.settings)
            logger.info("Index build completed. Loading retriever...")
            print("[retriever] index build completed")
            self.load()

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        if not query or not query.strip():
            return []
        if self._index is None:
            self.ensure_loaded(build_if_missing=True)

        k = max(1, int(top_k or self.settings.top_k))
        k = min(k, max(1, len(self._chunks)))
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)

        scores, idxs = self._index.search(q_emb, k)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        out: List[RetrievedChunk] = []
        for rank, (s, i) in enumerate(zip(scores, idxs), start=1):
            if i < 0 or i >= len(self._chunks):
                continue
            c = self._chunks[i]
            out.append(
                RetrievedChunk(
                    rank=rank,
                    score=float(s),
                    chunk_id=c["chunk_id"],
                    doc_id=c["doc_id"],
                    doc_title=c["doc_title"],
                    source_path=c["source_path"],
                    text=c["text"],
                )
            )
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run top-k dense semantic search over FAISS index")
    parser.add_argument("--query", required=True, help="Query text for semantic search")
    parser.add_argument("--top-k", type=int, default=None, help="Number of results to return")
    parser.add_argument(
        "--build-if-missing",
        action="store_true",
        help="Build index automatically if artifacts are missing",
    )
    args = parser.parse_args()

    setup_logging()
    retriever = DenseRetriever()
    retriever.ensure_loaded(build_if_missing=args.build_if_missing)
    results = retriever.search(args.query, top_k=args.top_k)

    print("\n=== Dense Retrieval Results ===")
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
