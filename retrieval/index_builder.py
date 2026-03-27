from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from utils import Settings, get_settings
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    doc_id: str
    title: str
    source_path: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    source_path: str
    text: str


_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_#+.-]*")


def tokenize_for_stats(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def iter_document_files(documents_dir: Path) -> Iterable[Path]:
    for p in sorted(documents_dir.glob("*.txt")):
        yield p


def read_document(path: Path) -> Tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    title = path.stem.replace("_", " ").strip()
    return title, text


def load_documents(documents_dir: Path) -> List[Document]:
    docs: list[Document] = []
    for path in iter_document_files(documents_dir):
        title, text = read_document(path)
        if not text:
            logger.warning("Skipping empty document: %s", path)
            continue
        docs.append(
            Document(
                doc_id=path.stem,
                title=title,
                source_path=str(path.as_posix()),
                text=text,
            )
        )
    return docs


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def chunk_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        parts = chunk_text(doc.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, part in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::chunk{i:03d}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    source_path=doc.source_path,
                    text=part,
                )
            )
    return chunks


def build_term_stats(chunks: List[Chunk]) -> Dict[str, Dict[str, float]]:
    """
    Build simple corpus term statistics for retrieval-aware rewriting:
    - df: in how many chunks the term appears
    - idf: log((N + 1)/(df + 1)) + 1
    """
    N = max(1, len(chunks))
    df: Dict[str, int] = {}
    for c in chunks:
        seen = set(tokenize_for_stats(c.text))
        for t in seen:
            df[t] = df.get(t, 0) + 1

    stats: Dict[str, Dict[str, float]] = {}
    for t, dfi in df.items():
        idf = math.log((N + 1) / (dfi + 1)) + 1.0
        stats[t] = {"df": float(dfi), "idf": float(idf)}
    return stats


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_faiss_index(embeddings: np.ndarray):
    try:
        import faiss
    except Exception as e:  # pragma: no cover
        raise ImportError("faiss-cpu is required. Install requirements.txt.") from e

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine if vectors are normalized
    index.add(embeddings.astype(np.float32))
    return index


def embed_chunks(
    chunks: List[Chunk],
    model_name: str,
    batch_size: int = 64,
    show_progress_bar: bool = True,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "sentence-transformers could not be imported. Check torch/sentence-transformers installation."
        ) from exc

    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    texts = [c.text for c in chunks]
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )
    return np.asarray(vectors, dtype=np.float32)


def save_artifacts(settings: Settings, index, chunks: List[Chunk], term_stats: Dict[str, Dict[str, float]]) -> None:
    try:
        import faiss
    except Exception as exc:  # pragma: no cover
        raise ImportError("faiss-cpu is required. Install requirements.txt.") from exc

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    index_path = settings.index_dir / "faiss.index"
    meta_path = settings.index_dir / "metadata.json"
    term_path = settings.index_dir / "term_stats.json"

    faiss.write_index(index, str(index_path))
    save_json(meta_path, {"chunks": [asdict(c) for c in chunks]})
    save_json(term_path, {"num_chunks": len(chunks), "terms": term_stats})

    logger.info("Saved index to %s", index_path)
    logger.info("Saved metadata to %s", meta_path)
    logger.info("Saved term stats to %s", term_path)


def build_index(settings: Optional[Settings] = None) -> int:
    return build_index_with_options(settings=settings, show_progress=True)


def build_index_with_options(settings: Optional[Settings] = None, *, show_progress: bool = True) -> int:
    settings = settings or get_settings()

    docs = load_documents(settings.documents_dir)
    if not docs:
        raise RuntimeError(f"No usable .txt documents found in {settings.documents_dir}")

    chunks = chunk_documents(docs, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    if not chunks:
        raise RuntimeError("No chunks were produced from documents.")

    logger.info("Loaded %d documents and produced %d chunks", len(docs), len(chunks))
    embeddings = embed_chunks(
        chunks,
        settings.embedding_model_name,
        show_progress_bar=show_progress,
    )

    logger.info("Building FAISS index")
    index = build_faiss_index(embeddings)
    term_stats = build_term_stats(chunks)
    save_artifacts(settings, index, chunks, term_stats)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dense FAISS index from data/documents/*.txt")
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show embedding progress bar.",
    )
    args = parser.parse_args()

    settings: Settings = get_settings()
    setup_logging()

    num_chunks = build_index_with_options(settings, show_progress=args.show_progress)
    logger.info("Index build completed successfully. Total chunks: %d", num_chunks)


if __name__ == "__main__":
    main()
