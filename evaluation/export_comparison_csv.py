from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rag.pipeline import RAGPipeline
from retrieval.index_builder import build_index
from utils import get_settings
from utils.logger import setup_logging


def load_queries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Queries JSON must be a list of objects.")
    return payload


def ensure_index_exists() -> None:
    settings = get_settings()
    index_path = settings.index_dir / "faiss.index"
    meta_path = settings.index_dir / "metadata.json"
    if index_path.exists() and meta_path.exists():
        return
    build_index(settings)


def _extract_query_text(row: dict[str, Any]) -> str:
    return str(row.get("asr_text") or row.get("query") or "").strip()


def _top1_info(result) -> tuple[str, float]:
    retrieved = result.retrieved or []
    if not retrieved:
        return "", 0.0
    first = retrieved[0]
    return str(first.get("doc_id", "")), float(first.get("score", 0.0) or 0.0)


def _rewrite_fields(result) -> tuple[str, str, str]:
    details = result.rewrite_details or {}
    cleaned = str(details.get("cleaned_query") or "")
    keyword = str(details.get("keyword_query") or "")
    rewritten = str(details.get("rewritten_query") or result.query_used or "")
    return cleaned, keyword, rewritten


def run_export(queries_path: Path, out_csv: Path) -> int:
    settings = get_settings()
    pipeline = RAGPipeline(settings)
    queries = load_queries(queries_path)

    rows: list[dict[str, Any]] = []
    for item in queries:
        original_query = _extract_query_text(item)
        if not original_query:
            continue

        baseline = pipeline.run(original_query, rewrite=False)
        rewritten = pipeline.run(original_query, rewrite=True)

        cleaned_query, keyword_query, rewritten_query = _rewrite_fields(rewritten)
        base_doc, base_score = _top1_info(baseline)
        rw_doc, rw_score = _top1_info(rewritten)

        rows.append(
            {
                "original_query": original_query,
                "cleaned_query": cleaned_query,
                "keyword_query": keyword_query,
                "rewritten_query": rewritten_query,
                "baseline_top1_doc": base_doc,
                "baseline_top1_score": f"{base_score:.6f}",
                "rewritten_top1_doc": rw_doc,
                "rewritten_top1_score": f"{rw_score:.6f}",
                "score_improvement": f"{(rw_score - base_score):+.6f}",
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "original_query",
        "cleaned_query",
        "keyword_query",
        "rewritten_query",
        "baseline_top1_doc",
        "baseline_top1_score",
        "rewritten_top1_doc",
        "rewritten_top1_score",
        "score_improvement",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export baseline vs rewritten retrieval comparison to CSV"
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Path to query list JSON (defaults to data/sample_queries.json)",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="evaluation/comparison_results.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    setup_logging()
    ensure_index_exists()

    settings = get_settings()
    queries_path = Path(args.queries) if args.queries else (settings.data_dir / "sample_queries.json")
    out_csv = Path(args.out_csv)

    n = run_export(queries_path, out_csv)
    print(f"Saved {n} rows to {out_csv.as_posix()}")


if __name__ == "__main__":
    main()
