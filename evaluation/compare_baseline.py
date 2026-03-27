from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from evaluation.metrics import hit_at_k, mean, mrr_at_k, recall_at_k
from rag.pipeline import RAGPipeline
from retrieval.index_builder import build_index
from utils import get_settings
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


def load_queries(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("sample_queries.json must be a list")
    return payload


def ensure_index_exists() -> None:
    s = get_settings()
    index_path = s.index_dir / "faiss.index"
    meta_path = s.index_dir / "metadata.json"
    if index_path.exists() and meta_path.exists():
        return
    logger.info("Index artifacts missing; building index now...")
    build_index(s)


def eval_one(p: RAGPipeline, query_text: str, expected_doc_ids: List[str], *, rewrite: bool, k: int) -> Dict[str, Any]:
    res = p.run(query_text, rewrite=rewrite)
    retrieved_doc_ids = [r["doc_id"] for r in res.retrieved]

    return {
        "pipeline_result": asdict(res),
        "retrieved_doc_ids": retrieved_doc_ids,
        "hit@k": hit_at_k(retrieved_doc_ids, expected_doc_ids, k=k),
        "recall@k": recall_at_k(retrieved_doc_ids, expected_doc_ids, k=k),
        "mrr@k": mrr_at_k(retrieved_doc_ids, expected_doc_ids, k=k),
    }


def _doc_id_list(result_payload: Dict[str, Any], top_k: int) -> List[str]:
    return [r.get("doc_id", "") for r in (result_payload.get("retrieved") or [])[:top_k]]


def _print_qualitative_item(
    idx: int,
    query_id: str,
    original_query: str,
    baseline_payload: Dict[str, Any],
    rewritten_payload: Dict[str, Any],
    top_docs_to_print: int,
) -> None:
    rw_details = rewritten_payload.get("rewrite_details") or {}
    rewritten_query = rw_details.get("cleaned_query") or rewritten_payload.get("query_used") or ""

    base_docs = _doc_id_list(baseline_payload, top_docs_to_print)
    rw_docs = _doc_id_list(rewritten_payload, top_docs_to_print)

    logger.info("[%d] Query ID: %s", idx, query_id)
    logger.info("  original : %s", original_query)
    logger.info("  rewritten: %s", rewritten_query)
    logger.info("  baseline top docs : %s", ", ".join(base_docs) if base_docs else "<none>")
    logger.info("  rewritten top docs: %s", ", ".join(rw_docs) if rw_docs else "<none>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs rewritten-query retrieval")
    parser.add_argument("--k", type=int, default=None, help="Cutoff for hit@k, recall@k, mrr@k")
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Path to queries JSON. Defaults to data/sample_queries.json",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional path to save detailed evaluation JSON",
    )
    parser.add_argument(
        "--top-docs",
        type=int,
        default=3,
        help="How many top docs to print in qualitative console comparison",
    )
    args = parser.parse_args()

    setup_logging()
    s = get_settings()
    ensure_index_exists()

    queries_path = Path(args.queries) if args.queries else (s.data_dir / "sample_queries.json")
    queries = load_queries(queries_path)
    p = RAGPipeline(s)

    k = int(args.k or s.top_k)
    base_hits: List[float] = []
    base_recalls: List[float] = []
    base_mrrs: List[float] = []
    rw_hits: List[float] = []
    rw_recalls: List[float] = []
    rw_mrrs: List[float] = []

    per_query_rows: list[dict[str, Any]] = []

    for q in queries:
        query_id = str(q.get("id") or "")
        qtext = str(q.get("asr_text") or q.get("query") or "").strip()
        if not qtext:
            continue
        expected = list(q.get("expected_doc_ids") or [])

        base = eval_one(p, qtext, expected, rewrite=False, k=k)
        rw = eval_one(p, qtext, expected, rewrite=True, k=k)

        base_hits.append(base["hit@k"])
        base_recalls.append(base["recall@k"])
        base_mrrs.append(base["mrr@k"])
        rw_hits.append(rw["hit@k"])
        rw_recalls.append(rw["recall@k"])
        rw_mrrs.append(rw["mrr@k"])

        _print_qualitative_item(
            idx=len(per_query_rows) + 1,
            query_id=query_id or f"q{len(per_query_rows)+1}",
            original_query=qtext,
            baseline_payload=base["pipeline_result"],
            rewritten_payload=rw["pipeline_result"],
            top_docs_to_print=max(1, int(args.top_docs)),
        )

        logger.info(
            "  metrics baseline[hit@%d=%.3f recall@%d=%.3f mrr@%d=%.3f] | rewritten[hit@%d=%.3f recall@%d=%.3f mrr@%d=%.3f]",
            k,
            base["hit@k"],
            k,
            base["recall@k"],
            k,
            base["mrr@k"],
            k,
            rw["hit@k"],
            k,
            rw["recall@k"],
            k,
            rw["mrr@k"],
        )

        per_query_rows.append(
            {
                "id": query_id,
                "query": qtext,
                "expected_doc_ids": expected,
                "baseline": {
                    "query_used": base["pipeline_result"].get("query_used"),
                    "top_doc_ids": base["retrieved_doc_ids"][:k],
                    "hit@k": base["hit@k"],
                    "recall@k": base["recall@k"],
                    "mrr@k": base["mrr@k"],
                },
                "rewritten": {
                    "query_used": rw["pipeline_result"].get("query_used"),
                    "rewrite_details": rw["pipeline_result"].get("rewrite_details"),
                    "top_doc_ids": rw["retrieved_doc_ids"][:k],
                    "hit@k": rw["hit@k"],
                    "recall@k": rw["recall@k"],
                    "mrr@k": rw["mrr@k"],
                },
            }
        )

    aggregate = {
        "n": len(per_query_rows),
        "k": k,
        "baseline": {
            "hit@k": mean(base_hits),
            "recall@k": mean(base_recalls),
            "mrr@k": mean(base_mrrs),
        },
        "rewritten": {
            "hit@k": mean(rw_hits),
            "recall@k": mean(rw_recalls),
            "mrr@k": mean(rw_mrrs),
        },
    }
    aggregate["delta"] = {
        "hit@k": aggregate["rewritten"]["hit@k"] - aggregate["baseline"]["hit@k"],
        "recall@k": aggregate["rewritten"]["recall@k"] - aggregate["baseline"]["recall@k"],
        "mrr@k": aggregate["rewritten"]["mrr@k"] - aggregate["baseline"]["mrr@k"],
    }

    logger.info("==== Aggregate Summary (n=%d, k=%d) ====", aggregate["n"], aggregate["k"])
    logger.info(
        "Baseline : hit@%d=%.3f | recall@%d=%.3f | mrr@%d=%.3f",
        k,
        aggregate["baseline"]["hit@k"],
        k,
        aggregate["baseline"]["recall@k"],
        k,
        aggregate["baseline"]["mrr@k"],
    )
    logger.info(
        "Rewritten: hit@%d=%.3f | recall@%d=%.3f | mrr@%d=%.3f",
        k,
        aggregate["rewritten"]["hit@k"],
        k,
        aggregate["rewritten"]["recall@k"],
        k,
        aggregate["rewritten"]["mrr@k"],
    )
    logger.info(
        "Delta    : hit@%d=%+.3f | recall@%d=%+.3f | mrr@%d=%+.3f",
        k,
        aggregate["delta"]["hit@k"],
        k,
        aggregate["delta"]["recall@k"],
        k,
        aggregate["delta"]["mrr@k"],
    )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "queries_path": str(queries_path.as_posix()),
            "aggregate": aggregate,
            "per_query": per_query_rows,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved detailed evaluation JSON to %s", out_path)


if __name__ == "__main__":
    main()

