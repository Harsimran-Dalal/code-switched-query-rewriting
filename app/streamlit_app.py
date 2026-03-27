from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rag.pipeline import RAGPipeline

from rewriting.rule_based import RuleBasedRewriter
try:
    from speech.asr_pipeline import ASRTranscriber
except Exception:
    ASRTranscriber = None
from utils import get_settings
from utils.logger import setup_logging


TMP_AUDIO_DIR = Path(".streamlit_tmp")
EVAL_CSV_PATH = Path("evaluation/comparison_results.csv")
ARCHITECTURE_IMAGE_PATH = Path("docs/architecture.png")
ARCHITECTURE_DOC_PATH = Path("docs/architecture.md")
MAX_HISTORY_ITEMS = 10

SAMPLE_QUERY_MAP = {
    "Admission Process": "Admission process ka pura step kya hota hai from registration to counselling?",
    "Scholarship Eligibility": "Agar income certificate nahi hai to scholarship milegi ya nahi?",
    "Fees and Seats": "Private college me fees zyada hoti hai ya govt me?",
    "Cutoff and Merit": "Merit list kab nikalti hai aur cutoff kaise decide hota hai?",
    "Required Documents": "Form bharte time kaunse documents scan karke rakhne chahiye?",
}

SAMPLE_QUERY_MAP = {
    "Admission Process": "Admission process ka pura step kya hota hai from registration to counselling?",
    "Scholarship Eligibility": "Agar income certificate nahi hai to scholarship milegi ya nahi?",
    "Fees and Seats": "Private college me fees zyada hoti hai ya govt me?",
    "Cutoff and Merit": "Merit list kab nikalti hai aur cutoff kaise decide hota hai?",
    "Required Documents": "Form bharte time kaunse documents scan karke rakhne chahiye?",
}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(1200px 500px at 10% -10%, #1f2937 0%, #0f172a 45%, #060b16 100%);
            color: #e5e7eb;
        }
        .main .block-container {
            max-width: 1200px;
            padding-top: 0.8rem;
            padding-bottom: 1.2rem;
        }
        .hero-card {
            background: linear-gradient(120deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 14px;
            padding: 0.95rem 1.05rem;
            margin-bottom: 0.55rem;
            box-shadow: 0 10px 24px rgba(2,6,23,0.45);
            animation: fadeInUp 260ms ease;
        }
        .hero-subtitle {
            color: #93c5fd;
            font-size: 0.92rem;
            margin-top: 0.2rem;
        }
        .section-divider {
            margin: 0.45rem 0 0.65rem 0;
            border: 0;
            border-top: 1px solid rgba(148,163,184,0.22);
        }
        .section-card {
            background: rgba(15,23,42,0.7);
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 12px;
            padding: 0.78rem 0.9rem;
            margin: 0.25rem 0 0.55rem 0;
            transition: transform 160ms ease, box-shadow 160ms ease;
            animation: fadeInUp 240ms ease;
        }
        .section-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 20px rgba(15,23,42,0.35);
        }
        .card-title {
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 0.28rem;
            font-weight: 600;
        }
        .tight-help {
            margin-top: -0.2rem;
            margin-bottom: 0.35rem;
            color: #94a3b8;
            font-size: 0.84rem;
        }
        .chip {
            display: inline-block;
            padding: 0.18rem 0.6rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .chip-baseline { background: rgba(59,130,246,0.18); color: #93c5fd; border: 1px solid rgba(147,197,253,0.35); }
        .chip-rewritten { background: rgba(16,185,129,0.18); color: #6ee7b7; border: 1px solid rgba(110,231,183,0.35); }
        .chip-improved { background: rgba(16,185,129,0.18); color: #6ee7b7; border: 1px solid rgba(110,231,183,0.35); }
        .chip-notimproved { background: rgba(239,68,68,0.18); color: #fca5a5; border: 1px solid rgba(252,165,165,0.35); }
        .rewrite-stage {
            border-radius: 12px;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.3rem;
            border: 1px solid rgba(148,163,184,0.24);
            background: rgba(15,23,42,0.72);
            animation: fadeInUp 280ms ease;
        }
        .stage-original { border-color: rgba(148,163,184,0.35); }
        .stage-cleaned { border-color: rgba(96,165,250,0.45); background: rgba(30,64,175,0.14); }
        .stage-keyword { border-color: rgba(139,92,246,0.42); background: rgba(76,29,149,0.14); }
        .stage-rewritten { border-color: rgba(16,185,129,0.55); background: rgba(5,150,105,0.14); box-shadow: 0 0 0 1px rgba(110,231,183,0.2) inset; }
        .flow-arrow {
            text-align: center;
            color: #60a5fa;
            font-size: 1.18rem;
            margin: 0.05rem 0 0.2rem 0;
        }
        .top-doc {
            background: linear-gradient(120deg, rgba(30,58,138,0.26), rgba(15,23,42,0.75));
            border: 1px solid rgba(96,165,250,0.45);
            border-radius: 12px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.45rem;
            animation: fadeInUp 260ms ease;
        }
        .score-badge {
            display: inline-block;
            padding: 0.16rem 0.55rem;
            border-radius: 999px;
            font-size: 0.73rem;
            font-weight: 700;
            border: 1px solid rgba(148,163,184,0.32);
            color: #e2e8f0;
            background: rgba(51,65,85,0.55);
            margin-left: 0.45rem;
            box-shadow: 0 0 12px rgba(59,130,246,0.18);
            transition: box-shadow 180ms ease;
        }
        .score-badge:hover {
            box-shadow: 0 0 16px rgba(59,130,246,0.32);
        }
        .compact-doc-note {
            color: #93c5fd;
            font-size: 0.8rem;
            margin: 0.1rem 0 0.3rem 0;
        }
        .insight-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.55rem;
            margin: 0.35rem 0 0.3rem 0;
        }
        .insight-card {
            border-radius: 12px;
            border: 1px solid rgba(148,163,184,0.25);
            background: rgba(15,23,42,0.72);
            padding: 0.7rem 0.8rem;
            transition: transform 150ms ease, box-shadow 150ms ease;
        }
        .insight-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 16px rgba(15,23,42,0.32);
        }
        .insight-label {
            color: #94a3b8;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .insight-value {
            color: #e2e8f0;
            font-size: 1rem;
            font-weight: 700;
            margin-top: 0.12rem;
        }
        .insight-positive { color: #6ee7b7; }
        .insight-negative { color: #fca5a5; }
        .architecture-card {
            border-radius: 12px;
            border: 1px solid rgba(148,163,184,0.24);
            background: rgba(15,23,42,0.7);
            padding: 0.7rem 0.8rem;
            margin: 0.2rem 0 0.45rem 0;
        }
        .architecture-flow {
            border-radius: 10px;
            border: 1px dashed rgba(148,163,184,0.34);
            background: rgba(2,6,23,0.45);
            padding: 0.75rem 0.85rem;
            color: #cbd5e1;
            line-height: 1.65;
            white-space: pre-wrap;
        }
        .sample-note {
            color: #93c5fd;
            font-size: 0.8rem;
            margin-top: 0.1rem;
        }
        .history-card {
            border-radius: 10px;
            border: 1px solid rgba(148,163,184,0.22);
            background: rgba(15,23,42,0.68);
            padding: 0.52rem 0.58rem;
            margin: 0.35rem 0;
            transition: transform 130ms ease, box-shadow 130ms ease;
        }
        .history-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 7px 16px rgba(2,6,23,0.35);
        }
        .history-query {
            color: #e2e8f0;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.2rem;
        }
        .history-preview {
            color: #93c5fd;
            font-size: 0.76rem;
            margin-bottom: 0.24rem;
        }
        .history-meta {
            color: #94a3b8;
            font-size: 0.72rem;
        }
        .status-pill {
            display: inline-block;
            padding: 0.12rem 0.48rem;
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 700;
            margin-left: 0.32rem;
            border: 1px solid rgba(148,163,184,0.3);
        }
        .status-improved { color: #6ee7b7; border-color: rgba(110,231,183,0.45); background: rgba(16,185,129,0.18); }
        .status-same { color: #cbd5e1; border-color: rgba(203,213,225,0.35); background: rgba(71,85,105,0.25); }
        .status-lower { color: #fca5a5; border-color: rgba(252,165,165,0.45); background: rgba(239,68,68,0.18); }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .subtle {
            color: #a5b4fc;
            font-size: 0.85rem;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15,23,42,0.97), rgba(2,6,23,0.98));
            border-right: 1px solid rgba(148,163,184,0.2);
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 0.8rem;
            padding-bottom: 0.8rem;
        }
        button[kind="primary"] {
            border-radius: 10px !important;
            border: 1px solid rgba(59,130,246,0.45) !important;
            box-shadow: 0 4px 14px rgba(59,130,246,0.2) !important;
            transition: transform 140ms ease, box-shadow 140ms ease !important;
        }
        button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(59,130,246,0.3) !important;
        }
        div[data-testid="stExpander"] {
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 10px;
            background: rgba(15,23,42,0.52);
        }
        .stTextArea textarea {
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _chip(label: str, style_class: str) -> str:
    return f"<span class='chip {style_class}'>{label}</span>"


def _score_badge(score: float) -> str:
    return f"<span class='score-badge'>{score:.3f}</span>"


def _insight_card(label: str, value: str, positive: Optional[bool] = None) -> str:
    value_class = "insight-value"
    if positive is True:
        value_class += " insight-positive"
    elif positive is False:
        value_class += " insight-negative"
    return (
        "<div class='insight-card'>"
        f"<div class='insight-label'>{label}</div>"
        f"<div class='{value_class}'>{value}</div>"
        "</div>"
    )


def _save_uploaded_audio(uploaded_file) -> Path:
    TMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TMP_AUDIO_DIR / uploaded_file.name
    out_path.write_bytes(uploaded_file.getbuffer())
    return out_path


def _load_evaluation_snapshot(path: Path) -> Optional[dict[str, float]]:
    if not path.exists():
        return None

    baseline_scores: list[float] = []
    rewritten_scores: list[float] = []
    improvements: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                baseline_scores.append(float(row.get("baseline_top1_score", "0") or 0.0))
                rewritten_scores.append(float(row.get("rewritten_top1_score", "0") or 0.0))
                improvements.append(float(row.get("score_improvement", "0") or 0.0))
            except ValueError:
                continue

    if not baseline_scores:
        return None

    n = float(len(baseline_scores))
    return {
        "n": n,
        "avg_baseline": sum(baseline_scores) / n,
        "avg_rewritten": sum(rewritten_scores) / n,
        "avg_improvement": sum(improvements) / n,
    }


def _render_evaluation_snapshot() -> None:
    snapshot = _load_evaluation_snapshot(EVAL_CSV_PATH)
    st.markdown("### Evaluation Snapshot")
    if snapshot is None:
        st.info("No evaluation snapshot found yet. Run evaluation export to populate this section.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Test Queries", f"{int(snapshot['n'])}")
    with c2:
        st.metric("Avg Baseline Score", f"{snapshot['avg_baseline']:.3f}")
    with c3:
        st.metric("Avg Rewritten Score", f"{snapshot['avg_rewritten']:.3f}")
    with c4:
        st.metric("Avg Improvement", f"{snapshot['avg_improvement']:+.3f}")


def _render_architecture_support() -> None:
    st.markdown("### Project Architecture")
    if ARCHITECTURE_IMAGE_PATH.exists():
        st.markdown("<div class='architecture-card'>", unsafe_allow_html=True)
        st.image(
            str(ARCHITECTURE_IMAGE_PATH.as_posix()),
            caption="System pipeline for retrieval-aware query rewriting",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    fallback_text = (
        "User Query (Speech/Text)\n"
        "    -> ASR (optional)\n"
        "    -> Query Rewriting Module\n"
        "    -> Dense Retrieval (FAISS)\n"
        "    -> Top-K Documents\n"
        "    -> Answer Generation\n"
        "    -> Baseline vs Rewritten Comparison"
    )
    st.markdown("<div class='architecture-card'><div class='card-title'>Architecture Fallback</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='architecture-flow'>{fallback_text}</div>", unsafe_allow_html=True)
    if ARCHITECTURE_DOC_PATH.exists():
        with st.expander("Architecture Notes", expanded=False):
            st.markdown(ARCHITECTURE_DOC_PATH.read_text(encoding="utf-8"))


def _render_retrieved_docs(title: str, docs: list[dict], max_docs: int) -> None:
    st.markdown(f"**{title}**")
    if not docs:
        st.info("No retrieved documents.")
        return

    top_doc = docs[0]
    top_title = top_doc.get("doc_title", "unknown")
    top_score = float(top_doc.get("score", 0.0) or 0.0)
    top_preview = (top_doc.get("text", "") or "")[:380]
    st.markdown(
        (
            "<div class='top-doc'>"
            "<div class='card-title'>Top Document</div>"
            f"<strong>{top_title}</strong>{_score_badge(top_score)}"
            f"<div style='margin-top:0.45rem'>{top_preview}...</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown("<div class='compact-doc-note'>Additional retrieved documents (compact view)</div>", unsafe_allow_html=True)

    for r in docs[1:max_docs]:
        rank = int(r.get("rank", 0) or 0)
        score = float(r.get("score", 0.0) or 0.0)
        doc_title = r.get("doc_title", "unknown")
        chunk_id = r.get("chunk_id", "n/a")
        with st.expander(f"{rank}. {doc_title} | score={score:.3f}", expanded=False):
            st.write(r.get("text", ""))
            st.caption(f"doc_id={r.get('doc_id', '')} | source={r.get('source_path', '')}")


def _render_result_block(label: str, result) -> None:
    chip_class = "chip-baseline" if "Baseline" in label else "chip-rewritten"
    st.markdown(f"### {label}")
    st.markdown(_chip(label, chip_class), unsafe_allow_html=True)
    st.markdown("**Query Used For Retrieval**")
    st.code(result.query_used)

    top_doc = (result.retrieved or [{}])[0] if result.retrieved else {}
    top_score = float(top_doc.get("score", 0.0) or 0.0)
    st.markdown(f"<span class='subtle'>Top-1 Score</span>{_score_badge(top_score)}", unsafe_allow_html=True)

    if result.rewrite_details:
        st.markdown("**Rewrite Details**")
        st.json(result.rewrite_details)

    st.markdown("**Short Summary**")
    st.write(result.summary)
    st.markdown("**Final Answer**")
    st.write(result.answer)
    st.caption(f"Generation mode: {result.generation_mode}")
    if result.citations:
        st.caption("Citations: " + ", ".join(result.citations))


def _top1(result) -> tuple[str, float]:
    retrieved = result.retrieved or []
    if not retrieved:
        return "", 0.0
    first = retrieved[0]
    return str(first.get("doc_id", "")), float(first.get("score", 0.0) or 0.0)


def _render_comparison_insights(baseline, rewritten) -> None:
    base_doc, base_score = _top1(baseline)
    rw_doc, rw_score = _top1(rewritten)

    score_improvement = rw_score - base_score
    top1_doc_same = bool(base_doc and rw_doc and base_doc == rw_doc)
    confidence_improved = rw_score > base_score
    improvement_chip = _chip("Improved", "chip-improved") if confidence_improved else _chip("Not Improved", "chip-notimproved")

    st.subheader("Comparison Insights")
    st.markdown(improvement_chip, unsafe_allow_html=True)
    st.markdown(
        "<div class='insight-grid'>"
        + _insight_card("Top-1 Score Improvement", f"{score_improvement:+.3f}", positive=(score_improvement >= 0))
        + _insight_card("Top-1 Document Same", "Yes" if top1_doc_same else "No", positive=top1_doc_same)
        + _insight_card("Confidence Increased", "Yes" if confidence_improved else "No", positive=confidence_improved)
        + _insight_card("Baseline Score", f"{base_score:.3f}")
        + _insight_card("Rewritten Score", f"{rw_score:.3f}")
        + _insight_card("Absolute Score Gain", f"{abs(score_improvement):.3f}")
        + "</div>",
        unsafe_allow_html=True,
    )

    if base_doc or rw_doc:
        st.caption(f"Top-1 baseline: {base_doc or 'n/a'} | Top-1 rewritten: {rw_doc or 'n/a'}")
        if not top1_doc_same:
            st.caption("Top-1 document changed after rewriting.")


def _render_sample_query_picker() -> None:
    st.markdown("<div class='card-title'>Quick Sample Queries</div>", unsafe_allow_html=True)
    labels = list(SAMPLE_QUERY_MAP.keys())
    cols = st.columns(len(labels))
    for col, label in zip(cols, labels):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state["asr_text"] = SAMPLE_QUERY_MAP[label]
                st.rerun()
    st.markdown(
        "<div class='sample-note'>Click any sample to populate the input box instantly.</div>",
        unsafe_allow_html=True,
    )


def _render_rewrite_pipeline(query_text: str, rewrite_preview) -> None:
    st.markdown(
        (
            "<div class='rewrite-stage stage-original'>"
            "<div class='card-title'>Original Query</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.code(query_text)
    st.markdown("<div class='flow-arrow'>↓</div>", unsafe_allow_html=True)

    st.markdown(
        (
            "<div class='rewrite-stage stage-cleaned'>"
            "<div class='card-title'>Cleaned Query</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.code(rewrite_preview.cleaned_query)
    st.markdown("<div class='flow-arrow'>↓</div>", unsafe_allow_html=True)

    st.markdown(
        (
            "<div class='rewrite-stage stage-keyword'>"
            "<div class='card-title'>Keyword Query</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.code(rewrite_preview.keyword_query)
    st.markdown("<div class='flow-arrow'>↓</div>", unsafe_allow_html=True)

    st.markdown(
        (
            "<div class='rewrite-stage stage-rewritten'>"
            "<div class='card-title'>Rewritten Query (Optimized)</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.code(rewrite_preview.rewritten_query)


@st.cache_resource
def _get_cached_runtime_components():
    settings = get_settings()
    pipeline = RAGPipeline(settings)
    rewriter = RuleBasedRewriter(settings)
    transcriber = ASRTranscriber(settings) if ASRTranscriber is not None else None
    return pipeline, rewriter, transcriber


def _init_session_state() -> None:
    if "asr_text" not in st.session_state:
        st.session_state["asr_text"] = ""
    if "history" not in st.session_state:
        st.session_state["history"] = []


def _truncate(text: str, max_len: int = 72) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3].rstrip() + "..."


def _status_for_gain(gain: float) -> str:
    if gain > 1e-6:
        return "Improved"
    if gain < -1e-6:
        return "Lower"
    return "Same"


def _status_class(status: str) -> str:
    s = status.lower()
    if s == "improved":
        return "status-improved"
    if s == "lower":
        return "status-lower"
    return "status-same"


def _append_history_item(comparison) -> None:
    baseline = comparison.baseline
    rewritten = comparison.rewritten
    rw_details = rewritten.rewrite_details or {}

    _, baseline_score = _top1(baseline)
    _, rewritten_score = _top1(rewritten)
    improvement = rewritten_score - baseline_score
    status = _status_for_gain(improvement)

    item = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_query": baseline.query_original,
        "cleaned_query": str(rw_details.get("cleaned_query") or ""),
        "keyword_query": str(rw_details.get("keyword_query") or ""),
        "rewritten_query": str(rw_details.get("rewritten_query") or rewritten.query_used or ""),
        "baseline_score": baseline_score,
        "rewritten_score": rewritten_score,
        "improvement": improvement,
        "status": status,
    }

    history: list[dict[str, Any]] = st.session_state.get("history", [])
    history.append(item)
    if len(history) > MAX_HISTORY_ITEMS:
        history = history[-MAX_HISTORY_ITEMS:]
    st.session_state["history"] = history


def _render_session_history_sidebar() -> None:
    st.sidebar.markdown("## 🕘 Session History")
    st.sidebar.write(len(st.session_state.get("history", [])))

    history: list[dict[str, Any]] = st.session_state.get("history", [])
    if not history:
        st.sidebar.caption("No history yet")
        return

    if st.sidebar.button("Clear History", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

    for rev_idx, item in enumerate(reversed(history)):
        idx = len(history) - 1 - rev_idx
        status = str(item.get("status", "Same"))
        status_class = _status_class(status)
        original_query = str(item.get("original_query", ""))
        rewritten_query = str(item.get("rewritten_query", ""))
        timestamp = str(item.get("timestamp", ""))
        improvement = float(item.get("improvement", 0.0) or 0.0)

        st.sidebar.markdown(
            (
                "<div class='history-card'>"
                f"<div class='history-query'>{_truncate(original_query, 70)}</div>"
                f"<div class='history-preview'>Rewrite: {_truncate(rewritten_query, 58)}</div>"
                "<div class='history-meta'>"
                f"{timestamp} | {improvement:+.3f}"
                f"<span class='status-pill {status_class}'>{status}</span>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        c1, c2 = st.sidebar.columns([1, 1])
        with c1:
            if st.button("Load", key=f"load_history_{idx}", use_container_width=True):
                st.session_state["asr_text"] = original_query
                st.rerun()
        with c2:
            if st.button("Delete", key=f"delete_history_{idx}", use_container_width=True):
                current = st.session_state.get("history", [])
                if 0 <= idx < len(current):
                    del current[idx]
                    st.session_state["history"] = current
                st.rerun()


def main() -> None:
    setup_logging()
    s = get_settings()
    st.set_page_config(page_title="Code-Switched Query Rewriting", layout="wide")
    _inject_styles()

    st.markdown(
        """
        <div class='hero-card'>
            <h2 style='margin:0'>Retrieval-Aware Query Rewriting for Code-Switched Spoken Questions</h2>
            <div class='hero-subtitle'>ASR transcript -> explainable rewriting -> retrieval -> answer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='section-divider' />", unsafe_allow_html=True)

    _init_session_state()

    with st.sidebar:
        st.header("Demo Settings")
        top_k = st.number_input("Top-K", min_value=1, max_value=20, value=int(s.top_k))
        show_segments = st.toggle("Show ASR timestamps", value=False)
        max_docs_to_show = st.slider("Retrieved docs to display", min_value=1, max_value=10, value=5)
        mode_options = ["Text (ASR transcript)"]
        if ASRTranscriber is not None:
            mode_options.append("Audio file (ASR)")
        mode = st.radio("Input mode", mode_options, index=0)
        if ASRTranscriber is None:
            st.caption("Audio/ASR mode is unavailable until optional speech dependencies are installed.")
    _render_session_history_sidebar()

    pipeline, rewriter, transcriber = _get_cached_runtime_components()

    if "asr_text" not in st.session_state:
        st.session_state["asr_text"] = ""

    st.subheader("1) Input")
    st.markdown(
        "<div class='section-card'><div class='card-title'>Input Container</div><div class='tight-help'>Paste a code-switched spoken query or ASR transcript here.</div></div>",
        unsafe_allow_html=True,
    )
    _render_sample_query_picker()
    if mode == "Text (ASR transcript)":
        st.session_state["asr_text"] = st.text_area(
            "Paste ASR transcript (code-switched, noisy is fine)",
            value=st.session_state.get(
                "asr_text",
                "bhai scholarship ke liye eligibility kya hai for undergrad admission",
            ),
            height=120,
        ).strip()
    else:
        audio = st.file_uploader("Upload audio (wav/mp3/etc)", type=["wav", "mp3", "m4a", "flac", "ogg"])
        if audio:
            audio_path = _save_uploaded_audio(audio)
            st.caption(f"Saved audio: {audio_path.as_posix()}")

            if st.button("Transcribe Speech", type="secondary"):
                if transcriber is None:
                    st.error("ASR backend is not available in this deployment.")
                else:
                    try:
                        with st.spinner("Running ASR..."):
                            asr = transcriber.transcribe(audio_path, return_timestamps=show_segments)
                        st.session_state["asr_text"] = asr.text.strip()
                        st.session_state["asr_meta"] = {
                            "language": asr.language,
                            "language_probability": asr.language_probability,
                            "duration_seconds": asr.duration_seconds,
                            "segments": [
                                {"start": s.start, "end": s.end, "text": s.text}
                                for s in (asr.segments or [])
                            ],
                        }
                    except Exception as exc:
                        st.error(f"ASR failed: {exc}")

        st.session_state["asr_text"] = st.text_area(
            "Raw ASR transcript (editable)",
            value=st.session_state.get("asr_text", ""),
            height=120,
        ).strip()

        asr_meta = st.session_state.get("asr_meta")
        if asr_meta:
            st.caption(
                f"ASR language: {asr_meta.get('language')} | duration(s): {asr_meta.get('duration_seconds')}"
            )
            if show_segments and asr_meta.get("segments"):
                with st.expander("ASR segments", expanded=False):
                    st.json(asr_meta.get("segments"))

    query_text = st.session_state.get("asr_text", "").strip()

    st.subheader("2) Query Rewrite Preview")
    if query_text:
        rewrite_preview = rewriter.rewrite_result(query_text)
        center_col = st.columns([1, 1.7, 1])[1]
        with center_col:
            _render_rewrite_pipeline(query_text, rewrite_preview)
    else:
        st.info("Provide text or transcribe audio first.")

    st.markdown("<hr class='section-divider' />", unsafe_allow_html=True)

    st.subheader("3) Run Retrieval + Answer Generation")
    if st.button("Run Baseline vs Rewritten", type="primary", disabled=not bool(query_text)):
        s.top_k = int(top_k)
        with st.spinner("Running baseline and rewritten modes..."):
            comparison = pipeline.compare_modes(query_text)
        st.session_state["comparison"] = comparison
        _append_history_item(comparison)

    comparison = st.session_state.get("comparison")
    if comparison:
        baseline = comparison.baseline
        rewritten = comparison.rewritten

        st.subheader("4) Baseline vs Rewritten")
        col_left, col_right = st.columns(2)
        with col_left:
            _render_result_block("Baseline Retrieval", baseline)
        with col_right:
            _render_result_block("Rewritten Retrieval", rewritten)

        _render_comparison_insights(baseline, rewritten)

        st.subheader("Top Retrieved Documents")
        col3, col4 = st.columns(2)
        with col3:
            _render_retrieved_docs("Baseline Top Docs", baseline.retrieved, max_docs=max_docs_to_show)
        with col4:
            _render_retrieved_docs("Rewritten Top Docs", rewritten.retrieved, max_docs=max_docs_to_show)

        with st.expander("Raw Comparison JSON", expanded=False):
            st.code(json.dumps(asdict(comparison), ensure_ascii=False, indent=2))

    st.markdown("<hr class='section-divider' />", unsafe_allow_html=True)

    st.subheader("5) Submission Support")
    _render_evaluation_snapshot()
    _render_architecture_support()


if __name__ == "__main__":
    main()

