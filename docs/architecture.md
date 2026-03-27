# Architecture Overview

This project demonstrates a lightweight, explainable retrieval-aware query rewriting pipeline for code-switched spoken questions.

## End-to-End Flow

User Query (Speech/Text)
    -> ASR (optional)
    -> Query Rewriting Module
    -> Dense Retrieval (FAISS)
    -> Top-K Documents
    -> Answer Generation
    -> Baseline vs Rewritten Comparison

## Module Breakdown

- Input Layer:
  - Text input or uploaded audio.
  - Audio input is transcribed by ASR before downstream processing.

- Rewriting Layer:
  - Rule-based transliteration normalization.
  - Keyword extraction for retrieval-focused terms.
  - Explainable query reformulation into cleaned, keyword, and rewritten forms.

- Retrieval Layer:
  - Dense retrieval using sentence-transformer embeddings.
  - FAISS index for efficient top-k nearest-neighbor search.

- Generation Layer:
  - Extractive answer generation from retrieved context.
  - Grounded response with citations.

- Comparison Layer:
  - Baseline retrieval on original query.
  - Rewritten retrieval on reformulated query.
  - Top-1 confidence and ranking comparison.

## Why This Design

- Explainable: every rewrite step is traceable.
- Lightweight: no external LLM dependency required for core flow.
- Demo-ready: Streamlit UI shows both retrieval paths side-by-side.
- Evaluation-friendly: CSV export supports report-ready analysis.
