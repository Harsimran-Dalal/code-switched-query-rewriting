# Methodology (high-level)

## Baseline pipeline

Speech → ASR → Dense retrieval (embeddings + FAISS) → Extractive RAG answer

## Proposed contribution: retrieval-aware query rewriting

The rewriting module takes ASR text and produces a query that better matches the retrieval index by:

- Removing fillers and discourse markers common in spoken Hinglish/Punjabi-English/Hindi-English
- Normalizing common romanized variants (e.g., “kya”, “kia”, “ky”)
- Translating a small curated lexicon of frequent mixed-language tokens into English retrieval keywords
- **Retrieval awareness**: using corpus term statistics (document frequency / IDF) to bias the rewrite towards discriminative terms present in the indexed documents

## Evaluation

Compare baseline vs rewritten queries using:

- Recall@k (did we retrieve any expected document in top-k?)
- MRR@k (how early do expected documents appear?)
