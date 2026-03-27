# Problem statement

Spoken queries produced by ASR are often **noisy** (misrecognitions, missing punctuation, repeated fillers). In multilingual settings, users frequently **code-switch** (e.g., Hindi-English / Punjabi-English / Hinglish), producing mixed-language queries that do not match the document vocabulary well.

This project studies a modular approach:

1. Transcribe speech with a pretrained ASR model (no training).
2. Rewrite the ASR text into a **retrieval-optimized query**, explicitly aiming to increase retrieval metrics.
3. Retrieve documents with BM25 and/or dense retrieval.
4. Generate answers with a small RAG pipeline.
5. Evaluate baseline vs rewriting on curated examples.
