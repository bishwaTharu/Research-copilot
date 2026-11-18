# ADR-1: Deterministic Ingestion Pipeline

## Status
Accepted

## Context
Scientific papers contain rich structural data: hierarchical sections, paragraphs, inline equations, mathematical formulas, embedded tables, figures with captions, and bibliography reference lists. Ingestion requires extracting and linking these diverse assets accurately.

Using autonomous LLM agents to orchestrate ingestion introduces non-determinism, significant token cost, high latency, and vulnerability to free-tier rate limits and hallucinated structures.

## Decision
Paper ingestion shall be executed as a deterministic, multi-stage pipeline with resumable checkpoints, rather than an LLM-driven agentic loop:
1. **Layout & Text Extraction**: Deterministic layout analysis extracts sections, paragraphs, bounding boxes, and coordinates into the unified `PaperDoc` schema.
2. **Visual Asset Extraction**: Deterministic cropping of figures and tables with attached captions based on bounding box coordinates.
3. **Reference Parsing**: Deterministic extraction of bibliography entries and citation markers (e.g., via GROBID / regex / structured parsers).
4. **Local Embedding**: Deterministic generation of dense vector representations for text chunks and figure captions using a fixed local model.
5. **Enrichment / Summary**: Optional model calls (e.g. vision summary of figures) are isolated as asynchronous, retryable, cached enrichment tasks that do not block document availability.

LLM agents are strictly reserved for query-time orchestration and claim verification.

## Consequences
- **Pros**: 100% reproducible extractions, lower latency, resilient to LLM gateway downtime, idempotent resumption on failure (NFR-REL-01).
- **Cons**: Requires specialized parser adapters (e.g. Docling, PyMuPDF) to handle diverse PDF layouts.

