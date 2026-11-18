# ADR-2: Own the Schema (Unified PaperDoc Format)

## Status
Accepted

## Context
Scientific PDF parsing tools (such as Docling, MinerU, Grobid, PyMuPDF, Marker, Nougat) produce idiosyncratic, incompatible outputs. If backend components (chunker, indexer, UI, verifier) bind directly to a specific parser's output format, swapping or cascading parsers requires rewriting major parts of the system.

## Decision
We define an internal, authoritative schema: `PaperDoc`. Every document parser must be wrapped in an adapter that translates the source output into `PaperDoc`.

### PaperDoc Specification
`PaperDoc` models a paper as:
- **Metadata**: `id`, `content_hash`, `title`, `authors`, `year`, `abstract`, `doi`, `privacy_setting`.
- **Text Blocks & Sections**: Hierarchical `section_path` (e.g. `["1. Introduction", "1.2 Scope"]`), `block_id`, `page`, `bbox` coordinates `[x0, y0, x1, y1]`, `text`, `type` (`heading`, `paragraph`, `equation`, `caption`).
- **Figures**: `figure_id`, `page`, `bbox`, `crop_path`, `caption`, `summary` (optional/pending).
- **Tables**: `table_id`, `page`, `bbox`, `crop_path`, `caption`, `csv_or_markdown` content.
- **References & Markers**: `references` list with raw bibliography string, parsed marker (e.g. `[12]`), title, authors, DOI, and `citation_mentions` (linking inline occurrences in blocks to target reference IDs).

## Consequences
- **Pros**: Completely decouples the ingestion engines from the rest of the application. Allows fallback parsers (e.g., PyMuPDF fallback when Docling fails or in resource-constrained environments).
- **Cons**: Requires mapping logic for each integrated parser.

