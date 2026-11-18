"""
Hybrid Ingestion Pipeline (GROBID + MinerU)

Architecture:
  Step 1: Metadata & Citations (GROBID)
          Extracts title, authors, abstract, publication year, DOI, and structured bibliography.
          Stores structured reference graph for verifiable citation tracing.
  Step 2: Full-Text & Formulas (MinerU)
          Extracts high-fidelity Markdown with LaTeX equations, tables, and visual crops.
  Step 3: Linkage & Chunking
          Attaches GROBID paper ID & metadata to every MinerU chunk.
          Extracts in-text citation markers and maps them to exact GROBID bibliography entries.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.ingestion.grobid_client import GROBIDClient, grobid_client
from backend.app.ingestion.mineru_client import MinerUClient, mineru_client
from backend.app.references.resolver import reference_resolver
from backend.app.schemas.domain import (
    CitationMention,
    Chunk,
    FigureItem,
    PaperDoc,
    PaperMetadata,
    ParseQualityReport,
    ReferenceItem,
    TableItem,
    TextBlock,
)
from backend.app.tools.layer import tool_layer

STAGE_VALIDATE = "validate"
STAGE_GROBID = "grobid_metadata"
STAGE_MINERU = "mineru_fulltext"
STAGE_RESOLVE_REFS = "resolve_references"
STAGE_LINK_AND_CHUNK = "link_and_chunk"
STAGE_EMBEDDING = "embedding"
STAGE_COMPLETE = "complete"

STAGES_ORDER = [
    STAGE_VALIDATE,
    STAGE_GROBID,
    STAGE_MINERU,
    STAGE_RESOLVE_REFS,
    STAGE_LINK_AND_CHUNK,
    STAGE_EMBEDDING,
    STAGE_COMPLETE,
]


class IngestionState:
    """Tracks state and checkpoints for a single paper ingestion job."""

    def __init__(self, paper_id: str, state_dir: Path):
        self.paper_id = paper_id
        self.state_file = state_dir / f"{paper_id}_state.json"
        self.data: Dict[str, Any] = {
            "paper_id": paper_id,
            "current_stage": STAGE_VALIDATE,
            "completed_stages": [],
            "error": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.load()

    def load(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as exc:
                logger.error(f"Failed to load state file {self.state_file}: {exc}")

    def save(self):
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def mark_stage_done(self, stage: str):
        if stage not in self.data["completed_stages"]:
            self.data["completed_stages"].append(stage)
        idx = STAGES_ORDER.index(stage) if stage in STAGES_ORDER else -1
        if idx != -1 and idx + 1 < len(STAGES_ORDER):
            self.data["current_stage"] = STAGES_ORDER[idx + 1]
        self.save()

    def is_stage_done(self, stage: str) -> bool:
        return stage in self.data["completed_stages"]


class IngestionPipeline:
    """
    Executes the 3-step Hybrid Ingestion Pipeline:
      1. GROBID: Extract metadata and structured bibliography.
      2. MinerU: Extract full-text Markdown, LaTeX formulas, tables, and visual crops.
      3. Linkage: Enrich MinerU chunks with GROBID paper metadata & citation references.
    """

    def __init__(
        self,
        grobid: Optional[GROBIDClient] = None,
        mineru: Optional[MinerUClient] = None,
        embedder_fn: Optional[
            Callable[[List[str]], Awaitable[List[List[float]]]]
        ] = None,
        state_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
    ):
        self.grobid = grobid or grobid_client
        self.mineru = mineru or mineru_client
        self.embedder_fn = embedder_fn
        self.data_dir = data_dir or Path(settings.DATA_DIR)
        self.state_dir = state_dir or self.data_dir / "states"
        self.crops_dir = Path(settings.CROPS_DIR)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        self.docs_store: Dict[str, PaperDoc] = {}
        self.chunks_store: Dict[str, List[Chunk]] = {}
        self._load_persisted_papers()

    def _load_persisted_papers(self):
        """Loads previously ingested PaperDoc and Chunks from disk into memory on startup."""
        for doc_file in self.data_dir.glob("paper_*_doc.json"):
            paper_id = doc_file.stem.replace("_doc", "")
            try:
                with open(doc_file, "r", encoding="utf-8") as f:
                    doc = PaperDoc.model_validate_json(f.read())
                    self.docs_store[paper_id] = doc
                chunks_file = self.data_dir / f"{paper_id}_chunks.json"
                if chunks_file.exists():
                    with open(chunks_file, "r", encoding="utf-8") as f:
                        self.chunks_store[paper_id] = [
                            Chunk.model_validate(c) for c in json.load(f)
                        ]
            except Exception as exc:
                logger.debug(f"Could not preload {paper_id}: {exc}")
        tool_layer.set_docs_store(self.docs_store)

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def chunk_paper(self, paper: PaperDoc, max_chunk_words: int = 300) -> List[Chunk]:
        """Backward-compatible alias for link_and_chunk."""
        return self.link_and_chunk(paper, max_chunk_words)

    def link_and_chunk(
        self, paper: PaperDoc, max_chunk_words: int = 300
    ) -> List[Chunk]:
        """
        Step 3: Chunking & Metadata Linkage.
        Chunks Markdown blocks respecting section boundaries, attaches GROBID paper metadata,
        and links in-text citation markers directly to GROBID structured references.
        """
        ref_lookup = {r.reference_id: r for r in paper.references}
        marker_lookup = {r.marker.strip("[]"): r for r in paper.references if r.marker}

        citation_bracket_regex = re.compile(r"\[(\d{1,3}(?:[,\s–\-]+\d{1,3})*)\]")

        chunks: List[Chunk] = []
        current_section = None
        current_blocks: List[TextBlock] = []
        current_words: List[str] = []
        chunk_idx = 0

        paper_meta_dict = {
            "title": paper.metadata.title,
            "authors": paper.metadata.authors,
            "year": paper.metadata.year,
            "doi": paper.metadata.doi,
        }

        def build_chunk(
            words: List[str], blocks: List[TextBlock], sec: Optional[List[str]]
        ) -> Chunk:
            nonlocal chunk_idx
            chunk_idx += 1
            chunk_text = " ".join(words)

            # Extract cited reference IDs from chunk text
            linked_refs: List[Dict[str, Any]] = []
            seen_ref_ids = set()

            for match in citation_bracket_regex.finditer(chunk_text):
                marker_content = match.group(1)
                nums = re.findall(r"\d+", marker_content)
                for num_str in nums:
                    ref_id = f"r{num_str}"
                    ref_obj = ref_lookup.get(ref_id) or marker_lookup.get(num_str)
                    if ref_obj and ref_obj.reference_id not in seen_ref_ids:
                        seen_ref_ids.add(ref_obj.reference_id)
                        linked_refs.append(
                            {
                                "reference_id": ref_obj.reference_id,
                                "marker": ref_obj.marker,
                                "title": ref_obj.title,
                                "authors": ref_obj.authors,
                                "year": ref_obj.year,
                                "doi": ref_obj.doi,
                                "venue": ref_obj.venue,
                                "raw_text": ref_obj.raw_text,
                            }
                        )

            start_page = min((b.page for b in blocks), default=1)
            end_page = max((b.page for b in blocks), default=1)

            return Chunk(
                chunk_id=f"c_{chunk_idx}",
                paper_id=paper.metadata.id,
                grobid_paper_id=paper.metadata.id,
                paper_metadata=paper_meta_dict,
                linked_references=linked_refs,
                block_ids=[b.block_id for b in blocks],
                section_path=list(sec or ["Document"]),
                text=chunk_text,
                page_start=start_page,
                page_end=end_page,
                embedding_model_id=(
                    settings.EMBEDDING_API_MODEL
                    if settings.EMBEDDING_BACKEND == "gateway"
                    else "local:hash-ngram"
                ),
            )

        for block in paper.blocks:
            if block.type == "fallback_page":
                continue

            sec_key = tuple(block.section_path)
            words = block.text.split()

            if sec_key != current_section or (
                len(current_words) + len(words) > max_chunk_words and current_words
            ):
                if current_words:
                    chunks.append(
                        build_chunk(current_words, current_blocks, current_section)
                    )
                    current_blocks = []
                    current_words = []

            current_section = sec_key
            current_blocks.append(block)
            current_words.extend(words)

        if current_words:
            chunks.append(build_chunk(current_words, current_blocks, current_section))

        return chunks

    async def run(self, file_path: Path, paper_id: str) -> PaperDoc:
        file_path = Path(file_path)
        state = IngestionState(paper_id, self.state_dir)

        paper_doc_file = self.data_dir / f"{paper_id}_doc.json"
        chunks_file = self.data_dir / f"{paper_id}_chunks.json"

        # ── Stage 1: Validate file ──────────────────────────────────────────
        if not state.is_stage_done(STAGE_VALIDATE):
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > settings.MAX_FILE_SIZE_MB:
                raise ValueError(
                    f"File size {size_mb:.1f}MB exceeds limit of {settings.MAX_FILE_SIZE_MB}MB"
                )
            state.mark_stage_done(STAGE_VALIDATE)

        content_hash = self.compute_sha256(file_path)

        # Stage 1 duplicate detection (Section 8.1: duplicate reuses existing paper)
        existing_doc = next(
            (
                d
                for d in self.docs_store.values()
                if d.metadata.content_hash == content_hash and d.metadata.id != paper_id
            ),
            None,
        )
        if existing_doc:
            logger.info(
                f"[{paper_id}] Duplicate detected by content hash ({content_hash[:10]}): reusing existing paper {existing_doc.metadata.id}"
            )
            return existing_doc

        # ── Step 1: GROBID (Metadata & Citations) ───────────────────────────
        grobid_cache_file = self.data_dir / f"{paper_id}_grobid.json"
        metadata: Optional[PaperMetadata] = None
        references: List[ReferenceItem] = []
        citation_mentions: List[CitationMention] = []

        if grobid_cache_file.exists():
            with open(grobid_cache_file, "r", encoding="utf-8") as f:
                cached_grobid = json.load(f)
                metadata = PaperMetadata.model_validate(cached_grobid["metadata"])
                references = [
                    ReferenceItem.model_validate(r) for r in cached_grobid["references"]
                ]
                citation_mentions = [
                    CitationMention.model_validate(cm)
                    for cm in cached_grobid.get("citation_mentions", [])
                ]
        else:
            logger.info(
                f"[{paper_id}] Step 1: Extracting metadata and bibliography via GROBID..."
            )
            metadata, references, citation_mentions = await self.grobid.process_pdf(
                file_path, paper_id
            )
            metadata.content_hash = content_hash
            with open(grobid_cache_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metadata": metadata.model_dump(),
                        "references": [r.model_dump() for r in references],
                        "citation_mentions": [
                            cm.model_dump() for cm in citation_mentions
                        ],
                    },
                    f,
                    indent=2,
                )
            state.mark_stage_done(STAGE_GROBID)

        # ── Step 2: MinerU (Full-Text, LaTeX Equations, Tables, Visuals) ─────
        mineru_cache_file = self.data_dir / f"{paper_id}_mineru.json"
        blocks: List[TextBlock] = []
        figures: List[FigureItem] = []
        tables: List[TableItem] = []

        if mineru_cache_file.exists():
            with open(mineru_cache_file, "r", encoding="utf-8") as f:
                cached_mineru = json.load(f)
                blocks = [TextBlock.model_validate(b) for b in cached_mineru["blocks"]]
                figures = [
                    FigureItem.model_validate(fig) for fig in cached_mineru["figures"]
                ]
                tables = [
                    TableItem.model_validate(tbl) for tbl in cached_mineru["tables"]
                ]
        else:
            logger.info(
                f"[{paper_id}] Step 2: Extracting clean Markdown, LaTeX formulas & figures via MinerU..."
            )
            blocks, figures, tables, _ = await self.mineru.process_pdf(
                file_path, paper_id, self.crops_dir
            )
            with open(mineru_cache_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "blocks": [b.model_dump() for b in blocks],
                        "figures": [fig.model_dump() for fig in figures],
                        "tables": [tbl.model_dump() for tbl in tables],
                    },
                    f,
                    indent=2,
                )
            state.mark_stage_done(STAGE_MINERU)

        # Stage 3: Figure Mentions Linking (Section 5.3, 6)
        for fig in figures:
            fig_num_match = re.search(r"\d+", fig.figure_id)
            patterns = [rf"\bfigure\s+{re.escape(fig.figure_id)}\b", rf"\bfig\.?\s+{re.escape(fig.figure_id)}\b"]
            if fig_num_match:
                num = fig_num_match.group(0)
                patterns.extend([rf"\bfigure\s+{num}\b", rf"\bfig\.?\s+{num}\b"])
            mentions = []
            for b in blocks:
                for pat in patterns:
                    m = re.search(pat, b.text, re.IGNORECASE)
                    if m:
                        start = max(0, m.start() - 35)
                        end = min(len(b.text), m.end() + 55)
                        mentions.append({
                            "page": b.page,
                            "block_id": b.block_id,
                            "excerpt": b.text[start:end].strip(),
                        })
                        break
            fig.mentions = mentions[:5]

        # Stage 4: Bibliography Marker Reconciliation Warning (Section 8.1, Stage 4)
        unique_cited_markers = {cm.reference_id for cm in citation_mentions}
        if references and unique_cited_markers and abs(len(unique_cited_markers) - len(references)) > 10:
            warn = f"Citation count mismatch: {len(unique_cited_markers)} distinct in-text markers vs {len(references)} bibliography entries."
            if warn not in metadata.warnings:
                metadata.warnings.append(warn)

        # ── Step 2.5: Enrich references via CrossRef/OpenAlex ────────────────
        if not state.is_stage_done(STAGE_RESOLVE_REFS):
            logger.info(
                f"[{paper_id}] Enriching {len(references)} bibliography entries..."
            )
            # Pre-group citing sentences by reference_id
            cites_by_ref: Dict[str, List[str]] = {}
            for cm in citation_mentions:
                cites_by_ref.setdefault(cm.reference_id, []).append(cm.sentence)

            allow_remote = (
                metadata.privacy_setting != "local_only"
                and settings.PRIVACY_DEFAULT != "local_only"
            )

            enriched_refs: List[ReferenceItem] = []
            for ref in references:
                if ref.reference_id in cites_by_ref:
                    ref.citing_sentences = cites_by_ref[ref.reference_id]
                try:
                    resolved = await reference_resolver.resolve_reference(
                        ref, allow_remote=allow_remote
                    )
                    enriched_refs.append(resolved)
                except Exception:
                    enriched_refs.append(ref)
            references = enriched_refs
            state.mark_stage_done(STAGE_RESOLVE_REFS)

        # Assemble PaperDoc
        max_page = max((b.page for b in blocks), default=metadata.total_pages or 1)
        metadata.total_pages = max(metadata.total_pages, max_page)

        paper_doc = PaperDoc(
            metadata=metadata,
            blocks=blocks,
            figures=figures,
            tables=tables,
            references=references,
            citation_mentions=citation_mentions,
        )

        with open(paper_doc_file, "w", encoding="utf-8") as f:
            f.write(paper_doc.model_dump_json(indent=2))

        # ── Step 3: Link Them Together & Chunk ───────────────────────────────
        chunks: List[Chunk] = []
        if not state.is_stage_done(STAGE_LINK_AND_CHUNK):
            logger.info(
                f"[{paper_id}] Step 3: Linking GROBID metadata & citations to MinerU text chunks..."
            )
            chunks = self.link_and_chunk(paper_doc)
            with open(chunks_file, "w", encoding="utf-8") as f:
                f.write(json.dumps([c.model_dump() for c in chunks], indent=2))
            state.mark_stage_done(STAGE_LINK_AND_CHUNK)
        else:
            if chunks_file.exists():
                with open(chunks_file, "r", encoding="utf-8") as f:
                    chunks = [Chunk.model_validate(c) for c in json.load(f)]

        # ── Embeddings Generation ───────────────────────────────────────────
        if not state.is_stage_done(STAGE_EMBEDDING):
            if self.embedder_fn and chunks:
                logger.info(f"[{paper_id}] Embedding {len(chunks)} hybrid chunks...")
                texts = [c.text for c in chunks]
                embeddings = await self.embedder_fn(texts)
                for chunk, emb in zip(chunks, embeddings):
                    chunk.embedding = emb
                with open(chunks_file, "w", encoding="utf-8") as f:
                    f.write(json.dumps([c.model_dump() for c in chunks], indent=2))
            state.mark_stage_done(STAGE_EMBEDDING)

        # ── Stage 7: Quality Report (Section 8.1, Stage 7) ─────────────────
        tier_counts = {"verified": 0, "probable": 0, "unresolved": 0}
        for r in references:
            t = r.tier if r.tier in tier_counts else "unresolved"
            tier_counts[t] += 1

        quality_report = ParseQualityReport(
            paper_id=paper_id,
            total_pages=metadata.total_pages,
            blocks_count=len(blocks),
            figures_count=len(figures),
            tables_count=len(tables),
            references_count=len(references),
            tier_counts=tier_counts,
            warnings=metadata.warnings,
            has_figures=len(figures) > 0,
            has_references=len(references) > 0,
            figure_summaries_pending=sum(
                1 for f in figures if f.summary_status == "pending"
            ),
        )
        quality_file = self.data_dir / f"{paper_id}_quality.json"
        with open(quality_file, "w", encoding="utf-8") as f:
            f.write(quality_report.model_dump_json(indent=2))

        paper_doc.metadata.status = "ready"
        state.mark_stage_done(STAGE_COMPLETE)

        self.docs_store[paper_id] = paper_doc
        self.chunks_store[paper_id] = chunks
        tool_layer.set_docs_store(self.docs_store)
        logger.info(
            f"[{paper_id}] Ingestion complete: {len(blocks)} blocks, {len(chunks)} chunks, {len(references)} refs."
        )
        return paper_doc
