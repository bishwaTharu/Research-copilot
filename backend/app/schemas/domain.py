"""
Domain models and schema definitions for Multimodal Research Co-pilot.
Compliant with ADR-2 (Own the schema) and SRS Section 9 (Data requirements).
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates on a page: [x0, y0, x1, y1] (normalized or point coordinates)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def to_list(self) -> List[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    @classmethod
    def from_list(cls, coords: List[float]) -> "BoundingBox":
        if len(coords) != 4:
            raise ValueError(
                "BoundingBox requires exactly 4 coordinates [x0, y0, x1, y1]"
            )
        return cls(
            x0=float(coords[0]),
            y0=float(coords[1]),
            x1=float(coords[2]),
            y1=float(coords[3]),
        )


class TextBlock(BaseModel):
    """Single textual or structural block inside a parsed paper (FR-ING-04)."""

    block_id: str = Field(..., description="Unique block ID e.g. 'b12'")
    page: int = Field(..., ge=1, description="1-indexed page number")
    bbox: List[float] = Field(..., description="[x0, y0, x1, y1] coordinates")
    type: Literal["heading", "paragraph", "equation", "caption", "fallback_page"] = (
        "paragraph"
    )
    text: str = Field(..., description="Extracted plain text")
    section_path: List[str] = Field(
        default_factory=list, description="Hierarchical section path"
    )


class FigureItem(BaseModel):
    """Visual figure extracted with bounding box and caption (FR-FIG-01 to 05)."""

    figure_id: str = Field(..., description="Unique figure ID e.g. 'fig1'")
    page: int = Field(..., ge=1)
    bbox: List[float]
    crop_path: str = Field(..., description="Relative path to cropped image asset")
    caption: str = Field(default="")
    summary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured vision summary: type, components, relations, key takeaway",
    )
    summary_status: Literal["pending", "completed", "failed", "skipped"] = "pending"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    embedding: Optional[List[float]] = None
    mentions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Text blocks and pages mentioning this figure e.g. [{'page': 2, 'block_id': 'b10', 'excerpt': '...'}]",
    )


class TableItem(BaseModel):
    """Table extracted with bounding box, caption and tabular content."""

    table_id: str = Field(..., description="Unique table ID e.g. 'tbl1'")
    page: int = Field(..., ge=1)
    bbox: List[float]
    crop_path: str
    caption: str = Field(default="")
    markdown: Optional[str] = None


class ReferenceItem(BaseModel):
    """Bibliography reference entry (FR-REF-01 to 07)."""

    reference_id: str = Field(..., description="Unique reference ID e.g. 'r1'")
    marker: str = Field(
        ...,
        description="Inline marker as used in text e.g. '[12]' or '(Smith et al., 2021)'",
    )
    raw_text: str = Field(..., description="Original raw bibliography string")
    doi: Optional[str] = None
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None
    resolution_status: Literal["resolved", "unresolved", "pending"] = "pending"
    open_access_url: Optional[str] = None
    tier: Literal["verified", "probable", "unresolved"] = "unresolved"
    match_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    is_retracted: bool = False
    retraction_details: Optional[str] = None
    bibtex: Optional[str] = None
    citing_sentences: List[str] = Field(default_factory=list)
    user_confirmed: bool = False


class CitationMention(BaseModel):
    """In-text reference mention linking a text block to a reference (FR-REF-02, 06)."""

    paper_id: str
    block_id: str
    reference_id: str
    sentence: str = Field(
        ..., description="Exact citing sentence in text for why-cited panel"
    )


class PaperMetadata(BaseModel):
    """High-level metadata for a paper."""

    id: str
    content_hash: str = Field(
        ..., description="SHA-256 hash of PDF binary for deduplication"
    )
    title: str = "Untitled Paper"
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    privacy_setting: Literal["allow_remote", "local_only"] = "allow_remote"
    status: Literal[
        "uploaded",
        "parsing",
        "extracted",
        "embedded",
        "ready",
        "failed",
        "waiting_quota",
    ] = "uploaded"
    total_pages: int = 0
    warnings: List[str] = Field(default_factory=list)


class PaperDoc(BaseModel):
    """
    Authoritative canonical paper representation (ADR-2).
    Every PDF parser (Docling, PyMuPDF, MinerU) maps into this schema.
    """

    metadata: PaperMetadata
    blocks: List[TextBlock] = Field(default_factory=list)
    figures: List[FigureItem] = Field(default_factory=list)
    tables: List[TableItem] = Field(default_factory=list)
    references: List[ReferenceItem] = Field(default_factory=list)
    citation_mentions: List[CitationMention] = Field(default_factory=list)


class ParseQualityReport(BaseModel):
    """
    Stage 7 Parse-Quality Report (Section 8.1, Stage 7).
    Counts and warnings for figures, tables, and reference resolution.
    """

    paper_id: str
    total_pages: int = 0
    blocks_count: int = 0
    figures_count: int = 0
    tables_count: int = 0
    references_count: int = 0
    tier_counts: Dict[str, int] = Field(
        default_factory=lambda: {"verified": 0, "probable": 0, "unresolved": 0}
    )
    warnings: List[str] = Field(default_factory=list)
    has_figures: bool = False
    has_references: bool = False
    figure_summaries_pending: int = 0
    duplicate_of: Optional[str] = None


class Chunk(BaseModel):
    """Searchable chunk for hybrid retrieval (FR-QA-02, 03).
    Step 3 of Hybrid Architecture: Links MinerU text to GROBID paper metadata & references.
    """

    chunk_id: str
    paper_id: str
    grobid_paper_id: Optional[str] = None
    paper_metadata: Optional[Dict[str, Any]] = None
    linked_references: List[Dict[str, Any]] = Field(default_factory=list)
    block_ids: List[str]
    section_path: List[str]
    text: str
    page_start: int
    page_end: int
    embedding: Optional[List[float]] = None
    embedding_model_id: str = "local:hash-ngram"


class EvidenceItem(BaseModel):
    """Discrete evidence item attached to a claim (SRS Page 8)."""

    type: Literal["text", "figure", "table", "ref"]
    block_id: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    bbox: Optional[List[float]] = None
    figure_id: Optional[str] = None
    table_id: Optional[str] = None
    reference_id: Optional[str] = None
    marker: Optional[str] = None
    doi: Optional[str] = None
    context: Optional[str] = None
    quote: Optional[str] = None


class VerificationChecks(BaseModel):
    """Deterministic validation checks executed before entailment (FR-VER-01)."""

    ids_exist: bool = False
    doi_resolves: bool = False
    quote_match: bool = False
    bbox_valid: bool = False


class ClaimRecord(BaseModel):
    """Claim unit returned by the orchestrator and audited by the verifier."""

    claim_id: str
    claim: str
    evidence: List[EvidenceItem] = Field(default_factory=list)
    verdict: Literal["supported", "partial", "unsupported", "unverified"] = "unverified"
    checks: VerificationChecks = Field(default_factory=VerificationChecks)
    routed_via: Optional[str] = None
    repair_attempted: bool = False
    provenance: Dict[str, Any] = Field(default_factory=dict)


class AnswerRecord(BaseModel):
    """Full grounded answer structure with evidence claims (FR-QA-01, 05, 06)."""

    answer_id: str
    paper_id: str
    question: str
    answer_markdown: str
    claims: List[ClaimRecord] = Field(default_factory=list)
    suggested_followups: List[str] = Field(default_factory=list)
    is_verified: bool = False
    abstained: bool = False
    abstention_reason: Optional[str] = None
    routed_via: Optional[str] = None
    created_at: str = ""
    provenance: Dict[str, Any] = Field(default_factory=dict)


class FeedbackRecord(BaseModel):
    """User rating and citation bug reporting (FR-UI-08)."""

    answer_id: str
    rating: int = Field(..., ge=1, le=5)
    flag_wrong_citation: bool = False
    comment: Optional[str] = None
    reported_evidence_id: Optional[str] = None
