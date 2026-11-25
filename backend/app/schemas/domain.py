"""
Domain models and schema definitions for Multimodal Research Co-pilot.
Compliant with ADR-2 (Own the schema), SRS Section 9 (Data requirements),
and Addendum v1.3 (Evidence Trail, Reference Reader, Formula Records, and Notes).
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
    """Single textual or structural block inside a parsed paper (FR-ING-04, Addendum v1.3)."""

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
    char_start: int = Field(default=0, description="Start character offset in section/page stream")
    char_end: int = Field(default=0, description="End character offset in section/page stream")


class FormulaItem(BaseModel):
    """Displayed formula extracted from paper (Addendum v1.3, Section 5.1)."""

    formula_id: str = Field(..., description="Unique formula ID e.g. 'eq1'")
    page: int = Field(..., ge=1, description="1-indexed page number")
    bbox: List[float] = Field(default_factory=list, description="[x0, y0, x1, y1] coordinates")
    crop_path: str = Field(default="", description="Relative path to formula crop image asset")
    latex: str = Field(default="", description="LaTeX source of the formula")
    number: str = Field(default="", description="Display number e.g. '(3)' or 'Eq. 3'")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence flag; low triggers page image fallback")
    mentions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Text blocks mentioning this formula e.g. [{'block_id': 'b5', 'page': 2, 'excerpt': '...'}]",
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
    """Bibliography reference entry (FR-REF-01 to 07, Addendum v1.3)."""

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
    """In-text reference mention linking a text block to a reference (FR-REF-02, 06, Addendum v1.3)."""

    paper_id: str
    block_id: str
    reference_id: str
    sentence: str = Field(
        ..., description="Exact citing sentence in text for why-cited panel"
    )
    char_start: int = Field(default=0, description="Character start offset of marker within sentence/block")
    char_end: int = Field(default=0, description="Character end offset of marker within sentence/block")


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
    Authoritative canonical paper representation (ADR-2, Addendum v1.3).
    Every PDF parser (Docling, PyMuPDF, MinerU) maps into this schema.
    """

    metadata: PaperMetadata
    blocks: List[TextBlock] = Field(default_factory=list)
    formulas: List[FormulaItem] = Field(default_factory=list, description="Displayed formulas (Addendum v1.3)")
    figures: List[FigureItem] = Field(default_factory=list)
    tables: List[TableItem] = Field(default_factory=list)
    references: List[ReferenceItem] = Field(default_factory=list)
    citation_mentions: List[CitationMention] = Field(default_factory=list)


class ParseQualityReport(BaseModel):
    """
    Stage 7 Parse-Quality Report (Section 8.1, Stage 7, Addendum v1.3).
    Counts and warnings for figures, formulas, tables, and reference resolution.
    """

    paper_id: str
    total_pages: int = 0
    blocks_count: int = 0
    formulas_count: int = 0
    figures_count: int = 0
    tables_count: int = 0
    references_count: int = 0
    tier_counts: Dict[str, int] = Field(
        default_factory=lambda: {"verified": 0, "probable": 0, "unresolved": 0}
    )
    warnings: List[str] = Field(default_factory=list)
    has_figures: bool = False
    has_formulas: bool = False
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
    """Discrete evidence item attached to a claim (SRS Page 8, Addendum v1.3)."""

    type: Literal["text", "figure", "table", "ref", "formula"]
    block_id: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    bbox: Optional[List[float]] = None
    formula_id: Optional[str] = None
    formula_latex: Optional[str] = None
    figure_id: Optional[str] = None
    table_id: Optional[str] = None
    reference_id: Optional[str] = None
    marker: Optional[str] = None
    doi: Optional[str] = None
    context: Optional[str] = None
    quote: Optional[str] = None


class VerificationChecks(BaseModel):
    """Deterministic validation checks executed before entailment (FR-VER-01, Addendum v1.3 Section 5.1)."""

    ids_exist: bool = False
    doi_resolves: bool = False
    quote_match: bool = False
    bbox_valid: bool = False
    formula_crop_valid: bool = False
    mention_in_block: bool = False


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


class NoteRecord(BaseModel):
    """Researcher note anchored to a paper element (Addendum v1.3 Section 4.2, 5.1, 6.2)."""

    note_id: str = Field(..., description="Unique note ID")
    paper_id: str
    anchor_type: Literal["block", "formula", "figure", "table", "ref"] = "block"
    anchor_id: str = Field(..., description="ID of the anchored element")
    quote: str = Field(default="", description="Text-quote selector for re-anchoring")
    text: str = Field(..., description="Note body text")
    color: str = Field(default="yellow", description="Note highlight color")
    created_at: str = Field(default="")
    updated_at: str = Field(default="")
    orphaned: bool = Field(default=False, description="True if anchor text changed after re-parse")
    tags: List[str] = Field(default_factory=list)


class CitedWorkPassage(BaseModel):
    """Passage from an open-access cited work (Addendum v1.3 Section 3.1, 5.1)."""

    reference_id: str
    passage_text: str
    page: Optional[int] = None
    section: Optional[str] = None
    match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    verified: bool = False
    label: str = "From the cited work"


class EvidenceTrailItem(BaseModel):
    """Single item in the evidence trail for an answer claim (Addendum v1.3 Section 3.1, 4.1)."""

    trail_item_id: str
    claim_id: str
    evidence_type: Literal["paragraph", "formula", "figure", "table", "cited_here", "cited_work_passage"]
    block_id: Optional[str] = None
    text_span: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    bbox: Optional[List[float]] = None
    formula_id: Optional[str] = None
    formula_latex: Optional[str] = None
    formula_number: Optional[str] = None
    formula_confidence: Optional[float] = None
    formula_crop_path: Optional[str] = None
    figure_id: Optional[str] = None
    table_id: Optional[str] = None
    crop_path: Optional[str] = None
    caption: Optional[str] = None
    reference_id: Optional[str] = None
    reference_marker: Optional[str] = None
    reference_tier: Optional[str] = None
    citing_sentence: Optional[str] = None
    cited_passage: Optional[CitedWorkPassage] = None
    checks: VerificationChecks = Field(default_factory=VerificationChecks)
    honest_state: Optional[str] = None


class ReferenceReaderPayload(BaseModel):
    """Payload for the Reference Reader tab (Addendum v1.3 Section 4.1, 6.3)."""

    reference_id: str
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    open_access_url: Optional[str] = None
    tier: str = "unresolved"
    fetch_status: Literal["idle", "fetching", "checking", "parsing", "ready", "failed", "no_copy"] = "idle"
    fetch_message: str = ""
    passages: List[CitedWorkPassage] = Field(default_factory=list)
    raw_entry: Optional[str] = None


# ── Agentic 4-Step Citation Verification Schemas ────────────────────────────


class CitationPassageMatch(BaseModel):
    """Exact snippet from Paper B matched during semantic search."""

    snippet: str
    page: Optional[int] = None
    section: Optional[str] = None
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    query_used: str = ""


class AgentStepTrace(BaseModel):
    """Trace record for one step of the 4-step agentic verification workflow."""

    step: int = Field(..., ge=1, le=4)
    stage: str
    message: str
    status: Literal["running", "done", "warning", "error"] = "done"
    details: Optional[Dict[str, Any]] = None


class CitationVerificationResult(BaseModel):
    """Structured output of the 4-step Agentic Citation Verification workflow."""

    paper_id: str
    citing_sentence: str
    extracted_markers: List[str] = Field(default_factory=list)
    resolved_reference: Optional[Dict[str, Any]] = None
    source_basis: Literal["full_text", "abstract_only", "unavailable"] = "unavailable"
    is_paywalled: bool = False
    paywall_notice: Optional[str] = None
    verdict: Literal[
        "supported",
        "contradicted",
        "partial",
        "context",
        "insufficient",
        "blocked_privacy",
    ] = "insufficient"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = ""
    matched_passages: List[CitationPassageMatch] = Field(default_factory=list)
    queries_tried: List[str] = Field(default_factory=list)
    self_corrected: bool = False
    steps: List[AgentStepTrace] = Field(default_factory=list)
    latency_ms: int = 0
