"""
FastAPI route handlers implementing SRS Section 8.2 Application API.
Versioned under /api/v1 with OpenAPI schema generation.
"""

from __future__ import annotations
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.ingestion.pipeline import IngestionPipeline
from backend.app.orchestrator.agent import qa_orchestrator
from backend.app.retrieval.embedder import local_embedder
from backend.app.retrieval.hybrid import HybridSearchIndex
from backend.app.schemas.domain import (
    AnswerRecord,
    CitationVerificationResult,
    ClaimRecord,
    EvidenceTrailItem,
    FeedbackRecord,
    FigureItem,
    FormulaItem,
    NoteRecord,
    PaperDoc,
    ParseQualityReport,
    ReferenceItem,
    ReferenceReaderPayload,
)
from backend.app.services.citation_verifier_agent import citation_verifier_agent
from backend.app.tools.layer import tool_layer
from backend.app.verifier.engine import verifier_engine

router = APIRouter()

# In-memory document & index registry for local speed and persistence fallback
pipeline = IngestionPipeline(embedder_fn=local_embedder.embed_texts)
paper_indices: Dict[str, HybridSearchIndex] = {}
answers_db: Dict[str, AnswerRecord] = {}
feedback_db: List[FeedbackRecord] = []
suggested_questions_cache: Dict[str, List[str]] = {}


class AskRequest(BaseModel):
    question: str
    stream: bool = False


class UploadResponse(BaseModel):
    paper_id: str
    job_id: str
    status: str
    title: str
    duplicate: bool = False
    message: Optional[str] = None


class PrivacyUpdateRequest(BaseModel):
    privacy_setting: str  # "allow_remote" or "local_only"


class NoteCreateRequest(BaseModel):
    anchor_type: str = "block"  # "block", "formula", "figure", "table", "ref"
    anchor_id: str
    text: str
    quote: str = ""
    color: str = "yellow"
    tags: List[str] = []


class NoteUpdateRequest(BaseModel):
    text: Optional[str] = None
    color: Optional[str] = None
    tags: Optional[List[str]] = None


class SupportCheckRequest(BaseModel):
    citing_statement: str
    cited_text: str


class CitationVerifyRequest(BaseModel):
    sentence: str
    reference_id: Optional[str] = None
    max_retries: int = 2


@router.post(
    "/papers", response_model=UploadResponse, status_code=status.HTTP_201_CREATED
)
async def upload_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accepts PDF upload, validates size, and kicks off resumable ingestion (FR-ING-01, 05).
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files are supported.",
        )

    paper_id = f"paper_{uuid.uuid4().hex[:10]}"
    job_id = f"job_{uuid.uuid4().hex[:8]}"

    uploads_dir = Path(settings.UPLOADS_DIR)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_path = uploads_dir / f"{paper_id}.pdf"

    # Save uploaded file
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run ingestion pipeline
    try:
        paper_doc = await pipeline.run(saved_path, paper_id)
        effective_paper_id = paper_doc.metadata.id
        chunks = pipeline.chunks_store.get(effective_paper_id, [])
        search_index = HybridSearchIndex(paper_doc, chunks, local_embedder)
        paper_indices[effective_paper_id] = search_index
        suggested_questions_cache.pop(effective_paper_id, None)
        is_duplicate = (effective_paper_id != paper_id)
        return UploadResponse(
            paper_id=effective_paper_id,
            job_id=job_id,
            status=paper_doc.metadata.status,
            title=paper_doc.metadata.title,
            duplicate=is_duplicate,
            message="Reused existing paper (duplicate content hash)" if is_duplicate else "Ingestion complete",
        )
    except Exception as exc:
        logger.error(f"Ingestion failed for {paper_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )


@router.get("/papers")
async def list_papers():
    """Lists all ingested papers with their processing status (FR-UI-01)."""
    return [
        {
            "id": doc.metadata.id,
            "title": doc.metadata.title,
            "authors": doc.metadata.authors,
            "total_pages": doc.metadata.total_pages,
            "status": doc.metadata.status,
            "warnings_count": len(doc.metadata.warnings),
        }
        for doc in pipeline.docs_store.values()
    ]


@router.get("/papers/{paper_id}")
async def get_paper(paper_id: str):
    """Returns paper details and parse-quality report (FR-ING-08)."""
    doc = pipeline.docs_store.get(paper_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Paper not found.")

    return {
        "metadata": doc.metadata,
        "blocks": doc.blocks,
        "blocks_count": len(doc.blocks),
        "figures_count": len(doc.figures),
        "tables_count": len(doc.tables),
        "references_count": len(doc.references),
        "parse_quality_report": {
            "warnings": doc.metadata.warnings,
            "has_figures": len(doc.figures) > 0,
            "has_references": len(doc.references) > 0,
        },
    }


@router.get("/papers/{paper_id}/figures", response_model=List[FigureItem])
async def get_paper_figures(paper_id: str):
    """Returns extracted figures with captions, crop paths, and summaries (FR-FIG-01 to 05)."""
    doc = pipeline.docs_store.get(paper_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return doc.figures


@router.get("/papers/{paper_id}/references", response_model=List[ReferenceItem])
async def get_paper_references(paper_id: str):
    """Returns references with resolution status and citing context (FR-REF-01 to 06)."""
    doc = pipeline.docs_store.get(paper_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return doc.references


def extract_suggested_questions(doc: PaperDoc) -> List[str]:
    """Generates targeted research exploration questions based on paper title, headings, formulas, figures, and citations."""
    title = (doc.metadata.title or "").strip()
    short_title = (title[:65] + "…") if len(title) > 65 else title
    questions: List[str] = [
        f"What is the primary contribution and hypothesis of \"{short_title}\"?"
        if short_title and short_title.lower() != "untitled paper"
        else "What is the primary contribution and hypothesis of this work?"
    ]

    # Extract clean technical section headings (from explicit heading blocks + section_path fallback)
    headings: List[str] = []
    seen = set()
    generic = {
        "introduction",
        "references",
        "bibliography",
        "conclusion",
        "conclusions",
        "abstract",
        "document",
        "fallback",
        "related work",
        "acknowledgements",
        "acknowledgments",
        "appendix",
        "original research",
        "research article",
        "review article",
        "preprint",
        "a preprint",
        "arxiv template",
    }

    def _add_heading(raw_h: str) -> None:
        h_clean = re.sub(r"^(\d+(\.\d+)*|[IVX]+\.)\s*", "", raw_h.strip()).strip()
        if h_clean and h_clean.lower() not in seen and 4 < len(h_clean) < 60:
            seen.add(h_clean.lower())
            headings.append(h_clean)

    for b in doc.blocks:
        if b.type == "heading" and len(b.text.strip()) > 3:
            _add_heading(b.text)
    if not headings:
        for b in doc.blocks:
            for seg in b.section_path or []:
                _add_heading(seg)

    technical_headings = [h for h in headings if h.lower() not in generic]

    if technical_headings:
        questions.append(f"How is {technical_headings[0]} formulated and implemented?")
    else:
        questions.append("What methodology or technical architecture is proposed in this paper?")

    # Empirical evaluations / experiments
    eval_headings = [
        h
        for h in technical_headings
        if any(
            k in h.lower()
            for k in ("eval", "experiment", "result", "benchmark", "study", "analysis", "ablation")
        )
    ]
    if eval_headings:
        questions.append(f"What key empirical results and benchmarks are reported in {eval_headings[0]}?")
    else:
        questions.append("What empirical evaluation benchmarks and baseline comparisons are reported?")

    # Formula-grounded question if paper contains extracted formulas
    if doc.formulas:
        eq = doc.formulas[0]
        eq_label = eq.number or eq.formula_id
        questions.append(f"Explain the derivation and role of Equation {eq_label} (page {eq.page}).")

    # Figure or second technical aspect
    if doc.figures:
        fig = doc.figures[0]
        questions.append(f"How does Figure {fig.figure_id} illustrate the system design or experimental setup?")
    elif len(technical_headings) > 1:
        questions.append(f"What key findings or details are discussed regarding {technical_headings[1]}?")

    # Cited-work comparison if references exist
    if doc.references:
        top_ref = next((r for r in doc.references if r.title and len(r.title) > 10), None)
        if top_ref and top_ref.title:
            ref_title = (top_ref.title[:50] + "…") if len(top_ref.title) > 50 else top_ref.title
            questions.append(f"How does this paper relate to cited work \"{ref_title}\" {top_ref.marker or ''}?".strip())

    # Deduplicate and cap at 5 curated paper-specific prompts
    deduped: List[str] = []
    seen_q = set()
    for q in questions:
        if q.lower() not in seen_q:
            seen_q.add(q.lower())
            deduped.append(q)

    if len(deduped) < 5:
        deduped.append("What are the primary limitations, failure modes, or threats to validity acknowledged?")

    return deduped[:5]


@router.get("/papers/{paper_id}/suggested-questions", response_model=List[str])
async def get_suggested_questions(paper_id: str):
    """Returns curated exploration questions for the paper (FR-UI-02)."""
    doc = pipeline.docs_store.get(paper_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Paper not found.")

    if paper_id in suggested_questions_cache:
        return suggested_questions_cache[paper_id]

    questions = extract_suggested_questions(doc)
    suggested_questions_cache[paper_id] = questions
    return questions


@router.post("/papers/{paper_id}/ask", response_model=AnswerRecord)
async def ask_question(paper_id: str, request: AskRequest):
    """
    Submits a research question and returns a grounded answer with evidence records (FR-QA-01).
    """
    doc = pipeline.docs_store.get(paper_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Paper not found.")

    search_index = paper_indices.get(paper_id)
    if not search_index:
        chunks = pipeline.chunks_store.get(paper_id, [])
        search_index = HybridSearchIndex(doc, chunks, local_embedder)
        paper_indices[paper_id] = search_index

    answer = await qa_orchestrator.answer_question(
        question=request.question,
        paper=doc,
        search_index=search_index,
    )
    answers_db[answer.answer_id] = answer
    return answer


@router.post("/papers/{paper_id}/ask/stream")
@router.get("/papers/{paper_id}/ask/stream")
async def ask_question_stream(
    paper_id: str,
    question: Optional[str] = None,
    request: Optional[AskRequest] = None,
):
    """
    Streams tokens and unverified claim markers in real-time, then upgrades to verified badges (FR-QA-01, Section 8.2).
    """
    query_text = (request.question if request else None) or question
    if not query_text:
        raise HTTPException(status_code=400, detail="Question is required.")

    doc = pipeline.docs_store.get(paper_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Paper not found.")

    search_index = paper_indices.get(paper_id)
    if not search_index:
        chunks = pipeline.chunks_store.get(paper_id, [])
        search_index = HybridSearchIndex(doc, chunks, local_embedder)
        paper_indices[paper_id] = search_index

    async def sse_stream():
        async for evt in qa_orchestrator.stream_answer(query_text, doc, search_index):
            if evt.get("type") == "final" and "answer" in evt:
                ans_obj = AnswerRecord.model_validate(evt["answer"])
                answers_db[ans_obj.answer_id] = ans_obj
            yield f"data: {json.dumps(evt)}\n\n"

    return StreamingResponse(sse_stream(), media_type="text/event-stream")


@router.get("/papers/{paper_id}/quality-report", response_model=ParseQualityReport)
async def get_paper_quality_report(paper_id: str):
    """Returns the comprehensive Stage 7 parse-quality report (Section 8.1, Stage 7)."""
    doc = pipeline.docs_store.get(paper_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Paper not found.")

    # Check disk cache
    q_file = Path(settings.DATA_DIR) / f"{paper_id}_quality.json"
    if q_file.exists():
        try:
            with open(q_file, "r", encoding="utf-8") as f:
                return ParseQualityReport.model_validate_json(f.read())
        except Exception:
            pass

    # Dynamic calculation
    tier_counts = {"verified": 0, "probable": 0, "unresolved": 0}
    for r in doc.references:
        t = r.tier if r.tier in tier_counts else "unresolved"
        tier_counts[t] += 1

    return ParseQualityReport(
        paper_id=paper_id,
        total_pages=doc.metadata.total_pages,
        blocks_count=len(doc.blocks),
        figures_count=len(doc.figures),
        tables_count=len(doc.tables),
        references_count=len(doc.references),
        tier_counts=tier_counts,
        warnings=doc.metadata.warnings,
        has_figures=len(doc.figures) > 0,
        has_references=len(doc.references) > 0,
        figure_summaries_pending=sum(
            1 for f in doc.figures if f.summary_status == "pending"
        ),
    )


@router.post("/papers/{paper_id}/privacy")
async def update_paper_privacy(paper_id: str, request: PrivacyUpdateRequest):
    """Updates privacy setting for a paper: allow_remote or local_only (NFR-PRIV-02)."""
    doc = pipeline.docs_store.get(paper_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Paper not found.")
    if request.privacy_setting not in ("allow_remote", "local_only"):
        raise HTTPException(
            status_code=400,
            detail="Invalid privacy setting. Must be 'allow_remote' or 'local_only'.",
        )
    doc.metadata.privacy_setting = request.privacy_setting  # type: ignore
    paper_doc_file = Path(settings.DATA_DIR) / f"{paper_id}_doc.json"
    if paper_doc_file.exists():
        with open(paper_doc_file, "w", encoding="utf-8") as f:
            f.write(doc.model_dump_json(indent=2))
    return {"paper_id": paper_id, "privacy_setting": doc.metadata.privacy_setting}


@router.post("/papers/{paper_id}/export")
async def export_paper_artifacts(paper_id: str, answer_id: Optional[str] = None):
    """Exports answer with verified citations as Markdown and BibTeX (FR-REF-07, Section 6)."""
    doc = pipeline.docs_store.get(paper_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Paper not found.")

    answer = answers_db.get(answer_id) if answer_id else None

    # Generate unified BibTeX
    bibtex_entries = [r.bibtex for r in doc.references if r.bibtex]
    full_bibtex = (
        "\n\n".join(bibtex_entries)
        if bibtex_entries
        else f"@article{{{doc.metadata.id},\n  title={{{doc.metadata.title}}},\n  year={{{doc.metadata.year or 2026}}}\n}}"
    )

    md_content = f"# {doc.metadata.title}\n\n"
    if answer:
        md_content += f"## Research Question\n{answer.question}\n\n## Grounded Answer\n{answer.answer_markdown}\n\n### Verified Evidence Claims\n"
        for c in answer.claims:
            md_content += f"- **[{c.verdict.upper()}]** {c.claim}\n"
    else:
        md_content += f"**Authors:** {', '.join(doc.metadata.authors)}\n\n**Abstract:**\n{doc.metadata.abstract or 'N/A'}\n"

    return {
        "paper_id": paper_id,
        "markdown": md_content,
        "bibtex": full_bibtex,
        "references_count": len(doc.references),
    }


# ── Section 7 Agent Tool Contract API Endpoints ──────────────────────────────


@router.get("/papers/{paper_id}/tools/search-text")
async def tool_search_text(paper_id: str, query: str, k: int = 5):
    """Section 7 Agent Tool: search_text(paper_id, query, k)"""
    search_index = paper_indices.get(paper_id)
    return tool_layer.search_text(paper_id, query, k=k, search_index=search_index)


@router.get("/papers/{paper_id}/tools/search-figures")
async def tool_search_figures(paper_id: str, query: str, k: int = 3):
    """Section 7 Agent Tool: search_figures(paper_id, query, k)"""
    return tool_layer.search_figures(paper_id, query, k=k)


@router.get("/papers/{paper_id}/tools/list-references")
async def tool_list_references(paper_id: str):
    """Section 7 Agent Tool: list_references(paper_id)"""
    return tool_layer.list_references(paper_id)


@router.get("/references/{ref_id}")
async def tool_get_reference(ref_id: str, paper_id: Optional[str] = None):
    """Section 7 Agent Tool: get_reference(reference_id)"""
    res = tool_layer.get_reference(ref_id, paper_id=paper_id)
    if not res:
        raise HTTPException(status_code=404, detail="Reference not found.")
    return res


@router.get("/references/{ref_id}/citing-context")
async def tool_get_citing_context(ref_id: str, paper_id: Optional[str] = None):
    """Section 7 Agent Tool: get_citing_context(reference_id)"""
    return tool_layer.get_citing_context(ref_id, paper_id=paper_id)


@router.get("/references/{ref_id}/open-access")
async def tool_find_open_access(ref_id: str, paper_id: Optional[str] = None):
    """Section 7 Agent Tool: find_open_access(reference_id)"""
    return await tool_layer.find_open_access(ref_id, paper_id=paper_id)


@router.get("/references/{ref_id}/retraction")
async def tool_retraction_check(ref_id: str, paper_id: Optional[str] = None):
    """Section 7 Agent Tool: retraction_check(reference_id)"""
    return tool_layer.retraction_check(ref_id, paper_id=paper_id)


@router.get("/answers/{answer_id}", response_model=AnswerRecord)
async def get_answer(answer_id: str):
    """Returns a persisted answer with evidence cards and model routing audit (FR-VER-05)."""
    answer = answers_db.get(answer_id)
    if not answer:
        raise HTTPException(status_code=404, detail="Answer not found.")
    return answer


@router.post(
    "/papers/{paper_id}/references/{ref_id}/confirm",
    response_model=ReferenceItem,
)
async def confirm_reference(paper_id: str, ref_id: str):
    """Promotes a Probable reference to Verified upon researcher confirmation (FR-REF-08, 10)."""
    doc = pipeline.docs_store.get(paper_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Paper not found.")
    target_ref = next((r for r in doc.references if r.reference_id == ref_id), None)
    if not target_ref:
        raise HTTPException(status_code=404, detail="Reference not found.")
    target_ref.tier = "verified"
    target_ref.user_confirmed = True
    target_ref.match_score = 1.0
    target_ref.resolution_status = "resolved"
    return target_ref


@router.get("/papers/{paper_id}/references/{ref_id}/open-access")
async def get_reference_open_access(paper_id: str, ref_id: str):
    """Returns free legal copy URL (Unpaywall/arXiv/DOI) for reference (FR-REF-09)."""
    doc = pipeline.docs_store.get(paper_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Paper not found.")
    target_ref = next((r for r in doc.references if r.reference_id == ref_id), None)
    if not target_ref:
        raise HTTPException(status_code=404, detail="Reference not found.")
    oa_url = target_ref.open_access_url or (
        f"https://doi.org/{target_ref.doi}" if target_ref.doi else None
    )
    return {
        "reference_id": ref_id,
        "title": target_ref.title,
        "open_access_url": oa_url,
        "has_free_copy": bool(oa_url),
    }


@router.get("/answers/{answer_id}/provenance")
async def get_answer_provenance(answer_id: str):
    """Returns detailed provenance and verification checks matrix for an answer (FR-VER-05)."""
    answer = answers_db.get(answer_id)
    if not answer:
        raise HTTPException(status_code=404, detail="Answer not found.")
    return answer.provenance or {
        "answer_id": answer.answer_id,
        "is_verified": answer.is_verified,
        "routed_via": answer.routed_via or "local",
        "claims_count": len(answer.claims),
    }


@router.post("/papers/{paper_id}/verify-claim", response_model=ClaimRecord)
async def verify_claim(paper_id: str, claim: ClaimRecord):
    """Independently verifies a single candidate claim against paper text and bounding boxes (FR-VER-01 to 04)."""
    doc = pipeline.docs_store.get(paper_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Paper not found.")
    verified_list = await verifier_engine.verify_answer([claim], doc)
    return verified_list[0] if verified_list else claim


@router.post("/feedback", status_code=status.HTTP_200_OK)
async def submit_feedback(feedback: FeedbackRecord):
    """Stores user ratings or citation error reports for gold-set improvement (FR-UI-08)."""
    feedback_db.append(feedback)
    return {"status": "recorded", "feedback_count": len(feedback_db)}


# ── Addendum v1.3: Evidence Trail, Formulas, Notes & Reference Reader ────────

@router.get("/answers/{answer_id}/trail", response_model=List[EvidenceTrailItem])
async def get_answer_trail(answer_id: str):
    """
    Returns the assembled Evidence Trail for an answer (Addendum v1.3 Section 6.3).
    Includes paragraphs, formulas, figures/tables, and 'cited-here' references.
    """
    answer = answers_db.get(answer_id)
    if not answer:
        raise HTTPException(status_code=404, detail="Answer not found.")
    doc = pipeline.docs_store.get(answer.paper_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Paper not found.")
    from backend.app.orchestrator.agent import build_evidence_trail
    return build_evidence_trail(answer, doc)


@router.get("/papers/{paper_id}/formulas", response_model=List[FormulaItem])
async def get_paper_formulas(paper_id: str):
    """Returns all displayed formulas extracted from paper (Addendum v1.3 Section 6.3)."""
    doc = pipeline.docs_store.get(paper_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return doc.formulas


@router.get("/papers/{paper_id}/formulas/{formula_id}", response_model=FormulaItem)
async def get_paper_formula(paper_id: str, formula_id: str):
    """Returns a single formula record with LaTeX, crop, and confidence (Addendum v1.3 Section 6.3)."""
    doc = pipeline.docs_store.get(paper_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Paper not found.")
    for f in doc.formulas:
        if f.formula_id == formula_id:
            return f
    raise HTTPException(status_code=404, detail=f"Formula {formula_id} not found.")


@router.get("/references/{reference_id}/reader", response_model=ReferenceReaderPayload)
async def get_reference_reader(reference_id: str, paper_id: Optional[str] = None):
    """Returns Reference Reader payload (abstract, OA status, passages) (Addendum v1.3 Section 6.3)."""
    ref, _ = tool_layer._find_ref(reference_id, paper_id)
    if not ref:
        raise HTTPException(status_code=404, detail=f"Reference {reference_id} not found.")
    from backend.app.services.reference_reader import build_reader_payload
    return build_reader_payload(ref)


@router.post("/references/{reference_id}/fetch")
async def fetch_reference_fulltext(
    reference_id: str, paper_id: Optional[str] = None, citing_sentence: str = ""
):
    """
    Fetches open-access full text of cited reference (Addendum v1.3 Section 6.3).
    Privacy check: local_only papers block remote fetch.
    """
    ref, doc = tool_layer._find_ref(reference_id, paper_id)
    if not ref:
        raise HTTPException(status_code=404, detail=f"Reference {reference_id} not found.")
    if doc and doc.metadata.privacy_setting == "local_only":
        raise HTTPException(
            status_code=403,
            detail="Local-only privacy mode is active. Remote paper fetching is blocked."
        )
    from backend.app.services.reference_reader import fetch_and_cache
    status_str, passages = await fetch_and_cache(ref, citing_sentence=citing_sentence)
    return {
        "reference_id": reference_id,
        "fetch_status": status_str,
        "passages": [p.model_dump() for p in passages],
    }


@router.get("/references/{reference_id}/fetch/status")
async def get_reference_fetch_status(reference_id: str, paper_id: Optional[str] = None):
    """Returns current fetch/parsing status for a reference."""
    ref, _ = tool_layer._find_ref(reference_id, paper_id)
    if not ref:
        raise HTTPException(status_code=404, detail=f"Reference {reference_id} not found.")
    from backend.app.services.reference_reader import build_reader_payload
    payload = build_reader_payload(ref)
    return {
        "reference_id": reference_id,
        "fetch_status": payload.fetch_status,
        "fetch_message": payload.fetch_message,
        "passages_count": len(payload.passages),
    }


@router.post("/references/{reference_id}/support-check")
async def reference_support_check(reference_id: str, req: SupportCheckRequest):
    """
    Optional check: does the cited work passage support author's citing statement?
    Uses pinned verifier model (Addendum v1.3 Section 6.3).
    """
    res = await verifier_engine.check_cited_work_support(
        req.citing_statement, req.cited_text
    )
    return {"reference_id": reference_id, **res}


@router.post(
    "/papers/{paper_id}/verify-citation",
    response_model=CitationVerificationResult,
)
async def verify_citation_agentic(paper_id: str, req: CitationVerifyRequest):
    """
    4-Step Agentic Citation Verification Pipeline:
    1. Citation Extraction & Resolution
    2. Paper B Retrieval (Semantic Scholar, OpenAlex, arXiv, Unpaywall + Paywall Abstract Fallback)
    3. Semantic Search & LLM Verification with Self-Correction Loop
    4. Structured Snippet, Page/Section, Confidence & Explanation
    """
    doc = pipeline.docs_store.get(paper_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return await citation_verifier_agent.verify_citation(
        paper=doc,
        sentence=req.sentence,
        reference_id=req.reference_id,
        max_retries=req.max_retries,
    )


@router.post("/papers/{paper_id}/verify-citation/stream")
async def stream_verify_citation_agentic(paper_id: str, req: CitationVerifyRequest):
    """
    Streams live status steps of the 4-step Agentic Citation Verification workflow
    over Server-Sent Events (SSE) so the UI remains responsive during retrieval & parsing.
    """
    doc = pipeline.docs_store.get(paper_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Paper not found.")

    async def sse_generator():
        async for evt in citation_verifier_agent.stream_verify_citation(
            paper=doc,
            sentence=req.sentence,
            reference_id=req.reference_id,
            max_retries=req.max_retries,
        ):
            yield f"data: {json.dumps(evt)}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


# ── Notes Service Endpoints ──────────────────────────────────────────────────

@router.get("/papers/{paper_id}/notes", response_model=List[NoteRecord])
async def get_paper_notes(paper_id: str):
    """Lists all researcher notes for a paper (Addendum v1.3 Section 6.3)."""
    from backend.app.services.notes_service import list_notes
    return list_notes(paper_id)


@router.post("/papers/{paper_id}/notes", response_model=NoteRecord)
async def create_paper_note(paper_id: str, req: NoteCreateRequest):
    """Creates a new note anchored to a block, formula, figure, or ref (Addendum v1.3)."""
    from backend.app.services.notes_service import create_note
    return create_note(
        paper_id=paper_id,
        anchor_type=req.anchor_type,
        anchor_id=req.anchor_id,
        text=req.text,
        quote=req.quote,
        color=req.color,
        tags=req.tags,
    )


@router.patch("/notes/{note_id}", response_model=NoteRecord)
@router.put("/notes/{note_id}", response_model=NoteRecord)
async def update_note_endpoint(note_id: str, req: NoteUpdateRequest, paper_id: Optional[str] = None):
    """Updates a note's text, color, or tags (Addendum v1.3 Section 6.3)."""
    from backend.app.services.notes_service import update_note
    target_paper_id = paper_id
    if not target_paper_id:
        # Search across all papers
        for pid in pipeline.docs_store.keys():
            updated = update_note(pid, note_id, text=req.text, color=req.color, tags=req.tags)
            if updated:
                return updated
        raise HTTPException(status_code=404, detail=f"Note {note_id} not found.")
    updated = update_note(target_paper_id, note_id, text=req.text, color=req.color, tags=req.tags)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Note {note_id} not found.")
    return updated


@router.delete("/notes/{note_id}")
async def delete_note_endpoint(note_id: str, paper_id: Optional[str] = None):
    """Deletes a note (Addendum v1.3 Section 6.3)."""
    from backend.app.services.notes_service import delete_note
    if paper_id:
        deleted = delete_note(paper_id, note_id)
        if deleted:
            return {"status": "deleted", "note_id": note_id}
    else:
        for pid in pipeline.docs_store.keys():
            if delete_note(pid, note_id):
                return {"status": "deleted", "note_id": note_id}
    return {"status": "deleted", "note_id": note_id}


@router.get("/papers/{paper_id}/notes/export")
async def export_notes(paper_id: str):
    """Exports paper notes in Markdown format (Addendum v1.3 Section 6.3)."""
    from backend.app.services.notes_service import export_notes_markdown
    md = export_notes_markdown(paper_id)
    return {"paper_id": paper_id, "markdown": md}


@router.get("/admin/llm-usage")
async def get_llm_usage():
    """Returns LLM usage statistics, token counts, and routed providers (FR-OPS-02)."""
    return {
        "gateway_url": settings.GATEWAY_BASE_URL,
        "model_mode": settings.MODEL_MODE,
        "embedding_backend": settings.EMBEDDING_BACKEND,
        "embedding_model": (
            settings.EMBEDDING_API_MODEL
            if settings.EMBEDDING_BACKEND == "gateway"
            else "local:hash-ngram"
        ),
        "total_answers_generated": len(answers_db),
    }


@router.get("/health")
async def health_check():
    """Service health endpoint (FR-OPS-03)."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "gateway_configured": bool(settings.GATEWAY_BASE_URL),
        "indexed_papers_count": len(pipeline.docs_store),
    }
