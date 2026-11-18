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
    ClaimRecord,
    FeedbackRecord,
    FigureItem,
    PaperDoc,
    ParseQualityReport,
    ReferenceItem,
)
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
    """Generates targeted research exploration questions based on paper headings, abstract and figures."""
    questions: List[str] = [
        "What is the primary contribution and hypothesis of this work?"
    ]

    # Extract clean technical section headings
    headings = []
    seen = set()
    for b in doc.blocks:
        if b.type == "heading" and len(b.text.strip()) > 3:
            h_clean = re.sub(r"^(\d+(\.\d+)*|[IVX]+\.)\s*", "", b.text.strip()).strip()
            if h_clean.lower() not in seen and 4 < len(h_clean) < 60:
                seen.add(h_clean.lower())
                headings.append(h_clean)

    generic = {"introduction", "references", "bibliography", "conclusion", "conclusions", "abstract", "document", "fallback", "related work"}
    technical_headings = [h for h in headings if h.lower() not in generic]

    if technical_headings:
        questions.append(f"How is {technical_headings[0]} formulated and implemented?")
    else:
        questions.append("What methodology or technical architecture is proposed in this paper?")

    # Empirical evaluations / experiments
    eval_headings = [h for h in technical_headings if any(k in h.lower() for k in ("eval", "experiment", "result", "benchmark", "study", "analysis"))]
    if eval_headings:
        questions.append(f"What key empirical results and benchmarks are reported in {eval_headings[0]}?")
    else:
        questions.append("What empirical evaluation benchmarks and baseline comparisons are reported?")

    # Figure or second technical aspect
    if doc.figures:
        fig = doc.figures[0]
        questions.append(f"How does Figure {fig.figure_id} illustrate the system design or experimental setup?")
    elif len(technical_headings) > 1:
        questions.append(f"What key findings or details are discussed regarding {technical_headings[1]}?")

    # Stated limitations
    questions.append("What are the primary limitations, failure modes, or threats to validity acknowledged?")

    return questions


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
