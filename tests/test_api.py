"""
API route unit tests (SRS 8.2, FR-OPS-03).
Tests endpoint logic directly without socket-bound ASGI transports.
"""

import asyncio
import pytest
from backend.app.api.routes import (
    answers_db,
    confirm_reference,
    extract_suggested_questions,
    get_answer_provenance,
    get_llm_usage,
    get_reference_open_access,
    health_check,
    list_papers,
    pipeline,
    verify_claim,
)
from backend.app.verifier.engine import verifier_engine
from backend.app.orchestrator.agent import generate_dynamic_followups
from backend.app.schemas.domain import (
    AnswerRecord,
    ClaimRecord,
    EvidenceItem,
    FigureItem,
    PaperDoc,
    PaperMetadata,
    ReferenceItem,
    TextBlock,
)


def test_health_endpoint():
    coro = health_check()
    try:
        data = coro.send(None)
    except StopIteration as e:
        data = e.value
    assert data["status"] == "healthy"
    assert "version" in data


def test_papers_list_endpoint():
    coro = list_papers()
    try:
        data = coro.send(None)
    except StopIteration as e:
        data = e.value
    assert isinstance(data, list)


def test_llm_usage_endpoint():
    coro = get_llm_usage()
    try:
        data = coro.send(None)
    except StopIteration as e:
        data = e.value
    assert "gateway_url" in data
    assert "model_mode" in data


def test_suggested_questions_generation():
    doc = PaperDoc(
        metadata=PaperMetadata(
            id="p_test_sq",
            content_hash="h_sq",
            title="Transformer Networks for Machine Translation",
            authors=["Vaswani et al."],
        ),
        blocks=[
            TextBlock(
                block_id="b1",
                page=1,
                bbox=[0, 0, 100, 100],
                type="heading",
                text="3. Multi-Head Self-Attention Architecture",
                section_path=["3. Architecture"],
            ),
            TextBlock(
                block_id="b2",
                page=2,
                bbox=[0, 0, 100, 100],
                type="heading",
                text="5. Experimental Evaluation on WMT 2014",
                section_path=["5. Evaluation"],
            ),
        ],
        figures=[
            FigureItem(
                figure_id="fig1",
                page=1,
                bbox=[0, 0, 100, 100],
                crop_path="p_test_sq/fig1.png",
                caption="Figure 1: The Transformer model architecture.",
            )
        ],
    )
    questions = extract_suggested_questions(doc)
    assert len(questions) >= 4
    # Check that section-specific and figure-specific questions were generated
    assert any("Multi-Head Self-Attention" in q for q in questions)
    assert any("WMT 2014" in q or "Experimental Evaluation" in q for q in questions)
    assert any("Figure fig1" in q for q in questions)


def test_dynamic_followup_generation():
    doc = PaperDoc(
        metadata=PaperMetadata(
            id="p_test_dyn",
            content_hash="h_dyn",
            title="Epistemic Agents for Cybersecurity",
            authors=["Chaudhary et al."],
        ),
        blocks=[
            TextBlock(
                block_id="b1",
                page=1,
                bbox=[0, 0, 100, 100],
                type="heading",
                text="3. Epistemic Self-Model Architecture",
                section_path=["3. Architecture"],
            ),
            TextBlock(
                block_id="b2",
                page=2,
                bbox=[0, 0, 100, 100],
                type="heading",
                text="4. Benchmark Evaluations on Phishing Triage",
                section_path=["4. Evaluation"],
            ),
        ],
        figures=[
            FigureItem(
                figure_id="fig1",
                page=1,
                bbox=[0, 0, 100, 100],
                crop_path="p_test_dyn/fig1.png",
                caption="Figure 1: Epistemic self-model gating diagram.",
            )
        ],
    )

    # 1. User asked about overview: should suggest questions probing architecture and evaluation
    followups_overview = generate_dynamic_followups(
        question="What is the main proposal of this paper?",
        paper=doc,
        claims=[],
    )
    assert len(followups_overview) >= 2
    assert any("Epistemic Self-Model" in q for q in followups_overview)

    # 2. User asked about method: should suggest baseline comparisons and ablation questions
    followups_method = generate_dynamic_followups(
        question="How does the epistemic self-model architecture operate?",
        paper=doc,
        claims=[],
    )
    assert len(followups_method) >= 2
    assert any("baseline" in q.lower() or "ablation" in q.lower() or "phishing" in q.lower() for q in followups_method)


@pytest.mark.asyncio
async def test_confirm_reference_endpoint():
    # Setup test paper with a probable reference in pipeline.docs_store
    paper_id = "test_paper_refs"
    ref = ReferenceItem(
        reference_id="ref_42",
        marker="[42]",
        raw_text="Vaswani et al. 2017.",
        tier="probable",
        match_score=0.78,
        user_confirmed=False,
    )
    doc = PaperDoc(
        metadata=PaperMetadata(
            id=paper_id,
            content_hash="h42",
            title="Transformer Paper",
        ),
        references=[ref],
    )
    pipeline.docs_store[paper_id] = doc

    # Call confirm_reference endpoint
    confirmed = await confirm_reference(paper_id=paper_id, ref_id="ref_42")
    assert confirmed.tier == "verified"
    assert confirmed.user_confirmed is True
    assert confirmed.match_score == 1.0


@pytest.mark.asyncio
async def test_open_access_endpoint():
    paper_id = "test_paper_oa"
    ref = ReferenceItem(
        reference_id="ref_oa",
        marker="[1]",
        raw_text="Sample Open Access Ref",
        doi="10.1234/test.doi",
        open_access_url="https://arxiv.org/pdf/1706.03762.pdf",
    )
    doc = PaperDoc(
        metadata=PaperMetadata(id=paper_id, content_hash="hoa", title="OA Paper"),
        references=[ref],
    )
    pipeline.docs_store[paper_id] = doc

    oa_data = await get_reference_open_access(paper_id=paper_id, ref_id="ref_oa")
    assert oa_data["reference_id"] == "ref_oa"
    assert oa_data["has_free_copy"] is True
    assert "arxiv.org" in oa_data["open_access_url"]


@pytest.mark.asyncio
async def test_answer_provenance_endpoint():
    ans_id = "ans_prov_test"
    answers_db[ans_id] = AnswerRecord(
        answer_id=ans_id,
        paper_id="p_test",
        question="What is the latency?",
        answer_markdown="Latency is minimal.",
        provenance={
            "correlation_id": "corr_999",
            "model": "llama-3.3-70b-versatile",
            "latency_ms": 350,
            "checks": {"ids_exist": True, "doi_resolves": True, "quote_match": True, "bbox_valid": True},
        },
    )

    prov = await get_answer_provenance(answer_id=ans_id)
    assert prov["correlation_id"] == "corr_999"
    assert prov["latency_ms"] == 350
    assert prov["checks"]["ids_exist"] is True


@pytest.mark.asyncio
async def test_verify_claim_endpoint():
    verifier_engine.adapter.mock_mode = True
    try:
        paper_id = "test_paper_claim"
        doc = PaperDoc(
            metadata=PaperMetadata(id=paper_id, content_hash="hc", title="Claim Check Paper"),
            blocks=[
                TextBlock(
                    block_id="b1",
                    page=1,
                    bbox=[10, 10, 100, 100],
                    type="paragraph",
                    text="The proposed model achieves 95% accuracy.",
                )
            ],
        )
        pipeline.docs_store[paper_id] = doc

        claim = ClaimRecord(
            claim_id="c_single",
            claim="The model achieves 95% accuracy.",
            evidence=[
                EvidenceItem(
                    type="text",
                    block_id="b1",
                    page=1,
                    bbox=[10, 10, 100, 100],
                    quote="achieves 95% accuracy",
                )
            ],
        )

        verified = await verify_claim(paper_id=paper_id, claim=claim)
        assert verified.checks.ids_exist is True
        assert verified.checks.quote_match is True
        assert verified.checks.bbox_valid is True
        assert verified.verdict in ("supported", "partial")
    finally:
        verifier_engine.adapter.mock_mode = False


