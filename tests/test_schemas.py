"""
Unit tests for domain schemas and bounding box calculations (ADR-2, FR-ING-04).
"""

import pytest
from backend.app.schemas.domain import (
    AnswerRecord,
    BoundingBox,
    ClaimRecord,
    EvidenceItem,
    PaperDoc,
    PaperMetadata,
    ReferenceItem,
    TextBlock,
    VerificationChecks,
)


def test_bounding_box_creation():
    bbox = BoundingBox(x0=10.0, y0=20.0, x1=200.0, y1=300.0)
    assert bbox.to_list() == [10.0, 20.0, 200.0, 300.0]
    
    from_list = BoundingBox.from_list([5.0, 15.0, 50.0, 80.0])
    assert from_list.x0 == 5.0
    assert from_list.y1 == 80.0


def test_bounding_box_invalid():
    with pytest.raises(ValueError):
        BoundingBox.from_list([10.0, 20.0])


def test_paper_doc_validation():
    metadata = PaperMetadata(
        id="test_paper_1",
        content_hash="abc123hash",
        title="Sample Paper on AI",
        total_pages=3,
        status="ready",
    )
    block1 = TextBlock(
        block_id="b1",
        page=1,
        bbox=[0.0, 0.0, 500.0, 100.0],
        type="heading",
        text="1. Introduction",
        section_path=["1. Introduction"],
    )
    doc = PaperDoc(metadata=metadata, blocks=[block1])
    assert doc.metadata.id == "test_paper_1"
    assert len(doc.blocks) == 1
    assert doc.blocks[0].block_id == "b1"


def test_claim_record_structure():
    ev = EvidenceItem(
        type="text",
        block_id="b1",
        page=1,
        bbox=[0.0, 0.0, 500.0, 100.0],
        quote="Introduction to AI",
    )
    claim = ClaimRecord(
        claim_id="c1",
        claim="The paper introduces AI fundamentals.",
        evidence=[ev],
        verdict="unverified",
        checks=VerificationChecks(ids_exist=True, quote_match=True, bbox_valid=True),
    )
    assert claim.claim_id == "c1"
    assert len(claim.evidence) == 1
    assert claim.checks.ids_exist is True


def test_reference_item_trust_tiers_and_fields():
    ref_verified = ReferenceItem(
        reference_id="r1",
        marker="[1]",
        raw_text="Vaswani et al. Attention is all you need. NeurIPS 2017.",
        doi="10.5555/3295222.3295349",
        title="Attention is all you need",
        authors=["Vaswani, Ashish", "Shazeer, Noam"],
        year=2017,
        venue="NeurIPS",
        tier="verified",
        match_score=0.98,
        open_access_url="https://arxiv.org/abs/1706.03762",
        is_retracted=False,
        bibtex="@article{vaswani2017_r1, title={Attention is all you need}}",
        citing_sentences=["We use multi-head attention as defined in [1]."],
    )
    assert ref_verified.tier == "verified"
    assert ref_verified.match_score == 0.98
    assert ref_verified.is_retracted is False
    assert len(ref_verified.citing_sentences) == 1

    ref_probable = ReferenceItem(
        reference_id="r2",
        marker="[2]",
        raw_text="Brown et al. Language Models are Few-Shot Learners. 2020.",
        title="Language Models are Few-Shot Learners",
        tier="probable",
        match_score=0.82,
        user_confirmed=False,
    )
    assert ref_probable.tier == "probable"
    assert ref_probable.user_confirmed is False


def test_answer_record_provenance():
    answer = AnswerRecord(
        answer_id="ans_test",
        paper_id="paper_test",
        question="What is the architecture?",
        answer_markdown="### Overview\nThe architecture uses multi-head attention.",
        claims=[],
        provenance={
            "correlation_id": "corr_12345",
            "model": "llama-3.3-70b-versatile",
            "latency_ms": 420,
            "checks": {
                "ids_exist": True,
                "doi_resolves": True,
                "quote_match": True,
                "bbox_valid": True,
            },
            "what_left_machine": "None (Local compute only)",
        },
    )
    assert answer.provenance["correlation_id"] == "corr_12345"
    assert answer.provenance["checks"]["ids_exist"] is True
    assert answer.provenance["latency_ms"] == 420

