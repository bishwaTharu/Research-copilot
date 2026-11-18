"""
Unit tests for Verifier Engine (FR-VER-01 to FR-VER-04).
Tests deterministic checks and fault injection (fake ID, altered quote, invalid bbox).
"""

import pytest
from backend.app.schemas.domain import (
    ClaimRecord,
    EvidenceItem,
    PaperDoc,
    PaperMetadata,
    TextBlock,
)
from backend.app.verifier.engine import VerifierEngine


@pytest.fixture
def sample_paper() -> PaperDoc:
    metadata = PaperMetadata(
        id="p_ver_1",
        content_hash="hash123",
        title="Verification Test Paper",
        total_pages=5,
        status="ready",
    )
    blocks = [
        TextBlock(
            block_id="b1",
            page=1,
            bbox=[50.0, 50.0, 550.0, 100.0],
            type="heading",
            text="1. Introduction",
            section_path=["1. Introduction"],
        ),
        TextBlock(
            block_id="b2",
            page=2,
            bbox=[50.0, 150.0, 550.0, 250.0],
            type="paragraph",
            text="Our system improves precision by 25% over the baseline model.",
            section_path=["2. Results"],
        ),
    ]
    return PaperDoc(metadata=metadata, blocks=blocks)


def test_deterministic_checks_valid_evidence(sample_paper):
    verifier = VerifierEngine()
    claim = ClaimRecord(
        claim_id="c1",
        claim="The system improves precision by 25%.",
        evidence=[
            EvidenceItem(
                type="text",
                block_id="b2",
                page=2,
                bbox=[50.0, 150.0, 550.0, 250.0],
                quote="improves precision by 25%",
            )
        ],
    )
    checks = verifier.run_deterministic_checks(claim, sample_paper)
    assert checks.ids_exist is True
    assert checks.quote_match is True
    assert checks.bbox_valid is True


def test_fault_injection_fake_id(sample_paper):
    """Injected fake block ID must be caught (FR-VER-01)."""
    verifier = VerifierEngine()
    claim = ClaimRecord(
        claim_id="c_fake",
        claim="An unsupported assertion with fabricated block ID.",
        evidence=[
            EvidenceItem(
                type="text",
                block_id="b999_fake",  # Fake ID
                page=2,
                bbox=[50.0, 50.0, 500.0, 100.0],
                quote="some quote",
            )
        ],
    )
    checks = verifier.run_deterministic_checks(claim, sample_paper)
    assert checks.ids_exist is False


def test_fault_injection_altered_quote(sample_paper):
    """Injected mismatched quote text must be caught (FR-VER-01)."""
    verifier = VerifierEngine()
    claim = ClaimRecord(
        claim_id="c_altered",
        claim="Precision improved by 99%.",
        evidence=[
            EvidenceItem(
                type="text",
                block_id="b2",
                page=2,
                bbox=[50.0, 150.0, 550.0, 250.0],
                quote="improves precision by 99%",  # Altered quote: actual text is 25%
            )
        ],
    )
    checks = verifier.run_deterministic_checks(claim, sample_paper)
    assert checks.ids_exist is True
    assert checks.quote_match is False


def test_fault_injection_invalid_bbox(sample_paper):
    """Injected negative bounding box must be caught (FR-VER-01)."""
    verifier = VerifierEngine()
    claim = ClaimRecord(
        claim_id="c_bbox",
        claim="Invalid bounding box claim.",
        evidence=[
            EvidenceItem(
                type="text",
                block_id="b2",
                page=2,
                bbox=[-100.0, -50.0, 50.0, 50.0],  # Negative coordinates
                quote="improves precision by 25%",
            )
        ],
    )
    checks = verifier.run_deterministic_checks(claim, sample_paper)
    assert checks.bbox_valid is False

