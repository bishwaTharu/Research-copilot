"""
Tests for Hybrid GROBID + MinerU Architecture (Steps 1, 2, and 3).
"""

import pytest
from pathlib import Path
from backend.app.schemas.domain import (
    PaperDoc,
    PaperMetadata,
    ReferenceItem,
    TextBlock,
)
from backend.app.ingestion.pipeline import IngestionPipeline


def test_step3_chunk_metadata_and_citation_linkage():
    """
    Verifies that Step 3 attaches the GROBID paper ID, metadata,
    and exact bibliographic references to every MinerU text chunk.
    """
    # Step 1 mock: GROBID outputs
    grobid_metadata = PaperMetadata(
        id="paper_hybrid_001",
        content_hash="hash123",
        title="Attention Is All You Need",
        authors=["Vaswani et al."],
        year=2017,
        doi="10.48550/arXiv.1706.03762",
    )
    grobid_references = [
        ReferenceItem(
            reference_id="r1",
            marker="[1]",
            raw_text="Hochreiter & Schmidhuber. Long short-term memory. Neural computation, 1997.",
            title="Long short-term memory",
            authors=["S. Hochreiter", "J. Schmidhuber"],
            year=1997,
            doi="10.1162/neco.1997.9.8.1735",
            venue="Neural Computation",
            resolution_status="resolved",
        ),
        ReferenceItem(
            reference_id="r2",
            marker="[2]",
            raw_text="Bahdanau et al. Neural machine translation by jointly learning to align and translate. ICLR 2015.",
            title="Neural machine translation by jointly learning to align and translate",
            authors=["D. Bahdanau", "K. Cho", "Y. Bengio"],
            year=2015,
            venue="ICLR",
            resolution_status="unresolved",
        ),
    ]

    # Step 2 mock: MinerU blocks with equations and citation markers
    mineru_blocks = [
        TextBlock(
            block_id="b1",
            page=1,
            bbox=[0, 0, 100, 100],
            type="heading",
            text="1. Introduction",
            section_path=["1. Introduction"],
        ),
        TextBlock(
            block_id="b2",
            page=1,
            bbox=[0, 100, 100, 200],
            type="paragraph",
            text="Recurrent models typically factor computation along symbol positions [1]. Attention mechanisms allow modeling dependencies [2].",
            section_path=["1. Introduction"],
        ),
        TextBlock(
            block_id="b3",
            page=1,
            bbox=[0, 200, 100, 300],
            type="equation",
            text="$$ Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}})V $$",
            section_path=["1. Introduction"],
        ),
    ]

    paper_doc = PaperDoc(
        metadata=grobid_metadata,
        blocks=mineru_blocks,
        references=grobid_references,
    )

    pipeline = IngestionPipeline()
    chunks = pipeline.link_and_chunk(paper_doc, max_chunk_words=100)

    assert len(chunks) >= 1
    chunk = chunks[0]

    # 1. GROBID Paper ID attached
    assert chunk.grobid_paper_id == "paper_hybrid_001"
    assert chunk.paper_id == "paper_hybrid_001"

    # 2. GROBID Metadata attached
    assert chunk.paper_metadata is not None
    assert chunk.paper_metadata["title"] == "Attention Is All You Need"
    assert chunk.paper_metadata["doi"] == "10.48550/arXiv.1706.03762"

    # 3. Exact bibliographic details linked
    ref_ids = [ref["reference_id"] for ref in chunk.linked_references]
    assert "r1" in ref_ids
    assert "r2" in ref_ids

    # Verify structured fields in linked citation
    r1_data = next(r for r in chunk.linked_references if r["reference_id"] == "r1")
    assert r1_data["title"] == "Long short-term memory"
    assert r1_data["doi"] == "10.1162/neco.1997.9.8.1735"
    assert r1_data["year"] == 1997

