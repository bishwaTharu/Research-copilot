"""
Unit tests for Section-aware Chunking (FR-QA-03).
Chunks must never cross section boundaries without being tracked.
"""

from backend.app.ingestion.pipeline import IngestionPipeline
from backend.app.schemas.domain import PaperDoc, PaperMetadata, TextBlock


def test_chunking_preserves_section_boundaries():
    metadata = PaperMetadata(
        id="chunk_test_p",
        content_hash="h1",
        title="Paper on Chunking",
        total_pages=2,
    )
    blocks = [
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
            text="First paragraph of introduction section discussing background.",
            section_path=["1. Introduction"],
        ),
        TextBlock(
            block_id="b3",
            page=2,
            bbox=[0, 0, 100, 100],
            type="heading",
            text="2. Methodology",
            section_path=["2. Methodology"],
        ),
        TextBlock(
            block_id="b4",
            page=2,
            bbox=[0, 100, 100, 200],
            type="paragraph",
            text="Methodology details and algorithmic equations.",
            section_path=["2. Methodology"],
        ),
    ]

    paper = PaperDoc(metadata=metadata, blocks=blocks)
    pipeline = IngestionPipeline()
    chunks = pipeline.chunk_paper(paper, max_chunk_words=100)

    # Chunks must be split by section
    assert len(chunks) == 2
    assert chunks[0].section_path == ["1. Introduction"]
    assert chunks[1].section_path == ["2. Methodology"]
    assert "b2" in chunks[0].block_ids
    assert "b4" in chunks[1].block_ids

