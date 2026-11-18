"""
Unit tests for Section 7: Agent Tool Contract.
Verifies that all tools strictly read from the database/index, return IDs and records,
and never synthesize or hallucinate references.
"""

import pytest
from backend.app.schemas.domain import (
    FigureItem,
    PaperDoc,
    PaperMetadata,
    ReferenceItem,
    TextBlock,
)
from backend.app.tools.layer import ToolLayer


@pytest.fixture
def mock_paper():
    metadata = PaperMetadata(
        id="paper_tool_test",
        content_hash="hash_tool_test",
        title="Attention Is All You Need",
        authors=["Vaswani et al."],
        year=2017,
        total_pages=15,
    )
    blocks = [
        TextBlock(
            block_id="b101",
            page=3,
            bbox=[50, 100, 400, 200],
            type="paragraph",
            text="The Transformer uses multi-head attention to attend to information from different representation subspaces.",
            section_path=["3. Model Architecture", "3.2 Attention"],
        ),
        TextBlock(
            block_id="b102",
            page=4,
            bbox=[50, 210, 400, 300],
            type="paragraph",
            text="Layer normalization is applied as described in Ba et al. [1].",
            section_path=["3. Model Architecture", "3.1 Encoder"],
        ),
    ]
    figures = [
        FigureItem(
            figure_id="fig1",
            page=3,
            bbox=[60, 350, 450, 700],
            crop_path="crops/paper_tool_test/fig1.png",
            caption="Figure 1: The Transformer model architecture.",
            summary={"key_takeaway": "Encoder-decoder stack with multi-head self-attention"},
            summary_status="completed",
            confidence=0.98,
        ),
        FigureItem(
            figure_id="fig2",
            page=5,
            bbox=[60, 100, 450, 300],
            crop_path="crops/paper_tool_test/fig2.png",
            caption="Figure 2: Scaled Dot-Product Attention.",
            summary_status="pending",
            confidence=0.60,
        ),
    ]
    references = [
        ReferenceItem(
            reference_id="r1",
            marker="[1]",
            raw_text="Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv:1607.06450, 2016.",
            title="Layer Normalization",
            authors=["Jimmy Lei Ba", "Jamie Ryan Kiros", "Geoffrey E Hinton"],
            year=2016,
            doi="10.48550/arXiv.1607.06450",
            tier="verified",
            match_score=0.99,
            open_access_url="https://arxiv.org/pdf/1607.06450.pdf",
            citing_sentences=["Layer normalization is applied as described in Ba et al. [1]."],
        ),
        ReferenceItem(
            reference_id="r2",
            marker="[2]",
            raw_text="Sample Probable Work 2021.",
            title="Sample Probable Work",
            authors=["Author A"],
            year=2021,
            tier="probable",
            match_score=0.74,
            is_retracted=True,
            retraction_details="Retracted by publisher due to duplicate submission.",
        ),
        ReferenceItem(
            reference_id="r3",
            marker="[3]",
            raw_text="Raw unresolvable bibliographic string without DOI.",
            tier="unresolved",
            match_score=0.30,
        ),
    ]
    return PaperDoc(
        metadata=metadata,
        blocks=blocks,
        figures=figures,
        references=references,
    )


def test_search_text_tool(mock_paper):
    tool = ToolLayer({"paper_tool_test": mock_paper})
    results = tool.search_text(
        paper_id="paper_tool_test",
        query="multi-head attention representation",
        k=2,
    )
    assert len(results) >= 1
    top = results[0]
    assert top["type"] == "text"
    assert "b101" in top["block_ids"]
    assert top["page"] == 3
    assert "Attention" in top["section"]


def test_search_figures_tool(mock_paper):
    tool = ToolLayer({"paper_tool_test": mock_paper})
    results = tool.search_figures(paper_id="paper_tool_test", query="architecture encoder", k=2)
    assert len(results) == 2
    # Check that pending summaries are flagged
    pending_fig = next(f for f in results if f["figure_id"] == "fig2")
    assert pending_fig["is_pending"] is True
    assert pending_fig["summary_status"] == "pending"


def test_list_references_tool(mock_paper):
    tool = ToolLayer({"paper_tool_test": mock_paper})
    listing = tool.list_references("paper_tool_test")
    assert listing["total"] == 3
    assert listing["counts"]["verified"] == 1
    assert listing["counts"]["probable"] == 1
    assert listing["counts"]["unresolved"] == 1
    assert len(listing["references"]) == 3


def test_get_reference_tool(mock_paper):
    tool = ToolLayer({"paper_tool_test": mock_paper})
    ref = tool.get_reference("r1", paper_id="paper_tool_test")
    assert ref is not None
    assert ref["reference_id"] == "r1"
    assert ref["tier"] == "verified"
    assert ref["doi"] == "10.48550/arXiv.1607.06450"
    assert "arxiv.org" in ref["open_access_url"]


def test_get_citing_context_tool(mock_paper):
    tool = ToolLayer({"paper_tool_test": mock_paper})
    ctx = tool.get_citing_context("r1", paper_id="paper_tool_test")
    assert ctx["reference_id"] == "r1"
    assert len(ctx["citing_sentences"]) >= 1
    assert "Layer normalization is applied" in ctx["citing_sentences"][0]


@pytest.mark.asyncio
async def test_find_open_access_tool(mock_paper):
    tool = ToolLayer({"paper_tool_test": mock_paper})
    oa = await tool.find_open_access("r1", paper_id="paper_tool_test")
    assert oa["has_free_copy"] is True
    assert "arxiv.org" in oa["open_access_url"]

    oa_unres = await tool.find_open_access("r3", paper_id="paper_tool_test")
    assert oa_unres["has_free_copy"] is False
    assert oa_unres["open_access_url"] is None


def test_retraction_check_tool(mock_paper):
    tool = ToolLayer({"paper_tool_test": mock_paper})
    retract_info = tool.retraction_check("r2", paper_id="paper_tool_test")
    assert retract_info["is_retracted"] is True
    assert "Retracted by publisher" in retract_info["retraction_details"]

    clean_info = tool.retraction_check("r1", paper_id="paper_tool_test")
    assert clean_info["is_retracted"] is False
