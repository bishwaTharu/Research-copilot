"""
Gold-set Benchmark Schema compliant with FR-EVAL-01 and FR-EVAL-04.
Used by the evaluation harness to validate parsing, retrieval, and citation accuracy.
"""

from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class GoldEvidence(BaseModel):
    type: Literal["text", "figure", "table", "ref"]
    target_id: Optional[str] = None
    section_name: Optional[str] = None
    expected_page: int
    quote_substring: Optional[str] = None


class GoldQuestion(BaseModel):
    question_id: str
    question: str
    is_out_of_scope: bool = False  # Should trigger abstention (FR-QA-07)
    expected_evidence: List[GoldEvidence] = Field(default_factory=list)
    gold_answer_points: List[str] = Field(default_factory=list)


class GoldCitation(BaseModel):
    marker: str
    raw_reference: str
    expected_doi: Optional[str] = None
    expected_year: Optional[int] = None


class GoldPaperItem(BaseModel):
    paper_id: str
    title: str
    pdf_filename: str
    expected_sections: List[str] = Field(default_factory=list)
    expected_figures_count: int = 0
    questions: List[GoldQuestion] = Field(default_factory=list)
    gold_citations: List[GoldCitation] = Field(default_factory=list)

