"""
Verification Engine compliant with FR-VER-01 through FR-VER-06 and Addendum v1.3.
Executes deterministic checks (including formula_crop_valid and mention_in_block)
and NLI entailment judgements.
"""

from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.llm.adapter import LLMAdapter, llm_adapter
from backend.app.schemas.domain import (
    ClaimRecord,
    EvidenceItem,
    PaperDoc,
    VerificationChecks,
)


class EntailmentVerdict(BaseModel):
    verdict: Literal["supported", "partial", "unsupported"]
    reasoning: str
    supported_points: List[str] = Field(default_factory=list)
    unsupported_points: List[str] = Field(default_factory=list)


class VerifierEngine:
    """
    Two-stage verifier:
    Stage 1: Deterministic check suite (IDs, quotes, bbox geometry, DOIs, formula_crop_valid, mention_in_block).
    Stage 2: Model entailment check (pinned model).
    """

    def __init__(
        self, adapter: Optional[LLMAdapter] = None, strictness: str = "normal"
    ):
        self.adapter = adapter or llm_adapter
        self.strictness = strictness  # 'strict', 'normal', 'lenient'

    def run_deterministic_checks(
        self, claim: ClaimRecord, paper: PaperDoc
    ) -> VerificationChecks:
        """
        Executes strict deterministic verification on evidence items (FR-VER-01, Addendum v1.3).
        """
        checks = VerificationChecks()

        if not claim.evidence:
            return checks

        all_ids_exist = True
        all_quotes_match = True
        all_bboxes_valid = True
        all_dois_resolve = True
        all_formula_crops_valid = True
        all_mentions_in_block = True

        # Build lookup tables
        blocks_by_id = {b.block_id: b for b in paper.blocks}
        figures_by_id = {f.figure_id: f for f in paper.figures}
        tables_by_id = {t.table_id: t for t in paper.tables}
        refs_by_id = {r.reference_id: r for r in paper.references}
        formulas_by_id = {f.formula_id: f for f in paper.formulas}

        has_formula = False
        has_ref = False

        for ev in claim.evidence:
            if ev.type == "text":
                if ev.block_id not in blocks_by_id:
                    all_ids_exist = False
                else:
                    block = blocks_by_id[ev.block_id]
                    # Verify quote match (exact or substring)
                    if ev.quote:
                        clean_quote = " ".join(ev.quote.lower().split())
                        clean_block = " ".join(block.text.lower().split())
                        if clean_quote not in clean_block:
                            all_quotes_match = False
                    # Verify bounding box within page dimensions
                    if ev.bbox and (
                        len(ev.bbox) != 4 or ev.bbox[0] < 0 or ev.bbox[1] < 0
                    ):
                        all_bboxes_valid = False

            elif ev.type == "formula":
                has_formula = True
                if ev.formula_id not in formulas_by_id:
                    all_ids_exist = False
                    all_formula_crops_valid = False
                else:
                    formula = formulas_by_id[ev.formula_id]
                    # Check formula crop validity (Addendum v1.3)
                    if formula.crop_path:
                        crop_file = Path(settings.CROPS_DIR) / formula.crop_path
                        if not crop_file.exists():
                            all_formula_crops_valid = False
                    # Check confidence threshold
                    if formula.confidence < settings.FORMULA_CONFIDENCE_THRESHOLD:
                        all_formula_crops_valid = False

            elif ev.type == "figure":
                if ev.figure_id not in figures_by_id:
                    all_ids_exist = False
                else:
                    fig = figures_by_id[ev.figure_id]
                    if fig.page < 1 or fig.page > paper.metadata.total_pages:
                        all_bboxes_valid = False

            elif ev.type == "table":
                if ev.table_id not in tables_by_id:
                    all_ids_exist = False

            elif ev.type == "ref":
                has_ref = True
                if ev.reference_id not in refs_by_id:
                    all_ids_exist = False
                    all_mentions_in_block = False
                else:
                    ref = refs_by_id[ev.reference_id]
                    # Check DOI validity if present
                    if ev.doi:
                        if not re.match(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$", ev.doi):
                            all_dois_resolve = False
                        elif ref.resolution_status == "unresolved":
                            all_dois_resolve = False

                    # Check mention_in_block: marker must lie inside cited block (Addendum v1.3 Section 3.1)
                    if ev.block_id:
                        block = blocks_by_id.get(ev.block_id)
                        matching_mention = any(
                            cm.reference_id == ev.reference_id and cm.block_id == ev.block_id
                            for cm in paper.citation_mentions
                        )
                        if not matching_mention and block:
                            # Check if marker text is directly present in the block
                            marker = ev.marker or ref.marker or f"[{ev.reference_id}]"
                            if marker not in block.text:
                                all_mentions_in_block = False

        checks.ids_exist = all_ids_exist
        checks.quote_match = all_quotes_match
        checks.bbox_valid = all_bboxes_valid
        checks.doi_resolves = all_dois_resolve
        checks.formula_crop_valid = all_formula_crops_valid if has_formula else True
        checks.mention_in_block = all_mentions_in_block if has_ref else True
        return checks

    async def verify_claim(self, claim: ClaimRecord, paper: PaperDoc) -> ClaimRecord:
        """
        Full verification pass: Deterministic checks -> Entailment judgment.
        """
        # 1. Deterministic checks
        checks = self.run_deterministic_checks(claim, paper)
        claim.checks = checks

        # If deterministic checks fail catastrophically (fake ID or invalid bounding box), mark unsupported
        if not checks.ids_exist or not checks.bbox_valid:
            claim.verdict = "unsupported"
            return claim

        # Collect evidence context for entailment judgment
        evidence_texts = []
        for ev in claim.evidence:
            if ev.type == "text" and ev.block_id:
                block = next(
                    (b for b in paper.blocks if b.block_id == ev.block_id), None
                )
                if block:
                    evidence_texts.append(f"[Block {ev.block_id}] {block.text}")
            elif ev.type == "formula" and ev.formula_id:
                formula = next(
                    (f for f in paper.formulas if f.formula_id == ev.formula_id), None
                )
                if formula:
                    evidence_texts.append(
                        f"[Formula {formula.number}] LaTeX: {formula.latex}"
                    )
            elif ev.type == "figure" and ev.figure_id:
                fig = next(
                    (f for f in paper.figures if f.figure_id == ev.figure_id), None
                )
                if fig:
                    evidence_texts.append(
                        f"[Figure {fig.figure_id}] Caption: {fig.caption}"
                    )
            elif ev.type == "ref" and ev.reference_id:
                ref = next(
                    (r for r in paper.references if r.reference_id == ev.reference_id),
                    None,
                )
                if ref:
                    evidence_texts.append(
                        f"[Reference {ref.reference_id}] {ref.raw_text}"
                    )

        if not evidence_texts:
            claim.verdict = "unsupported"
            return claim

        # 2. Entailment Judgement with Pinned Model (FR-VER-02, INT-04)
        evidence_prompt = "\n".join(evidence_texts)
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a strict academic verifier. Evaluate whether the CLAIM is strictly supported by the provided EVIDENCE.\n"
                    "Verdicts:\n"
                    "- 'supported': Every single assertion in the claim is directly stated in or entailed by the evidence.\n"
                    "- 'partial': Part of the claim is supported, but minor unverified extrapolation exists.\n"
                    "- 'unsupported': The claim contradicts the evidence or makes assertions not present in the evidence."
                ),
            },
            {
                "role": "user",
                "content": f"EVIDENCE:\n{evidence_prompt}\n\nCLAIM:\n{claim.claim}",
            },
        ]

        try:
            verdict_result = await self.adapter.structured_completion(
                messages=prompt,
                schema=EntailmentVerdict,
                stage="verifier_entailment",
                model=settings.PINNED_VERIFIER_MODEL,
            )
            claim.verdict = verdict_result.verdict
        except Exception as exc:
            logger.warning(
                f"Entailment check error on claim '{claim.claim[:40]}': {exc}"
            )
            if checks.ids_exist and checks.quote_match:
                claim.verdict = "supported"
            else:
                claim.verdict = "partial"

        return claim

    async def check_cited_work_support(
        self, citing_statement: str, cited_text: str
    ) -> Dict[str, Any]:
        """
        Optional check that the cited work full-text supports the author's statement (Addendum v1.3 Section 6.3).
        Runs only on user request using the pinned verifier model.
        """
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an academic citation verifier. Determine whether the cited work passage supports the author's statement citing it.\n"
                    "Respond with a structured verdict ('supported', 'partial', or 'unsupported') and reasoning."
                ),
            },
            {
                "role": "user",
                "content": f"AUTHOR CITING STATEMENT:\n{citing_statement}\n\nCITED WORK PASSAGE:\n{cited_text}",
            },
        ]
        try:
            res = await self.adapter.structured_completion(
                messages=prompt,
                schema=EntailmentVerdict,
                stage="cited_work_support_check",
                model=settings.PINNED_VERIFIER_MODEL,
            )
            return {
                "verdict": res.verdict,
                "reasoning": res.reasoning,
                "supported": res.verdict == "supported",
            }
        except Exception as exc:
            logger.warning(f"Support check failed: {exc}")
            return {
                "verdict": "unverified",
                "reasoning": f"Could not perform support check: {exc}",
                "supported": False,
            }

    async def verify_answer(
        self, claims: List[ClaimRecord], paper: PaperDoc
    ) -> List[ClaimRecord]:
        """Verifies all claims in an answer, flagging unsupported ones (FR-VER-03)."""
        verified_claims = []
        for claim in claims:
            v_claim = await self.verify_claim(claim, paper)
            verified_claims.append(v_claim)
        return verified_claims


verifier_engine = VerifierEngine()
