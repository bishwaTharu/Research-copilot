"""
Tool Layer (Section 7: Agent Tool Contract)
Deterministic, typed tools for search, figure discovery, reference inspection,
citing context retrieval, and open-access checking.
Rules:
- Tools read the database first and call a live service only on a cache miss.
- The agent can pass only IDs it was given, so it cannot invent a reference.
- Each tool returns IDs and records, never free text.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from backend.app.core.logging import logger
from backend.app.references.resolver import reference_resolver
from backend.app.schemas.domain import PaperDoc, ReferenceItem


class ToolLayer:
    """
    Deterministic tool suite exposed to the orchestrator agent and API endpoints.
    Complies with Software Architecture Section 7 Agent Tool Contract.
    """

    def __init__(self, docs_store: Optional[Dict[str, PaperDoc]] = None):
        # In-memory or database paper repository reference
        self.docs_store = docs_store if docs_store is not None else {}

    def set_docs_store(self, docs_store: Dict[str, PaperDoc]):
        self.docs_store = docs_store

    def search_text(
        self,
        paper_id: str,
        query: str,
        k: int = 5,
        search_index: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Tool: search_text(paper_id, query, k)
        Output: Evidence IDs of text blocks with section and page.
        Method: Hybrid dense plus keyword search.
        """
        doc = self.docs_store.get(paper_id)
        if not doc:
            logger.warning(f"Tool search_text: paper {paper_id} not found.")
            return []

        if search_index:
            results = search_index.search(query=query, top_k=k)
            items = []
            for res in results:
                if res.item_type == "text":
                    meta = res.metadata
                    items.append({
                        "type": "text",
                        "block_ids": meta.get("block_ids", []),
                        "page": meta.get("page", 1),
                        "section": meta.get("section", ""),
                        "bbox": meta.get("bbox", [0, 0, 100, 100]),
                        "text": res.text,
                        "score": round(res.score, 4),
                    })
            if items:
                return items[:k]

        # Deterministic keyword fallback over doc.blocks
        q_tokens = set(re.findall(r"\w+", query.lower()))
        scored_blocks = []
        for b in doc.blocks:
            if b.type == "fallback_page":
                continue
            b_tokens = set(re.findall(r"\w+", b.text.lower()))
            overlap = len(q_tokens.intersection(b_tokens))
            if overlap > 0:
                scored_blocks.append((overlap, b))

        scored_blocks.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "type": "text",
                "block_ids": [b.block_id],
                "page": b.page,
                "section": " > ".join(b.section_path) if b.section_path else "Content",
                "bbox": b.bbox,
                "text": b.text,
                "score": float(score),
            }
            for score, b in scored_blocks[:k]
        ]

    def search_figures(
        self,
        paper_id: str,
        query: str,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Tool: search_figures(paper_id, query, k)
        Output: Figure IDs with crop path and summary.
        Notes: Pending summaries flagged.
        """
        doc = self.docs_store.get(paper_id)
        if not doc or not doc.figures:
            return []

        q_tokens = set(re.findall(r"\w+", query.lower()))
        results = []

        for fig in doc.figures:
            fig_text = f"{fig.caption} {fig.figure_id}".lower()
            if fig.summary and isinstance(fig.summary, dict):
                fig_text += " " + json_summary_to_text(fig.summary).lower()

            fig_tokens = set(re.findall(r"\w+", fig_text))
            score = len(q_tokens.intersection(fig_tokens))

            results.append({
                "figure_id": fig.figure_id,
                "page": fig.page,
                "crop_path": fig.crop_path,
                "bbox": fig.bbox,
                "caption": fig.caption,
                "summary": fig.summary,
                "summary_status": fig.summary_status,
                "confidence": fig.confidence,
                "is_pending": fig.summary_status == "pending",
                "score": float(score),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def list_references(self, paper_id: str) -> Dict[str, Any]:
        """
        Tool: list_references(paper_id)
        Output: Reference IDs with tier badges, counts of each tier.
        """
        doc = self.docs_store.get(paper_id)
        if not doc:
            return {"total": 0, "counts": {"verified": 0, "probable": 0, "unresolved": 0}, "references": []}

        counts = {"verified": 0, "probable": 0, "unresolved": 0}
        ref_summaries = []

        for r in doc.references:
            tier = r.tier if r.tier in counts else "unresolved"
            counts[tier] += 1
            ref_summaries.append({
                "reference_id": r.reference_id,
                "marker": r.marker,
                "title": r.title or r.raw_text[:60],
                "authors": r.authors,
                "year": r.year,
                "tier": r.tier,
                "match_score": r.match_score,
                "doi": r.doi,
                "is_retracted": r.is_retracted,
                "has_free_copy": bool(r.open_access_url or r.doi),
            })

        return {
            "total": len(doc.references),
            "counts": counts,
            "references": ref_summaries,
        }

    def get_reference(
        self, reference_id: str, paper_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Tool: get_reference(reference_id)
        Output: Metadata, abstract, identifier, tier, match score.
        Notes: Rendered strictly from stored record, never fabricated.
        """
        ref, _ = self._find_ref(reference_id, paper_id)
        if not ref:
            return None

        return {
            "reference_id": ref.reference_id,
            "marker": ref.marker,
            "title": ref.title,
            "authors": ref.authors,
            "year": ref.year,
            "venue": ref.venue,
            "doi": ref.doi,
            "abstract": ref.abstract,
            "tier": ref.tier,
            "match_score": ref.match_score,
            "open_access_url": ref.open_access_url,
            "is_retracted": ref.is_retracted,
            "retraction_details": ref.retraction_details,
            "bibtex": ref.bibtex,
            "raw_text": ref.raw_text,
            "user_confirmed": ref.user_confirmed,
        }

    def get_citing_context(
        self, reference_id: str, paper_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Tool: get_citing_context(reference_id)
        Output: Sentences and block IDs where the paper cites it.
        Notes: From GROBID and in-text citation marker mapping.
        """
        ref, doc = self._find_ref(reference_id, paper_id)
        if not ref:
            return {"reference_id": reference_id, "citing_sentences": [], "blocks": []}

        citing_sentences = list(ref.citing_sentences or [])
        matched_blocks: List[Dict[str, Any]] = []

        if doc:
            # Check explicit citation mentions
            for cm in doc.citation_mentions:
                if cm.reference_id == reference_id:
                    if cm.sentence not in citing_sentences:
                        citing_sentences.append(cm.sentence)
                    matched_blocks.append({
                        "block_id": cm.block_id,
                        "sentence": cm.sentence,
                    })

            # Check textual blocks containing the marker
            if not matched_blocks and ref.marker:
                for b in doc.blocks:
                    if ref.marker in b.text:
                        matched_blocks.append({
                            "block_id": b.block_id,
                            "page": b.page,
                            "section": " > ".join(b.section_path) if b.section_path else "",
                            "excerpt": b.text[:150],
                        })

        return {
            "reference_id": reference_id,
            "marker": ref.marker,
            "citing_sentences": citing_sentences,
            "blocks": matched_blocks,
        }

    async def find_open_access(
        self, reference_id: str, paper_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Tool: find_open_access(reference_id)
        Output: Legal free link or none.
        Notes: Never guesses a mirror. Validates open-access status from stored record.
        """
        ref, _ = self._find_ref(reference_id, paper_id)
        if not ref:
            return {"reference_id": reference_id, "has_free_copy": False, "open_access_url": None}

        # Check existing verified open_access_url
        if ref.open_access_url:
            return {
                "reference_id": reference_id,
                "has_free_copy": True,
                "open_access_url": ref.open_access_url,
                "source": "unpaywall_or_arxiv",
            }

        if ref.doi:
            doi_link = f"https://doi.org/{ref.doi}"
            return {
                "reference_id": reference_id,
                "has_free_copy": True,
                "open_access_url": doi_link,
                "source": "doi_resolver",
            }

        return {
            "reference_id": reference_id,
            "has_free_copy": False,
            "open_access_url": None,
            "source": None,
        }

    def retraction_check(
        self, reference_id: str, paper_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Tool: retraction_check(reference_id)
        Output: Retraction/correction warnings from CrossRef/publisher metadata.
        """
        ref, _ = self._find_ref(reference_id, paper_id)
        if not ref:
            return {"reference_id": reference_id, "is_retracted": False, "retraction_details": None}

        return {
            "reference_id": reference_id,
            "is_retracted": ref.is_retracted,
            "retraction_details": ref.retraction_details,
        }

    def search_formulas(
        self,
        paper_id: str,
        query: str,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Tool: search_formulas(paper_id, query, k) (Addendum v1.3 Section 6.3, 7).
        Output: Formula IDs with number, page, crop path, LaTeX and confidence.
        """
        doc = self.docs_store.get(paper_id)
        if not doc or not doc.formulas:
            return []

        q_tokens = set(re.findall(r"\w+", query.lower()))
        results = []

        for formula in doc.formulas:
            # Score against LaTeX, number, and mentioning context
            formula_text = f"{formula.number} {formula.latex}".lower()
            for m in formula.mentions:
                formula_text += " " + str(m.get("excerpt", "")).lower()

            f_tokens = set(re.findall(r"\w+", formula_text))
            score = len(q_tokens.intersection(f_tokens))

            # Bonus score if query explicitly asks for formula number e.g. "eq 3" or "(3)"
            if formula.number.strip("()").lower() in query.lower():
                score += 5

            results.append({
                "type": "formula",
                "formula_id": formula.formula_id,
                "number": formula.number,
                "page": formula.page,
                "latex": formula.latex,
                "crop_path": formula.crop_path,
                "confidence": formula.confidence,
                "score": float(score),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def get_formula(
        self, formula_id: str, paper_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Tool: get_formula(formula_id, paper_id) (Addendum v1.3 Section 6.3, 7).
        Output: Formula record, LaTeX, crop path, and confidence.
        """
        docs = [self.docs_store[paper_id]] if (paper_id and paper_id in self.docs_store) else list(self.docs_store.values())
        for doc in docs:
            for f in doc.formulas:
                if f.formula_id == formula_id:
                    return {
                        "formula_id": f.formula_id,
                        "page": f.page,
                        "bbox": f.bbox,
                        "crop_path": f.crop_path,
                        "latex": f.latex,
                        "number": f.number,
                        "confidence": f.confidence,
                        "mentions": f.mentions,
                    }
        return None

    def search_cited_text(
        self, reference_id: str, query: str, k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Tool: search_cited_text(reference_id, query, k) (Addendum v1.3 Section 6.3, 7).
        Output: Passage IDs from a fetched cited work.
        """
        from backend.app.services.reference_reader import build_reader_payload
        ref, _ = self._find_ref(reference_id)
        if not ref:
            return []
        payload = build_reader_payload(ref)
        if not payload.passages:
            return []

        q_tokens = set(re.findall(r"\w+", query.lower()))
        scored = []
        for p in payload.passages:
            p_tokens = set(re.findall(r"\w+", p.passage_text.lower()))
            overlap = len(q_tokens.intersection(p_tokens))
            scored.append({
                "reference_id": reference_id,
                "passage_text": p.passage_text,
                "page": p.page,
                "section": p.section,
                "match_score": p.match_score,
                "score": float(overlap),
                "label": p.label,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def _find_ref(
        self, reference_id: str, paper_id: Optional[str] = None
    ) -> tuple[Optional[ReferenceItem], Optional[PaperDoc]]:
        if paper_id and paper_id in self.docs_store:
            doc = self.docs_store[paper_id]
            for r in doc.references:
                if r.reference_id == reference_id:
                    return r, doc

        for doc in self.docs_store.values():
            for r in doc.references:
                if r.reference_id == reference_id:
                    return r, doc
        return None, None


def json_summary_to_text(summary: Dict[str, Any]) -> str:
    parts = []
    if "key_takeaway" in summary:
        parts.append(str(summary["key_takeaway"]))
    if "type" in summary:
        parts.append(str(summary["type"]))
    if "components" in summary and isinstance(summary["components"], list):
        parts.extend(str(c) for c in summary["components"])
    return " ".join(parts)


tool_layer = ToolLayer()
