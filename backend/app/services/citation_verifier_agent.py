"""
Agentic 4-Step Citation Verification Pipeline:
1. Citation Extraction & Resolution (in-text markers -> Paper A bibliography via GROBID/layout-aware data)
2. Paper Retrieval (Semantic Scholar, OpenAlex, arXiv, CrossRef + Unpaywall OA PDF; Paywalled Abstract Fallback)
3. Semantic Search & Verification (Column-aware PDF chunking, hybrid semantic embedding search, LLM evaluation + Self-Correction loop)
4. Streaming UI Presentation (Real-time SSE step streaming, exact snippet, page/section, confidence score, explanation)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple

import aiohttp
import numpy as np
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.llm.adapter import LLMAdapter, llm_adapter
from backend.app.retrieval.embedder import Embedder, local_embedder
from backend.app.schemas.domain import (
    AgentStepTrace,
    CitationPassageMatch,
    CitationVerificationResult,
    PaperDoc,
    ReferenceItem,
)


# ── Internal LLM Schemas ────────────────────────────────────────────────────


class QueryFormulationLLM(BaseModel):
    search_query: str = Field(..., description="Focused semantic search query derived from the claim")
    focus_aspect: str = Field(default="results", description="Aspect targeted: results, methodology, theory, or definition")


class CitationVerdictLLM(BaseModel):
    verdict: Literal["supported", "contradicted", "partial", "context", "insufficient"] = Field(
        ..., description="Whether Paper B supports, contradicts, partially supports, provides context, or lacks evidence"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    explanation: str = Field(
        ..., description="Concise 1-3 sentence explanation of how Paper B supports, contradicts, or contextualizes Paper A's claim"
    )
    best_passage_indices: List[int] = Field(
        default_factory=list, description="0-based indices of the most relevant passages provided"
    )
    needs_retry: bool = Field(
        default=False, description="True if passages did not strongly match and a rephrased query could find better evidence"
    )
    refined_query: Optional[str] = Field(
        default=None, description="Alternative search query (e.g., targeting methodology vs results) if needs_retry is True"
    )


# ── Step 1: In-Text Citation Extraction Helpers ─────────────────────────────


def extract_citation_markers(sentence: str) -> List[str]:
    """
    Extracts numeric ([12], [1, 3], [4-6]) and author-year ((Smith et al., 2023), Smith et al. (2023))
    in-text citation markers from a highlighted sentence in Paper A.
    """
    markers: List[str] = []
    if not sentence:
        return markers

    # 1. Numeric bracket citations: [12], [1, 3], [4-6], [2, 5–7]
    for m in re.finditer(r"\[(\s*\d+(?:\s*[-–,]\s*\d+)*\s*)\]", sentence):
        inner = m.group(1)
        for part in re.split(r"\s*,\s*", inner):
            part = part.strip()
            range_m = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", part)
            if range_m:
                start, end = int(range_m.group(1)), int(range_m.group(2))
                for num in range(start, min(end + 1, start + 15)):
                    tok = f"[{num}]"
                    if tok not in markers:
                        markers.append(tok)
            elif part.isdigit():
                tok = f"[{part}]"
                if tok not in markers:
                    markers.append(tok)

    # 2. Parenthetical author-year citations: (Smith et al., 2023; Doe, 2021)
    for m in re.finditer(
        r"\(([A-Z][A-Za-z\-']+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z\-']+|&\s+[A-Z][A-Za-z\-']+))?,?\s*(?:19|20)\d{2}[a-z]?(?:\s*;\s*[A-Z][A-Za-z\-']+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z\-']+|&\s+[A-Z][A-Za-z\-']+))?,?\s*(?:19|20)\d{2}[a-z]?)*)\)",
        sentence,
    ):
        for seg in m.group(1).split(";"):
            tok = f"({seg.strip()})"
            if tok not in markers:
                markers.append(tok)

    # 3. Narrative author-year citations: Smith et al. (2023)
    for m in re.finditer(
        r"\b([A-Z][A-Za-z\-']+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z\-']+|&\s+[A-Z][A-Za-z\-']+))?)\s*\(((?:19|20)\d{2}[a-z]?)\)",
        sentence,
    ):
        tok = f"{m.group(1)} ({m.group(2)})"
        if tok not in markers and f"({m.group(1)}, {m.group(2)})" not in markers:
            markers.append(tok)

    return markers


def strip_citation_markers(sentence: str) -> str:
    """Removes bracket and parenthetical citation markers to isolate the core scientific claim."""
    cleaned = re.sub(r"\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]", "", sentence)
    cleaned = re.sub(
        r"\([A-Z][A-Za-z\-']+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z\-']+|&\s+[A-Z][A-Za-z\-']+))?,?\s*(?:19|20)\d{2}[a-z]?(?:\s*;\s*[^)]+)?\)",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or sentence.strip()


def resolve_marker_in_paper(
    paper: PaperDoc,
    sentence: str,
    markers: List[str],
    explicit_reference_id: Optional[str] = None,
) -> Optional[ReferenceItem]:
    """
    Matches an explicit reference_id or extracted in-text marker against Paper A's
    GROBID/layout-parsed bibliography (`paper.references`) and citation mentions.
    """
    refs = paper.references or []
    if not refs:
        return None

    # 1. Explicit reference_id always wins
    if explicit_reference_id:
        for r in refs:
            if r.reference_id == explicit_reference_id:
                return r

    # 2. Match against extracted markers
    for marker in markers:
        # Numeric marker e.g. [12]
        num_m = re.match(r"^\[(\d+)\]$", marker)
        if num_m:
            num_str = num_m.group(1)
            num_idx = int(num_str)
            # Direct marker equality
            for r in refs:
                if r.marker and r.marker.strip() in (marker, num_str, f"({num_str})", f"{num_str}."):
                    return r
                if r.reference_id in (f"r{num_str}", f"ref_{num_str}", f"b{num_idx - 1}"):
                    return r
            # 1-based positional fallback when bibliography is ordered
            if 1 <= num_idx <= len(refs):
                return refs[num_idx - 1]

        # Author-year marker e.g. (Smith et al., 2023) or Smith et al. (2023)
        ay_m = re.search(r"([A-Z][A-Za-z\-']+).*?((?:19|20)\d{2})", marker)
        if ay_m:
            surname = ay_m.group(1).lower()
            year_int = int(ay_m.group(2))
            best_ref: Optional[ReferenceItem] = None
            best_score = 0
            for r in refs:
                score = 0
                authors_blob = " ".join(r.authors or []).lower()
                raw_lower = (r.raw_text or "").lower()
                if surname in authors_blob:
                    score += 3
                elif surname in raw_lower:
                    score += 2
                if r.year == year_int or str(year_int) in raw_lower:
                    score += 2
                if score > best_score and score >= 3:
                    best_score = score
                    best_ref = r
            if best_ref:
                return best_ref

    # 3. Check paper.citation_mentions for sentence overlap
    sent_words = set(re.findall(r"[a-z]{4,}", sentence.lower()))
    if sent_words and paper.citation_mentions:
        best_mention_ref_id: Optional[str] = None
        best_overlap = 0
        for cm in paper.citation_mentions:
            cm_words = set(re.findall(r"[a-z]{4,}", (cm.sentence or "").lower()))
            overlap = len(sent_words & cm_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_mention_ref_id = cm.reference_id
        if best_mention_ref_id and best_overlap >= 3:
            for r in refs:
                if r.reference_id == best_mention_ref_id:
                    return r

    # 4. Last fallback: lexical title/author match against sentence
    if sent_words:
        best_ref = None
        best_ov = 0
        for r in refs:
            ref_words = set(re.findall(r"[a-z]{4,}", f"{r.title or ''} {r.raw_text or ''}".lower()))
            ov = len(sent_words & ref_words)
            if ov > best_ov:
                best_ov = ov
                best_ref = r
        if best_ref and best_ov >= 2:
            return best_ref

    return None


# ── Step 2: Layout-Aware PDF Parser & Multi-Source Academic Retrieval ───────


def _reconstruct_openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """Reconstructs plain-text abstract from OpenAlex abstract_inverted_index."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return None
    pairs: List[Tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            pairs.append((pos, word))
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[0])
    return " ".join(w for _, w in pairs).strip()


def _detect_section_heading(text: str) -> Optional[str]:
    """Detects common academic section headings to attach section metadata to chunks."""
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not first_line or len(first_line) > 85:
        return None
    if re.match(
        r"^(?:(\d+(?:\.\d+)*|[IVX]+)\.?\s+)?(Abstract|Introduction|Background|Related Work|Methodology|Methods?|Approach|Model|Architecture|Experiments?|Experimental Setup|Evaluation|Results?|Discussion|Limitations|Conclusion|Conclusions|Appendix)\b",
        first_line,
        re.I,
    ):
        return first_line
    # Numbered short heading e.g. "3.2 Attention Mechanism"
    if re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z0-9\s\-:]{2,65}$", first_line):
        return first_line
    return None


def parse_pdf_layout_aware(raw_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Layout-aware multi-column PDF parser using PyMuPDF (fitz) block sorting.
    Detects 2-column academic layouts so reading order is preserved and every chunk
    retains its exact page number and detected section heading.
    """
    chunks: List[Dict[str, Any]] = []
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        current_section = "Body"

        for page_idx, page in enumerate(doc, start=1):
            page_w = page.rect.width or 612.0
            mid_x = page_w * 0.5
            blocks = page.get_text("blocks") or []

            # Keep only text blocks (type == 0) with content
            text_blocks = [b for b in blocks if len(b) >= 7 and b[6] == 0 and str(b[4]).strip()]
            if not text_blocks:
                continue

            # Detect 2-column academic layout: significant blocks strictly left and right of page center
            left_col = [b for b in text_blocks if b[2] <= mid_x + 18]
            right_col = [b for b in text_blocks if b[0] >= mid_x - 18]
            spanning = [b for b in text_blocks if b[0] < mid_x - 18 and b[2] > mid_x + 18]

            if len(left_col) >= 2 and len(right_col) >= 2:
                # Top spanning (title/abstract) -> Left column -> Right column -> Bottom spanning
                top_span = sorted([b for b in spanning if b[1] < page.rect.height * 0.45], key=lambda b: (b[1], b[0]))
                bot_span = sorted([b for b in spanning if b[1] >= page.rect.height * 0.45], key=lambda b: (b[1], b[0]))
                ordered = (
                    top_span
                    + sorted(left_col, key=lambda b: (b[1], b[0]))
                    + sorted(right_col, key=lambda b: (b[1], b[0]))
                    + bot_span
                )
            else:
                ordered = sorted(text_blocks, key=lambda b: (round(b[1] / 6), b[0]))

            page_buf: List[str] = []
            buf_len = 0
            for b in ordered:
                raw_txt = re.sub(r"-\n(\w)", r"\1", str(b[4]).strip())
                raw_txt = re.sub(r"\s+", " ", raw_txt).strip()
                if len(raw_txt) < 15:
                    continue

                heading = _detect_section_heading(raw_txt)
                if heading and len(raw_txt) <= 95:
                    # Flush current buffer before starting new section
                    if page_buf:
                        chunks.append(
                            {
                                "text": " ".join(page_buf),
                                "page": page_idx,
                                "section": current_section,
                            }
                        )
                        page_buf = []
                        buf_len = 0
                    current_section = heading
                    continue

                page_buf.append(raw_txt)
                buf_len += len(raw_txt)
                if buf_len >= 650:
                    chunks.append(
                        {
                            "text": " ".join(page_buf),
                            "page": page_idx,
                            "section": current_section,
                        }
                    )
                    page_buf = []
                    buf_len = 0

            if page_buf:
                chunks.append(
                    {
                        "text": " ".join(page_buf),
                        "page": page_idx,
                        "section": current_section,
                    }
                )

        doc.close()
    except Exception as exc:
        logger.warning(f"Layout-aware PDF parse error: {exc}")

    return chunks


def _chunk_plain_text(text: str, section_default: str = "Body") -> List[Dict[str, Any]]:
    """Chunks plain text (e.g., HTML fallback or abstract) with section detection."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 25]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]
    chunks: List[Dict[str, Any]] = []
    cur_sec = section_default
    for idx, para in enumerate(paragraphs):
        h = _detect_section_heading(para)
        if h and len(para) <= 95:
            cur_sec = h
            continue
        chunks.append({"text": para[:1200], "page": max(1, (idx // 5) + 1), "section": cur_sec})
    return chunks


# ── Step 3: Semantic + Lexical Hybrid Search over Paper B ───────────────────


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _lexical_overlap_score(query: str, text: str) -> float:
    stop = {
        "the", "and", "for", "that", "with", "this", "from", "are", "was",
        "were", "have", "has", "not", "but", "can", "into", "their", "they",
        "which", "also", "such", "than", "more", "most", "using", "used",
    }
    q_tokens = [w for w in re.findall(r"[a-z0-9]{3,}", query.lower()) if w not in stop]
    if not q_tokens:
        return 0.0
    t_tokens = set(re.findall(r"[a-z0-9]{3,}", text.lower()))
    uni_hits = sum(1 for w in set(q_tokens) if w in t_tokens)
    uni_score = uni_hits / max(len(set(q_tokens)), 1)

    # Bigram bonus for technical phrases
    q_bigrams = {f"{q_tokens[i]} {q_tokens[i+1]}" for i in range(len(q_tokens) - 1)}
    t_lower = text.lower()
    bi_hits = sum(1 for bg in q_bigrams if bg in t_lower)
    bi_score = bi_hits / max(len(q_bigrams), 1) if q_bigrams else 0.0

    return min(1.0, 0.72 * uni_score + 0.28 * bi_score)


# ── Main Agent Class ────────────────────────────────────────────────────────


class CitationVerifierAgent:
    """
    4-Step Agentic Citation Verification Engine:
      Step 1: Citation Extraction & Resolution
      Step 2: Paper Retrieval (Semantic Scholar, OpenAlex, arXiv, CrossRef, Unpaywall + Paywall Abstract Fallback)
      Step 3: Semantic Search & LLM Verification with Self-Correction Loop
      Step 4: Structured Verdict & Live SSE Streaming Presentation
    """

    def __init__(
        self,
        adapter: Optional[LLMAdapter] = None,
        embedder: Optional[Embedder] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.adapter = adapter or llm_adapter
        self.embedder = embedder or local_embedder
        self.cache_dir = cache_dir or Path(settings.CITED_WORKS_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _agent_cache_path(self, ref: ReferenceItem) -> Path:
        raw_key = ref.doi or ref.title or ref.reference_id or ref.raw_text
        digest = hashlib.sha256(raw_key.strip().lower().encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"agent_paper_b_{digest}.json"

    def _load_paper_b_cache(self, ref: ReferenceItem) -> Optional[Dict[str, Any]]:
        path = self._agent_cache_path(ref)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_paper_b_cache(self, ref: ReferenceItem, payload: Dict[str, Any]) -> None:
        path = self._agent_cache_path(ref)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            logger.debug(f"Could not save Paper B cache: {exc}")

    # ── Academic API Clients (Step 2) ───────────────────────────────────────

    async def _query_unpaywall(self, session: aiohttp.ClientSession, doi: str) -> Dict[str, Any]:
        email = settings.CROSSREF_MAILTO or "research-copilot@example.edu"
        url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={urllib.parse.quote(email)}"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=6)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    best = data.get("best_oa_location") or {}
                    pdf_url = best.get("url_for_pdf") or best.get("url")
                    return {"pdf_url": pdf_url, "is_oa": bool(data.get("is_oa"))}
        except Exception:
            pass
        return {}

    async def _query_semantic_scholar(
        self, session: aiohttp.ClientSession, doi: Optional[str], title: Optional[str]
    ) -> Dict[str, Any]:
        fields = "title,authors,year,abstract,openAccessPdf,externalIds"
        headers = {"User-Agent": "ResearchCopilot/1.3 (academic verification)"}
        try:
            if doi:
                url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi)}?fields={fields}"
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=7)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "title": data.get("title"),
                            "abstract": data.get("abstract"),
                            "pdf_url": (data.get("openAccessPdf") or {}).get("url"),
                            "arxiv_id": (data.get("externalIds") or {}).get("ArXiv"),
                            "doi": (data.get("externalIds") or {}).get("DOI") or doi,
                        }
            if title and len(title) > 6:
                q = urllib.parse.quote(title[:160])
                url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit=1&fields={fields}"
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=7)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("data") or []
                        if items:
                            top = items[0]
                            return {
                                "title": top.get("title"),
                                "abstract": top.get("abstract"),
                                "pdf_url": (top.get("openAccessPdf") or {}).get("url"),
                                "arxiv_id": (top.get("externalIds") or {}).get("ArXiv"),
                                "doi": (top.get("externalIds") or {}).get("DOI"),
                            }
        except Exception:
            pass
        return {}

    async def _query_openalex(
        self, session: aiohttp.ClientSession, doi: Optional[str], title: Optional[str]
    ) -> Dict[str, Any]:
        mailto = settings.OPENALEX_MAILTO or "research-copilot@example.edu"
        try:
            if doi:
                clean_doi = doi.replace("https://doi.org/", "").strip()
                url = f"https://api.openalex.org/works/https://doi.org/{urllib.parse.quote(clean_doi)}?mailto={urllib.parse.quote(mailto)}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=7)) as resp:
                    if resp.status == 200:
                        w = await resp.json()
                        best_oa = w.get("best_oa_location") or {}
                        oa_info = w.get("open_access") or {}
                        return {
                            "title": w.get("title"),
                            "abstract": _reconstruct_openalex_abstract(w.get("abstract_inverted_index")),
                            "pdf_url": best_oa.get("pdf_url") or oa_info.get("oa_url"),
                            "doi": (w.get("doi") or "").replace("https://doi.org/", "") or doi,
                        }
            if title and len(title) > 6:
                url = f"https://api.openalex.org/works?search={urllib.parse.quote(title[:160])}&per-page=1&mailto={urllib.parse.quote(mailto)}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=7)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results") or []
                        if results:
                            w = results[0]
                            best_oa = w.get("best_oa_location") or {}
                            oa_info = w.get("open_access") or {}
                            return {
                                "title": w.get("title"),
                                "abstract": _reconstruct_openalex_abstract(w.get("abstract_inverted_index")),
                                "pdf_url": best_oa.get("pdf_url") or oa_info.get("oa_url"),
                                "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                            }
        except Exception:
            pass
        return {}

    async def _query_arxiv(self, session: aiohttp.ClientSession, title: Optional[str]) -> Dict[str, Any]:
        if not title or len(title) < 8:
            return {}
        clean_title = re.sub(r"[^\w\s]", " ", title).strip()[:140]
        url = f"https://export.arxiv.org/api/query?search_query=ti:{urllib.parse.quote(clean_title)}&max_results=1"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=7)) as resp:
                if resp.status == 200:
                    xml_text = await resp.text()
                    root = ET.fromstring(xml_text)
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    entry = root.find("atom:entry", ns)
                    if entry is not None:
                        entry_title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
                        # Check reasonable title overlap so we don't fetch unrelated arXiv papers
                        if _lexical_overlap_score(title, entry_title) >= 0.45:
                            entry_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
                            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
                            pdf_url = None
                            m = re.search(r"arxiv\.org/abs/([0-9]+\.[0-9]+(?:v\d+)?)", entry_id)
                            if m:
                                pdf_url = f"https://arxiv.org/pdf/{m.group(1)}.pdf"
                            return {
                                "title": re.sub(r"\s+", " ", entry_title),
                                "abstract": re.sub(r"\s+", " ", summary),
                                "pdf_url": pdf_url,
                            }
        except Exception:
            pass
        return {}

    async def _download_and_parse_pdf(
        self, session: aiohttp.ClientSession, url: str
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Downloads OA PDF (or HTML full text) and parses with layout-aware column sorting."""
        # Normalize arXiv /abs/ -> /pdf/
        m = re.search(r"arxiv\.org/abs/([0-9]+\.[0-9]+(?:v\d+)?)", url, re.I)
        if m:
            url = f"https://arxiv.org/pdf/{m.group(1)}.pdf"

        headers = {"User-Agent": "ResearchCopilot/1.3 (academic verification)"}
        max_bytes = settings.MAX_CITED_WORK_SIZE_MB * 1024 * 1024
        try:
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=settings.CITED_WORK_FETCH_TIMEOUT)
            ) as resp:
                if resp.status != 200:
                    return [], None
                content_type = (resp.headers.get("Content-Type") or "").lower()
                raw = await resp.content.read(max_bytes)
                if not raw:
                    return [], None

                is_pdf = "pdf" in content_type or url.lower().endswith(".pdf") or raw[:4] == b"%PDF"
                if is_pdf:
                    chunks = parse_pdf_layout_aware(raw)
                    if chunks:
                        return chunks, url
                elif "html" in content_type or "text" in content_type:
                    decoded = raw.decode("utf-8", errors="replace")
                    # Reject login/paywall walls that have very little body text
                    cleaned = re.sub(r"<script[^>]*>.*?</script>", " ", decoded, flags=re.S | re.I)
                    cleaned = re.sub(r"<style[^>]*>.*?</style>", " ", cleaned, flags=re.S | re.I)
                    cleaned = re.sub(r"<[^>]+>", "\n", cleaned)
                    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
                    if len(cleaned) > 1500 and "purchase pdf" not in cleaned.lower()[:2000]:
                        return _chunk_plain_text(cleaned), url
        except Exception as exc:
            logger.debug(f"Paper B download failed for {url}: {exc}")
        return [], None

    async def _retrieve_paper_b(
        self, ref: ReferenceItem
    ) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str], Literal["full_text", "abstract_only", "unavailable"], List[str]]:
        """
        Step 2: Queries Semantic Scholar, OpenAlex, Unpaywall, and arXiv for Paper B.
        Returns: (chunks, abstract, pdf_url_used, source_basis, sources_queried)
        """
        cached = self._load_paper_b_cache(ref)
        if cached and cached.get("chunks"):
            return (
                cached["chunks"],
                cached.get("abstract"),
                cached.get("pdf_url"),
                cached.get("source_basis", "full_text"),
                ["local_cache"],
            )

        sources_queried: List[str] = []
        candidate_pdf_urls: List[str] = []
        abstract = ref.abstract

        if ref.open_access_url:
            if "arxiv.org" in ref.open_access_url or ref.open_access_url.lower().endswith(".pdf"):
                candidate_pdf_urls.append(ref.open_access_url)

        async with aiohttp.ClientSession() as session:
            # 1. Unpaywall (if DOI present)
            if ref.doi:
                sources_queried.append("Unpaywall")
                upw = await self._query_unpaywall(session, ref.doi)
                if upw.get("pdf_url"):
                    candidate_pdf_urls.append(upw["pdf_url"])

            # 2. Semantic Scholar
            sources_queried.append("Semantic Scholar")
            s2 = await self._query_semantic_scholar(session, ref.doi, ref.title)
            if s2.get("pdf_url"):
                candidate_pdf_urls.append(s2["pdf_url"])
            if s2.get("arxiv_id"):
                candidate_pdf_urls.append(f"https://arxiv.org/pdf/{s2['arxiv_id']}.pdf")
            if not abstract and s2.get("abstract"):
                abstract = s2["abstract"]

            # 3. OpenAlex
            if not candidate_pdf_urls or not abstract:
                sources_queried.append("OpenAlex")
                oa = await self._query_openalex(session, ref.doi or s2.get("doi"), ref.title)
                if oa.get("pdf_url"):
                    candidate_pdf_urls.append(oa["pdf_url"])
                if not abstract and oa.get("abstract"):
                    abstract = oa["abstract"]

            # 4. arXiv API
            if not candidate_pdf_urls:
                sources_queried.append("arXiv")
                ax = await self._query_arxiv(session, ref.title)
                if ax.get("pdf_url"):
                    candidate_pdf_urls.append(ax["pdf_url"])
                if not abstract and ax.get("abstract"):
                    abstract = ax["abstract"]

            # Deduplicate candidate URLs preserving priority
            seen_u = set()
            unique_urls = []
            for u in candidate_pdf_urls:
                if u and u not in seen_u:
                    seen_u.add(u)
                    unique_urls.append(u)

            # Attempt download & layout-aware parse
            for pdf_url in unique_urls[:3]:
                chunks, used_url = await self._download_and_parse_pdf(session, pdf_url)
                if chunks:
                    self._save_paper_b_cache(
                        ref,
                        {
                            "chunks": chunks,
                            "abstract": abstract,
                            "pdf_url": used_url,
                            "source_basis": "full_text",
                        },
                    )
                    return chunks, abstract, used_url, "full_text", sources_queried

        # Paywalled Papers Fallback: use abstract from Semantic Scholar / OpenAlex / arXiv / stored ref
        if abstract and len(abstract.strip()) > 30:
            abs_chunks = [
                {
                    "text": abstract.strip(),
                    "page": 1,
                    "section": "Abstract (Paywalled Fallback)",
                }
            ]
            self._save_paper_b_cache(
                ref,
                {
                    "chunks": abs_chunks,
                    "abstract": abstract.strip(),
                    "pdf_url": None,
                    "source_basis": "abstract_only",
                },
            )
            return abs_chunks, abstract.strip(), None, "abstract_only", sources_queried

        return [], None, None, "unavailable", sources_queried

    # ── Semantic Search & LLM Verification Helpers (Step 3) ─────────────────

    async def _semantic_search_chunks(
        self, query: str, chunks: List[Dict[str, Any]], top_k: int = 4
    ) -> List[CitationPassageMatch]:
        if not chunks:
            return []

        texts = [c["text"] for c in chunks]
        q_vec_list = await self.embedder.embed_query(query)
        c_vecs_list = await self.embedder.embed_texts(texts)

        q_vec = np.array(q_vec_list, dtype=np.float32)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for chunk, vec_list in zip(chunks, c_vecs_list):
            c_vec = np.array(vec_list, dtype=np.float32)
            sem = max(0.0, _cosine_sim(q_vec, c_vec))
            lex = _lexical_overlap_score(query, chunk["text"])
            combined = min(1.0, 0.55 * sem + 0.45 * lex)
            scored.append((combined, chunk))

        scored.sort(key=lambda x: -x[0])
        matches: List[CitationPassageMatch] = []
        for score, ch in scored[:top_k]:
            matches.append(
                CitationPassageMatch(
                    snippet=ch["text"][:850],
                    page=ch.get("page"),
                    section=ch.get("section"),
                    similarity_score=round(float(score), 3),
                    query_used=query,
                )
            )
        return matches

    async def _formulate_initial_query(self, clean_claim: str) -> str:
        """Converts Paper A's claim into a focused search query for Paper B."""
        if self.adapter.mock_mode:
            return clean_claim
        try:
            out = await self.adapter.structured_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Convert the user's scientific claim from Paper A into a concise semantic search query "
                            "(8-18 words) to locate supporting or contradicting evidence inside the cited Paper B."
                        ),
                    },
                    {"role": "user", "content": f"Claim from Paper A: {clean_claim}"},
                ],
                schema=QueryFormulationLLM,
                stage="citation_query_formulation",
                max_validation_retries=1,
            )
            return out.search_query.strip() or clean_claim
        except Exception:
            return clean_claim

    def _heuristic_rephrase_query(self, clean_claim: str, previous_query: str) -> str:
        """Deterministic fallback self-correction query focusing on methodology & empirical results."""
        key_terms = re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", clean_claim)
        stop = {"that", "this", "with", "from", "have", "were", "their", "which", "shows", "showed", "demonstrated"}
        filtered = [t for t in key_terms if t.lower() not in stop]
        base = " ".join(filtered[:10]) or clean_claim
        if "method" not in previous_query.lower():
            return f"{base} experiments results methodology evaluation"
        return f"{base} theorem proof empirical analysis findings"

    async def _evaluate_passages_with_llm(
        self,
        citing_sentence: str,
        passages: List[CitationPassageMatch],
        source_basis: str,
        attempt: int,
        queries_tried: List[str],
    ) -> CitationVerdictLLM:
        """Evaluates whether retrieved passages from Paper B support, contradict, or contextualize Paper A's claim."""
        passage_block = "\n\n".join(
            f"[Passage {i} | Page {p.page or '?'} | Section: {p.section or 'Unknown'} | Sim: {p.similarity_score}]\n{p.snippet}"
            for i, p in enumerate(passages)
        )
        basis_note = (
            "IMPORTANT: Full text of Paper B is paywalled; you are evaluating ONLY the abstract. "
            "Start your explanation with 'Full text is paywalled, but based on the abstract...'."
            if source_basis == "abstract_only"
            else "You are evaluating passages extracted from Paper B's full text."
        )

        # Offline / Mock deterministic evaluator
        if self.adapter.mock_mode:
            return self._heuristic_verdict(citing_sentence, passages, source_basis, attempt)

        try:
            res = await self.adapter.structured_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a rigorous academic citation verification agent. "
                            "Given a citing claim from Paper A and retrieved passages from cited Paper B, "
                            "determine whether Paper B actually supports, contradicts, partially supports, "
                            "provides background context, or has insufficient evidence for the claim.\n"
                            f"{basis_note}\n"
                            "If the passages are weakly related and this is attempt 1 of full-text search, "
                            "set needs_retry=true and propose a refined_query targeting methodology, experiments, "
                            "or alternative terminology in Paper B."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"ATTEMPT: {attempt}\n"
                            f"PREVIOUS QUERIES: {', '.join(queries_tried)}\n\n"
                            f"PAPER A CLAIM:\n{citing_sentence}\n\n"
                            f"PAPER B RETRIEVED PASSAGES:\n{passage_block}"
                        ),
                    },
                ],
                schema=CitationVerdictLLM,
                stage="citation_verifier_judge",
                model=settings.PINNED_VERIFIER_MODEL,
                max_validation_retries=1,
            )
            if source_basis == "abstract_only" and not res.explanation.lower().startswith("full text is paywalled"):
                res.explanation = f"Full text is paywalled, but based on the abstract, {res.explanation}"
            return res
        except Exception as exc:
            logger.warning(f"Citation LLM evaluation fallback triggered: {exc}")
            return self._heuristic_verdict(citing_sentence, passages, source_basis, attempt)

    def _heuristic_verdict(
        self,
        citing_sentence: str,
        passages: List[CitationPassageMatch],
        source_basis: str,
        attempt: int,
    ) -> CitationVerdictLLM:
        top = passages[0] if passages else None
        top_sim = top.similarity_score if top else 0.0
        clean_claim = strip_citation_markers(citing_sentence)
        lex = _lexical_overlap_score(clean_claim, top.snippet) if top else 0.0
        score = max(top_sim, lex)

        # Simple contradiction cues
        neg_claim = bool(re.search(r"\b(not|never|no|fail|cannot|does not|do not)\b", clean_claim.lower()))
        neg_passage = bool(top and re.search(r"\b(not|never|no|fail|cannot|does not|do not|contrary)\b", top.snippet.lower()))

        prefix = "Full text is paywalled, but based on the abstract, " if source_basis == "abstract_only" else ""
        if score >= 0.48:
            if neg_claim != neg_passage and score >= 0.55:
                return CitationVerdictLLM(
                    verdict="contradicted",
                    confidence=round(min(0.88, 0.55 + score * 0.35), 2),
                    explanation=f"{prefix}the matched passage in Paper B appears to state an opposing polarity relative to Paper A's claim.",
                    best_passage_indices=[0],
                    needs_retry=False,
                )
            return CitationVerdictLLM(
                verdict="supported",
                confidence=round(min(0.94, 0.55 + score * 0.42), 2),
                explanation=f"{prefix}Paper B directly discusses and aligns with Paper A's claim (semantic alignment {int(score * 100)}%).",
                best_passage_indices=[0],
                needs_retry=False,
            )
        elif score >= 0.28:
            return CitationVerdictLLM(
                verdict="partial",
                confidence=round(min(0.75, 0.4 + score * 0.4), 2),
                explanation=f"{prefix}Paper B provides related context or partial support for Paper A's statement.",
                best_passage_indices=[0],
                needs_retry=(attempt == 1 and source_basis == "full_text"),
                refined_query=self._heuristic_rephrase_query(clean_claim, clean_claim) if attempt == 1 else None,
            )
        else:
            return CitationVerdictLLM(
                verdict="insufficient",
                confidence=round(max(0.15, score), 2),
                explanation=f"{prefix}no passage in Paper B strongly matched Paper A's claim.",
                best_passage_indices=[0] if top else [],
                needs_retry=(attempt == 1 and source_basis == "full_text"),
                refined_query=self._heuristic_rephrase_query(clean_claim, clean_claim) if attempt == 1 else None,
            )

    # ── End-to-End Streaming 4-Step Pipeline ────────────────────────────────

    async def stream_verify_citation(
        self,
        paper: PaperDoc,
        sentence: str,
        reference_id: Optional[str] = None,
        max_retries: int = 2,
        prebuilt_paper_b_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executes the 4-step Agentic Citation Verification workflow while streaming
        live status updates to keep the UI responsive during PDF retrieval & parsing.
        """
        t0 = time.perf_counter()
        steps_trace: List[AgentStepTrace] = []

        def _record(step: int, stage: str, message: str, status: Literal["running", "done", "warning", "error"] = "done", details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            trace = AgentStepTrace(step=step, stage=stage, message=message, status=status, details=details)
            steps_trace.append(trace)
            return {"type": "step", **trace.model_dump()}

        # ── STEP 1: Citation Extraction & Resolution ────────────────────────
        yield _record(1, "extract_resolve", "Step 1/4: Extracting in-text citation markers and resolving against Paper A's bibliography...", "running")

        markers = extract_citation_markers(sentence)
        ref = resolve_marker_in_paper(paper, sentence, markers, explicit_reference_id=reference_id)

        if not ref:
            yield _record(
                1,
                "extract_resolve",
                "Could not match an in-text citation marker to any bibliography entry in Paper A.",
                "error",
            )
            res = CitationVerificationResult(
                paper_id=paper.metadata.id,
                citing_sentence=sentence,
                extracted_markers=markers,
                resolved_reference=None,
                source_basis="unavailable",
                verdict="insufficient",
                confidence=0.0,
                explanation=(
                    "No matching bibliography entry was found in Paper A for this sentence. "
                    "Include a citation marker such as [12] or (Author, Year), or select a specific reference."
                ),
                steps=steps_trace,
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
            yield {"type": "result", "result": res.model_dump()}
            return

        ref_summary = {
            "reference_id": ref.reference_id,
            "marker": ref.marker,
            "title": ref.title or ref.raw_text[:120],
            "authors": ref.authors,
            "year": ref.year,
            "doi": ref.doi,
            "venue": ref.venue,
            "open_access_url": ref.open_access_url,
            "tier": ref.tier,
        }
        yield _record(
            1,
            "extract_resolve",
            f"Resolved citation {ref.marker or ref.reference_id} → \"{(ref.title or ref.raw_text)[:75]}\"",
            "done",
            {"reference_id": ref.reference_id, "doi": ref.doi},
        )

        # Privacy check (NFR-PRIV-02): block outbound Paper B fetch when paper is local_only
        if paper.metadata.privacy_setting == "local_only" and prebuilt_paper_b_chunks is None:
            yield _record(
                2,
                "retrieve_paper",
                "Local-Only Privacy Mode is active — external API lookups for Paper B are blocked.",
                "warning",
            )
            res = CitationVerificationResult(
                paper_id=paper.metadata.id,
                citing_sentence=sentence,
                extracted_markers=markers,
                resolved_reference=ref_summary,
                source_basis="unavailable",
                verdict="blocked_privacy",
                confidence=0.0,
                explanation=(
                    "Paper A is set to Local-Only Privacy Mode, so querying Semantic Scholar/OpenAlex/Unpaywall "
                    "for Paper B is disabled. Switch to Remote AI Active to retrieve and verify Paper B."
                ),
                steps=steps_trace,
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
            yield {"type": "result", "result": res.model_dump()}
            return

        # ── STEP 2: Paper Retrieval (Open-Access PDF or Paywalled Abstract) ──
        yield _record(
            2,
            "retrieve_paper",
            "Step 2/4: Querying Unpaywall, Semantic Scholar, OpenAlex & arXiv for Paper B full text...",
            "running",
        )

        if prebuilt_paper_b_chunks is not None:
            chunks = prebuilt_paper_b_chunks
            abstract = ref.abstract
            pdf_url_used = ref.open_access_url
            source_basis: Literal["full_text", "abstract_only", "unavailable"] = (
                "full_text" if chunks else ("abstract_only" if abstract else "unavailable")
            )
            sources_queried = ["injected_test_chunks"]
        else:
            chunks, abstract, pdf_url_used, source_basis, sources_queried = await self._retrieve_paper_b(ref)

        is_paywalled = source_basis == "abstract_only"
        paywall_notice: Optional[str] = None

        if source_basis == "full_text":
            yield _record(
                2,
                "retrieve_paper",
                f"Downloaded & layout-parsed Paper B ({len(chunks)} chunks via {', '.join(sources_queried)}).",
                "done",
                {"chunks_count": len(chunks), "pdf_url": pdf_url_used, "sources": sources_queried},
            )
        elif source_basis == "abstract_only":
            paywall_notice = "Full text is paywalled, but based on the abstract..."
            yield _record(
                2,
                "retrieve_paper",
                "Full text of Paper B is paywalled — falling back to abstract retrieved via academic metadata APIs.",
                "warning",
                {"sources": sources_queried},
            )
        else:
            yield _record(
                2,
                "retrieve_paper",
                "Could not locate an open-access PDF or abstract for Paper B across Semantic Scholar, OpenAlex, arXiv, or Unpaywall.",
                "error",
                {"sources": sources_queried},
            )
            res = CitationVerificationResult(
                paper_id=paper.metadata.id,
                citing_sentence=sentence,
                extracted_markers=markers,
                resolved_reference=ref_summary,
                source_basis="unavailable",
                is_paywalled=True,
                paywall_notice="Full text is paywalled and no public abstract was found.",
                verdict="insufficient",
                confidence=0.0,
                explanation=(
                    "Neither an open-access PDF nor a public abstract could be retrieved for Paper B. "
                    "Please check the DOI or upload Paper B directly."
                ),
                steps=steps_trace,
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
            yield {"type": "result", "result": res.model_dump()}
            return

        # ── STEP 3: Semantic Search & Verification (+ Self-Correction Loop) ──
        clean_claim = strip_citation_markers(sentence)
        yield _record(
            3,
            "semantic_search",
            f"Step 3/4: Converting claim to semantic query and scanning Paper B ({len(chunks)} passages)...",
            "running",
        )

        initial_query = await self._formulate_initial_query(clean_claim)
        queries_tried: List[str] = [initial_query]
        matched_passages = await self._semantic_search_chunks(initial_query, chunks, top_k=4)

        verdict_obj = await self._evaluate_passages_with_llm(
            citing_sentence=sentence,
            passages=matched_passages,
            source_basis=source_basis,
            attempt=1,
            queries_tried=queries_tried,
        )

        top_sim = matched_passages[0].similarity_score if matched_passages else 0.0
        self_corrected = False

        # Agentic Self-Correction Loop: if weak match or judge requests retry on full-text Paper B
        should_retry = (
            max_retries > 1
            and source_basis == "full_text"
            and (verdict_obj.needs_retry or verdict_obj.verdict == "insufficient" or top_sim < 0.35)
        )
        if should_retry:
            refined_q = (
                verdict_obj.refined_query.strip()
                if verdict_obj.refined_query and verdict_obj.refined_query.strip().lower() != initial_query.lower()
                else self._heuristic_rephrase_query(clean_claim, initial_query)
            )
            queries_tried.append(refined_q)
            self_corrected = True
            yield _record(
                3,
                "self_correct",
                f"Step 3b/4 (Self-Correction): Weak initial match — rephrasing query to \"{refined_q[:70]}\" and re-scanning Paper B...",
                "running",
                {"refined_query": refined_q},
            )

            retry_passages = await self._semantic_search_chunks(refined_q, chunks, top_k=4)
            # Merge unique best passages by snippet prefix
            merged_by_key: Dict[str, CitationPassageMatch] = {}
            for p in retry_passages + matched_passages:
                k = p.snippet[:80]
                if k not in merged_by_key or p.similarity_score > merged_by_key[k].similarity_score:
                    merged_by_key[k] = p
            matched_passages = sorted(merged_by_key.values(), key=lambda x: -x.similarity_score)[:4]

            verdict_obj = await self._evaluate_passages_with_llm(
                citing_sentence=sentence,
                passages=matched_passages,
                source_basis=source_basis,
                attempt=2,
                queries_tried=queries_tried,
            )

        yield _record(
            3,
            "semantic_search",
            f"Semantic verification finished across {len(queries_tried)} search pass(es) (top match {int((matched_passages[0].similarity_score if matched_passages else 0) * 100)}%).",
            "done",
            {"queries_tried": queries_tried, "self_corrected": self_corrected},
        )

        # ── STEP 4: Final UI Presentation Payload ───────────────────────────
        # Re-order matched passages so LLM's chosen best indices come first
        ordered_passages = list(matched_passages)
        if verdict_obj.best_passage_indices:
            prioritized: List[CitationPassageMatch] = []
            seen_idx = set()
            for idx in verdict_obj.best_passage_indices:
                if 0 <= idx < len(matched_passages) and idx not in seen_idx:
                    prioritized.append(matched_passages[idx])
                    seen_idx.add(idx)
            for idx, p in enumerate(matched_passages):
                if idx not in seen_idx:
                    prioritized.append(p)
            ordered_passages = prioritized

        yield _record(
            4,
            "verify_verdict",
            f"Step 4/4: Verdict ready — {verdict_obj.verdict.upper()} ({int(verdict_obj.confidence * 100)}% confidence).",
            "done",
            {"verdict": verdict_obj.verdict, "confidence": verdict_obj.confidence},
        )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        final_result = CitationVerificationResult(
            paper_id=paper.metadata.id,
            citing_sentence=sentence,
            extracted_markers=markers,
            resolved_reference=ref_summary,
            source_basis=source_basis,
            is_paywalled=is_paywalled,
            paywall_notice=paywall_notice,
            verdict=verdict_obj.verdict,
            confidence=round(float(verdict_obj.confidence), 3),
            explanation=verdict_obj.explanation,
            matched_passages=ordered_passages[:3],
            queries_tried=queries_tried,
            self_corrected=self_corrected,
            steps=steps_trace,
            latency_ms=elapsed_ms,
        )
        yield {"type": "result", "result": final_result.model_dump()}

    async def verify_citation(
        self,
        paper: PaperDoc,
        sentence: str,
        reference_id: Optional[str] = None,
        max_retries: int = 2,
        prebuilt_paper_b_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> CitationVerificationResult:
        """Non-streaming wrapper returning the final CitationVerificationResult."""
        final: Optional[CitationVerificationResult] = None
        async for evt in self.stream_verify_citation(
            paper=paper,
            sentence=sentence,
            reference_id=reference_id,
            max_retries=max_retries,
            prebuilt_paper_b_chunks=prebuilt_paper_b_chunks,
        ):
            if evt.get("type") == "result":
                final = CitationVerificationResult.model_validate(evt["result"])
        assert final is not None
        return final


citation_verifier_agent = CitationVerifierAgent()
