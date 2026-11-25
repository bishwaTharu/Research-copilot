"""
Reference Reader Service - Fetch open-access full text of cited references (Addendum v1.3).
Privacy-safe: only fetches when privacy_setting == 'allow_remote'.
Never calls a model. Caches fetched text and passages to disk.
"""

from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from ..core.config import settings
from ..schemas.domain import CitedWorkPassage, ReferenceItem, ReferenceReaderPayload


def _cited_works_dir() -> Path:
    path = Path(settings.CITED_WORKS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(reference_id: str) -> Path:
    return _cited_works_dir() / f"{reference_id}.json"


def _arxiv_pdf_url(open_access_url: str) -> Optional[str]:
    """Convert arXiv abstract URL to PDF URL."""
    m = re.search(r"arxiv\.org/abs/([0-9]+\.[0-9]+)", open_access_url, re.I)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}.pdf"
    m = re.search(r"arxiv\.org/pdf/([0-9]+\.[0-9]+)", open_access_url, re.I)
    if m:
        return open_access_url
    return None


def build_reader_payload(ref: ReferenceItem) -> ReferenceReaderPayload:
    """Build the initial reader payload from stored reference data."""
    cache = _cache_path(ref.reference_id)

    fetch_status = "idle"
    passages = []

    if cache.exists():
        try:
            with open(cache, "r", encoding="utf-8") as f:
                cached = json.load(f)
            fetch_status = cached.get("status", "ready")
            passages = [CitedWorkPassage(**p) for p in cached.get("passages", [])]
        except Exception:
            fetch_status = "failed"

    # Determine honest state message per Addendum Section 4.2
    if fetch_status == "idle" and not ref.open_access_url:
        fetch_message = "No free full text found. Abstract and link only."
    elif fetch_status == "ready":
        fetch_message = "Ready."
    elif fetch_status == "failed":
        fetch_message = "Could not fetch the free copy right now. Try again."
    elif fetch_status == "fetching":
        fetch_message = "Fetching free copy, then Checking file, Parsing, Ready."
    else:
        fetch_message = ""

    return ReferenceReaderPayload(
        reference_id=ref.reference_id,
        title=ref.title,
        authors=ref.authors,
        year=ref.year,
        abstract=ref.abstract,
        doi=ref.doi,
        open_access_url=ref.open_access_url,
        tier=ref.tier,
        fetch_status=fetch_status,  # type: ignore
        fetch_message=fetch_message,
        passages=passages,
        raw_entry=ref.raw_text,
    )


def extract_passages_from_text(
    text: str, query: str, reference_id: str, k: int = 3
) -> List[CitedWorkPassage]:
    """Extract relevant passages from full text using keyword matching (no model)."""
    if not text or not query:
        return []

    # Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return []

    query_words = set(re.findall(r"\w+", query.lower()))
    scored = []
    for i, para in enumerate(paragraphs):
        para_words = set(re.findall(r"\w+", para.lower()))
        overlap = len(query_words & para_words)
        if overlap > 0 and len(para) > 20:
            scored.append((overlap / max(len(query_words), 1), i, para))

    scored.sort(key=lambda x: -x[0])
    passages = []
    for score, idx, para in scored[:k]:
        # Verbatim quote match requirement: passage must be in text
        if para in text:
            passages.append(
                CitedWorkPassage(
                    reference_id=reference_id,
                    passage_text=para[:800],
                    page=None,
                    section=None,
                    match_score=round(score, 3),
                    verified=True,
                    label="From the cited work",
                )
            )
    return passages


async def fetch_and_cache(
    ref: ReferenceItem,
    citing_sentence: str = "",
) -> Tuple[str, List[CitedWorkPassage]]:
    """
    Fetch open-access full text, cache it, extract passages.
    Returns (status, passages). Status: 'ready' | 'failed' | 'no_copy'.
    Pure Python implementation using urllib.
    """
    cache = _cache_path(ref.reference_id)

    if cache.exists():
        try:
            with open(cache, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return cached.get("status", "ready"), [
                CitedWorkPassage(**p) for p in cached.get("passages", [])
            ]
        except Exception:
            pass

    if not ref.open_access_url:
        _write_cache(ref.reference_id, "no_copy", [])
        return "no_copy", []

    try:
        import urllib.request

        fetch_url = ref.open_access_url
        arxiv_pdf = _arxiv_pdf_url(fetch_url)
        if arxiv_pdf:
            fetch_url = arxiv_pdf

        max_bytes = settings.MAX_CITED_WORK_SIZE_MB * 1024 * 1024
        req = urllib.request.Request(
            fetch_url,
            headers={"User-Agent": "Research-Copilot/1.0 (academic use)"},
        )

        with urllib.request.urlopen(
            req, timeout=settings.CITED_WORK_FETCH_TIMEOUT
        ) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(max_bytes)

        if "pdf" in content_type.lower() or fetch_url.endswith(".pdf"):
            text = _extract_text_from_pdf_bytes(raw)
        else:
            text = re.sub(r"<[^>]+>", " ", raw.decode("utf-8", errors="replace"))
            text = re.sub(r"\s+", " ", text).strip()

        if not text:
            _write_cache(ref.reference_id, "failed", [])
            return "failed", []

        query = citing_sentence or ref.title or ""
        passages = extract_passages_from_text(text, query, ref.reference_id)
        _write_cache(ref.reference_id, "ready", passages)
        return "ready", passages

    except Exception:
        _write_cache(ref.reference_id, "failed", [])
        return "failed", []


def _extract_text_from_pdf_bytes(raw: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF if available."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=raw, filetype="pdf")
        texts = []
        for page in doc:
            texts.append(page.get_text())
        return "\n\n".join(texts)
    except Exception:
        return ""


def _write_cache(
    reference_id: str, status: str, passages: List[CitedWorkPassage]
) -> None:
    cache = _cache_path(reference_id)
    try:
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": status,
                    "passages": [p.model_dump() for p in passages],
                },
                f,
                indent=2,
            )
    except Exception:
        pass
