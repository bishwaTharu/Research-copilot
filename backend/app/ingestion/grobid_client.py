"""
GROBID Client — Step 1 of the hybrid ingestion pipeline.

Responsibilities (strict, no overlap with MinerU):
  - Paper title, authors, abstract, year, DOI
  - Structured bibliography with per-field extraction
  - In-text citation marker → reference ID mapping
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.schemas.domain import CitationMention, PaperMetadata, ReferenceItem

_NS = {
    "tei": "http://www.tei-c.org/ns/1.0",
}


def _text(el: Optional[ET.Element], default: str = "") -> str:
    return (el.text or "").strip() if el is not None else default


class GROBIDClient:
    """
    Thin async wrapper around the GROBID REST API.
    Uses /api/processFulltextDocument for full structured extraction.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.GROBID_URL).rstrip("/")

    async def is_available(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/isalive",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def process_pdf(
        self, pdf_path: Path, paper_id: str
    ) -> Tuple[PaperMetadata, List[ReferenceItem], List[CitationMention]]:
        """
        Runs the PDF through GROBID and returns:
          - PaperMetadata (title, authors, abstract, year, doi)
          - List[ReferenceItem] (structured bibliography)
          - List[CitationMention] (in-text citation → reference mappings)
        Falls back gracefully if GROBID service is unreachable.
        """
        try:
            tei_xml = await self._call_grobid(pdf_path)
            return self._parse_tei(tei_xml, paper_id)
        except Exception as exc:
            logger.warning(
                f"GROBID service unavailable ({exc}); falling back to local metadata & bib extraction."
            )
            return self._fallback_extract(pdf_path, paper_id)

    def _fallback_extract(
        self, pdf_path: Path, paper_id: str
    ) -> Tuple[PaperMetadata, List[ReferenceItem], List[CitationMention]]:
        """Fallback extractor using PyMuPDF and regex when GROBID service is offline."""
        import pymupdf as fitz

        doc = fitz.open(str(pdf_path))
        meta = doc.metadata or {}
        raw_title = meta.get("title", "").strip() or pdf_path.stem.replace("_", " ")
        authors_raw = meta.get("author", "")
        authors = (
            [a.strip() for a in authors_raw.split(",") if a.strip()]
            if authors_raw
            else []
        )

        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"

        # Find References section
        ref_header_match = re.search(
            r"(?i)\n(?:references|bibliography|works cited)\s*\n", full_text
        )
        references: List[ReferenceItem] = []
        if ref_header_match:
            ref_body = full_text[ref_header_match.end() :]
            splits = list(re.finditer(r"(?:^|\n)(?:\[(\d+)\]|(\d+)\.)\s+", ref_body))
            for idx, m in enumerate(splits, start=1):
                marker_num = m.group(1) or m.group(2)
                start_pos = m.end()
                end_pos = splits[idx].start() if idx < len(splits) else len(ref_body)
                entry_text = ref_body[start_pos:end_pos].strip().replace("\n", " ")

                doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", entry_text)
                doi = doi_match.group(0).rstrip(".") if doi_match else None
                year_match = re.search(r"\b(19\d\d|20\d\d)\b", entry_text)
                year = int(year_match.group(0)) if year_match else None

                references.append(
                    ReferenceItem(
                        reference_id=f"r{marker_num}",
                        marker=f"[{marker_num}]",
                        raw_text=entry_text,
                        doi=doi,
                        year=year,
                        resolution_status="resolved" if doi else "unresolved",
                    )
                )

        citation_mentions: List[CitationMention] = []
        for m in re.finditer(r"\[(\d{1,3}(?:[,\s–\-]+\d{1,3})*)\]", full_text):
            marker = m.group(0)
            nums = re.findall(r"\d+", marker)
            for n in nums:
                citation_mentions.append(
                    CitationMention(
                        paper_id=paper_id,
                        block_id="",
                        reference_id=f"r{n}",
                        sentence=full_text[
                            max(0, m.start() - 80) : min(len(full_text), m.end() + 80)
                        ].strip(),
                    )
                )

        metadata = PaperMetadata(
            id=paper_id,
            content_hash="",
            title=raw_title,
            authors=authors,
            total_pages=len(doc),
        )
        return metadata, references, citation_mentions

    # ── Private ──────────────────────────────────────────────────────────────

    async def _call_grobid(self, pdf_path: Path) -> str:
        url = f"{self.base_url}/api/processFulltextDocument"
        data = aiohttp.FormData()
        data.add_field(
            "input",
            open(pdf_path, "rb"),
            filename=pdf_path.name,
            content_type="application/pdf",
        )
        data.add_field("consolidateReferences", "0")  # no external lookups from GROBID
        data.add_field("consolidateHeader", "0")

        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, data=data) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"GROBID returned HTTP {resp.status}: {text[:200]}"
                    )
                return await resp.text()

    def _parse_tei(
        self, tei_xml: str, paper_id: str
    ) -> Tuple[PaperMetadata, List[ReferenceItem], List[CitationMention]]:
        root = ET.fromstring(tei_xml)
        header = root.find(".//tei:teiHeader", _NS)

        # ── Metadata ─────────────────────────────────────────────────────────
        title = _text(
            root.find(".//tei:titleStmt/tei:title[@level='a'][@type='main']", _NS)
        ) or _text(root.find(".//tei:titleStmt/tei:title", _NS))

        authors: List[str] = []
        for author_el in root.findall(".//tei:fileDesc//tei:author", _NS):
            forename = _text(author_el.find(".//tei:forename", _NS))
            surname = _text(author_el.find(".//tei:surname", _NS))
            full = f"{forename} {surname}".strip()
            if full:
                authors.append(full)

        abstract_el = root.find(".//tei:abstract//tei:p", _NS)
        abstract = _text(abstract_el) if abstract_el is not None else None

        # Year
        date_el = root.find(".//tei:publicationStmt/tei:date[@type='published']", _NS)
        year_str = date_el.get("when", "") if date_el is not None else ""
        year = int(year_str[:4]) if re.match(r"\d{4}", year_str) else None

        # DOI
        doi_el = root.find(".//tei:idno[@type='DOI']", _NS)
        doi = _text(doi_el) or None

        metadata = PaperMetadata(
            id=paper_id,
            content_hash="",  # filled in by pipeline
            title=title or "Untitled Paper",
            authors=authors,
            abstract=abstract,
            year=year,
            doi=doi,
        )

        # ── Bibliography ─────────────────────────────────────────────────────
        references: List[ReferenceItem] = []
        ref_id_map: Dict[str, str] = {}  # xml:id → reference_id ("r1", "r2", …)

        for idx, bib_el in enumerate(
            root.findall(".//tei:listBibl/tei:biblStruct", _NS), start=1
        ):
            ref_id = f"r{idx}"
            xml_id = bib_el.get("{http://www.w3.org/XML/1998/namespace}id", "")
            if xml_id:
                ref_id_map[xml_id] = ref_id

            # Title
            ref_title = _text(bib_el.find(".//tei:title[@level='a']", _NS)) or _text(
                bib_el.find(".//tei:title[@level='m']", _NS)
            )
            # Authors
            ref_authors: List[str] = []
            for a in bib_el.findall(".//tei:author", _NS):
                fn = _text(a.find("tei:forename", _NS))
                sn = _text(a.find("tei:surname", _NS))
                name = f"{fn} {sn}".strip()
                if name:
                    ref_authors.append(name)
            # Year
            ref_date = bib_el.find(".//tei:date", _NS)
            ref_year_str = ref_date.get("when", "") if ref_date is not None else ""
            ref_year = (
                int(ref_year_str[:4]) if re.match(r"\d{4}", ref_year_str) else None
            )
            # DOI
            ref_doi_el = bib_el.find(".//tei:idno[@type='DOI']", _NS)
            ref_doi = _text(ref_doi_el) or None
            # Venue
            venue_el = bib_el.find(".//tei:title[@level='j']", _NS) or bib_el.find(
                ".//tei:title[@level='s']", _NS
            )
            venue = _text(venue_el) or None

            # Reconstruct a readable raw string
            raw_parts = []
            if ref_authors:
                raw_parts.append(", ".join(ref_authors[:3]))
            if ref_title:
                raw_parts.append(f'"{ref_title}"')
            if venue:
                raw_parts.append(venue)
            if ref_year:
                raw_parts.append(str(ref_year))
            if ref_doi:
                raw_parts.append(f"DOI:{ref_doi}")
            raw_text = ". ".join(raw_parts) or f"Reference {idx}"

            references.append(
                ReferenceItem(
                    reference_id=ref_id,
                    marker=f"[{idx}]",
                    raw_text=raw_text,
                    doi=ref_doi,
                    title=ref_title or None,
                    authors=ref_authors,
                    year=ref_year,
                    venue=venue,
                    resolution_status="resolved" if ref_doi else "unresolved",
                )
            )

        # ── In-text citation mentions ─────────────────────────────────────────
        citation_mentions: List[CitationMention] = []
        block_counter = 0
        for div in root.findall(".//tei:body//tei:div", _NS):
            for p_el in div.findall(".//tei:p", _NS):
                block_counter += 1
                block_id = f"b{block_counter}"
                # Collect sentence text
                sentence = "".join(p_el.itertext()).strip()[:300]
                for ref_el in p_el.findall(".//tei:ref[@type='bibr']", _NS):
                    target = ref_el.get("target", "").lstrip("#")
                    if target in ref_id_map:
                        citation_mentions.append(
                            CitationMention(
                                paper_id=paper_id,
                                block_id=block_id,
                                reference_id=ref_id_map[target],
                                sentence=sentence,
                            )
                        )

        logger.info(
            f"GROBID: '{metadata.title}' — "
            f"{len(authors)} authors, {len(references)} refs, "
            f"{len(citation_mentions)} citation mentions"
        )
        return metadata, references, citation_mentions


grobid_client = GROBIDClient()
