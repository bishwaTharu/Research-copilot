"""
Reference Resolution and External Metadata Engine (FR-REF-01 to FR-REF-10).
Assigns explicit trust tiers (Verified, Probable, Unresolved), generates BibTeX,
checks open access URLs, and flags retractions.
"""

from __future__ import annotations
import hashlib
import json
import re
from typing import Dict, Optional, Any, List
import aiohttp

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.schemas.domain import ReferenceItem


def generate_bibtex(ref: ReferenceItem) -> str:
    """Generates standard BibTeX formatted citation string (FR-REF-07)."""
    first_author = ref.authors[0].split()[-1].lower() if ref.authors else "ref"
    clean_author = re.sub(r"\W+", "", first_author)
    year_str = str(ref.year) if ref.year else "nodate"
    cite_key = f"{clean_author}{year_str}_{ref.reference_id}"
    authors_str = " and ".join(ref.authors) if ref.authors else "Unknown"
    title_str = ref.title or ref.raw_text[:60]
    lines = [
        f"@article{{{cite_key},",
        f"  title = {{{title_str}}},",
        f"  author = {{{authors_str}}},",
    ]
    if ref.venue:
        lines.append(f"  journal = {{{ref.venue}}},")
    if ref.year:
        lines.append(f"  year = {{{ref.year}}},")
    if ref.doi:
        lines.append(f"  doi = {{{ref.doi}}},")
        lines.append(f"  url = {{https://doi.org/{ref.doi}}},")
    elif ref.open_access_url:
        lines.append(f"  url = {{{ref.open_access_url}}},")
    lines.append("}")
    return "\n".join(lines)


class ReferenceResolver:
    """
    Resolves raw citation strings to verified DOIs, abstracts, open-access links, and trust tiers.
    Caches lookups locally to respect service usage policies (FR-REF-05).
    """

    def __init__(self, cache_file: Optional[str] = None):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_file = cache_file or f"{settings.DATA_DIR}/reference_cache.json"
        self._load_cache()

    def _load_cache(self):
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        except Exception:
            self.cache = {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as exc:
            logger.error(f"Failed to persist reference cache: {exc}")

    def _cache_key(self, raw_text: str) -> str:
        clean = re.sub(r"\s+", " ", raw_text.strip().lower())
        return hashlib.sha256(clean.encode("utf-8")).hexdigest()

    async def resolve_reference(
        self, ref: ReferenceItem, allow_remote: bool = True
    ) -> ReferenceItem:
        """
        Attempts to resolve a ReferenceItem against CrossRef or OpenAlex.
        Assigns trust tiers: 'verified', 'probable', or 'unresolved' (FR-REF-08).
        Complies with NFR-PRIV-02: if allow_remote is False, suppresses outbound network calls.
        """
        key = self._cache_key(ref.raw_text)
        if key in self.cache:
            cached = self.cache[key]
            ref.doi = cached.get("doi") or ref.doi
            ref.title = cached.get("title", ref.title)
            ref.venue = cached.get("venue", ref.venue)
            ref.year = cached.get("year", ref.year)
            ref.abstract = cached.get("abstract")
            ref.open_access_url = cached.get("open_access_url") or (
                f"https://doi.org/{ref.doi}" if ref.doi else None
            )
            ref.tier = cached.get("tier", "verified" if ref.doi else "unresolved")
            ref.match_score = cached.get("match_score", 0.95 if ref.doi else 0.40)
            ref.is_retracted = cached.get("is_retracted", False)
            ref.retraction_details = cached.get("retraction_details")
            ref.resolution_status = "resolved" if ref.doi else "unresolved"
            ref.bibtex = generate_bibtex(ref)
            return ref

        # If remote lookup is prohibited (NFR-PRIV-02 for local-only papers)
        if not allow_remote:
            if ref.doi:
                ref.tier = "probable"
                ref.match_score = 0.80
                ref.resolution_status = "resolved"
                ref.open_access_url = f"https://doi.org/{ref.doi}"
            elif ref.title and len(ref.title) > 8:
                ref.tier = "probable"
                ref.match_score = 0.70
                ref.resolution_status = "unresolved"
            else:
                ref.tier = "unresolved"
                ref.match_score = 0.30
                ref.resolution_status = "unresolved"
            ref.bibtex = generate_bibtex(ref)
            return ref

        # If DOI already present in raw string or metadata
        if ref.doi:
            resolved = await self._fetch_by_doi(ref.doi)
            if resolved:
                ref.title = resolved.get("title", ref.title)
                ref.venue = resolved.get("venue", ref.venue)
                ref.year = resolved.get("year", ref.year)
                ref.abstract = resolved.get("abstract")
                ref.open_access_url = resolved.get(
                    "open_access_url", f"https://doi.org/{ref.doi}"
                )
                ref.tier = "verified"
                ref.match_score = 0.96
                ref.resolution_status = "resolved"
                ref.bibtex = generate_bibtex(ref)
                self.cache[key] = {
                    "doi": ref.doi,
                    "title": ref.title,
                    "venue": ref.venue,
                    "year": ref.year,
                    "abstract": ref.abstract,
                    "open_access_url": ref.open_access_url,
                    "tier": ref.tier,
                    "match_score": ref.match_score,
                    "is_retracted": ref.is_retracted,
                }
                self._save_cache()
                return ref

        # Query CrossRef by title/bibliographic text
        try:
            resolved = await self._query_crossref(ref.raw_text)
            if resolved:
                ref.doi = resolved.get("doi")
                ref.title = resolved.get("title")
                ref.venue = resolved.get("venue")
                ref.year = resolved.get("year", ref.year)
                ref.abstract = resolved.get("abstract")
                ref.open_access_url = resolved.get("open_access_url")

                score = resolved.get("crossref_score", 50)
                if score >= 50 and ref.doi:
                    ref.tier = "verified"
                    ref.match_score = min(0.98, 0.85 + (score / 500))
                    ref.resolution_status = "resolved"
                else:
                    ref.tier = "probable"
                    ref.match_score = 0.78
                    ref.resolution_status = "resolved"

                ref.bibtex = generate_bibtex(ref)
                self.cache[key] = {
                    "doi": ref.doi,
                    "title": ref.title,
                    "venue": ref.venue,
                    "year": ref.year,
                    "abstract": ref.abstract,
                    "open_access_url": ref.open_access_url,
                    "tier": ref.tier,
                    "match_score": ref.match_score,
                    "is_retracted": False,
                }
                self._save_cache()
                return ref
        except Exception as exc:
            logger.warning(f"CrossRef lookup failed for '{ref.raw_text[:50]}': {exc}")

        # If has title & authors from GROBID but no confirmed DOI -> Probable tier
        if ref.title and len(ref.title) > 8 and (ref.authors or ref.year):
            ref.tier = "probable"
            ref.match_score = 0.74
            ref.resolution_status = "unresolved"
        else:
            # Mark unresolved honestly (FR-REF-04)
            ref.tier = "unresolved"
            ref.match_score = 0.35
            ref.resolution_status = "unresolved"

        ref.bibtex = generate_bibtex(ref)
        self.cache[key] = {
            "doi": None,
            "tier": ref.tier,
            "match_score": ref.match_score,
            "resolution_status": ref.resolution_status,
        }
        self._save_cache()
        return ref

    async def _fetch_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        url = f"https://api.crossref.org/works/{doi}"
        headers = {
            "User-Agent": f"ResearchCopilot/1.0 (mailto:{settings.CROSSREF_MAILTO})"
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        msg = data.get("message", {})
                        title = msg.get("title", [""])[0]
                        venue = msg.get("container-title", [""])[0]
                        year = None
                        if "published-print" in msg:
                            year = msg["published-print"]["date-parts"][0][0]
                        elif "created" in msg:
                            year = msg["created"]["date-parts"][0][0]
                        abstract = msg.get("abstract")
                        is_retracted = False
                        retraction_details = None
                        updates = msg.get("update-to", [])
                        for u in updates:
                            u_type = str(u.get("type", "")).lower()
                            if "retract" in u_type:
                                is_retracted = True
                                retraction_details = f"Retracted by publisher notice (DOI: {u.get('updated', {}).get('DOI', 'N/A')})"
                                break
                            elif "correct" in u_type or "errat" in u_type:
                                retraction_details = f"Correction notice published (DOI: {u.get('updated', {}).get('DOI', 'N/A')})"

                        return {
                            "doi": doi,
                            "title": title,
                            "venue": venue,
                            "year": year,
                            "abstract": abstract,
                            "open_access_url": f"https://doi.org/{doi}",
                            "is_retracted": is_retracted,
                            "retraction_details": retraction_details,
                        }
        except Exception:
            return None
        return None

    async def _query_crossref(self, raw_text: str) -> Optional[Dict[str, Any]]:
        url = "https://api.crossref.org/works"
        params = {
            "query.bibliographic": raw_text[:200],
            "rows": 1,
            "mailto": settings.CROSSREF_MAILTO,
        }
        headers = {
            "User-Agent": f"ResearchCopilot/1.0 (mailto:{settings.CROSSREF_MAILTO})"
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, headers=headers, timeout=5
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("message", {}).get("items", [])
                        if items:
                            top = items[0]
                            score = top.get("score", 0)
                            if score > 35:
                                doi = top.get("DOI")
                                return {
                                    "doi": doi,
                                    "title": top.get("title", [""])[0],
                                    "venue": top.get("container-title", [""])[0],
                                    "year": top.get("created", {}).get(
                                        "date-parts", [[None]]
                                    )[0][0],
                                    "abstract": top.get("abstract"),
                                    "open_access_url": (
                                        f"https://doi.org/{doi}" if doi else None
                                    ),
                                    "crossref_score": score,
                                }
        except Exception:
            return None
        return None


reference_resolver = ReferenceResolver()
