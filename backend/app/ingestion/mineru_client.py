"""
MinerU Client — Step 2 of the hybrid ingestion pipeline.

Responsibilities (strict, no overlap with GROBID):
  - Clean Markdown full-text with LaTeX equations preserved
  - HTML tables (converted to Markdown)
  - Figure extraction with captions and crop images
  - Section-aware chunking anchors

MinerU is called as a subprocess (magic-pdf CLI) since it has no REST API.
Falls back to PyMuPDF if MinerU is unavailable.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.schemas.domain import FigureItem, TableItem, TextBlock


class MinerUClient:
    """
    Wraps the `magic-pdf` CLI to extract clean Markdown from a PDF.

    magic-pdf writes its output to a directory structured as:
      <output_dir>/<stem>/auto/<stem>.md
      <output_dir>/<stem>/auto/images/*.png   (figures)
    """

    def __init__(self, output_base: Optional[Path] = None):
        self.output_base = output_base or Path(settings.DATA_DIR) / "mineru_output"
        self.output_base.mkdir(parents=True, exist_ok=True)

    async def is_available(self) -> bool:
        return shutil.which("magic-pdf") is not None

    async def process_pdf(
        self, pdf_path: Path, paper_id: str, crops_dir: Path
    ) -> Tuple[List[TextBlock], List[FigureItem], List[TableItem], List[str]]:
        """
        Runs magic-pdf on the PDF and returns parsed content.
        Falls back to PyMuPDF on failure.
        Returns: (blocks, figures, tables, page_image_paths)
        """
        if not await self.is_available():
            logger.warning(
                "magic-pdf not found — falling back to PyMuPDF for full-text."
            )
            return await self._fallback_pymupdf(pdf_path, paper_id, crops_dir)

        try:
            return await self._run_mineru(pdf_path, paper_id, crops_dir)
        except Exception as exc:
            logger.error(
                f"MinerU failed for {paper_id}: {exc}. Falling back to PyMuPDF."
            )
            return await self._fallback_pymupdf(pdf_path, paper_id, crops_dir)

    # ── MinerU path ───────────────────────────────────────────────────────────

    async def _run_mineru(
        self, pdf_path: Path, paper_id: str, crops_dir: Path
    ) -> Tuple[List[TextBlock], List[FigureItem], List[TableItem], List[str]]:
        out_dir = self.output_base / paper_id
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "magic-pdf",
            "-p",
            str(pdf_path),
            "-o",
            str(out_dir),
            "-m",
            "auto",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

        if proc.returncode != 0:
            raise RuntimeError(
                f"magic-pdf exited {proc.returncode}: {stderr.decode()[:300]}"
            )

        # Locate output markdown
        stem = pdf_path.stem
        md_path = out_dir / stem / "auto" / f"{stem}.md"
        if not md_path.exists():
            # Try alternative layout
            candidates = list(out_dir.rglob("*.md"))
            if not candidates:
                raise FileNotFoundError(f"MinerU produced no markdown in {out_dir}")
            md_path = candidates[0]

        markdown_text = md_path.read_text(encoding="utf-8")

        # Copy MinerU figure images into our crops dir
        paper_crops = crops_dir / paper_id
        paper_crops.mkdir(parents=True, exist_ok=True)
        mineru_images_dir = md_path.parent / "images"
        if mineru_images_dir.exists():
            for img in mineru_images_dir.glob("*.png"):
                shutil.copy2(img, paper_crops / img.name)

        blocks, figures, tables = self._parse_markdown(
            markdown_text, paper_id, paper_crops
        )
        # Always ensure full-page images exist for the UI viewer
        self._render_page_images(pdf_path, paper_crops)
        page_images = sorted(str(p) for p in paper_crops.glob("page_*.png"))

        logger.info(
            f"MinerU: {len(blocks)} blocks, {len(figures)} figures, "
            f"{len(tables)} tables for {paper_id}"
        )
        return blocks, figures, tables, page_images

    def _render_page_images(self, pdf_path: Path, paper_crops: Path) -> List[str]:
        """Pre-renders page PNG images for high-fidelity UI viewing."""
        images = []
        try:
            import pymupdf as fitz

            doc = fitz.open(str(pdf_path))
            for page_num in range(1, len(doc) + 1):
                img_path = paper_crops / f"page_{page_num}.png"
                if not img_path.exists():
                    pix = doc.load_page(page_num - 1).get_pixmap(dpi=150)
                    pix.save(str(img_path))
                images.append(str(img_path))
        except Exception as exc:
            logger.warning(f"Could not render page images: {exc}")
        return images

    def _parse_markdown(
        self, markdown: str, paper_id: str, crops_dir: Path
    ) -> Tuple[List[TextBlock], List[FigureItem], List[TableItem]]:
        """
        Converts MinerU Markdown into TextBlock / FigureItem / TableItem lists.
        Section hierarchy is determined by ATX headings (# / ##).
        """
        blocks: List[TextBlock] = []
        figures: List[FigureItem] = []
        tables: List[TableItem] = []

        section_path: List[str] = ["Document"]
        block_counter = 0
        fig_counter = 0
        tbl_counter = 0
        page = 1  # MinerU embeds <!-- page X --> comments in its markdown

        # Regex patterns
        heading_re = re.compile(r"^(#{1,4})\s+(.+)$")
        page_re = re.compile(r"<!--\s*page\s+(\d+)\s*-->", re.IGNORECASE)
        img_re = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
        table_start_re = re.compile(r"^\|.+\|$")

        lines = markdown.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]

            # Page marker
            m = page_re.search(line)
            if m:
                page = int(m.group(1))
                i += 1
                continue

            # Heading
            m = heading_re.match(line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                # Rebuild section path at this depth
                section_path = section_path[: level - 1] + [title]
                block_counter += 1
                blocks.append(
                    TextBlock(
                        block_id=f"b{block_counter}",
                        page=page,
                        bbox=[0.0, 0.0, 0.0, 0.0],
                        type="heading",
                        text=title,
                        section_path=list(section_path),
                    )
                )
                i += 1
                continue

            # Markdown image (figure)
            m = img_re.search(line)
            if m:
                caption = m.group(1).strip()
                img_rel = m.group(2).strip()
                img_name = Path(img_rel).name
                fig_counter += 1
                figures.append(
                    FigureItem(
                        figure_id=f"fig{fig_counter}",
                        page=page,
                        bbox=[0.0, 0.0, 0.0, 0.0],
                        crop_path=f"{paper_id}/{img_name}",
                        caption=caption or f"Figure {fig_counter}",
                        summary_status="pending",
                    )
                )
                i += 1
                continue

            # Markdown table
            if table_start_re.match(line):
                table_lines = []
                while i < len(lines) and table_start_re.match(lines[i]):
                    table_lines.append(lines[i])
                    i += 1
                table_md = "\n".join(table_lines)
                # Also store as a text block so it's searchable
                tbl_counter += 1
                tables.append(
                    TableItem(
                        table_id=f"tbl{tbl_counter}",
                        page=page,
                        bbox=[0.0, 0.0, 0.0, 0.0],
                        crop_path=f"{paper_id}/tbl{tbl_counter}.png",
                        caption=f"Table {tbl_counter}",
                        markdown=table_md,
                    )
                )
                block_counter += 1
                blocks.append(
                    TextBlock(
                        block_id=f"b{block_counter}",
                        page=page,
                        bbox=[0.0, 0.0, 0.0, 0.0],
                        type="caption",
                        text=table_md,
                        section_path=list(section_path),
                    )
                )
                continue

            # LaTeX equation block
            if line.strip().startswith("$$"):
                eq_lines = [line]
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("$$"):
                    eq_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    eq_lines.append(lines[i])
                    i += 1
                eq_text = "\n".join(eq_lines)
                block_counter += 1
                blocks.append(
                    TextBlock(
                        block_id=f"b{block_counter}",
                        page=page,
                        bbox=[0.0, 0.0, 0.0, 0.0],
                        type="equation",
                        text=eq_text,
                        section_path=list(section_path),
                    )
                )
                continue

            # Paragraph — accumulate consecutive non-empty lines
            if line.strip():
                para_lines = []
                while (
                    i < len(lines)
                    and lines[i].strip()
                    and not heading_re.match(lines[i])
                ):
                    para_lines.append(lines[i])
                    i += 1
                text = " ".join(para_lines).strip()
                if text:
                    block_counter += 1
                    blocks.append(
                        TextBlock(
                            block_id=f"b{block_counter}",
                            page=page,
                            bbox=[0.0, 0.0, 0.0, 0.0],
                            type="paragraph",
                            text=text,
                            section_path=list(section_path),
                        )
                    )
                continue

            i += 1

        return blocks, figures, tables

    def _find_page_images(self, out_dir: Path, paper_id: str) -> List[str]:
        """Return sorted list of page PNG paths MinerU may have produced."""
        return sorted(str(p) for p in out_dir.rglob("page_*.png"))

    # ── PyMuPDF fallback ─────────────────────────────────────────────────────

    async def _fallback_pymupdf(
        self, pdf_path: Path, paper_id: str, crops_dir: Path
    ) -> Tuple[List[TextBlock], List[FigureItem], List[TableItem], List[str]]:
        """
        Minimal PyMuPDF extraction used only when MinerU is not installed.
        Does NOT parse bibliography (that is GROBID's job).
        Renders per-page PNG images for the PDF viewer.
        """
        import pymupdf as fitz

        doc = fitz.open(str(pdf_path))
        paper_crops = crops_dir / paper_id
        paper_crops.mkdir(parents=True, exist_ok=True)

        blocks: List[TextBlock] = []
        figures: List[FigureItem] = []
        tables: List[TableItem] = []
        page_images: List[str] = []

        section_path = ["Document"]
        block_counter = 0
        fig_counter = 0

        section_heading_re = re.compile(
            r"^(\d+(\.\d+)*|[IVX]+\.)\s+([A-Z][A-Za-z0-9\s\-–]{2,50})$"
        )
        caption_re = re.compile(r"^(Figure|Fig\.|Table|Tab\.)\s+\d+", re.IGNORECASE)

        for page_num in range(1, len(doc) + 1):
            page = doc.load_page(page_num - 1)

            # Render page image for viewer
            pix = page.get_pixmap(dpi=150)
            img_path = paper_crops / f"page_{page_num}.png"
            if not img_path.exists():
                pix.save(str(img_path))
            page_images.append(str(img_path))

            page_rect = page.rect
            for pb in page.get_text("blocks"):
                x0, y0, x1, y1, text, _, b_type = pb[:7]
                if b_type != 0:  # skip image blocks
                    continue
                clean = text.strip()
                if not clean:
                    continue

                first_line = clean.split("\n")[0].strip()
                if section_heading_re.match(first_line) or (
                    len(first_line) < 60 and first_line.isupper()
                ):
                    section_path = [first_line]
                    block_type = "heading"
                elif caption_re.match(first_line):
                    block_type = "caption"
                    fig_counter += 1
                    fig_id = f"fig{fig_counter}"
                    crop_rect = fitz.Rect(
                        max(0, x0 - 20),
                        max(0, y0 - 150),
                        min(page_rect.width, x1 + 20),
                        min(page_rect.height, y1 + 10),
                    )
                    pix2 = page.get_pixmap(clip=crop_rect, dpi=200)
                    crop_file = paper_crops / f"{fig_id}.png"
                    pix2.save(str(crop_file))
                    figures.append(
                        FigureItem(
                            figure_id=fig_id,
                            page=page_num,
                            bbox=[
                                crop_rect.x0,
                                crop_rect.y0,
                                crop_rect.x1,
                                crop_rect.y1,
                            ],
                            crop_path=f"{paper_id}/{fig_id}.png",
                            caption=clean,
                            summary_status="pending",
                        )
                    )
                else:
                    block_type = "paragraph"

                block_counter += 1
                blocks.append(
                    TextBlock(
                        block_id=f"b{block_counter}",
                        page=page_num,
                        bbox=[float(x0), float(y0), float(x1), float(y1)],
                        type=block_type,
                        text=clean,
                        section_path=list(section_path),
                    )
                )

        return blocks, figures, tables, page_images


mineru_client = MinerUClient()
