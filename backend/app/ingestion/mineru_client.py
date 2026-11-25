"""
MinerU Client — Step 2 of the hybrid ingestion pipeline.

Responsibilities (strict, no overlap with GROBID):
  - Clean Markdown full-text with LaTeX equations preserved
  - HTML tables (converted to Markdown)
  - Figure extraction with captions and crop images
  - Formula extraction (LaTeX, numbering, crops, confidence) per Addendum v1.3
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
from backend.app.schemas.domain import FigureItem, FormulaItem, TableItem, TextBlock


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
    ) -> Tuple[
        List[TextBlock], List[FigureItem], List[TableItem], List[str], List[FormulaItem]
    ]:
        """
        Runs magic-pdf on the PDF and returns parsed content including formulas.
        Falls back to PyMuPDF on failure.
        Returns: (blocks, figures, tables, page_image_paths, formulas)
        """
        if not await self.is_available():
            logger.warning(
                "magic-pdf not found — falling back to PyMuPDF for full-text and formula extraction."
            )
            return await self._fallback_pymupdf(pdf_path, paper_id, crops_dir)

        try:
            return await self._run_mineru(pdf_path, paper_id, crops_dir)
        except Exception as exc:
            logger.error(f"MinerU execution failed ({exc}) — falling back to PyMuPDF.")
            return await self._fallback_pymupdf(pdf_path, paper_id, crops_dir)

    # ── Subprocess runner ────────────────────────────────────────────────────

    async def _run_mineru(
        self, pdf_path: Path, paper_id: str, crops_dir: Path
    ) -> Tuple[
        List[TextBlock], List[FigureItem], List[TableItem], List[str], List[FormulaItem]
    ]:
        paper_out = self.output_base / paper_id
        paper_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            "magic-pdf",
            "-p",
            str(pdf_path),
            "-o",
            str(paper_out),
            "-m",
            "auto",
        ]

        logger.info(f"Running MinerU: {' '.join(cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"magic-pdf failed with code {proc.returncode}: {stderr.decode()[:500]}"
            )

        # Locate output .md
        md_files = list(paper_out.rglob("*.md"))
        if not md_files:
            raise FileNotFoundError(
                f"No markdown found in MinerU output directory {paper_out}"
            )
        markdown_text = md_files[0].read_text(encoding="utf-8")

        # Copy cropped figure images into shared crops_dir
        paper_crops = crops_dir / paper_id
        paper_crops.mkdir(parents=True, exist_ok=True)
        mineru_images_dir = paper_out / pdf_path.stem / "auto" / "images"
        if mineru_images_dir.exists():
            for img in mineru_images_dir.glob("*.png"):
                shutil.copy2(img, paper_crops / img.name)

        blocks, figures, tables, formulas = self._parse_markdown(
            markdown_text, paper_id, paper_crops, pdf_path
        )
        # Always ensure full-page images exist for the UI viewer
        self._render_page_images(pdf_path, paper_crops)
        page_images = sorted(str(p) for p in paper_crops.glob("page_*.png"))

        self._link_formula_mentions(blocks, formulas)

        logger.info(
            f"MinerU: {len(blocks)} blocks, {len(figures)} figures, "
            f"{len(tables)} tables, {len(formulas)} formulas for {paper_id}"
        )
        return blocks, figures, tables, page_images, formulas

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
        self,
        markdown: str,
        paper_id: str,
        crops_dir: Path,
        pdf_path: Optional[Path] = None,
    ) -> Tuple[List[TextBlock], List[FigureItem], List[TableItem], List[FormulaItem]]:
        """
        Converts MinerU Markdown into TextBlock / FigureItem / TableItem / FormulaItem lists.
        Section hierarchy is determined by ATX headings (# / ##).
        """
        blocks: List[TextBlock] = []
        figures: List[FigureItem] = []
        tables: List[TableItem] = []
        formulas: List[FormulaItem] = []

        section_path: List[str] = ["Document"]
        block_counter = 0
        fig_counter = 0
        tbl_counter = 0
        formula_counter = 0
        page = 1

        heading_re = re.compile(r"^(#{1,4})\s+(.+)$")
        page_re = re.compile(r"<!--\s*page\s+(\d+)\s*-->", re.IGNORECASE)
        img_re = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
        table_start_re = re.compile(r"^\|.+\|$")

        lines = markdown.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]

            m = page_re.search(line)
            if m:
                page = int(m.group(1))
                i += 1
                continue

            m = heading_re.match(line)
            if m:
                level = len(m.group(1))
                heading_text = m.group(2).strip()
                section_path = section_path[: level - 1] + [heading_text]
                block_counter += 1
                blocks.append(
                    TextBlock(
                        block_id=f"b{block_counter}",
                        page=page,
                        bbox=[0.0, 0.0, 0.0, 0.0],
                        type="heading",
                        text=heading_text,
                        section_path=list(section_path),
                    )
                )
                i += 1
                continue

            m = img_re.search(line)
            if m:
                caption = m.group(1).strip()
                src_filename = Path(m.group(2)).name
                fig_counter += 1
                fig_id = f"fig{fig_counter}"
                figures.append(
                    FigureItem(
                        figure_id=fig_id,
                        page=page,
                        bbox=[0.0, 0.0, 0.0, 0.0],
                        crop_path=f"{paper_id}/{src_filename}",
                        caption=caption or f"Figure {fig_counter}",
                        summary_status="pending",
                    )
                )
                block_counter += 1
                blocks.append(
                    TextBlock(
                        block_id=f"b{block_counter}",
                        page=page,
                        bbox=[0.0, 0.0, 0.0, 0.0],
                        type="caption",
                        text=caption or f"Figure {fig_counter}",
                        section_path=list(section_path),
                    )
                )
                i += 1
                continue

            if table_start_re.match(line):
                table_lines = []
                while i < len(lines) and lines[i].strip():
                    table_lines.append(lines[i])
                    i += 1
                table_md = "\n".join(table_lines)
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

            # LaTeX equation block ($$ ... $$)
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

                # Extract formula record
                clean_latex = eq_text.replace("$$", "").strip()
                formula_counter += 1
                fid = f"eq{formula_counter}"
                # Detect numbering like \tag{3} or (3) at end
                num_match = re.search(
                    r"\\tag\{([^}]+)\}|\(([0-9]+[a-zA-Z]?)\)$", clean_latex
                )
                display_num = (
                    f"({num_match.group(1) or num_match.group(2)})"
                    if num_match
                    else f"({formula_counter})"
                )
                garbled = sum(1 for c in clean_latex if ord(c) > 127)
                conf = 0.5 if garbled > 4 else 1.0

                formulas.append(
                    FormulaItem(
                        formula_id=fid,
                        page=page,
                        bbox=[0.0, 0.0, 0.0, 0.0],
                        crop_path=f"{paper_id}/{fid}.png",
                        latex=clean_latex,
                        number=display_num,
                        confidence=conf,
                        mentions=[],
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
                    and not lines[i].strip().startswith("$$")
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

        return blocks, figures, tables, formulas

    def _link_formula_mentions(
        self, blocks: List[TextBlock], formulas: List[FormulaItem]
    ) -> None:
        """Scan text blocks for 'Eq. N' / 'Equation N' and add mentions to FormulaItem."""
        if not formulas:
            return
        num_to_formula = {}
        for f in formulas:
            m = re.search(r"\d+", f.number)
            if m:
                num_to_formula[m.group(0)] = f

        eq_pattern = re.compile(
            r"\b(?:Eq(?:uation)?s?\.?\s*\(?(\d+)\)?)", re.IGNORECASE
        )
        for block in blocks:
            text = block.text or ""
            for match in eq_pattern.finditer(text):
                num = match.group(1)
                if num in num_to_formula:
                    formula = num_to_formula[num]
                    excerpt = text[
                        max(0, match.start() - 30) : min(len(text), match.end() + 30)
                    ].strip()
                    if not any(
                        m.get("block_id") == block.block_id for m in formula.mentions
                    ):
                        formula.mentions.append(
                            {
                                "block_id": block.block_id,
                                "page": block.page,
                                "excerpt": excerpt,
                            }
                        )

    # ── PyMuPDF fallback ─────────────────────────────────────────────────────

    async def _fallback_pymupdf(
        self, pdf_path: Path, paper_id: str, crops_dir: Path
    ) -> Tuple[
        List[TextBlock], List[FigureItem], List[TableItem], List[str], List[FormulaItem]
    ]:
        return self._fallback_pymupdf_sync(pdf_path, paper_id, crops_dir)

    def _fallback_pymupdf_sync(
        self, pdf_path: Path, paper_id: str, crops_dir: Path
    ) -> Tuple[
        List[TextBlock], List[FigureItem], List[TableItem], List[str], List[FormulaItem]
    ]:
        """
        Minimal PyMuPDF extraction used only when MinerU is not installed.
        Renders per-page PNG images and crops figures and formulas.
        """
        import pymupdf as fitz

        doc = fitz.open(str(pdf_path))
        paper_crops = crops_dir / paper_id
        paper_crops.mkdir(parents=True, exist_ok=True)

        blocks: List[TextBlock] = []
        figures: List[FigureItem] = []
        tables: List[TableItem] = []
        page_images: List[str] = []
        formulas: List[FormulaItem] = []

        section_path = ["Document"]
        block_counter = 0
        fig_counter = 0
        formula_counter = 0

        section_heading_re = re.compile(
            r"^(\d+(\.\d+)*|[IVX]+\.)\s+([A-Z][A-Za-z0-9\s\-–]{2,50})$"
        )
        caption_re = re.compile(r"^(Figure|Fig\.|Table|Tab\.)\s+\d+", re.IGNORECASE)
        # Math detection pattern
        math_pattern = re.compile(
            r"([a-zA-Z\d_^{}\\]+\s*=\s*[a-zA-Z\d_^{}\\+\-*/()]+|\b\d+\s*=\s*|\$\$.*?\$\$|\\\[.*?\\\])"
        )
        eq_label_pattern = re.compile(r"\(([0-9]+[a-zA-Z]?)\)\s*$")

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
                is_formula = False

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
                elif eq_label_pattern.search(clean) or (
                    math_pattern.search(clean)
                    and len(clean) < 150
                    and "\n" not in clean
                ):
                    block_type = "equation"
                    is_formula = True
                    formula_counter += 1
                    fid = f"eq{formula_counter}"
                    num_m = eq_label_pattern.search(clean)
                    disp_num = (
                        f"({num_m.group(1)})" if num_m else f"({formula_counter})"
                    )

                    # Crop formula region from page
                    eq_crop_rect = fitz.Rect(
                        max(0, x0 - 10),
                        max(0, y0 - 5),
                        min(page_rect.width, x1 + 10),
                        min(page_rect.height, y1 + 5),
                    )
                    pix_eq = page.get_pixmap(clip=eq_crop_rect, dpi=200)
                    eq_crop_file = paper_crops / f"{fid}.png"
                    pix_eq.save(str(eq_crop_file))

                    garbled = sum(1 for c in clean if ord(c) > 127)
                    conf = 0.5 if garbled > 3 else 1.0

                    formulas.append(
                        FormulaItem(
                            formula_id=fid,
                            page=page_num,
                            bbox=[
                                eq_crop_rect.x0,
                                eq_crop_rect.y0,
                                eq_crop_rect.x1,
                                eq_crop_rect.y1,
                            ],
                            crop_path=f"{paper_id}/{fid}.png",
                            latex=clean,
                            number=disp_num,
                            confidence=conf,
                            mentions=[],
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

        self._link_formula_mentions(blocks, formulas)
        return blocks, figures, tables, page_images, formulas


mineru_client = MinerUClient()
