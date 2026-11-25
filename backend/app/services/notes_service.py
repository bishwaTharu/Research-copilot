"""
Notes Service - Create, read, update, delete, and re-anchor notes (Addendum v1.3).
Notes are stored locally per paper as JSON files. Never sent to any model.
"""

from __future__ import annotations
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from ..core.config import settings
from ..schemas.domain import NoteRecord


def _notes_path(paper_id: str) -> Path:
    notes_dir = Path(settings.NOTES_DIR)
    notes_dir.mkdir(parents=True, exist_ok=True)
    return notes_dir / f"{paper_id}_notes.json"


def _load_notes(paper_id: str) -> List[NoteRecord]:
    path = _notes_path(paper_id)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [NoteRecord(**item) for item in data]
    except Exception:
        return []


def _save_notes(paper_id: str, notes: List[NoteRecord]) -> None:
    path = _notes_path(paper_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([n.model_dump() for n in notes], f, indent=2, ensure_ascii=False)


def create_note(
    paper_id: str,
    anchor_type: str,
    anchor_id: str,
    text: str,
    quote: str = "",
    color: str = "yellow",
    tags: Optional[List[str]] = None,
) -> NoteRecord:
    """Create and persist a new note."""
    now = datetime.now(timezone.utc).isoformat()
    note = NoteRecord(
        note_id=str(uuid.uuid4()),
        paper_id=paper_id,
        anchor_type=anchor_type,  # type: ignore
        anchor_id=anchor_id,
        quote=quote,
        text=text,
        color=color,
        created_at=now,
        updated_at=now,
        orphaned=False,
        tags=tags or [],
    )
    notes = _load_notes(paper_id)
    notes.append(note)
    _save_notes(paper_id, notes)
    return note


def list_notes(paper_id: str) -> List[NoteRecord]:
    """Return all notes for a paper."""
    return _load_notes(paper_id)


def get_note(paper_id: str, note_id: str) -> Optional[NoteRecord]:
    """Return a single note by ID."""
    for note in _load_notes(paper_id):
        if note.note_id == note_id:
            return note
    return None


def update_note(
    paper_id: str,
    note_id: str,
    text: Optional[str] = None,
    color: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Optional[NoteRecord]:
    """Update a note's text, color, or tags."""
    notes = _load_notes(paper_id)
    for i, note in enumerate(notes):
        if note.note_id == note_id:
            if text is not None:
                note.text = text
            if color is not None:
                note.color = color
            if tags is not None:
                note.tags = tags
            note.updated_at = datetime.now(timezone.utc).isoformat()
            notes[i] = note
            _save_notes(paper_id, notes)
            return note
    return None


def delete_note(paper_id: str, note_id: str) -> bool:
    """Delete a note. Returns True if deleted."""
    notes = _load_notes(paper_id)
    original_count = len(notes)
    notes = [n for n in notes if n.note_id != note_id]
    if len(notes) < original_count:
        _save_notes(paper_id, notes)
        return True
    return False


def delete_all_notes(paper_id: str) -> int:
    """Delete all notes for a paper. Returns count deleted."""
    notes = _load_notes(paper_id)
    count = len(notes)
    if count > 0:
        path = _notes_path(paper_id)
        path.unlink(missing_ok=True)
    return count


def reanchor_notes(paper_id: str, blocks: list) -> List[NoteRecord]:
    """
    After re-parse, try to re-anchor orphaned notes by text-quote matching (Addendum v1.3 Section 4.2).
    If the quote text is found in a block, update anchor_id. Otherwise mark orphaned.
    """
    notes = _load_notes(paper_id)
    if not notes:
        return notes

    # Build text lookup from blocks
    block_texts = {}
    for block in blocks:
        if hasattr(block, "block_id") and hasattr(block, "text"):
            block_texts[block.block_id] = block.text

    changed = False
    for i, note in enumerate(notes):
        if not note.quote:
            continue
        # Check if current anchor still valid
        current_text = block_texts.get(note.anchor_id, "")
        if note.quote in current_text:
            if note.orphaned:
                note.orphaned = False
                note.updated_at = datetime.now(timezone.utc).isoformat()
                notes[i] = note
                changed = True
            continue
        # Try to find the quote in another block
        found = False
        for block_id, block_text in block_texts.items():
            if note.quote in block_text:
                note.anchor_id = block_id
                note.orphaned = False
                note.updated_at = datetime.now(timezone.utc).isoformat()
                notes[i] = note
                changed = True
                found = True
                break
        if not found and not note.orphaned:
            note.orphaned = True
            note.updated_at = datetime.now(timezone.utc).isoformat()
            notes[i] = note
            changed = True

    if changed:
        _save_notes(paper_id, notes)
    return notes


def export_notes_markdown(paper_id: str) -> str:
    """Export all notes as Markdown."""
    notes = _load_notes(paper_id)
    if not notes:
        return f"# Notes for paper {paper_id}\n\nNo notes yet.\n"

    lines = [f"# Notes for paper {paper_id}\n"]
    for note in notes:
        orphan_flag = " ⚠️ ORPHANED" if note.orphaned else ""
        lines.append(f"## Note ({note.anchor_type}: {note.anchor_id}){orphan_flag}")
        if note.quote:
            lines.append(f"> {note.quote}")
        lines.append(f"\n{note.text}")
        if note.tags:
            lines.append(f"\nTags: {', '.join(note.tags)}")
        lines.append(f"\n*Created: {note.created_at}*\n")
    return "\n".join(lines)
