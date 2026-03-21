"""Anki deck, media, and note management."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aqt.main import AnkiQt  # type: ignore[import-not-found]


def _get_mw() -> AnkiQt:
    """Get the Anki main window (mw).

    Raises:
        RuntimeError: If Anki runtime is not available.
    """
    try:
        from aqt import mw  # type: ignore[import-not-found]
    except ImportError as e:
        msg = "Anki runtime is not available"
        raise RuntimeError(msg) from e

    if mw is None:
        msg = "Anki main window is not initialized"
        raise RuntimeError(msg)

    return mw


def get_decks() -> list[str]:
    """Return a list of existing deck names.

    Returns:
        List of deck name strings.
    """
    mw = _get_mw()
    return [d.name for d in mw.col.decks.all_names_and_ids()]


def create_deck(name: str) -> None:
    """Create a new deck if it doesn't already exist.

    Args:
        name: Deck name.

    Raises:
        ValueError: If name is empty.
    """
    if not name.strip():
        msg = "Deck name cannot be empty"
        raise ValueError(msg)

    mw = _get_mw()

    if mw.col.decks.id_for_name(name) is not None:
        return

    mw.col.decks.add_normal_deck_with_name(name)


def save_media(filename: str, data: bytes) -> str:
    """Save a file to Anki's media collection with a unique name.

    Args:
        filename: Original filename (e.g. 'hello.mp3').
        data: File bytes.

    Returns:
        Unique filename (with UUID prefix).

    Raises:
        ValueError: If data is empty.
    """
    if not data:
        msg = "File data cannot be empty"
        raise ValueError(msg)

    mw = _get_mw()

    unique_name = f"{uuid.uuid4().hex[:12]}_{filename}"
    mw.col.media.write_data(unique_name, data)
    return unique_name


def add_note(deck_name: str, note_type: str, fields: dict[str, str]) -> None:
    """Add a note to the specified deck.

    Args:
        deck_name: Deck name.
        note_type: Note type name (model).
        fields: Dict of {field_name: value}.

    Raises:
        ValueError: If note type is not found.
    """
    mw = _get_mw()

    model = mw.col.models.by_name(note_type)
    if model is None:
        msg = f"Note type '{note_type}' not found"
        raise ValueError(msg)

    note = mw.col.new_note(model)
    for field_name, value in fields.items():
        note[field_name] = value

    deck_id = mw.col.decks.id_for_name(deck_name)
    if deck_id is not None:
        note.note_type()["did"] = deck_id

    mw.col.add_note(note, deck_id or 1)
