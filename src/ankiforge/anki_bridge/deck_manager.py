"""Управление колодами Anki, медиа-файлами и заметками."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aqt.main import AnkiQt  # type: ignore[import-not-found]


def _get_mw() -> AnkiQt:
    """Получает главное окно Anki (mw).

    Raises:
        RuntimeError: Если Anki runtime недоступен.
    """
    try:
        from aqt import mw  # type: ignore[import-not-found]
    except ImportError as e:
        msg = "Anki runtime недоступен"
        raise RuntimeError(msg) from e

    if mw is None:
        msg = "Главное окно Anki не инициализировано"
        raise RuntimeError(msg)

    return mw


def get_decks() -> list[str]:
    """Возвращает список имён существующих колод.

    Returns:
        Список строк — имена колод.
    """
    mw = _get_mw()
    return [d.name for d in mw.col.decks.all_names_and_ids()]


def create_deck(name: str) -> None:
    """Создаёт новую колоду, если она ещё не существует.

    Args:
        name: Имя колоды.

    Raises:
        ValueError: Если имя пустое.
    """
    if not name.strip():
        msg = "Имя колоды не может быть пустым"
        raise ValueError(msg)

    mw = _get_mw()

    if mw.col.decks.id_for_name(name) is not None:
        return

    mw.col.decks.add_normal_deck_with_name(name)


def save_media(filename: str, data: bytes) -> str:
    """Сохраняет файл в медиа-коллекцию Anki с уникальным именем.

    Args:
        filename: Оригинальное имя файла (например, 'hello.mp3').
        data: Байты файла.

    Returns:
        Уникальное имя файла (с UUID-префиксом).

    Raises:
        ValueError: Если данные пустые.
    """
    if not data:
        msg = "Данные файла не могут быть пустыми"
        raise ValueError(msg)

    mw = _get_mw()

    unique_name = f"{uuid.uuid4().hex[:12]}_{filename}"
    mw.col.media.write_data(unique_name, data)
    return unique_name


def add_note(deck_name: str, note_type: str, fields: dict[str, str]) -> None:
    """Добавляет заметку в указанную колоду.

    Args:
        deck_name: Имя колоды.
        note_type: Имя типа заметки (note type / model).
        fields: Словарь {имя_поля: значение}.

    Raises:
        ValueError: Если note type не найден.
    """
    mw = _get_mw()

    model = mw.col.models.by_name(note_type)
    if model is None:
        msg = f"Note type '{note_type}' не найден"
        raise ValueError(msg)

    note = mw.col.new_note(model)
    for field_name, value in fields.items():
        note[field_name] = value

    deck_id = mw.col.decks.id_for_name(deck_name)
    if deck_id is not None:
        note.note_type()["did"] = deck_id

    mw.col.add_note(note, deck_id or 1)
