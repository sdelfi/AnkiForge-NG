"""Тесты для ankiforge.anki_bridge.deck_manager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ankiforge.anki_bridge.deck_manager import add_note, create_deck, get_decks, save_media

# ---------------------------------------------------------------------------
# Fixtures — мок Anki runtime (mw, mw.col, mw.col.decks, mw.col.media, etc.)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_mw() -> MagicMock:
    """Мок главного окна Anki (mw) с col, decks, media, models."""
    mw = MagicMock()

    # decks
    mw.col.decks.all_names_and_ids.return_value = [
        MagicMock(name="Default", id=1),
        MagicMock(name="Spanish", id=2),
    ]
    # Настраиваем .name вручную (MagicMock(name=...) — это имя самого мока)
    mw.col.decks.all_names_and_ids.return_value[0].name = "Default"
    mw.col.decks.all_names_and_ids.return_value[1].name = "Spanish"

    mw.col.decks.id_for_name.return_value = None  # колода не существует по умолчанию

    # media
    mw.col.media.write_data.return_value = None

    # models (note types)
    mw.col.models.by_name.return_value = {
        "name": "AnkiForge QA",
        "flds": [{"name": "Question"}, {"name": "Answer"}],
    }

    # notes
    mock_note = MagicMock()
    mw.col.new_note.return_value = mock_note

    return mw


# ---------------------------------------------------------------------------
# get_decks
# ---------------------------------------------------------------------------


class TestGetDecks:
    """Тесты get_decks()."""

    def test_returns_deck_names(self, mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.deck_manager._get_mw", return_value=mock_mw):
            result = get_decks()

        assert result == ["Default", "Spanish"]

    def test_empty_collection(self, mock_mw: MagicMock) -> None:
        mock_mw.col.decks.all_names_and_ids.return_value = []

        with patch("ankiforge.anki_bridge.deck_manager._get_mw", return_value=mock_mw):
            result = get_decks()

        assert result == []


# ---------------------------------------------------------------------------
# create_deck
# ---------------------------------------------------------------------------


class TestCreateDeck:
    """Тесты create_deck()."""

    def test_creates_new_deck(self, mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.deck_manager._get_mw", return_value=mock_mw):
            create_deck("NewDeck")

        mock_mw.col.decks.id_for_name.assert_called_once_with("NewDeck")
        mock_mw.col.decks.add_normal_deck_with_name.assert_called_once_with("NewDeck")

    def test_does_not_duplicate_existing_deck(self, mock_mw: MagicMock) -> None:
        mock_mw.col.decks.id_for_name.return_value = 42  # колода уже существует

        with patch("ankiforge.anki_bridge.deck_manager._get_mw", return_value=mock_mw):
            create_deck("ExistingDeck")

        mock_mw.col.decks.add_normal_deck_with_name.assert_not_called()

    def test_empty_name_raises(self, mock_mw: MagicMock) -> None:
        with (
            patch("ankiforge.anki_bridge.deck_manager._get_mw", return_value=mock_mw),
            pytest.raises(ValueError, match="Имя колоды не может быть пустым"),
        ):
            create_deck("")


# ---------------------------------------------------------------------------
# save_media
# ---------------------------------------------------------------------------


class TestSaveMedia:
    """Тесты save_media()."""

    def test_saves_file_with_uuid_prefix(self, mock_mw: MagicMock) -> None:
        data = b"fake-audio-data"

        with patch("ankiforge.anki_bridge.deck_manager._get_mw", return_value=mock_mw):
            result = save_media("hello.mp3", data)

        # Имя файла должно начинаться с UUID и заканчиваться оригинальным именем
        assert result.endswith("_hello.mp3")
        assert len(result) > len("hello.mp3")
        mock_mw.col.media.write_data.assert_called_once()

    def test_different_calls_produce_unique_names(self, mock_mw: MagicMock) -> None:
        data = b"data"

        with patch("ankiforge.anki_bridge.deck_manager._get_mw", return_value=mock_mw):
            name1 = save_media("file.png", data)
            name2 = save_media("file.png", data)

        assert name1 != name2

    def test_empty_data_raises(self, mock_mw: MagicMock) -> None:
        with (
            patch("ankiforge.anki_bridge.deck_manager._get_mw", return_value=mock_mw),
            pytest.raises(ValueError, match="Данные файла не могут быть пустыми"),
        ):
            save_media("file.mp3", b"")


# ---------------------------------------------------------------------------
# add_note
# ---------------------------------------------------------------------------


class TestAddNote:
    """Тесты add_note()."""

    def test_adds_note_to_deck(self, mock_mw: MagicMock) -> None:
        fields: dict[str, str] = {"Question": "What is Python?", "Answer": "A programming language"}

        with patch("ankiforge.anki_bridge.deck_manager._get_mw", return_value=mock_mw):
            add_note("Default", "AnkiForge QA", fields)

        mock_mw.col.models.by_name.assert_called_once_with("AnkiForge QA")
        mock_mw.col.add_note.assert_called_once()

    def test_unknown_note_type_raises(self, mock_mw: MagicMock) -> None:
        mock_mw.col.models.by_name.return_value = None

        with (
            patch("ankiforge.anki_bridge.deck_manager._get_mw", return_value=mock_mw),
            pytest.raises(ValueError, match="Note type .* не найден"),
        ):
            add_note("Default", "NonExistent", {"Q": "A"})

    def test_sets_deck_id_on_note(self, mock_mw: MagicMock) -> None:
        mock_mw.col.decks.id_for_name.return_value = 42
        fields: dict[str, str] = {"Question": "Q", "Answer": "A"}

        with patch("ankiforge.anki_bridge.deck_manager._get_mw", return_value=mock_mw):
            add_note("MyDeck", "AnkiForge QA", fields)

        # Проверяем что add_note был вызван с правильной колодой
        mock_mw.col.decks.id_for_name.assert_called_with("MyDeck")

    def test_populates_note_fields(self, mock_mw: MagicMock) -> None:
        fields: dict[str, str] = {"Question": "What?", "Answer": "That!"}
        mock_note = MagicMock()
        mock_mw.col.new_note.return_value = mock_note

        with patch("ankiforge.anki_bridge.deck_manager._get_mw", return_value=mock_mw):
            add_note("Default", "AnkiForge QA", fields)

        # Проверяем что поля были установлены
        assert mock_note.__setitem__.call_count == 2
