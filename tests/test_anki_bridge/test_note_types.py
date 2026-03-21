"""Tests for ankiforge.anki_bridge.note_types."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ankiforge.anki_bridge.note_types import (
    ensure_language_note_type,
    ensure_qa_audio_note_type,
    ensure_qa_image_note_type,
    ensure_qa_note_type,
)

QA_AUDIO_NOTE_TYPE_NAME = "AnkiForge QA+Audio"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NOTE_TYPE_NAME = "AnkiForge QA"
QA_IMAGE_NOTE_TYPE_NAME = "AnkiForge QA+Image"
LANGUAGE_NOTE_TYPE_NAME = "AnkiForge Language"


def _make_mock_mw() -> MagicMock:
    """Create a mock Anki main window (mw) with col.models."""
    mw = MagicMock()
    mw.col.models.by_name.return_value = None  # note type does not exist

    model: dict[str, Any] = {
        "name": "",
        "flds": [],
        "tmpls": [],
        "css": "",
    }
    mw.col.models.new.return_value = model

    # Simulate Anki behavior: new_field/new_template return dicts,
    # add_field/add_template append to model lists.
    mw.col.models.new_field.side_effect = lambda name: {"name": name}
    mw.col.models.add_field.side_effect = lambda m, f: m["flds"].append(f)
    mw.col.models.new_template.side_effect = lambda name: {"name": name, "qfmt": "", "afmt": ""}
    mw.col.models.add_template.side_effect = lambda m, t: m["tmpls"].append(t)

    return mw


@pytest.fixture()
def mock_mw() -> MagicMock:
    """Mock Anki main window (mw) with col.models."""
    return _make_mock_mw()


@pytest.fixture()
def language_mock_mw() -> MagicMock:
    """Separate mock for Language tests (own model dict)."""
    return _make_mock_mw()


# ---------------------------------------------------------------------------
# ensure_qa_note_type — creating new
# ---------------------------------------------------------------------------


class TestEnsureQaNoteTypeCreatesNew:
    """Tests for creating a new note type when it does not exist yet."""

    def test_creates_note_type_when_not_exists(self, mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=mock_mw):
            result = ensure_qa_note_type()

        mock_mw.col.models.add.assert_called_once()
        assert result is not None

    def test_sets_correct_name(self, mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=mock_mw):
            result = ensure_qa_note_type()

        assert result["name"] == NOTE_TYPE_NAME

    def test_has_question_and_answer_fields(self, mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=mock_mw):
            result = ensure_qa_note_type()

        field_names = [f["name"] for f in result["flds"]]
        assert "Question" in field_names
        assert "Answer" in field_names
        assert len(field_names) == 2

    def test_has_one_template(self, mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=mock_mw):
            result = ensure_qa_note_type()

        assert len(result["tmpls"]) == 1

    def test_front_template_contains_question(self, mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=mock_mw):
            result = ensure_qa_note_type()

        front = result["tmpls"][0]["qfmt"]
        assert "{{Question}}" in front

    def test_back_template_contains_answer(self, mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=mock_mw):
            result = ensure_qa_note_type()

        back = result["tmpls"][0]["afmt"]
        assert "{{Answer}}" in back

    def test_css_supports_night_mode(self, mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=mock_mw):
            result = ensure_qa_note_type()

        css = result["css"]
        assert ".night_mode" in css or ".nightMode" in css

    def test_css_uses_sans_serif(self, mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=mock_mw):
            result = ensure_qa_note_type()

        assert "sans-serif" in result["css"]

    def test_has_ankiforge_branding(self, mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=mock_mw):
            result = ensure_qa_note_type()

        back = result["tmpls"][0]["afmt"]
        assert "AnkiForge" in back


# ---------------------------------------------------------------------------
# ensure_qa_note_type — existing note type
# ---------------------------------------------------------------------------


class TestEnsureQaNoteTypeExisting:
    """Tests when the note type already exists."""

    def test_returns_existing_note_type(self, mock_mw: MagicMock) -> None:
        existing = {"name": NOTE_TYPE_NAME, "flds": [], "tmpls": [], "css": ""}
        mock_mw.col.models.by_name.return_value = existing

        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=mock_mw):
            result = ensure_qa_note_type()

        assert result is existing
        mock_mw.col.models.add.assert_not_called()


# ---------------------------------------------------------------------------
# HTML validity (QA)
# ---------------------------------------------------------------------------


class TestTemplateHtmlValidity:
    """Basic HTML validity check for templates."""

    def test_front_template_is_valid_html(self, mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=mock_mw):
            result = ensure_qa_note_type()

        front = result["tmpls"][0]["qfmt"]
        # Basic check: contains HTML tags
        assert "<div" in front or "<p" in front or "<span" in front

    def test_back_template_is_valid_html(self, mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=mock_mw):
            result = ensure_qa_note_type()

        back = result["tmpls"][0]["afmt"]
        assert "<div" in back or "<p" in back or "<span" in back


# ===========================================================================
# ensure_language_note_type
# ===========================================================================


class TestEnsureLanguageNoteTypeCreatesNew:
    """Tests for creating a new Language note type."""

    def test_creates_note_type_when_not_exists(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        language_mock_mw.col.models.add.assert_called_once()
        assert result is not None

    def test_sets_correct_name(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        assert result["name"] == LANGUAGE_NOTE_TYPE_NAME

    def test_has_nine_fields(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        field_names = [f["name"] for f in result["flds"]]
        assert field_names == [
            "Word",
            "Audio",
            "Definition",
            "Example",
            "Image",
            "AudioDefinition",
            "AudioSilence",
            "AudioExample",
            "Transcription",
        ]

    def test_has_word_field(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        field_names = [f["name"] for f in result["flds"]]
        assert "Word" in field_names

    def test_has_audio_field(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        field_names = [f["name"] for f in result["flds"]]
        assert "Audio" in field_names

    def test_has_definition_field(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        field_names = [f["name"] for f in result["flds"]]
        assert "Definition" in field_names

    def test_has_example_field(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        field_names = [f["name"] for f in result["flds"]]
        assert "Example" in field_names

    def test_has_image_field(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        field_names = [f["name"] for f in result["flds"]]
        assert "Image" in field_names

    def test_has_one_template(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        assert len(result["tmpls"]) == 1

    def test_front_template_contains_word(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        front = result["tmpls"][0]["qfmt"]
        assert "{{Word}}" in front

    def test_front_template_contains_audio(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        front = result["tmpls"][0]["qfmt"]
        assert "{{Audio}}" in front

    def test_back_template_contains_definition(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        back = result["tmpls"][0]["afmt"]
        assert "{{Definition}}" in back

    def test_back_template_contains_example(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        back = result["tmpls"][0]["afmt"]
        assert "{{Example}}" in back

    def test_back_template_contains_image(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        back = result["tmpls"][0]["afmt"]
        assert "{{Image}}" in back

    def test_css_supports_night_mode(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        css = result["css"]
        assert ".night_mode" in css

    def test_css_uses_sans_serif(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        assert "sans-serif" in result["css"]

    def test_has_ankiforge_branding(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        back = result["tmpls"][0]["afmt"]
        assert "AnkiForge" in back

    def test_image_has_max_width(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        css = result["css"]
        assert "320px" in css

    def test_image_has_rounded_corners(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        css = result["css"]
        assert "border-radius" in css

    def test_definition_visually_highlighted(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        css = result["css"]
        assert ".definition" in css

    def test_example_italic_style(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        css = result["css"]
        assert "italic" in css


# ---------------------------------------------------------------------------
# ensure_language_note_type — existing
# ---------------------------------------------------------------------------


class TestEnsureLanguageNoteTypeExisting:
    """Tests when Language note type already exists."""

    def test_returns_existing_note_type_and_upgrades(self, language_mock_mw: MagicMock) -> None:
        existing: dict[str, Any] = {
            "name": LANGUAGE_NOTE_TYPE_NAME,
            "flds": [{"name": n} for n in ("Word", "Audio", "Definition", "Example", "Image")],
            "tmpls": [{"qfmt": "old", "afmt": "old"}],
            "css": "old",
        }
        language_mock_mw.col.models.by_name.return_value = existing

        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        assert result is existing
        language_mock_mw.col.models.add.assert_not_called()
        # Upgrade adds 4 new fields (AudioDefinition, AudioSilence, AudioExample, Transcription)
        assert language_mock_mw.col.models.add_field.call_count == 4
        language_mock_mw.col.models.save.assert_called_once()


# ---------------------------------------------------------------------------
# HTML validity (Language)
# ---------------------------------------------------------------------------


class TestLanguageTemplateHtmlValidity:
    """Basic HTML validity check for Language templates."""

    def test_front_template_is_valid_html(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        front = result["tmpls"][0]["qfmt"]
        assert "<div" in front

    def test_back_template_is_valid_html(self, language_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=language_mock_mw):
            result = ensure_language_note_type()

        back = result["tmpls"][0]["afmt"]
        assert "<div" in back


# ===========================================================================
# ensure_qa_image_note_type
# ===========================================================================

QA_IMAGE_NOTE_TYPE_NAME = "AnkiForge QA+Image"


@pytest.fixture()
def qa_image_mock_mw() -> MagicMock:
    """Mock for QA+Image tests."""
    return _make_mock_mw()


class TestEnsureQaImageNoteTypeCreatesNew:
    """Tests for creating a new QA+Image note type."""

    def test_creates_note_type_when_not_exists(self, qa_image_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_image_mock_mw):
            result = ensure_qa_image_note_type()

        qa_image_mock_mw.col.models.add.assert_called_once()
        assert result is not None

    def test_sets_correct_name(self, qa_image_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_image_mock_mw):
            result = ensure_qa_image_note_type()

        assert result["name"] == QA_IMAGE_NOTE_TYPE_NAME

    def test_has_three_fields(self, qa_image_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_image_mock_mw):
            result = ensure_qa_image_note_type()

        field_names = [f["name"] for f in result["flds"]]
        assert field_names == ["Question", "Answer", "Image"]

    def test_has_question_field(self, qa_image_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_image_mock_mw):
            result = ensure_qa_image_note_type()

        field_names = [f["name"] for f in result["flds"]]
        assert "Question" in field_names

    def test_has_answer_field(self, qa_image_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_image_mock_mw):
            result = ensure_qa_image_note_type()

        field_names = [f["name"] for f in result["flds"]]
        assert "Answer" in field_names

    def test_has_image_field(self, qa_image_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_image_mock_mw):
            result = ensure_qa_image_note_type()

        field_names = [f["name"] for f in result["flds"]]
        assert "Image" in field_names

    def test_has_one_template(self, qa_image_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_image_mock_mw):
            result = ensure_qa_image_note_type()

        assert len(result["tmpls"]) == 1

    def test_front_template_contains_question(self, qa_image_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_image_mock_mw):
            result = ensure_qa_image_note_type()

        front = result["tmpls"][0]["qfmt"]
        assert "{{Question}}" in front

    def test_back_template_contains_answer(self, qa_image_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_image_mock_mw):
            result = ensure_qa_image_note_type()

        back = result["tmpls"][0]["afmt"]
        assert "{{Answer}}" in back

    def test_back_template_contains_image(self, qa_image_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_image_mock_mw):
            result = ensure_qa_image_note_type()

        back = result["tmpls"][0]["afmt"]
        assert "{{Image}}" in back

    def test_image_has_max_width(self, qa_image_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_image_mock_mw):
            result = ensure_qa_image_note_type()

        assert "320px" in result["css"]

    def test_image_has_rounded_corners(self, qa_image_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_image_mock_mw):
            result = ensure_qa_image_note_type()

        assert "border-radius" in result["css"]

    def test_css_supports_night_mode(self, qa_image_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_image_mock_mw):
            result = ensure_qa_image_note_type()

        assert ".night_mode" in result["css"]

    def test_css_uses_sans_serif(self, qa_image_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_image_mock_mw):
            result = ensure_qa_image_note_type()

        assert "sans-serif" in result["css"]

    def test_has_ankiforge_branding(self, qa_image_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_image_mock_mw):
            result = ensure_qa_image_note_type()

        back = result["tmpls"][0]["afmt"]
        assert "AnkiForge" in back


class TestEnsureQaImageNoteTypeExisting:
    """Tests when QA+Image note type already exists."""

    def test_returns_existing_note_type(self, qa_image_mock_mw: MagicMock) -> None:
        existing = {"name": QA_IMAGE_NOTE_TYPE_NAME, "flds": [], "tmpls": [], "css": ""}
        qa_image_mock_mw.col.models.by_name.return_value = existing

        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_image_mock_mw):
            result = ensure_qa_image_note_type()

        assert result is existing
        qa_image_mock_mw.col.models.add.assert_not_called()


class TestQaImageTemplateHtmlValidity:
    """Basic HTML validity check for QA+Image templates."""

    def test_front_template_is_valid_html(self, qa_image_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_image_mock_mw):
            result = ensure_qa_image_note_type()

        front = result["tmpls"][0]["qfmt"]
        assert "<div" in front

    def test_back_template_is_valid_html(self, qa_image_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_image_mock_mw):
            result = ensure_qa_image_note_type()

        back = result["tmpls"][0]["afmt"]
        assert "<div" in back


# ===========================================================================
# ensure_qa_audio_note_type
# ===========================================================================

QA_AUDIO_NOTE_TYPE_NAME = "AnkiForge QA+Audio"


@pytest.fixture()
def qa_audio_mock_mw() -> MagicMock:
    """Mock for QA+Audio tests."""
    return _make_mock_mw()


class TestEnsureQaAudioNoteTypeCreatesNew:
    """Tests for creating a new QA+Audio note type."""

    def test_creates_note_type_when_not_exists(self, qa_audio_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_audio_mock_mw):
            result = ensure_qa_audio_note_type()

        qa_audio_mock_mw.col.models.add.assert_called_once()
        assert result is not None

    def test_sets_correct_name(self, qa_audio_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_audio_mock_mw):
            result = ensure_qa_audio_note_type()

        assert result["name"] == QA_AUDIO_NOTE_TYPE_NAME

    def test_has_three_fields(self, qa_audio_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_audio_mock_mw):
            result = ensure_qa_audio_note_type()

        field_names = [f["name"] for f in result["flds"]]
        assert field_names == ["Question", "Answer", "Audio"]

    def test_has_question_field(self, qa_audio_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_audio_mock_mw):
            result = ensure_qa_audio_note_type()

        field_names = [f["name"] for f in result["flds"]]
        assert "Question" in field_names

    def test_has_answer_field(self, qa_audio_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_audio_mock_mw):
            result = ensure_qa_audio_note_type()

        field_names = [f["name"] for f in result["flds"]]
        assert "Answer" in field_names

    def test_has_audio_field(self, qa_audio_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_audio_mock_mw):
            result = ensure_qa_audio_note_type()

        field_names = [f["name"] for f in result["flds"]]
        assert "Audio" in field_names

    def test_has_one_template(self, qa_audio_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_audio_mock_mw):
            result = ensure_qa_audio_note_type()

        assert len(result["tmpls"]) == 1

    def test_front_template_contains_question(self, qa_audio_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_audio_mock_mw):
            result = ensure_qa_audio_note_type()

        front = result["tmpls"][0]["qfmt"]
        assert "{{Question}}" in front

    def test_front_template_contains_audio_autoplay(self, qa_audio_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_audio_mock_mw):
            result = ensure_qa_audio_note_type()

        front = result["tmpls"][0]["qfmt"]
        assert "{{Audio}}" in front

    def test_back_template_contains_answer(self, qa_audio_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_audio_mock_mw):
            result = ensure_qa_audio_note_type()

        back = result["tmpls"][0]["afmt"]
        assert "{{Answer}}" in back

    def test_css_supports_night_mode(self, qa_audio_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_audio_mock_mw):
            result = ensure_qa_audio_note_type()

        assert ".night_mode" in result["css"]

    def test_css_uses_sans_serif(self, qa_audio_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_audio_mock_mw):
            result = ensure_qa_audio_note_type()

        assert "sans-serif" in result["css"]

    def test_has_ankiforge_branding(self, qa_audio_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_audio_mock_mw):
            result = ensure_qa_audio_note_type()

        back = result["tmpls"][0]["afmt"]
        assert "AnkiForge" in back


class TestEnsureQaAudioNoteTypeExisting:
    """Tests when QA+Audio note type already exists."""

    def test_returns_existing_note_type(self, qa_audio_mock_mw: MagicMock) -> None:
        existing = {"name": QA_AUDIO_NOTE_TYPE_NAME, "flds": [], "tmpls": [], "css": ""}
        qa_audio_mock_mw.col.models.by_name.return_value = existing

        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_audio_mock_mw):
            result = ensure_qa_audio_note_type()

        assert result is existing
        qa_audio_mock_mw.col.models.add.assert_not_called()


class TestQaAudioTemplateHtmlValidity:
    """Basic HTML validity check for QA+Audio templates."""

    def test_front_template_is_valid_html(self, qa_audio_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_audio_mock_mw):
            result = ensure_qa_audio_note_type()

        front = result["tmpls"][0]["qfmt"]
        assert "<div" in front

    def test_back_template_is_valid_html(self, qa_audio_mock_mw: MagicMock) -> None:
        with patch("ankiforge.anki_bridge.note_types._get_mw", return_value=qa_audio_mock_mw):
            result = ensure_qa_audio_note_type()

        back = result["tmpls"][0]["afmt"]
        assert "<div" in back
