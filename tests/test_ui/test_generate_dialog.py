"""Tests for ankiforge.ui.generate_dialog — generation mode selection dialog."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ankiforge.models import GenerationMode
from ankiforge.ui.generate_dialog import (
    _get_mode_info,
    _should_open_settings_first,
)

# ---------------------------------------------------------------------------
# Tests for _should_open_settings_first
# ---------------------------------------------------------------------------


class TestShouldOpenSettingsFirst:
    """Tests auto-open settings logic."""

    @patch("ankiforge.ui.generate_dialog.is_configured", return_value=False)
    def test_opens_settings_when_not_configured(self, mock_cfg: MagicMock) -> None:
        """If API key is not configured — should open settings."""
        assert _should_open_settings_first() is True

    @patch("ankiforge.ui.generate_dialog.is_configured", return_value=True)
    def test_skips_settings_when_configured(self, mock_cfg: MagicMock) -> None:
        """If API key is configured — skip settings."""
        assert _should_open_settings_first() is False


# ---------------------------------------------------------------------------
# Tests for _get_mode_info
# ---------------------------------------------------------------------------


class TestGetModeInfo:
    """Tests mode info retrieval."""

    def test_returns_info_for_all_modes(self) -> None:
        """Returns info for every mode."""
        for mode in GenerationMode:
            info = _get_mode_info(mode)
            assert "title" in info
            assert "subtitle" in info
            assert "icon" in info

    def test_questions_mode_has_correct_info(self) -> None:
        """Questions mode has correct info."""
        info = _get_mode_info(GenerationMode.QUESTIONS)
        assert len(info["title"]) > 0
        assert len(info["icon"]) > 0

    def test_language_mode_has_correct_info(self) -> None:
        """Language mode has correct info."""
        info = _get_mode_info(GenerationMode.LANGUAGE)
        assert len(info["title"]) > 0

    def test_all_modes_have_unique_titles(self) -> None:
        """All modes have unique titles."""
        titles = [_get_mode_info(m)["title"] for m in GenerationMode]
        assert len(titles) == len(set(titles))

    def test_all_modes_have_unique_icons(self) -> None:
        """All modes have unique icons."""
        icons = [_get_mode_info(m)["icon"] for m in GenerationMode]
        assert len(icons) == len(set(icons))
