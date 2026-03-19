"""Тесты для ankiforge.ui.generate_dialog — диалог выбора режима генерации."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ankiforge.models import GenerationMode
from ankiforge.ui.generate_dialog import (
    _get_mode_info,
    _should_open_settings_first,
)

# ---------------------------------------------------------------------------
# Тесты _should_open_settings_first
# ---------------------------------------------------------------------------


class TestShouldOpenSettingsFirst:
    """Проверяет логику автооткрытия настроек."""

    @patch("ankiforge.ui.generate_dialog.is_configured", return_value=False)
    def test_opens_settings_when_not_configured(self, mock_cfg: MagicMock) -> None:
        """Если API-ключ не настроен — нужно открыть настройки."""
        assert _should_open_settings_first() is True

    @patch("ankiforge.ui.generate_dialog.is_configured", return_value=True)
    def test_skips_settings_when_configured(self, mock_cfg: MagicMock) -> None:
        """Если API-ключ настроен — пропускаем настройки."""
        assert _should_open_settings_first() is False


# ---------------------------------------------------------------------------
# Тесты _get_mode_info
# ---------------------------------------------------------------------------


class TestGetModeInfo:
    """Проверяет информацию о режимах."""

    def test_returns_info_for_all_modes(self) -> None:
        """Возвращает info для каждого режима."""
        for mode in GenerationMode:
            info = _get_mode_info(mode)
            assert "title" in info
            assert "subtitle" in info
            assert "icon" in info

    def test_questions_mode_has_correct_info(self) -> None:
        """Режим вопросов имеет корректную информацию."""
        info = _get_mode_info(GenerationMode.QUESTIONS)
        assert len(info["title"]) > 0
        assert len(info["icon"]) > 0

    def test_language_mode_has_correct_info(self) -> None:
        """Языковой режим имеет корректную информацию."""
        info = _get_mode_info(GenerationMode.LANGUAGE)
        assert len(info["title"]) > 0

    def test_all_modes_have_unique_titles(self) -> None:
        """Все режимы имеют уникальные названия."""
        titles = [_get_mode_info(m)["title"] for m in GenerationMode]
        assert len(titles) == len(set(titles))

    def test_all_modes_have_unique_icons(self) -> None:
        """Все режимы имеют уникальные иконки."""
        icons = [_get_mode_info(m)["icon"] for m in GenerationMode]
        assert len(icons) == len(set(icons))
