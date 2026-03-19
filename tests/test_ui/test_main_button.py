"""Тесты для ankiforge.ui.main_button — кнопка Generate Cards на главном экране."""

from __future__ import annotations

from unittest.mock import MagicMock

from ankiforge.ui.main_button import (
    MODE_DESCRIPTIONS,
    setup_main_button,
)

# ---------------------------------------------------------------------------
# Тесты MODE_DESCRIPTIONS
# ---------------------------------------------------------------------------


class TestModeDescriptions:
    """Проверяет что описания режимов корректны."""

    def test_all_five_modes_present(self) -> None:
        """Все 5 режимов генерации имеют описания."""
        from ankiforge.models import GenerationMode

        for mode in GenerationMode:
            assert mode in MODE_DESCRIPTIONS, f"Нет описания для {mode}"

    def test_descriptions_have_title_and_subtitle(self) -> None:
        """Каждое описание содержит title и subtitle."""
        for mode, desc in MODE_DESCRIPTIONS.items():
            assert "title" in desc, f"Нет title для {mode}"
            assert "subtitle" in desc, f"Нет subtitle для {mode}"
            assert len(desc["title"]) > 0
            assert len(desc["subtitle"]) > 0


# ---------------------------------------------------------------------------
# Тесты setup_main_button
# ---------------------------------------------------------------------------


class TestSetupMainButton:
    """Проверяет регистрацию кнопки."""

    def test_setup_main_button_adds_action(self) -> None:
        """setup_main_button добавляет кнопку в toolbar."""
        mw = MagicMock()
        mw.form.deckBrowser.web = MagicMock()

        setup_main_button(mw)

        # Проверяем что addToolBar был вызван или кнопка добавлена
        assert mw.form.menuTools.addSeparator.called or mw.form.menuTools.addAction.called

    def test_setup_main_button_registers_menu_action(self) -> None:
        """setup_main_button добавляет пункт в меню Tools."""
        mw = MagicMock()
        action = MagicMock()
        mw.form.menuTools.addAction.return_value = action

        setup_main_button(mw)

        mw.form.menuTools.addAction.assert_called()
        # Проверяем текст содержит Generate Cards
        call_args = mw.form.menuTools.addAction.call_args
        assert "Generate Cards" in str(call_args)
