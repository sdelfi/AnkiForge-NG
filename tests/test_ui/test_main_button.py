"""Тесты для ankiforge.ui.main_button — кнопка Generate Cards на главном экране."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ankiforge.ui.main_button import (
    MODE_DESCRIPTIONS,
    _on_deck_browser_content,
    _on_js_message,
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
    """Проверяет регистрацию кнопки в меню Tools."""

    def test_setup_main_button_registers_menu_action(self) -> None:
        """setup_main_button добавляет пункт в меню Tools."""
        mw = MagicMock()
        action = MagicMock()
        mw.form.menuTools.addAction.return_value = action

        setup_main_button(mw)

        mw.form.menuTools.addAction.assert_called()
        call_args = mw.form.menuTools.addAction.call_args
        assert "Generate Cards" in str(call_args)


# ---------------------------------------------------------------------------
# Тесты DeckBrowser кнопки
# ---------------------------------------------------------------------------


class TestDeckBrowserButton:
    """Проверяет функции для DeckBrowser."""

    def test_on_deck_browser_content_adds_link(self) -> None:
        """_on_deck_browser_content добавляет кнопку AnkiForge в drawLinks DeckBrowser."""
        deck_browser = MagicMock()
        deck_browser.drawLinks = [
            ["", "shared", "Get Shared"],
            ["", "create", "Create Deck"],
        ]

        _on_deck_browser_content(deck_browser, MagicMock())

        assert len(deck_browser.drawLinks) == 3
        link = deck_browser.drawLinks[2]
        assert link[1] == "_ankiforgeGenerate"
        assert link[2] == "Generate Cards"

    def test_on_deck_browser_content_no_duplicates(self) -> None:
        """_on_deck_browser_content не дублирует кнопку при повторном вызове."""
        deck_browser = MagicMock()
        deck_browser.drawLinks = [["", "_ankiforgeGenerate", "Generate Cards"]]

        _on_deck_browser_content(deck_browser, MagicMock())

        assert len(deck_browser.drawLinks) == 1

    @patch("ankiforge.ui.main_button._on_generate_clicked")
    def test_js_message_handler_handles_ankiforge(self, mock_generate: MagicMock) -> None:
        """_on_js_message обрабатывает _ankiforgeGenerate."""
        mw = MagicMock()
        result = _on_js_message(mw, (False, None), "_ankiforgeGenerate", MagicMock())

        assert result[0] is True
        mock_generate.assert_called_once_with(mw)

    def test_js_message_handler_ignores_other(self) -> None:
        """_on_js_message пропускает чужие сообщения."""
        mw = MagicMock()
        handled = (False, None)
        result = _on_js_message(mw, handled, "other_message", MagicMock())

        assert result == handled

    def test_setup_deck_browser_button_registers_hooks(self) -> None:
        """setup_deck_browser_button регистрирует хуки."""
        mock_gui_hooks = MagicMock()

        with patch.dict("sys.modules", {"aqt": MagicMock(gui_hooks=mock_gui_hooks)}):
            from ankiforge.ui.main_button import setup_deck_browser_button

            mw = MagicMock()
            setup_deck_browser_button(mw)

            mock_gui_hooks.deck_browser_will_render_content.append.assert_called_once()
            mock_gui_hooks.webview_did_receive_js_message.append.assert_called_once()
