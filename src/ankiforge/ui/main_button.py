"""Кнопка 'Generate Cards' на главном экране Anki + интеграция с меню."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ankiforge.models import GenerationMode

if TYPE_CHECKING:
    from aqt.main import AnkiQt  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# Описания режимов генерации
# ---------------------------------------------------------------------------

MODE_DESCRIPTIONS: dict[GenerationMode, dict[str, str]] = {
    GenerationMode.QUESTIONS: {
        "title": "Questions → Answers",
        "subtitle": "Enter questions — AI generates answers",
    },
    GenerationMode.LANGUAGE: {
        "title": "Language Cards",
        "subtitle": "Enter words — get definition, example, audio and image",
    },
    GenerationMode.MATERIAL: {
        "title": "From Material",
        "subtitle": "Paste text — AI extracts facts and creates cards",
    },
    GenerationMode.IMAGE: {
        "title": "QA + Image",
        "subtitle": "Questions with AI-generated illustrations",
    },
    GenerationMode.AUDIO: {
        "title": "QA + Audio",
        "subtitle": "Questions with text-to-speech audio",
    },
}


# ---------------------------------------------------------------------------
# Обработчик клика
# ---------------------------------------------------------------------------


def _on_generate_clicked(mw: AnkiQt) -> None:
    """Обработчик клика — проверяет конфиг и открывает диалог выбора режима.

    Поддерживает навигацию "Назад" из InputDialog в GenerateDialog.
    """
    from ankiforge.ui.generate_dialog import _should_open_settings_first

    if _should_open_settings_first():
        from ankiforge.config.dialog import SettingsDialog

        settings_dialog = SettingsDialog(mw)
        result = settings_dialog.run()
        if result == 0:  # QDialog.DialogCode.Rejected
            return

    from ankiforge.ui.generate_dialog import GenerateDialog, InputDialog

    while True:
        generate_dialog = GenerateDialog(mw)
        selected_mode = generate_dialog.run()

        if selected_mode is None:
            return  # пользователь отменил

        input_dialog = InputDialog(mw, selected_mode)
        go_back = input_dialog.run()

        if not go_back:
            return  # пользователь закрыл или завершил генерацию


# ---------------------------------------------------------------------------
# DeckBrowser: кнопка в нижней панели
# ---------------------------------------------------------------------------

_ANKIFORGE_CMD = "_ankiforgeGenerate"


def _on_deck_browser_content(deck_browser: object, content: object) -> None:  # noqa: ANN001
    """Добавляет кнопку AnkiForge в нижнюю панель DeckBrowser."""
    # drawLinks — атрибут DeckBrowser, не DeckBrowserContent
    # Формат: [shortcut, command, label]
    for link in deck_browser.drawLinks:  # type: ignore[attr-defined]
        if link[1] == _ANKIFORGE_CMD:
            return  # уже добавлена
    deck_browser.drawLinks.append(["", _ANKIFORGE_CMD, "Generate Cards"])  # type: ignore[attr-defined]


def _on_js_message(
    mw: AnkiQt,
    handled: tuple[bool, object],
    message: str,
    context: object,
) -> tuple[bool, object]:
    """Обрабатывает JS-сообщение от кнопки AnkiForge в DeckBrowser."""
    if message == _ANKIFORGE_CMD:
        _on_generate_clicked(mw)
        return (True, None)
    return handled


def setup_deck_browser_button(mw: AnkiQt) -> None:
    """Регистрирует кнопку AnkiForge в нижней панели DeckBrowser.

    Args:
        mw: Главное окно Anki.
    """
    from aqt import gui_hooks  # type: ignore[import-not-found]

    gui_hooks.deck_browser_will_render_content.append(_on_deck_browser_content)
    gui_hooks.webview_did_receive_js_message.append(lambda handled, msg, ctx: _on_js_message(mw, handled, msg, ctx))


# ---------------------------------------------------------------------------
# Меню Tools
# ---------------------------------------------------------------------------


def setup_main_button(mw: AnkiQt) -> None:
    """Добавляет пункт 'Generate Cards' в меню Tools.

    Args:
        mw: Главное окно Anki.
    """
    action = mw.form.menuTools.addAction("Generate Cards...")
    action.triggered.connect(lambda: _on_generate_clicked(mw))
