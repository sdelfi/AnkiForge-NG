"""'Generate Cards' button on Anki main screen + menu integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ankiforge.models import GenerationMode

if TYPE_CHECKING:
    from aqt.main import AnkiQt  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# Generation mode descriptions
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
# Click handler
# ---------------------------------------------------------------------------


def _on_generate_clicked(mw: AnkiQt) -> None:
    """Click handler — checks config and opens mode selection dialog.

    Supports "Back" navigation from InputDialog to GenerateDialog.
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
            return  # user cancelled

        input_dialog = InputDialog(mw, selected_mode)
        go_back = input_dialog.run()

        if not go_back:
            return  # user closed or finished generation


# ---------------------------------------------------------------------------
# DeckBrowser: bottom panel button
# ---------------------------------------------------------------------------

_ANKIFORGE_CMD = "_ankiforgeGenerate"


def _on_deck_browser_content(deck_browser: object, content: object) -> None:  # noqa: ANN001
    """Add AnkiForge button to DeckBrowser bottom panel."""
    # drawLinks — attribute of DeckBrowser, not DeckBrowserContent
    # Format: [shortcut, command, label]
    for link in deck_browser.drawLinks:  # type: ignore[attr-defined]
        if link[1] == _ANKIFORGE_CMD:
            return  # already added
    deck_browser.drawLinks.append(["", _ANKIFORGE_CMD, "Generate Cards"])  # type: ignore[attr-defined]


def _on_js_message(
    mw: AnkiQt,
    handled: tuple[bool, object],
    message: str,
    context: object,
) -> tuple[bool, object]:
    """Handle JS message from AnkiForge button in DeckBrowser."""
    if message == _ANKIFORGE_CMD:
        _on_generate_clicked(mw)
        return (True, None)
    return handled


def setup_deck_browser_button(mw: AnkiQt) -> None:
    """Register AnkiForge button in DeckBrowser bottom panel.

    Args:
        mw: Anki main window.
    """
    from aqt import gui_hooks  # type: ignore[import-not-found]

    gui_hooks.deck_browser_will_render_content.append(_on_deck_browser_content)
    gui_hooks.webview_did_receive_js_message.append(lambda handled, msg, ctx: _on_js_message(mw, handled, msg, ctx))


# ---------------------------------------------------------------------------
# Tools menu
# ---------------------------------------------------------------------------


def setup_main_button(mw: AnkiQt) -> None:
    """Add 'Generate Cards' item to Tools menu.

    Args:
        mw: Anki main window.
    """
    action = mw.form.menuTools.addAction("Generate Cards...")
    action.triggered.connect(lambda: _on_generate_clicked(mw))
