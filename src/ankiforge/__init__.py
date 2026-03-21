"""AnkiForge — AI-powered flashcard generation for Anki via OpenRouter."""

__version__ = "0.1.0"


def _register_addon() -> None:
    """Register the add-on in Anki (called only inside Anki runtime)."""
    try:
        from aqt import gui_hooks, mw  # type: ignore[import-not-found]
    except ImportError:
        return

    if mw is None:
        return

    def on_main_window_init() -> None:
        """Initialize after main window is loaded."""
        from ankiforge.config.dialog import setup_settings_menu
        from ankiforge.ui.main_button import setup_deck_browser_button, setup_main_button

        setup_settings_menu(mw)
        setup_main_button(mw)
        setup_deck_browser_button(mw)

    gui_hooks.main_window_did_init.append(on_main_window_init)


_register_addon()
