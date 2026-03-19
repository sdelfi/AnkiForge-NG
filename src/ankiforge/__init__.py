"""AnkiForge — AI-powered flashcard generation for Anki via OpenRouter."""

__version__ = "0.1.0"


def _register_addon() -> None:
    """Регистрация add-on в Anki (вызывается только внутри Anki runtime)."""
    try:
        from aqt import gui_hooks, mw  # type: ignore[import-not-found]
    except ImportError:
        return

    if mw is None:
        return

    def on_main_window_init() -> None:
        """Инициализация после загрузки главного окна."""
        from ankiforge.config.dialog import setup_settings_menu
        from ankiforge.ui.main_button import setup_main_button

        setup_settings_menu(mw)
        setup_main_button(mw)

    gui_hooks.main_window_did_init.append(on_main_window_init)


_register_addon()
