"""AnkiForge NG — fork of AnkiForge with local/custom API support and fixes.

Fork of AnkiForge (https://github.com/Xpom1/AnkiForge) by Xpom1, GPL-3.0.
"""

__version__ = "0.2.8"


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
        from ankiforge.ui.editor_button import setup_editor_button
        from ankiforge.ui.main_button import setup_deck_browser_button, setup_main_button

        setup_settings_menu(mw)
        setup_main_button(mw)
        setup_deck_browser_button(mw)
        setup_editor_button()

    gui_hooks.main_window_did_init.append(on_main_window_init)


# Auto-register when used as dev symlink (addons21/ankiforge/).
# In packaged mode, the root __init__.py entry point calls _register_addon() instead.
if __name__ == "ankiforge":
    from pathlib import Path as _Path

    _addon_dir = _Path(__file__).parent
    if _addon_dir.parent.name == "addons21":
        _register_addon()
