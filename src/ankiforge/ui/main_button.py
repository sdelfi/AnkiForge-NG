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
        "subtitle": "Введите вопросы — AI сгенерирует ответы",
    },
    GenerationMode.LANGUAGE: {
        "title": "Language Cards",
        "subtitle": "Введите слова — получите definition, example, аудио и картинку",
    },
    GenerationMode.MATERIAL: {
        "title": "From Material",
        "subtitle": "Вставьте текст — AI извлечёт факты и создаст карточки",
    },
    GenerationMode.IMAGE: {
        "title": "QA + Image",
        "subtitle": "Вопросы с AI-сгенерированными иллюстрациями",
    },
    GenerationMode.AUDIO: {
        "title": "QA + Audio",
        "subtitle": "Вопросы с аудио-озвучкой ответов",
    },
}


# ---------------------------------------------------------------------------
# Регистрация кнопки
# ---------------------------------------------------------------------------


def setup_main_button(mw: AnkiQt) -> None:
    """Добавляет пункт 'Generate Cards' в меню Tools.

    При клике проверяет конфигурацию и открывает диалог выбора режима.

    Args:
        mw: Главное окно Anki.
    """
    action = mw.form.menuTools.addAction("Generate Cards...")
    action.triggered.connect(lambda: _on_generate_clicked(mw))


def _on_generate_clicked(mw: AnkiQt) -> None:
    """Обработчик клика — проверяет конфиг и открывает диалог выбора режима."""
    from ankiforge.ui.generate_dialog import _should_open_settings_first

    if _should_open_settings_first():
        from ankiforge.config.dialog import SettingsDialog

        settings_dialog = SettingsDialog(mw)
        result = settings_dialog.run()
        # Если пользователь отменил настройки — не открываем генерацию
        if result == 0:  # QDialog.DialogCode.Rejected
            return

    from ankiforge.ui.generate_dialog import GenerateDialog

    generate_dialog = GenerateDialog(mw)
    selected_mode = generate_dialog.run()

    if selected_mode is not None:
        from ankiforge.ui.generate_dialog import InputDialog

        input_dialog = InputDialog(mw, selected_mode)
        input_dialog.run()
