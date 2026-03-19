"""Диалог выбора режима генерации карточек."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ankiforge.config.manager import is_configured
from ankiforge.models import GenerationMode

if TYPE_CHECKING:
    from collections.abc import Callable

    from aqt.main import AnkiQt  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Информация о режимах (тестируемая без Qt)
# ---------------------------------------------------------------------------

_MODE_INFO: dict[GenerationMode, dict[str, str]] = {
    GenerationMode.QUESTIONS: {
        "title": "Questions → Answers",
        "subtitle": "Введите вопросы — AI сгенерирует развёрнутые ответы",
        "icon": "\u2753",
    },
    GenerationMode.LANGUAGE: {
        "title": "Language Cards",
        "subtitle": "Слова → definition, example, аудио, картинка",
        "icon": "\U0001f30d",
    },
    GenerationMode.MATERIAL: {
        "title": "From Material",
        "subtitle": "Текст → атомарные QA-карточки (опц. с картинками)",
        "icon": "\U0001f4da",
    },
    GenerationMode.IMAGE: {
        "title": "QA + Image",
        "subtitle": "Вопросы с AI-сгенерированными иллюстрациями",
        "icon": "\U0001f5bc\ufe0f",
    },
    GenerationMode.AUDIO: {
        "title": "QA + Audio",
        "subtitle": "Вопросы с аудио-озвучкой",
        "icon": "\U0001f3a7",
    },
}


def _should_open_settings_first() -> bool:
    """Проверяет, нужно ли открыть настройки перед генерацией.

    Returns:
        True если API-ключ не настроен.
    """
    return not is_configured()


def _get_mode_info(mode: GenerationMode) -> dict[str, str]:
    """Возвращает информацию о режиме генерации.

    Args:
        mode: Режим генерации.

    Returns:
        Словарь с title, subtitle и icon.
    """
    return _MODE_INFO[mode]


# ---------------------------------------------------------------------------
# GenerateDialog
# ---------------------------------------------------------------------------


class GenerateDialog:
    """Диалог выбора режима генерации карточек.

    Показывает 5 режимов с описаниями и иконками.
    Пользователь выбирает режим — диалог возвращает выбранный GenerationMode.
    """

    def __init__(self, mw: AnkiQt) -> None:
        """Инициализация диалога выбора режима.

        Args:
            mw: Главное окно Anki.
        """
        from aqt.qt import (
            QDialog,
            QDialogButtonBox,
            QFont,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSizePolicy,
            QVBoxLayout,
        )

        self._mw = mw
        self._selected_mode: GenerationMode | None = None

        self._dialog = QDialog(mw)
        self._dialog.setWindowTitle("AnkiForge — Generate Cards")
        self._dialog.setMinimumWidth(480)

        layout = QVBoxLayout()
        self._dialog.setLayout(layout)

        # --- Заголовок ---
        title_label = QLabel("Выберите режим генерации")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # --- Кнопки режимов ---
        for mode in GenerationMode:
            info = _get_mode_info(mode)

            btn_layout = QHBoxLayout()

            # Иконка
            icon_label = QLabel(info["icon"])
            icon_font = QFont()
            icon_font.setPointSize(24)
            icon_label.setFont(icon_font)
            icon_label.setFixedWidth(48)
            btn_layout.addWidget(icon_label)

            # Текст (title + subtitle)
            text_layout = QVBoxLayout()
            mode_title = QLabel(info["title"])
            mode_title_font = QFont()
            mode_title_font.setBold(True)
            mode_title.setFont(mode_title_font)
            text_layout.addWidget(mode_title)

            mode_subtitle = QLabel(info["subtitle"])
            mode_subtitle.setStyleSheet("color: gray;")
            text_layout.addWidget(mode_subtitle)
            btn_layout.addLayout(text_layout)

            # Кнопка выбора
            select_btn = QPushButton("Выбрать")
            select_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            select_btn.clicked.connect(self._make_mode_handler(mode))
            btn_layout.addWidget(select_btn)

            layout.addLayout(btn_layout)

        # --- Cancel ---
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        button_box.rejected.connect(self._dialog.reject)
        layout.addWidget(button_box)

    def _make_mode_handler(self, mode: GenerationMode) -> Callable[[], None]:
        """Создаёт обработчик для кнопки режима.

        Args:
            mode: Режим генерации.

        Returns:
            Callable для подключения к clicked signal.
        """

        def handler() -> None:
            self._selected_mode = mode
            self._dialog.accept()

        return handler

    @property
    def selected_mode(self) -> GenerationMode | None:
        """Возвращает выбранный режим или None если отменено."""
        return self._selected_mode

    def run(self) -> GenerationMode | None:
        """Показывает диалог модально.

        Returns:
            Выбранный GenerationMode или None если пользователь отменил.
        """
        result = self._dialog.exec()
        if result and self._selected_mode is not None:
            return self._selected_mode
        return None
