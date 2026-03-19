"""Диалог выбора режима генерации и ввода данных для генерации карточек."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ankiforge.anki_bridge.deck_manager import add_note, create_deck, save_media
from ankiforge.anki_bridge.note_types import (
    LANGUAGE_NOTE_TYPE_NAME,
    QA_AUDIO_NOTE_TYPE_NAME,
    QA_IMAGE_NOTE_TYPE_NAME,
)
from ankiforge.config.manager import get_config, is_configured
from ankiforge.models import CardRequest, GeneratedCard, GenerationMode, GenerationProgress

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Protocol

    from aqt.main import AnkiQt  # type: ignore[import-not-found]

    from ankiforge.openrouter.client import OpenRouterClient

    class _CardGenerator(Protocol):
        def generate(
            self,
            request: CardRequest,
            progress_callback: Callable[[GenerationProgress], None],
        ) -> list[GeneratedCard]: ...


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

_INPUT_PLACEHOLDERS: dict[GenerationMode, str] = {
    GenerationMode.QUESTIONS: "Введите вопросы (по одному на строку):\n\nЧто такое Python?\nКак работает GIL?",
    GenerationMode.LANGUAGE: "Введите слова (по одному на строку или через запятую):\n\nabandon\nserendipity",
    GenerationMode.MATERIAL: "Вставьте учебный текст для генерации карточек...",
    GenerationMode.IMAGE: "Введите вопросы (по одному на строку):\n\nКак выглядит митохондрия?\nСтруктура ДНК",
    GenerationMode.AUDIO: "Введите вопросы (по одному на строку):\n\nWhat is photosynthesis?\nExplain osmosis",
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


def _get_input_placeholder(mode: GenerationMode) -> str:
    """Возвращает placeholder-текст для поля ввода в зависимости от режима.

    Args:
        mode: Режим генерации.

    Returns:
        Текст-подсказка.
    """
    return _INPUT_PLACEHOLDERS[mode]


def _should_show_images_checkbox(mode: GenerationMode) -> bool:
    """Определяет, нужно ли показывать чекбокс 'Добавить картинки'.

    Args:
        mode: Режим генерации.

    Returns:
        True только для режима MATERIAL.
    """
    return mode == GenerationMode.MATERIAL


# ---------------------------------------------------------------------------
# Фабрика генераторов (тестируемая без Qt)
# ---------------------------------------------------------------------------


def _create_generator(
    *,
    mode: GenerationMode,
    client: OpenRouterClient,
    text_model: str,
    image_model: str,
    audio_model: str,
    include_images: bool = False,
) -> _CardGenerator:
    """Создаёт генератор карточек по режиму.

    Args:
        mode: Режим генерации.
        client: OpenRouter клиент.
        text_model: ID текстовой модели.
        image_model: ID image модели.
        audio_model: ID audio модели.
        include_images: Добавлять ли картинки (для material).

    Returns:
        Экземпляр генератора.
    """
    if mode == GenerationMode.QUESTIONS:
        from ankiforge.generators.questions import QuestionsGenerator

        return QuestionsGenerator(client, text_model)

    if mode == GenerationMode.LANGUAGE:
        from ankiforge.generators.language import LanguageGenerator

        return LanguageGenerator(client, text_model, audio_model, image_model)

    if mode == GenerationMode.MATERIAL:
        from ankiforge.generators.material import MaterialGenerator

        return MaterialGenerator(client, text_model, image_model=image_model if include_images else None)

    if mode == GenerationMode.IMAGE:
        from ankiforge.generators.image import ImageGenerator

        return ImageGenerator(client, text_model, image_model)

    if mode == GenerationMode.AUDIO:
        from ankiforge.generators.audio import AudioGenerator

        return AudioGenerator(client, text_model, audio_model)

    msg = f"Неизвестный режим генерации: {mode}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Сборка CardRequest (тестируемая без Qt)
# ---------------------------------------------------------------------------


def _build_card_request(
    *,
    mode: GenerationMode,
    input_text: str,
    deck_name: str,
    create_new_deck: bool,
    include_images: bool,
    language: str,
) -> CardRequest:
    """Собирает CardRequest из параметров формы.

    Args:
        mode: Режим генерации.
        input_text: Введённый текст.
        deck_name: Имя колоды.
        create_new_deck: Создать новую колоду.
        include_images: Добавлять картинки (для material).
        language: Язык карточек.

    Returns:
        CardRequest.

    Raises:
        ValueError: Если данные невалидны.
    """
    if not input_text.strip():
        msg = "Введите данные для генерации"
        raise ValueError(msg)
    if not deck_name.strip():
        msg = "Укажите имя колоды"
        raise ValueError(msg)

    return CardRequest(
        mode=mode,
        input_text=input_text.strip(),
        target_deck=deck_name.strip(),
        create_new_deck=create_new_deck,
        include_images=include_images,
        language=language,
    )


# ---------------------------------------------------------------------------
# Сохранение карточек в колоду (тестируемое без Qt)
# ---------------------------------------------------------------------------


def _build_note_fields(card: GeneratedCard) -> dict[str, str]:
    """Формирует словарь полей для add_note из GeneratedCard.

    Args:
        card: Сгенерированная карточка.

    Returns:
        Словарь {имя_поля: значение}.
    """
    if card.note_type == LANGUAGE_NOTE_TYPE_NAME:
        audio_ref = ""
        if card.audio_data:
            media_name = save_media("audio.mp3", card.audio_data)
            audio_ref = f"[sound:{media_name}]"

        image_ref = ""
        if card.image_data:
            media_name = save_media("image.png", card.image_data)
            image_ref = f'<img src="{media_name}">'

        return {
            "Word": card.word,
            "Definition": card.definition or "",
            "Example": card.example or "",
            "Audio": audio_ref,
            "Image": image_ref,
        }

    if card.note_type == QA_IMAGE_NOTE_TYPE_NAME:
        image_ref = ""
        if card.image_data:
            media_name = save_media("image.png", card.image_data)
            image_ref = f'<img src="{media_name}">'

        return {
            "Question": card.word,
            "Answer": card.answer or "",
            "Image": image_ref,
        }

    if card.note_type == QA_AUDIO_NOTE_TYPE_NAME:
        audio_ref = ""
        if card.audio_data:
            media_name = save_media("audio.mp3", card.audio_data)
            audio_ref = f"[sound:{media_name}]"

        return {
            "Question": card.word,
            "Answer": card.answer or "",
            "Audio": audio_ref,
        }

    # QA (default)
    return {
        "Question": card.word,
        "Answer": card.answer or "",
    }


def _save_cards_to_deck(
    cards: list[GeneratedCard],
    deck_name: str,
    *,
    create_new: bool,
) -> None:
    """Сохраняет сгенерированные карточки в колоду Anki.

    Args:
        cards: Список карточек.
        deck_name: Имя целевой колоды.
        create_new: Создать колоду если не существует.
    """
    if not cards:
        return

    if create_new:
        create_deck(deck_name)

    for card in cards:
        fields = _build_note_fields(card)
        add_note(deck_name, card.note_type, fields)


# ---------------------------------------------------------------------------
# GenerateDialog — выбор режима
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
        from aqt.qt import (  # type: ignore[import-not-found]
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


# ---------------------------------------------------------------------------
# InputDialog — ввод данных и запуск генерации
# ---------------------------------------------------------------------------

_NEW_DECK_OPTION = "— Создать новую —"


class InputDialog:
    """Диалог ввода данных для генерации карточек.

    Показывает: текстовое поле ввода, выбор колоды, опцию создания новой колоды,
    чекбокс картинок (для material), кнопку Generate.
    """

    def __init__(self, mw: AnkiQt, mode: GenerationMode) -> None:
        """Инициализация диалога ввода данных.

        Args:
            mw: Главное окно Anki.
            mode: Выбранный режим генерации.
        """
        from aqt.qt import (
            QCheckBox,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QFont,
            QFormLayout,
            QLabel,
            QLineEdit,
            QPlainTextEdit,
            QVBoxLayout,
        )

        self._mw = mw
        self._mode = mode

        self._dialog = QDialog(mw)
        info = _get_mode_info(mode)
        self._dialog.setWindowTitle(f"AnkiForge — {info['title']}")
        self._dialog.setMinimumWidth(550)
        self._dialog.setMinimumHeight(400)

        layout = QVBoxLayout()
        self._dialog.setLayout(layout)

        # --- Заголовок ---
        header = QLabel(f"{info['icon']} {info['title']}")
        header_font = QFont()
        header_font.setPointSize(13)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)

        # --- Текстовое поле ввода ---
        self._input_text = QPlainTextEdit()
        self._input_text.setPlaceholderText(_get_input_placeholder(mode))
        self._input_text.setMinimumHeight(150)
        layout.addWidget(self._input_text)

        # --- Форма настроек ---
        form = QFormLayout()

        # Выбор колоды
        self._deck_combo = QComboBox()
        self._deck_combo.addItem(_NEW_DECK_OPTION, "")
        try:
            from ankiforge.anki_bridge.deck_manager import get_decks

            for name in get_decks():
                self._deck_combo.addItem(name, name)
            # По умолчанию выбираем первую реальную колоду если есть
            if self._deck_combo.count() > 1:
                self._deck_combo.setCurrentIndex(1)
        except RuntimeError:
            pass
        self._deck_combo.currentIndexChanged.connect(self._on_deck_changed)
        form.addRow("Колода:", self._deck_combo)

        # Поле для имени новой колоды
        self._new_deck_input = QLineEdit()
        self._new_deck_input.setPlaceholderText("Имя новой колоды")
        self._new_deck_input.setVisible(self._deck_combo.currentData() == "")
        form.addRow("Новая колода:", self._new_deck_input)

        # Чекбокс картинок (только для material)
        self._images_checkbox = QCheckBox("Добавить картинки к карточкам")
        self._images_checkbox.setVisible(_should_show_images_checkbox(mode))
        form.addRow("", self._images_checkbox)

        layout.addLayout(form)

        # --- Статус ---
        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        # --- Кнопки ---
        button_box = QDialogButtonBox()
        self._generate_btn = button_box.addButton("Generate", QDialogButtonBox.ButtonRole.AcceptRole)
        self._generate_btn.clicked.connect(self._on_generate)
        button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        button_box.rejected.connect(self._dialog.reject)
        layout.addWidget(button_box)

    def _on_deck_changed(self, _index: int) -> None:
        """Показывает/скрывает поле новой колоды."""
        is_new = self._deck_combo.currentData() == ""
        self._new_deck_input.setVisible(is_new)

    def _get_deck_name(self) -> tuple[str, bool]:
        """Возвращает имя колоды и флаг создания новой.

        Returns:
            (deck_name, create_new).
        """
        if self._deck_combo.currentData() == "":
            return self._new_deck_input.text(), True
        return str(self._deck_combo.currentData()), False

    def _on_generate(self) -> None:
        """Обработчик кнопки Generate — валидация, генерация, сохранение."""
        deck_name, create_new = self._get_deck_name()
        config = get_config()

        try:
            request = _build_card_request(
                mode=self._mode,
                input_text=self._input_text.toPlainText(),
                deck_name=deck_name,
                create_new_deck=create_new,
                include_images=self._images_checkbox.isChecked(),
                language=config.language,
            )
        except ValueError as e:
            self._status_label.setText(str(e))
            self._status_label.setStyleSheet("color: red;")
            return

        self._generate_btn.setEnabled(False)
        self._status_label.setText("Генерация...")
        self._status_label.setStyleSheet("color: blue;")

        try:
            from ankiforge.openrouter.client import OpenRouterClient

            client = OpenRouterClient(api_key=config.api_key)
            generator = _create_generator(
                mode=self._mode,
                client=client,
                text_model=config.text_model,
                image_model=config.image_model,
                audio_model=config.audio_model,
                include_images=request.include_images,
            )

            def progress_cb(progress: GenerationProgress) -> None:
                self._status_label.setText(f"Генерация: {progress.completed_cards} / {progress.total_cards}")

            cards = generator.generate(request, progress_cb)

            # Ensure note types exist
            self._ensure_note_types(self._mode)

            _save_cards_to_deck(cards, request.target_deck, create_new=request.create_new_deck)

            self._status_label.setText(f"Готово! Создано карточек: {len(cards)}")
            self._status_label.setStyleSheet("color: green;")
        except Exception as e:  # noqa: BLE001
            self._status_label.setText(f"Ошибка: {e}")
            self._status_label.setStyleSheet("color: red;")
        finally:
            self._generate_btn.setEnabled(True)

    def _ensure_note_types(self, mode: GenerationMode) -> None:
        """Создаёт нужные note types для выбранного режима."""
        from ankiforge.anki_bridge.note_types import (
            ensure_language_note_type,
            ensure_qa_audio_note_type,
            ensure_qa_image_note_type,
            ensure_qa_note_type,
        )

        if mode == GenerationMode.LANGUAGE:
            ensure_language_note_type()
        elif mode == GenerationMode.IMAGE:
            ensure_qa_image_note_type()
        elif mode == GenerationMode.AUDIO:
            ensure_qa_audio_note_type()
        elif mode == GenerationMode.MATERIAL:
            ensure_qa_note_type()
            ensure_qa_image_note_type()
        else:
            ensure_qa_note_type()

    def run(self) -> None:
        """Показывает диалог модально."""
        self._dialog.exec()
