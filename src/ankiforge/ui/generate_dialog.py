"""Диалог выбора режима генерации и ввода данных для генерации карточек."""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING

from ankiforge.anki_bridge.deck_manager import add_note, create_deck, save_media
from ankiforge.anki_bridge.note_types import (
    LANGUAGE_NOTE_TYPE_NAME,
    QA_AUDIO_NOTE_TYPE_NAME,
    QA_IMAGE_NOTE_TYPE_NAME,
)
from ankiforge.config.manager import get_config, is_configured
from ankiforge.models import (
    AddonConfig,
    AnswerDetail,
    CardRequest,
    GeneratedCard,
    GenerationMode,
    GenerationProgress,
    MaterialOptions,
)
from ankiforge.ui.progress_widget import (
    GenerationWorker,
    ProgressWidget,
    _count_input_items,
    _format_cost,
)
from ankiforge.ui.styles import DIALOG_QSS

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Protocol

    from aqt.main import AnkiQt  # type: ignore[import-not-found]
    from aqt.qt import QComboBox, QLabel, QSpinBox  # type: ignore[import-not-found]

    from ankiforge.models import LanguageOptions
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
        "subtitle": "Enter questions — AI will generate detailed answers",
        "icon": "\u2753",
    },
    GenerationMode.LANGUAGE: {
        "title": "Language Cards",
        "subtitle": "Words → definition, example, audio, image",
        "icon": "\U0001f30d",
    },
    GenerationMode.MATERIAL: {
        "title": "From Material",
        "subtitle": "Text → atomic QA cards (optionally with images)",
        "icon": "\U0001f4da",
    },
    GenerationMode.IMAGE: {
        "title": "QA + Image",
        "subtitle": "Questions with AI-generated illustrations",
        "icon": "\U0001f5bc\ufe0f",
    },
    GenerationMode.AUDIO: {
        "title": "QA + Audio",
        "subtitle": "Questions with text-to-speech audio",
        "icon": "\U0001f3a7",
    },
}

_INPUT_PLACEHOLDERS: dict[GenerationMode, str] = {
    GenerationMode.QUESTIONS: "Enter questions (one per line):\n\nWhat is Python?\nHow does GIL work?",
    GenerationMode.LANGUAGE: "Enter words (one per line or comma-separated):\n\nabandon\nserendipity",
    GenerationMode.MATERIAL: "Paste study text to generate cards from...",
    GenerationMode.IMAGE: "Enter questions (one per line):\n\nWhat does a mitochondria look like?\nDNA structure",
    GenerationMode.AUDIO: "Enter questions (one per line):\n\nWhat is photosynthesis?\nExplain osmosis",
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


def _should_show_custom_prompt(mode: GenerationMode) -> bool:
    """Определяет, нужно ли показывать поле кастомного промпта.

    Args:
        mode: Режим генерации.

    Returns:
        True для всех режимов.
    """
    return True


def _get_deck_name_from_combo(typed_text: str, existing_decks: set[str]) -> tuple[str, bool]:
    """Определяет имя колоды и нужно ли создавать новую.

    Args:
        typed_text: Текст из editable combo.
        existing_decks: Набор существующих колод.

    Returns:
        (deck_name, is_new) — имя колоды и флаг создания новой.
    """
    name = typed_text.strip()
    if not name:
        return "", True
    return name, name not in existing_decks


def _estimate_cost_from_config(
    config: AddonConfig,
    mode: GenerationMode,
    card_count: int,
    language_options: LanguageOptions | None = None,
    *,
    material_options: MaterialOptions | None = None,
    image_size: str = "auto",
) -> float | None:
    """Расчёт стоимости из кэшированного pricing в конфиге.

    Args:
        config: Конфигурация с pricing моделей.
        mode: Режим генерации.
        card_count: Количество карточек.
        language_options: Опции генерации для language режима.

    Returns:
        Стоимость в долларах или None если pricing не загружен.
    """
    if card_count <= 0:
        return 0.0

    # Средние значения токенов/символов на вызов
    avg_prompt_tokens = 200
    avg_completion_tokens = 150
    avg_audio_chars = 100

    # Средняя стоимость генерации одного изображения по размеру (USD).
    # OpenRouter API не отдаёт image output pricing (/models не содержит image_output),
    # поэтому используем эмпирические средние на основе реальных запросов.
    # Примеры: Gemini 2.5 Flash Image 0.5K=$0.02, 1K=$0.04; Gemini 3.1 Flash 1K=$0.07
    _avg_image_cost_by_size: dict[str, float] = {
        "0.5K": 0.02,
        "1K": 0.04,
        "2K": 0.10,
        "4K": 0.25,
        "auto": 0.04,
    }

    tp = config.text_model_pricing
    ip = config.image_model_pricing
    ap = config.audio_model_pricing

    # Если нет pricing — не можем посчитать
    mode_val = mode.value
    if tp is None and mode_val in ("questions", "language", "material", "image", "audio"):
        return None

    cost = 0.0

    if tp is not None and mode_val in ("questions", "language", "material", "image", "audio"):
        text_multiplier = 2 if mode_val == "material" else 1  # map-reduce: 2 вызова на абзац
        cost += (tp.prompt * avg_prompt_tokens + tp.completion * avg_completion_tokens) * card_count * text_multiplier

    if ip is not None and mode_val in ("language", "image", "material"):
        include_img = True
        img_size = image_size
        if mode_val == "language" and language_options:
            include_img = language_options.include_photo
            img_size = language_options.image_size
        elif mode_val == "material":
            include_img = material_options.include_images if material_options else False
            img_size = material_options.image_size if material_options else image_size
        if include_img:
            cost += _avg_image_cost_by_size.get(img_size, 0.04) * card_count

    if ap is not None and mode_val in ("language", "audio"):
        if mode_val == "language":
            if language_options:
                audio_count = sum(
                    [
                        language_options.include_audio_word,
                        language_options.include_audio_definition,
                        language_options.include_audio_example,
                    ]
                )
            else:
                audio_count = 3
            cost += ap.prompt * avg_audio_chars * card_count * audio_count
        else:
            cost += ap.prompt * avg_audio_chars * card_count

    return cost


def _log_cost(mode: GenerationMode, card_count: int, cost: float) -> None:
    """Записывает расход в лог-файл.

    Args:
        mode: Режим генерации.
        card_count: Количество карточек.
        cost: Стоимость в долларах.
    """
    import json
    from datetime import datetime
    from pathlib import Path

    try:
        from aqt import mw  # type: ignore[import-not-found]

        if mw is None or mw.addonManager is None:
            return
        addon_dir = Path(mw.addonManager.addonsFolder("ankiforge"))
    except Exception:  # noqa: BLE001
        return

    log_path = addon_dir / "cost_log.json"
    entries: list[dict[str, object]] = []
    if log_path.exists():
        try:
            entries = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            entries = []

    entries.append(
        {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "mode": mode.value,
            "cards": card_count,
            "cost_usd": round(cost, 6),
        }
    )

    import contextlib

    with contextlib.suppress(Exception):
        log_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def _get_default_custom_prompt(mode: GenerationMode = GenerationMode.LANGUAGE) -> str:
    """Возвращает дефолтный промпт для указанного режима.

    Args:
        mode: Режим генерации.

    Returns:
        Текст промпта по умолчанию.
    """
    if mode == GenerationMode.LANGUAGE:
        from ankiforge.generators.language import _DEFAULT_SYSTEM_PROMPT

        return _DEFAULT_SYSTEM_PROMPT

    if mode == GenerationMode.QUESTIONS:
        from ankiforge.generators.questions import _SYSTEM_PROMPT

        return _SYSTEM_PROMPT

    if mode == GenerationMode.IMAGE:
        from ankiforge.generators.image import _SYSTEM_PROMPT

        return _SYSTEM_PROMPT

    if mode == GenerationMode.AUDIO:
        from ankiforge.generators.audio import _SYSTEM_PROMPT

        return _SYSTEM_PROMPT

    if mode == GenerationMode.MATERIAL:
        from ankiforge.generators.material import _EXTRACT_PROMPT, _GENERATE_PROMPT

        return f"--- Extract phase ---\n{_EXTRACT_PROMPT}\n\n--- Generate phase ---\n{_GENERATE_PROMPT}"

    return ""


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
    config: AddonConfig | None = None,
    image_size: str = "auto",
    voice: str = "alloy",
) -> _CardGenerator:
    """Создаёт генератор карточек по режиму.

    Args:
        mode: Режим генерации.
        client: OpenRouter клиент.
        text_model: ID текстовой модели.
        image_model: ID image модели.
        audio_model: ID audio модели.
        include_images: Добавлять ли картинки (для material).
        config: Конфигурация с pricing (для real-time cost tracking).
        image_size: Размер изображения (для image/material).
        voice: Голос диктора (для audio).

    Returns:
        Экземпляр генератора.
    """
    if mode == GenerationMode.QUESTIONS:
        from ankiforge.generators.questions import QuestionsGenerator

        return QuestionsGenerator(client, text_model)

    if mode == GenerationMode.LANGUAGE:
        from ankiforge.generators.language import LanguageGenerator

        return LanguageGenerator(
            client,
            text_model,
            audio_model,
            image_model,
            text_pricing=config.text_model_pricing if config else None,
            image_pricing=config.image_model_pricing if config else None,
            audio_pricing=config.audio_model_pricing if config else None,
        )

    if mode == GenerationMode.MATERIAL:
        from ankiforge.generators.material import MaterialGenerator

        return MaterialGenerator(
            client, text_model, image_model=image_model if include_images else None, image_size=image_size
        )

    if mode == GenerationMode.IMAGE:
        from ankiforge.generators.image import ImageGenerator

        return ImageGenerator(client, text_model, image_model, image_size=image_size)

    if mode == GenerationMode.AUDIO:
        from ankiforge.generators.audio import AudioGenerator

        return AudioGenerator(client, text_model, audio_model, voice=voice)

    msg = f"Unknown generation mode: {mode}"
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
    custom_prompt: str | None = None,
    language_options: LanguageOptions | None = None,
    material_options: MaterialOptions | None = None,
    voice: str = "alloy",
    image_size: str = "auto",
) -> CardRequest:
    """Собирает CardRequest из параметров формы.

    Args:
        mode: Режим генерации.
        input_text: Введённый текст.
        deck_name: Имя колоды.
        create_new_deck: Создать новую колоду.
        include_images: Добавлять картинки (для material).
        language: Язык карточек.
        custom_prompt: Кастомный промпт.
        language_options: Опции генерации для language режима.
        material_options: Опции генерации для material режима.
        voice: Голос диктора (для audio).
        image_size: Размер изображения (для image/material).

    Returns:
        CardRequest.

    Raises:
        ValueError: Если данные невалидны.
    """
    if not input_text.strip():
        msg = "Enter data to generate"
        raise ValueError(msg)
    if not deck_name.strip():
        msg = "Specify a deck name"
        raise ValueError(msg)

    # Пустой/пробельный custom_prompt → None
    cleaned_prompt = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else None

    return CardRequest(
        mode=mode,
        input_text=input_text.strip(),
        target_deck=deck_name.strip(),
        create_new_deck=create_new_deck,
        include_images=include_images,
        language=language,
        custom_prompt=cleaned_prompt,
        language_options=language_options,
        material_options=material_options,
        voice=voice,
        image_size=image_size,
    )


# ---------------------------------------------------------------------------
# Сохранение карточек в колоду (тестируемое без Qt)
# ---------------------------------------------------------------------------


def _markdown_to_html(text: str) -> str:
    """Конвертирует markdown code blocks и inline code в HTML для Anki.

    Args:
        text: Текст с markdown-разметкой.

    Returns:
        HTML-строка с ``<pre><code>`` и ``<code>`` тегами.
    """
    import re as _re

    if not text:
        return text

    def _escape(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 1. Fenced code blocks: ```lang\n...\n```
    def _replace_block(m: _re.Match[str]) -> str:
        lang = m.group(1) or ""
        code = _escape(m.group(2))
        cls = f' class="language-{lang}"' if lang else ""
        return f"<pre><code{cls}>{code}</code></pre>"

    result = _re.sub(r"```(\w*)\n(.*?)```", _replace_block, text, flags=_re.DOTALL)

    # 2. Inline code: `...`
    def _replace_inline(m: _re.Match[str]) -> str:
        return f"<code>{_escape(m.group(1))}</code>"

    result = _re.sub(r"`([^`]+)`", _replace_inline, result)

    # 3. Newlines → <br> ТОЛЬКО вне <pre>...</pre>
    parts = _re.split(r"(<pre>.*?</pre>)", result, flags=_re.DOTALL)
    for i, part in enumerate(parts):
        if not part.startswith("<pre>"):
            parts[i] = part.replace("\n", "<br>")
    return "".join(parts)


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
            media_name = save_media("audio.wav", card.audio_data)
            audio_ref = f"[sound:{media_name}]"

        image_ref = ""
        if card.image_data:
            media_name = save_media("image.png", card.image_data)
            image_ref = f'<img src="{media_name}">'

        audio_def_ref = ""
        if card.audio_definition:
            media_name = save_media("audio_def.wav", card.audio_definition)
            audio_def_ref = f"[sound:{media_name}]"

        audio_ex_ref = ""
        if card.audio_example:
            media_name = save_media("audio_ex.wav", card.audio_example)
            audio_ex_ref = f"[sound:{media_name}]"

        audio_silence_ref = ""
        if card.audio_silence:
            media_name = save_media("silence.wav", card.audio_silence)
            audio_silence_ref = f"[sound:{media_name}]"

        return {
            "Word": card.word,
            "Definition": card.definition or "",
            "Example": card.example or "",
            "Audio": audio_ref,
            "Image": image_ref,
            "AudioDefinition": audio_def_ref,
            "AudioSilence": audio_silence_ref,
            "AudioExample": audio_ex_ref,
            "Transcription": card.transcription or "",
        }

    if card.note_type == QA_IMAGE_NOTE_TYPE_NAME:
        image_ref = ""
        if card.image_data:
            media_name = save_media("image.png", card.image_data)
            image_ref = f'<img src="{media_name}">'

        return {
            "Question": _markdown_to_html(card.word),
            "Answer": _markdown_to_html(card.answer or ""),
            "Image": image_ref,
        }

    if card.note_type == QA_AUDIO_NOTE_TYPE_NAME:
        audio_ref = ""
        if card.audio_data:
            media_name = save_media("audio.wav", card.audio_data)
            audio_ref = f"[sound:{media_name}]"

        return {
            "Question": _markdown_to_html(card.word),
            "Answer": _markdown_to_html(card.answer or ""),
            "Audio": audio_ref,
        }

    # QA (default)
    return {
        "Question": _markdown_to_html(card.word),
        "Answer": _markdown_to_html(card.answer or ""),
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
        from aqt.qt import (
            QDialog,
            QFont,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSizePolicy,
            QVBoxLayout,
        )  # noqa: F811

        self._mw = mw
        self._selected_mode: GenerationMode | None = None

        self._dialog = QDialog(mw)
        self._dialog.setWindowTitle("AnkiForge — Generate Cards")
        self._dialog.setMinimumWidth(480)
        self._dialog.setStyleSheet(DIALOG_QSS)

        layout = QVBoxLayout()
        layout.setSpacing(16)
        self._dialog.setLayout(layout)

        # --- Заголовок ---
        title_label = QLabel("Choose generation mode")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # --- Кнопки режимов ---
        for mode in GenerationMode:
            info = _get_mode_info(mode)

            btn_layout = QHBoxLayout()
            btn_layout.setSpacing(12)

            # Иконка
            icon_label = QLabel(info["icon"])
            icon_font = QFont()
            icon_font.setPointSize(24)
            icon_label.setFont(icon_font)
            icon_label.setFixedWidth(48)
            btn_layout.addWidget(icon_label)

            # Текст (title + subtitle)
            text_layout = QVBoxLayout()
            text_layout.setSpacing(2)
            mode_title = QLabel(info["title"])
            mode_title_font = QFont()
            mode_title_font.setBold(True)
            mode_title.setFont(mode_title_font)
            text_layout.addWidget(mode_title)

            mode_subtitle = QLabel(info["subtitle"])
            mode_subtitle.setStyleSheet("color: palette(mid);")
            text_layout.addWidget(mode_subtitle)
            btn_layout.addLayout(text_layout)

            # Кнопка выбора
            select_btn = QPushButton("Select")
            select_btn.setObjectName("selectBtn")
            select_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            select_btn.clicked.connect(self._make_mode_handler(mode))
            btn_layout.addWidget(select_btn)

            layout.addLayout(btn_layout)

        layout.addStretch()

        # --- Настройки / Отмена ---
        buttons_layout = QHBoxLayout()
        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self._on_open_settings)
        buttons_layout.addWidget(settings_btn)
        buttons_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._dialog.reject)
        buttons_layout.addWidget(cancel_btn)
        layout.addLayout(buttons_layout)

    def _on_open_settings(self) -> None:
        """Открывает диалог настроек."""
        from ankiforge.config.dialog import SettingsDialog

        SettingsDialog(self._mw).run()

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


class InputDialog:
    """Диалог ввода данных для генерации карточек.

    Показывает: текстовое поле ввода, выбор/создание колоды (editable combo),
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
            QCompleter,
            QDialog,
            QFont,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QPlainTextEdit,
            QPushButton,
            Qt,
            QVBoxLayout,
        )

        self._mw = mw
        self._mode = mode
        self._worker: GenerationWorker | None = None

        self._dialog = QDialog(mw)
        info = _get_mode_info(mode)
        self._dialog.setWindowTitle(f"AnkiForge — {info['title']}")
        self._dialog.setMinimumWidth(520)
        self._dialog.setStyleSheet(DIALOG_QSS)

        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(14, 14, 14, 14)
        self._dialog.setLayout(layout)

        # --- Заголовок ---
        header = QLabel(info["title"])
        header_font = QFont()
        header_font.setPointSize(13)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)

        subtitle = QLabel(info["subtitle"])
        subtitle.setStyleSheet("color: palette(mid);")
        layout.addWidget(subtitle)

        # === Секция 1: Данные для генерации ===
        input_group = QGroupBox("Generation input")
        input_vlayout = QVBoxLayout()
        input_vlayout.setContentsMargins(8, 6, 8, 8)
        self._input_text = QPlainTextEdit()
        self._input_text.setPlaceholderText(_get_input_placeholder(mode))
        self._input_text.setMinimumHeight(140)
        input_vlayout.addWidget(self._input_text)
        input_group.setLayout(input_vlayout)
        layout.addWidget(input_group)

        # === Секция 2: Настройки ===
        settings_group = QGroupBox("Settings")
        settings_form = QFormLayout()
        settings_form.setSpacing(8)
        settings_form.setContentsMargins(8, 6, 8, 8)

        # Editable combo — выбор существующей или ввод новой колоды
        self._deck_combo = QComboBox()
        self._deck_combo.setEditable(True)
        self._deck_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._deck_combo.lineEdit().setPlaceholderText("Select or create a deck...")

        deck_completer = QCompleter()
        deck_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        deck_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        deck_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._deck_combo.setCompleter(deck_completer)

        self._existing_decks: set[str] = set()
        try:
            from ankiforge.anki_bridge.deck_manager import get_decks

            for name in get_decks():
                self._deck_combo.addItem(name, name)
                self._existing_decks.add(name)
        except RuntimeError:
            pass

        # Не выбираем колоду автоматически — пусть пользователь вводит сам
        self._deck_combo.setCurrentIndex(-1)
        deck_completer.setModel(self._deck_combo.model())
        settings_form.addRow("Deck:", self._deck_combo)

        deck_hint = QLabel("Pick an existing deck from the list, or type a new name to create one")
        deck_hint.setStyleSheet("color: palette(mid); font-size: 11px;")
        deck_hint.setWordWrap(True)
        settings_form.addRow("", deck_hint)

        # Язык карточек
        self._language_combo = QComboBox()
        self._language_combo.addItem("English", "en")
        self._language_combo.addItem("Russian", "ru")
        self._language_combo.addItem("Deutsch", "de")
        self._language_combo.addItem("Français", "fr")
        self._language_combo.addItem("Español", "es")
        self._language_combo.addItem("日本語", "ja")
        self._language_combo.addItem("中文", "zh")

        # Выставляем язык из конфига
        config = get_config()
        for i in range(self._language_combo.count()):
            if self._language_combo.itemData(i) == config.language:
                self._language_combo.setCurrentIndex(i)
                break
        settings_form.addRow("Language:", self._language_combo)

        # Чекбокс картинок — создаём здесь, но добавляем в «Опции генерации» для material
        self._images_checkbox = QCheckBox("Add images to cards")

        settings_group.setLayout(settings_form)
        layout.addWidget(settings_group)

        # === Секция 2.5: Опции генерации ===
        self._lang_options_checkboxes: dict[str, QCheckBox] = {}
        self._image_size_combo: QComboBox | None = None
        self._image_size_label: QLabel | None = None
        self._voice_combo: QComboBox | None = None
        self._max_cards_spin: QSpinBox | None = None
        self._answer_detail_combo: QComboBox | None = None

        if mode == GenerationMode.LANGUAGE:
            opts_group = QGroupBox("Generation options")
            opts_form = QFormLayout()
            opts_form.setSpacing(8)
            opts_form.setContentsMargins(8, 6, 8, 8)
            opts_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
            opts_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            checkbox_defs = [
                ("include_photo", "Generate photo"),
                ("include_audio_word", "Audio: word pronunciation"),
                ("include_audio_definition", "Audio: definition narration"),
                ("include_audio_example", "Audio: example narration"),
                ("include_transcription", "Phonetic transcription (IPA)"),
                ("detailed_image", "Detailed image (HD prompt)"),
            ]
            for key, label in checkbox_defs:
                cb = QCheckBox(label)
                cb.setChecked(key not in ("detailed_image",))
                cb.stateChanged.connect(self._update_cost_estimate)
                self._lang_options_checkboxes[key] = cb
                opts_form.addRow(cb)

            # Голос диктора
            self._voice_combo = self._create_voice_combo()
            opts_form.addRow("Voice:", self._voice_combo)

            # Размер изображения
            self._image_size_combo = self._create_image_size_combo()
            opts_form.addRow("Image size:", self._image_size_combo)

            opts_group.setLayout(opts_form)
            layout.addWidget(opts_group)

        elif mode == GenerationMode.MATERIAL:
            from aqt.qt import QSpinBox

            opts_group = QGroupBox("Generation options")
            opts_form = QFormLayout()
            opts_form.setSpacing(8)
            opts_form.setContentsMargins(8, 6, 8, 8)
            opts_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
            opts_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            self._max_cards_spin = QSpinBox()
            self._max_cards_spin.setMinimum(1)
            self._max_cards_spin.setMaximum(5)
            self._max_cards_spin.setValue(1)
            self._max_cards_spin.valueChanged.connect(lambda _: self._update_cost_estimate())
            opts_form.addRow("Max cards per paragraph:", self._max_cards_spin)

            self._answer_detail_combo = QComboBox()
            self._answer_detail_combo.addItem("Short (1-2 sentences)", AnswerDetail.SHORT.value)
            self._answer_detail_combo.addItem("Medium (2-4 sentences)", AnswerDetail.MEDIUM.value)
            self._answer_detail_combo.addItem("Detailed (comprehensive)", AnswerDetail.DETAILED.value)
            opts_form.addRow("Answer detail:", self._answer_detail_combo)

            # Чекбокс картинок
            opts_form.addRow(self._images_checkbox)

            self._image_size_combo = self._create_image_size_combo()
            self._image_size_combo.setVisible(False)
            self._image_size_label = QLabel("Image size:")
            self._image_size_label.setVisible(False)
            opts_form.addRow(self._image_size_label, self._image_size_combo)

            # Показывать image_size + label когда включены картинки
            def _toggle_image_size(state: int) -> None:
                visible = bool(state)
                if self._image_size_combo is not None:
                    self._image_size_combo.setVisible(visible)
                if self._image_size_label is not None:
                    self._image_size_label.setVisible(visible)

            self._images_checkbox.stateChanged.connect(_toggle_image_size)

            opts_group.setLayout(opts_form)
            layout.addWidget(opts_group)

        elif mode == GenerationMode.IMAGE:
            opts_group = QGroupBox("Generation options")
            opts_form = QFormLayout()
            opts_form.setSpacing(8)
            opts_form.setContentsMargins(8, 6, 8, 8)
            opts_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
            opts_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            self._image_size_combo = self._create_image_size_combo()
            opts_form.addRow("Image size:", self._image_size_combo)

            opts_group.setLayout(opts_form)
            layout.addWidget(opts_group)

        elif mode == GenerationMode.AUDIO:
            opts_group = QGroupBox("Generation options")
            opts_form = QFormLayout()
            opts_form.setSpacing(8)
            opts_form.setContentsMargins(8, 6, 8, 8)
            opts_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
            opts_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            self._voice_combo = self._create_voice_combo()
            opts_form.addRow("Voice:", self._voice_combo)

            opts_group.setLayout(opts_form)
            layout.addWidget(opts_group)

        # === Секция 3: Custom prompt (только для language) ===
        if _should_show_custom_prompt(mode):
            prompt_group = QGroupBox("Custom prompt")
            prompt_group.setCheckable(True)
            prompt_group.setChecked(False)
            prompt_vlayout = QVBoxLayout()
            prompt_vlayout.setContentsMargins(8, 6, 8, 8)

            self._custom_prompt_input = QPlainTextEdit()
            self._custom_prompt_input.setPlaceholderText(_get_default_custom_prompt(mode))
            self._custom_prompt_input.setMaximumHeight(90)
            self._custom_prompt_input.setEnabled(False)
            prompt_vlayout.addWidget(self._custom_prompt_input)
            prompt_group.setLayout(prompt_vlayout)
            prompt_group.toggled.connect(self._custom_prompt_input.setEnabled)

            self._custom_prompt_toggle = prompt_group
            layout.addWidget(prompt_group)
        else:
            self._custom_prompt_toggle = None
            self._custom_prompt_input = None

        # --- Единственная строка статуса/стоимости ---
        self._cost_label = QLabel("")
        self._cost_label.setStyleSheet("color: palette(mid);")
        layout.addWidget(self._cost_label)

        # Обновляем оценку при изменении текста
        self._input_text.textChanged.connect(self._update_cost_estimate)

        # --- Прогресс-виджет (бар + Cancel) ---
        self._progress_widget = ProgressWidget(self._dialog)
        layout.addWidget(self._progress_widget.widget)

        layout.addStretch()

        # --- Кнопки нижней панели ---
        buttons_layout = QHBoxLayout()

        self._back_btn = QPushButton("\u2190 Back")
        self._back_btn.clicked.connect(self._on_back)
        buttons_layout.addWidget(self._back_btn)

        buttons_layout.addStretch()

        self._again_btn = QPushButton("Again")
        self._again_btn.setObjectName("againBtn")
        self._again_btn.clicked.connect(self._on_generate_again)
        self._again_btn.setVisible(False)
        buttons_layout.addWidget(self._again_btn)

        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setObjectName("generateBtn")
        self._generate_btn.setDefault(True)
        self._generate_btn.clicked.connect(self._on_generate)
        buttons_layout.addWidget(self._generate_btn)

        layout.addLayout(buttons_layout)

        self._go_back = False

    def _get_material_options(self) -> MaterialOptions | None:
        """Собирает MaterialOptions из UI (только для material режима).

        Returns:
            MaterialOptions или None если не material режим.
        """
        if self._mode != GenerationMode.MATERIAL:
            return None

        max_cards = 3
        if self._max_cards_spin is not None:
            max_cards = self._max_cards_spin.value()

        include_images = self._images_checkbox.isChecked()
        image_size = "auto"
        if self._image_size_combo is not None:
            image_size = self._image_size_combo.currentData() or "auto"

        answer_detail = AnswerDetail.SHORT
        if self._answer_detail_combo is not None:
            value = self._answer_detail_combo.currentData()
            answer_detail = AnswerDetail(value) if value else AnswerDetail.SHORT

        return MaterialOptions(
            max_cards_per_paragraph=max_cards,
            include_images=include_images,
            image_size=image_size,
            answer_detail=answer_detail,
        )

    def _create_voice_combo(self) -> QComboBox:
        """Создаёт комбобокс выбора голоса диктора."""
        from aqt.qt import QComboBox

        combo = QComboBox()
        combo.addItem("Alloy (neutral)", "alloy")
        combo.addItem("Ash (male, warm)", "ash")
        combo.addItem("Ballad (male, soft)", "ballad")
        combo.addItem("Coral (female, warm)", "coral")
        combo.addItem("Echo (male, clear)", "echo")
        combo.addItem("Fable (male, British)", "fable")
        combo.addItem("Nova (female, energetic)", "nova")
        combo.addItem("Onyx (male, deep)", "onyx")
        combo.addItem("Sage (female, calm)", "sage")
        combo.addItem("Shimmer (female, bright)", "shimmer")
        combo.addItem("Verse (male, expressive)", "verse")
        return combo

    def _create_image_size_combo(self) -> QComboBox:
        """Создаёт комбобокс выбора размера изображения."""
        from aqt.qt import QComboBox

        combo = QComboBox()
        combo.addItem("Auto (1K, default)", "auto")
        combo.addItem("0.5K (cheaper, sufficient for cards)", "0.5K")
        combo.addItem("1K (standard)", "1K")
        combo.addItem("2K (high quality)", "2K")
        combo.addItem("4K (maximum)", "4K")
        combo.currentIndexChanged.connect(self._update_cost_estimate)
        return combo

    def _get_deck_name(self) -> tuple[str, bool]:
        """Возвращает имя колоды и флаг создания новой.

        Если введённое имя совпадает с существующей колодой — использует её.
        Если нет — помечает как новую для создания.

        Returns:
            (deck_name, create_new).
        """
        return _get_deck_name_from_combo(self._deck_combo.currentText(), self._existing_decks)

    def _get_language_options(self) -> LanguageOptions | None:
        """Собирает LanguageOptions из чекбоксов (только для language режима).

        Returns:
            LanguageOptions или None если не language режим.
        """
        if not self._lang_options_checkboxes:
            return None
        from ankiforge.models import LanguageOptions

        image_size = "auto"
        if self._image_size_combo is not None:
            image_size = self._image_size_combo.currentData() or "auto"

        voice = "alloy"
        if self._voice_combo is not None:
            voice = self._voice_combo.currentData() or "alloy"

        return LanguageOptions(
            include_photo=self._lang_options_checkboxes["include_photo"].isChecked(),
            include_audio_word=self._lang_options_checkboxes["include_audio_word"].isChecked(),
            include_audio_definition=self._lang_options_checkboxes["include_audio_definition"].isChecked(),
            include_audio_example=self._lang_options_checkboxes["include_audio_example"].isChecked(),
            include_transcription=self._lang_options_checkboxes["include_transcription"].isChecked(),
            image_size=image_size,
            detailed_image=self._lang_options_checkboxes["detailed_image"].isChecked(),
            voice=voice,
        )

    def _get_max_cards_per_paragraph(self) -> int:
        """Возвращает текущее значение max_cards_per_paragraph из спинбокса."""
        if self._max_cards_spin is not None:
            return int(self._max_cards_spin.value())
        return 3

    def _update_cost_estimate(self) -> None:
        """Обновляет оценку стоимости из кэшированного pricing в конфиге."""
        self._cost_label.setStyleSheet("color: palette(mid);")
        text = self._input_text.toPlainText()
        card_count = _count_input_items(text, self._mode, max_cards_per_paragraph=self._get_max_cards_per_paragraph())
        if card_count == 0:
            self._cost_label.setText("")
            return

        config = get_config()
        lang_opts = self._get_language_options()
        cost = _estimate_cost_from_config(config, self._mode, card_count, lang_opts)
        if cost is None:
            self._cost_label.setText(f"{card_count} cards (pricing not loaded \u2014 open settings)")
            return

        cost_text = _format_cost(cost)
        self._cost_label.setText(f"Estimate: {cost_text} ({card_count} cards)")

    def _on_back(self) -> None:
        """Обработчик кнопки Назад — возврат к выбору режима."""
        self._go_back = True
        self._dialog.reject()

    # --- Три состояния кнопок: idle / generating / done ---

    def _reconnect_generate_btn(self, slot: Callable[[], None]) -> None:
        """Отключает все слоты от clicked и подключает новый."""
        import contextlib

        with contextlib.suppress(TypeError, RuntimeError):
            self._generate_btn.clicked.disconnect()
        self._generate_btn.clicked.connect(slot)

    def _set_buttons_idle(self) -> None:
        """Состояние idle: до генерации или после сброса."""
        self._back_btn.setEnabled(True)
        self._again_btn.setVisible(False)
        self._generate_btn.setText("Generate")
        self._generate_btn.setObjectName("generateBtn")
        self._generate_btn.setEnabled(True)
        self._generate_btn.setVisible(True)
        self._reconnect_generate_btn(self._on_generate)

    def _set_buttons_generating(self) -> None:
        """Состояние generating: идёт генерация."""
        self._back_btn.setEnabled(False)
        self._again_btn.setVisible(False)
        self._generate_btn.setText("Cancel")
        self._generate_btn.setEnabled(True)
        self._reconnect_generate_btn(self._on_cancel)

    def _set_buttons_done(self) -> None:
        """Состояние done: генерация завершена."""
        self._back_btn.setEnabled(True)
        self._again_btn.setVisible(True)
        self._generate_btn.setText("Close")
        self._generate_btn.setEnabled(True)
        self._reconnect_generate_btn(self._dialog.accept)

    def _on_generate_again(self) -> None:
        """Сбрасывает UI для повторной генерации."""
        self._progress_widget.hide()
        self._cost_label.setText("")
        self._set_buttons_idle()
        self._update_cost_estimate()

    def _on_generate(self) -> None:
        """Обработчик кнопки Generate — валидация, оценка стоимости, асинхронная генерация."""
        deck_name, create_new = self._get_deck_name()
        config = get_config()

        # Custom prompt
        custom_prompt: str | None = None
        if (
            self._custom_prompt_input is not None
            and self._custom_prompt_toggle is not None
            and self._custom_prompt_toggle.isChecked()
        ):
            custom_prompt = self._custom_prompt_input.toPlainText()

        # Собираем MaterialOptions
        mat_opts = self._get_material_options()

        # Собираем voice и image_size
        voice = self._voice_combo.currentData() or "alloy" if self._voice_combo is not None else "alloy"
        image_size = self._image_size_combo.currentData() or "auto" if self._image_size_combo is not None else "auto"

        try:
            request = _build_card_request(
                mode=self._mode,
                input_text=self._input_text.toPlainText(),
                deck_name=deck_name,
                create_new_deck=create_new,
                include_images=self._images_checkbox.isChecked(),
                language=self._language_combo.currentData() or "en",
                custom_prompt=custom_prompt,
                language_options=self._get_language_options(),
                material_options=mat_opts,
                voice=voice,
                image_size=image_size,
            )
        except ValueError as e:
            self._cost_label.setText(str(e))
            self._cost_label.setStyleSheet("color: #f44336;")
            return

        from ankiforge.openrouter.client import OpenRouterClient

        client = OpenRouterClient(api_key=config.api_key)

        # Подтягиваем актуальный pricing из OpenRouter (модель могла смениться)
        self._ensure_pricing(config, client)

        # Подсчёт карточек
        card_count = _count_input_items(
            request.input_text, self._mode, max_cards_per_paragraph=self._get_max_cards_per_paragraph()
        )

        # Переключаем кнопки и показываем прогресс
        self._set_buttons_generating()
        self._progress_widget.show(card_count)

        # Создаём генератор
        generator = _create_generator(
            mode=self._mode,
            client=client,
            text_model=config.text_model,
            image_model=config.image_model,
            audio_model=config.audio_model,
            include_images=request.include_images,
            config=config,
            image_size=image_size,
            voice=voice,
        )

        # Ensure note types exist (до запуска потока — работает с Anki API из главного потока)
        self._ensure_note_types(self._mode)

        # Сохраняем request для использования в callbacks
        self._current_request = request

        # Запускаем генерацию в фоновом потоке
        self._worker = GenerationWorker(generator, request)
        self._worker.connect_progress(self._on_progress_updated)
        self._worker.connect_finished(self._on_generation_finished)
        self._worker.connect_error(self._on_generation_error)
        self._worker.start()

    def _on_progress_updated(self, progress: GenerationProgress) -> None:
        """Обновляет UI по сигналу прогресса из рабочего потока."""
        self._last_progress = progress
        self._progress_widget.update_progress(progress)
        cost_text = _format_cost(progress.current_cost) if progress.current_cost > 0 else ""
        parts = [f"{progress.completed_cards}/{progress.total_cards} cards"]
        if cost_text:
            parts.append(f"spent: {cost_text}")
        self._cost_label.setText(" · ".join(parts))
        self._cost_label.setStyleSheet("color: palette(mid);")

    def _on_generation_finished(self, cards: list[GeneratedCard]) -> None:
        """Обработка успешного завершения генерации."""
        request = self._current_request

        _save_cards_to_deck(cards, request.target_deck, create_new=request.create_new_deck)

        # Вычисляем итоговую стоимость (из последнего прогресса)
        total_cost = 0.0
        if hasattr(self, "_last_progress"):
            total_cost = self._last_progress.current_cost

        # Логируем расходы
        _log_cost(self._mode, len(cards), total_cost)

        self._progress_widget.finish()
        self._cost_label.setText(f"Done \u00b7 {len(cards)} cards \u00b7 {_format_cost(total_cost)}")
        self._cost_label.setStyleSheet("color: #4caf50;")
        self._set_buttons_done()
        self._worker = None

        # Обновляем главное окно Anki чтобы показать новые карточки
        import contextlib

        with contextlib.suppress(Exception):
            self._mw.reset()

    def _on_generation_error(self, error_msg: str) -> None:
        """Обработка ошибки генерации."""
        self._progress_widget.hide()
        self._cost_label.setText(f"Error: {error_msg}")
        self._cost_label.setStyleSheet("color: #f44336;")
        self._set_buttons_idle()
        self._worker = None

    def _on_cancel(self) -> None:
        """Отменяет текущую генерацию."""
        if self._worker is not None:
            self._worker.cancel()

    def _ensure_pricing(self, config: AddonConfig, client: OpenRouterClient) -> None:
        """Подтягивает pricing моделей — всегда обновляет все три из API."""
        import contextlib

        from ankiforge.config.dialog import _extract_pricing
        from ankiforge.config.manager import save_config

        with contextlib.suppress(Exception):
            models = client.fetch_models()
            # Всегда обновляем pricing — модель могла смениться
            if config.text_model:
                config.text_model_pricing = _extract_pricing(config.text_model, models)
            if config.image_model:
                config.image_model_pricing = _extract_pricing(config.image_model, models)
            if config.audio_model:
                config.audio_model_pricing = _extract_pricing(config.audio_model, models)
            save_config(config)

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

    def run(self) -> bool:
        """Показывает диалог модально.

        Returns:
            True если пользователь нажал Назад (для возврата к выбору режима).
        """
        self._dialog.exec()
        return self._go_back
