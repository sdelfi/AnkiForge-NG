"""Mode selection and input dialog for card generation."""

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
    from aqt.qt import QComboBox, QSpinBox, QWidget  # type: ignore[import-not-found]

    from ankiforge.models import LanguageOptions
    from ankiforge.openrouter.client import OpenRouterClient

    class _CardGenerator(Protocol):
        def generate(
            self,
            request: CardRequest,
            progress_callback: Callable[[GenerationProgress], None],
        ) -> list[GeneratedCard]: ...


# ---------------------------------------------------------------------------
# Mode info (testable without Qt)
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
    """Check if settings should be opened before generation.

    Returns:
        True if API key is not configured.
    """
    return not is_configured()


def _get_mode_info(mode: GenerationMode) -> dict[str, str]:
    """Return generation mode info.

    Args:
        mode: Generation mode.

    Returns:
        Dict with title, subtitle, and icon.
    """
    return _MODE_INFO[mode]


def _get_input_placeholder(mode: GenerationMode) -> str:
    """Return placeholder text for the input field depending on mode.

    Args:
        mode: Generation mode.

    Returns:
        Placeholder text.
    """
    return _INPUT_PLACEHOLDERS[mode]


def _should_show_images_checkbox(mode: GenerationMode) -> bool:
    """Determine if 'Add images' checkbox should be shown.

    Args:
        mode: Generation mode.

    Returns:
        True only for MATERIAL mode.
    """
    return mode == GenerationMode.MATERIAL


def _should_show_custom_prompt(mode: GenerationMode) -> bool:
    """Determine if custom prompt field should be shown.

    Args:
        mode: Generation mode.

    Returns:
        True for all modes.
    """
    return True


def _get_deck_name_from_combo(typed_text: str, existing_decks: set[str]) -> tuple[str, bool]:
    """Determine deck name and whether to create a new one.

    Args:
        typed_text: Text from editable combo.
        existing_decks: Set of existing decks.

    Returns:
        (deck_name, is_new) — deck name and new creation flag.
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
    """Estimate cost from cached pricing in config.

    Args:
        config: Config with model pricing.
        mode: Generation mode.
        card_count: Number of cards.
        language_options: Generation options for language mode.

    Returns:
        Cost in USD or None if pricing is not loaded.
    """
    if card_count <= 0:
        return 0.0

    # Average tokens/chars per call
    avg_prompt_tokens = 200
    avg_completion_tokens = 150
    avg_audio_chars = 100

    # Average image generation cost by size (USD).
    # OpenRouter API doesn't return image output pricing (/models lacks image_output),
    # so we use empirical averages based on real requests.
    # Examples: Gemini 2.5 Flash Image 0.5K=$0.02, 1K=$0.04; Gemini 3.1 Flash 1K=$0.07
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

    # If no pricing — can't calculate
    mode_val = mode.value
    if tp is None and mode_val in ("questions", "language", "material", "image", "audio"):
        return None

    cost = 0.0

    if tp is not None and mode_val in ("questions", "language", "material", "image", "audio"):
        text_multiplier = 2 if mode_val == "material" else 1  # map-reduce: 2 calls per paragraph
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
    """Write cost to log file.

    Args:
        mode: Generation mode.
        card_count: Number of cards.
        cost: Cost in USD.
    """
    import json
    from datetime import datetime
    from pathlib import Path

    try:
        from aqt import mw  # type: ignore[import-not-found]

        if mw is None or mw.addonManager is None:
            return
        addon_dir = Path(__file__).parent.parent
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
    """Return default prompt for the specified mode.

    Args:
        mode: Generation mode.

    Returns:
        Default prompt text.
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


def _add_help_icon(widget: object, tooltip: str) -> QWidget:
    """Wraps a widget with a help icon in a horizontal layout.

    Args:
        widget: Any QWidget (QLabel, QCheckBox, etc.).
        tooltip: Tooltip text for the help icon.

    Returns:
        QWidget containing the original widget and a help icon.
    """
    from aqt.qt import QHBoxLayout, QWidget

    from ankiforge.ui.styles import make_help_icon

    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(4)
    h.addWidget(widget)
    h.addWidget(make_help_icon(tooltip))
    h.addStretch()
    return w


# ---------------------------------------------------------------------------
# Generator factory (testable without Qt)
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
    """Create a card generator for the given mode.

    Args:
        mode: Generation mode.
        client: OpenRouter client.
        text_model: Text model ID.
        image_model: Image model ID.
        audio_model: Audio model ID.
        include_images: Whether to add images (for material).
        config: Config with pricing (for real-time cost tracking).
        image_size: Image size (for image/material).
        voice: Narrator voice (for audio).

    Returns:
        Generator instance.
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
# CardRequest building (testable without Qt)
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
    """Build CardRequest from form parameters.

    Args:
        mode: Generation mode.
        input_text: Entered text.
        deck_name: Deck name.
        create_new_deck: Create new deck.
        include_images: Add images (for material).
        language: Card language.
        custom_prompt: Custom prompt.
        language_options: Generation options for language mode.
        material_options: Generation options for material mode.
        voice: Narrator voice (for audio).
        image_size: Image size (for image/material).

    Returns:
        CardRequest.

    Raises:
        ValueError: If data is invalid.
    """
    if not input_text.strip():
        msg = "Enter data to generate"
        raise ValueError(msg)
    if not deck_name.strip():
        msg = "Specify a deck name"
        raise ValueError(msg)

    # Empty/whitespace custom_prompt → None
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
# Saving cards to deck (testable without Qt)
# ---------------------------------------------------------------------------


def _markdown_to_html(text: str) -> str:
    """Convert markdown code blocks and inline code to HTML for Anki.

    Args:
        text: Text with markdown markup.

    Returns:
        HTML string with ``<pre><code>`` and ``<code>`` tags.
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

    # 3. Newlines → <br> ONLY outside <pre>...</pre>
    parts = _re.split(r"(<pre>.*?</pre>)", result, flags=_re.DOTALL)
    for i, part in enumerate(parts):
        if not part.startswith("<pre>"):
            parts[i] = part.replace("\n", "<br>")
    return "".join(parts)


def _build_note_fields(card: GeneratedCard) -> dict[str, str]:
    """Build field dict for add_note from GeneratedCard.

    Args:
        card: Generated card.

    Returns:
        Dict of {field_name: value}.
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
    """Save generated cards to an Anki deck.

    Args:
        cards: List of cards.
        deck_name: Target deck name.
        create_new: Create deck if it doesn't exist.
    """
    if not cards:
        return

    if create_new:
        create_deck(deck_name)

    for card in cards:
        fields = _build_note_fields(card)
        add_note(deck_name, card.note_type, fields)


# ---------------------------------------------------------------------------
# GenerateDialog — mode selection
# ---------------------------------------------------------------------------


class GenerateDialog:
    """Card generation mode selection dialog.

    Shows 5 modes with descriptions and icons.
    User selects a mode — dialog returns the chosen GenerationMode.
    """

    def __init__(self, mw: AnkiQt) -> None:
        """Initialize mode selection dialog.

        Args:
            mw: Anki main window.
        """
        from aqt.qt import (
            QDialog,
            QFont,
            QFrame,
            QHBoxLayout,
            QLabel,
            QPushButton,
            Qt,
            QVBoxLayout,
        )  # noqa: F811

        from ankiforge.ui.styles import get_dialog_size, wrap_in_scroll_area

        self._mw = mw
        self._selected_mode: GenerationMode | None = None

        self._dialog = QDialog(mw)
        self._dialog.setWindowTitle("AnkiForge — Generate Cards")
        self._dialog.setMinimumWidth(480)
        self._dialog.setStyleSheet(DIALOG_QSS)

        # Main dialog layout: scroll + buttons at the bottom
        dialog_layout = QVBoxLayout()
        dialog_layout.setContentsMargins(0, 0, 0, 0)
        dialog_layout.setSpacing(0)
        self._dialog.setLayout(dialog_layout)

        # Content inside scroll area
        from aqt.qt import QWidget as _QWidget

        content = _QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 6)
        content.setLayout(layout)

        # --- Title ---
        title_label = QLabel("Choose generation mode")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # --- Mode buttons (entire row is clickable) ---
        _row_qss = (
            "QFrame#modeRow {"
            "  border: 1px solid palette(mid);"
            "  border-radius: 8px;"
            "  background: transparent;"
            "}"
            "QFrame#modeRow:hover {"
            "  border-color: palette(highlight);"
            "  background-color: palette(midlight);"
            "}"
        )

        for mode in GenerationMode:
            info = _get_mode_info(mode)
            handler = self._make_mode_handler(mode)

            row = QFrame()
            row.setObjectName("modeRow")
            row.setFrameShape(QFrame.Shape.StyledPanel)
            row.setCursor(Qt.CursorShape.PointingHandCursor)
            row.setStyleSheet(_row_qss)
            # Make entire row clickable via mousePressEvent
            row.mousePressEvent = lambda _event, h=handler: h()

            row_layout = QHBoxLayout()
            row_layout.setSpacing(12)
            row_layout.setContentsMargins(12, 10, 12, 10)
            row.setLayout(row_layout)

            # Icon
            icon_label = QLabel(info["icon"])
            icon_font = QFont()
            icon_font.setPointSize(24)
            icon_label.setFont(icon_font)
            icon_label.setFixedWidth(48)
            icon_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            row_layout.addWidget(icon_label)

            # Text (title + subtitle)
            text_layout = QVBoxLayout()
            text_layout.setSpacing(2)
            mode_title = QLabel(info["title"])
            mode_title_font = QFont()
            mode_title_font.setBold(True)
            mode_title.setFont(mode_title_font)
            mode_title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            text_layout.addWidget(mode_title)

            mode_subtitle = QLabel(info["subtitle"])
            mode_subtitle.setStyleSheet("color: palette(text); font-size: 11px;")
            mode_subtitle.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            text_layout.addWidget(mode_subtitle)
            row_layout.addLayout(text_layout)

            row_layout.addStretch()

            layout.addWidget(row)

        layout.addStretch()

        # Scroll area with content
        dialog_layout.addWidget(wrap_in_scroll_area(content))

        # --- Settings / Cancel (outside scroll, always visible) ---
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(14, 6, 14, 14)
        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self._on_open_settings)
        buttons_layout.addWidget(settings_btn)

        donate_btn = QPushButton("❤")
        donate_btn.setFixedSize(28, 28)
        donate_btn.setStyleSheet(
            "QPushButton { color: #e74c3c; background: transparent; border: none;"
            " font-size: 16px; }"
            "QPushButton:hover { color: #ff6b6b; }"
        )
        donate_btn.setToolTip("Support the Project")
        donate_btn.clicked.connect(self._on_donate)
        buttons_layout.addWidget(donate_btn)

        buttons_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._dialog.reject)
        buttons_layout.addWidget(cancel_btn)
        dialog_layout.addLayout(buttons_layout)

        # Scale to screen size
        w, h = get_dialog_size(width_pct=0.35, height_pct=0.55, min_w=480, min_h=400)
        self._dialog.resize(w, h)

    def _on_donate(self) -> None:
        """Show dialog with crypto address for donation."""
        from aqt.qt import QApplication, QDialog, QLabel, QPushButton, Qt, QVBoxLayout

        address = "0x34f58CF2BE6073f12b2c3c6aE9f8c31983A3f5fE"

        dlg = QDialog(self._dialog)
        dlg.setWindowTitle("Support the Project")
        dlg.setMinimumWidth(420)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(12)
        lay.setContentsMargins(20, 20, 20, 20)

        info = QLabel(
            "AnkiForge is a free, open-source project.\nIf you find it useful, consider supporting the development:"
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        addr_label = QLabel(address)
        addr_label.setStyleSheet(
            "font-family: monospace; font-size: 13px; padding: 8px;"
            " background: palette(base); border: 1px solid palette(mid);"
            " border-radius: 4px;"
        )
        addr_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(addr_label)

        networks = QLabel("Networks: Ethereum, Base, Arbitrum, Avalanche")
        networks.setStyleSheet("color: gray; font-size: 12px;")
        lay.addWidget(networks)

        copy_btn = QPushButton("Copy Address")
        clipboard = QApplication.clipboard()
        if clipboard:
            copy_btn.clicked.connect(lambda: clipboard.setText(address))
        lay.addWidget(copy_btn)

        dlg.exec()

    def _on_open_settings(self) -> None:
        """Open settings dialog."""
        from ankiforge.config.dialog import SettingsDialog

        SettingsDialog(self._mw).run()

    def _make_mode_handler(self, mode: GenerationMode) -> Callable[[], None]:
        """Create handler for mode button.

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
        """Return selected mode or None if cancelled."""
        return self._selected_mode

    def run(self) -> GenerationMode | None:
        """Show dialog modally.

        Returns:
            Выбранный GenerationMode или None если пользователь отменил.
        """
        result = self._dialog.exec()
        if result and self._selected_mode is not None:
            return self._selected_mode
        return None


# ---------------------------------------------------------------------------
# InputDialog — data input and generation launch
# ---------------------------------------------------------------------------


class InputDialog:
    """Data input dialog for card generation.

    Показывает: текстовое поле ввода, выбор/создание колоды (editable combo),
    чекбокс картинок (для material), кнопку Generate.
    """

    def __init__(self, mw: AnkiQt, mode: GenerationMode) -> None:
        """Initialize data input dialog.

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
        from aqt.qt import (
            QWidget as _QWidget,
        )

        from ankiforge.ui.styles import get_dialog_size, wrap_in_scroll_area

        self._mw = mw
        self._mode = mode
        self._worker: GenerationWorker | None = None

        self._dialog = QDialog(mw)
        info = _get_mode_info(mode)
        self._dialog.setWindowTitle(f"AnkiForge — {info['title']}")
        self._dialog.setMinimumWidth(520)
        self._dialog.setStyleSheet(DIALOG_QSS)

        # Main dialog layout: scroll + buttons at the bottom
        dialog_layout = QVBoxLayout()
        dialog_layout.setContentsMargins(0, 0, 0, 0)
        dialog_layout.setSpacing(0)
        self._dialog.setLayout(dialog_layout)

        # Content inside scroll area
        content = _QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(14, 14, 14, 6)
        content.setLayout(layout)

        # --- Title ---
        header = QLabel(info["title"])
        header_font = QFont()
        header_font.setPointSize(13)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)

        subtitle = QLabel(info["subtitle"])
        subtitle.setStyleSheet("color: palette(text); font-size: 11px;")
        layout.addWidget(subtitle)

        # === Section 1: Generation data ===
        input_group = QGroupBox("Generation input")
        input_vlayout = QVBoxLayout()
        input_vlayout.setContentsMargins(8, 6, 8, 8)
        self._input_text = QPlainTextEdit()
        self._input_text.setPlaceholderText(_get_input_placeholder(mode))
        self._input_text.setMinimumHeight(140)
        input_vlayout.addWidget(self._input_text)
        input_group.setLayout(input_vlayout)
        layout.addWidget(input_group)

        # === Section 2: Settings ===
        settings_group = QGroupBox("Settings")
        settings_form = QFormLayout()
        settings_form.setSpacing(8)
        settings_form.setContentsMargins(8, 6, 8, 8)

        # Editable combo — select existing or type new deck
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

        # Don't auto-select a deck — let the user type
        self._deck_combo.setCurrentIndex(-1)
        deck_completer.setModel(self._deck_combo.model())
        settings_form.addRow("Deck:", self._deck_combo)

        deck_hint = QLabel("Pick an existing deck from the list, or type a new name to create one")
        deck_hint.setStyleSheet("color: palette(text); font-size: 11px;")
        deck_hint.setWordWrap(True)
        settings_form.addRow("", deck_hint)

        # Card language
        self._language_combo = QComboBox()
        self._language_combo.addItem("English", "en")
        self._language_combo.addItem("Russian", "ru")
        self._language_combo.addItem("Deutsch", "de")
        self._language_combo.addItem("Français", "fr")
        self._language_combo.addItem("Español", "es")
        self._language_combo.addItem("日本語", "ja")
        self._language_combo.addItem("中文", "zh")

        # Set language from config
        config = get_config()
        for i in range(self._language_combo.count()):
            if self._language_combo.itemData(i) == config.language:
                self._language_combo.setCurrentIndex(i)
                break
        settings_form.addRow("Language:", self._language_combo)

        # Images checkbox — created here, added to "Generation options" for material
        self._images_checkbox = QCheckBox("Add images to cards")

        settings_group.setLayout(settings_form)
        layout.addWidget(settings_group)

        # === Section 2.5: Generation options ===
        self._lang_options_checkboxes: dict[str, QCheckBox] = {}
        self._image_size_combo: QComboBox | None = None
        self._image_size_label: QWidget | None = None
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
            _cb_tooltips: dict[str, str] = {
                "detailed_image": "Uses a richer prompt for higher quality images. Costs more per image",
            }
            for key, label in checkbox_defs:
                cb = QCheckBox(label)
                cb.setChecked(key not in ("detailed_image",))
                cb.stateChanged.connect(self._update_cost_estimate)
                self._lang_options_checkboxes[key] = cb
                tip = _cb_tooltips.get(key)
                if tip:
                    opts_form.addRow(_add_help_icon(cb, tip))
                else:
                    opts_form.addRow(cb)

            # Narrator voice
            self._voice_combo = self._create_voice_combo()
            opts_form.addRow(
                _add_help_icon(QLabel("Voice:"), "Text-to-speech voice used for audio generation"),
                self._voice_combo,
            )

            # Image size
            self._image_size_combo = self._create_image_size_combo()
            opts_form.addRow(
                _add_help_icon(
                    QLabel("Image size:"), "Larger images cost more. 0.5K is usually sufficient for flashcards"
                ),
                self._image_size_combo,
            )

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
            opts_form.addRow(
                _add_help_icon(
                    QLabel("Max cards per paragraph:"),
                    "Maximum number of Q&A cards generated from each paragraph of the source text",
                ),
                self._max_cards_spin,
            )

            self._answer_detail_combo = QComboBox()
            self._answer_detail_combo.addItem("Short (1-2 sentences)", AnswerDetail.SHORT.value)
            self._answer_detail_combo.addItem("Medium (2-4 sentences)", AnswerDetail.MEDIUM.value)
            self._answer_detail_combo.addItem("Detailed (comprehensive)", AnswerDetail.DETAILED.value)
            opts_form.addRow(
                _add_help_icon(QLabel("Answer detail:"), "Controls the length and depth of generated answers"),
                self._answer_detail_combo,
            )

            # Images checkbox
            opts_form.addRow(self._images_checkbox)

            self._image_size_combo = self._create_image_size_combo()
            self._image_size_combo.setVisible(False)
            self._image_size_label = _add_help_icon(
                QLabel("Image size:"),
                "Larger images cost more. 0.5K is usually sufficient for flashcards",
            )
            self._image_size_label.setVisible(False)
            opts_form.addRow(self._image_size_label, self._image_size_combo)

            # Show image_size + label when images are enabled
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
            opts_form.addRow(
                _add_help_icon(
                    QLabel("Image size:"), "Larger images cost more. 0.5K is usually sufficient for flashcards"
                ),
                self._image_size_combo,
            )

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
            opts_form.addRow(
                _add_help_icon(QLabel("Voice:"), "Text-to-speech voice used for audio generation"),
                self._voice_combo,
            )

            opts_group.setLayout(opts_form)
            layout.addWidget(opts_group)

        # === Section 3: Custom prompt ===
        if _should_show_custom_prompt(mode):
            prompt_group = QGroupBox("Custom prompt")
            prompt_group.setCheckable(True)
            prompt_group.setChecked(False)
            prompt_vlayout = QVBoxLayout()
            prompt_vlayout.setContentsMargins(8, 6, 8, 8)

            prompt_vlayout.addWidget(
                _add_help_icon(
                    QLabel(""),
                    "Override the default AI prompt. Leave empty to use the built-in prompt optimized for this mode",
                )
            )

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

        # --- Single status/cost line ---
        self._cost_label = QLabel("")
        self._cost_label.setStyleSheet("color: palette(text);")
        layout.addWidget(self._cost_label)

        # Update estimate when text changes
        self._input_text.textChanged.connect(self._update_cost_estimate)

        # --- Progress widget (bar + Cancel) ---
        self._progress_widget = ProgressWidget(self._dialog)
        layout.addWidget(self._progress_widget.widget)

        layout.addStretch()

        # Scroll area with content
        dialog_layout.addWidget(wrap_in_scroll_area(content))

        # --- Bottom panel buttons (outside scroll, always visible) ---
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(14, 6, 14, 14)

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

        dialog_layout.addLayout(buttons_layout)

        # Scale to screen size
        w, h = get_dialog_size(width_pct=0.4, height_pct=0.65, min_w=520, min_h=450)
        self._dialog.resize(w, h)

        self._go_back = False

    def _get_material_options(self) -> MaterialOptions | None:
        """Build MaterialOptions from UI (only for material mode).

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
        """Create narrator voice combobox."""
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
        """Create image size combobox."""
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
        """Return deck name and new creation flag.

        Если введённое имя совпадает с существующей колодой — использует её.
        Если нет — помечает как новую для создания.

        Returns:
            (deck_name, create_new).
        """
        return _get_deck_name_from_combo(self._deck_combo.currentText(), self._existing_decks)

    def _get_language_options(self) -> LanguageOptions | None:
        """Build LanguageOptions from checkboxes (only for language mode).

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
        """Return current max_cards_per_paragraph value from spinbox."""
        if self._max_cards_spin is not None:
            return int(self._max_cards_spin.value())
        return 3

    def _update_cost_estimate(self) -> None:
        """Update cost estimate from cached pricing in config."""
        self._cost_label.setStyleSheet("color: palette(text);")
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
        """Back button handler — return to mode selection."""
        self._go_back = True
        self._dialog.reject()

    # --- Three button states: idle / generating / done ---

    def _reconnect_generate_btn(self, slot: Callable[[], None]) -> None:
        """Disconnect all clicked slots and connect a new one."""
        import contextlib

        with contextlib.suppress(TypeError, RuntimeError):
            self._generate_btn.clicked.disconnect()
        self._generate_btn.clicked.connect(slot)

    def _set_buttons_idle(self) -> None:
        """Idle state: before generation or after reset."""
        self._back_btn.setEnabled(True)
        self._again_btn.setVisible(False)
        self._generate_btn.setText("Generate")
        self._generate_btn.setObjectName("generateBtn")
        self._generate_btn.setEnabled(True)
        self._generate_btn.setVisible(True)
        self._reconnect_generate_btn(self._on_generate)

    def _set_buttons_generating(self) -> None:
        """Generating state: generation in progress."""
        self._back_btn.setEnabled(False)
        self._again_btn.setVisible(False)
        self._generate_btn.setText("Cancel")
        self._generate_btn.setEnabled(True)
        self._reconnect_generate_btn(self._on_cancel)

    def _set_buttons_done(self) -> None:
        """Done state: generation complete."""
        self._back_btn.setEnabled(True)
        self._again_btn.setVisible(True)
        self._generate_btn.setText("Close")
        self._generate_btn.setEnabled(True)
        self._reconnect_generate_btn(self._dialog.accept)

    def _on_generate_again(self) -> None:
        """Reset UI for new generation."""
        self._progress_widget.hide()
        self._cost_label.setText("")
        self._set_buttons_idle()
        self._update_cost_estimate()

    def _on_generate(self) -> None:
        """Generate button handler — validation, cost estimation, async generation."""
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

        # Build MaterialOptions
        mat_opts = self._get_material_options()

        # Build voice and image_size
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

        # Fetch up-to-date pricing from OpenRouter (model may have changed)
        self._ensure_pricing(config, client)

        # Count cards
        card_count = _count_input_items(
            request.input_text, self._mode, max_cards_per_paragraph=self._get_max_cards_per_paragraph()
        )

        # Switch buttons and show progress
        self._set_buttons_generating()
        self._progress_widget.show(card_count)

        # Create generator
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

        # Ensure note types exist (before thread start — works with Anki API from main thread)
        self._ensure_note_types(self._mode)

        # Save request for use in callbacks
        self._current_request = request

        # Start generation in background thread
        self._worker = GenerationWorker(generator, request)
        self._worker.connect_progress(self._on_progress_updated)
        self._worker.connect_finished(self._on_generation_finished)
        self._worker.connect_error(self._on_generation_error)
        self._worker.start()

    def _on_progress_updated(self, progress: GenerationProgress) -> None:
        """Update UI on progress signal from worker thread."""
        self._last_progress = progress
        self._progress_widget.update_progress(progress)
        cost_text = _format_cost(progress.current_cost) if progress.current_cost > 0 else ""
        parts = [f"{progress.completed_cards}/{progress.total_cards} cards"]
        if cost_text:
            parts.append(f"spent: {cost_text}")
        self._cost_label.setText(" · ".join(parts))
        self._cost_label.setStyleSheet("color: palette(text);")

    def _on_generation_finished(self, cards: list[GeneratedCard]) -> None:
        """Handle successful generation completion."""
        request = self._current_request

        _save_cards_to_deck(cards, request.target_deck, create_new=request.create_new_deck)

        # Calculate final cost (from last progress)
        total_cost = 0.0
        if hasattr(self, "_last_progress"):
            total_cost = self._last_progress.current_cost

        # Log costs
        _log_cost(self._mode, len(cards), total_cost)

        self._progress_widget.finish()
        self._cost_label.setText(f"Done \u00b7 {len(cards)} cards \u00b7 {_format_cost(total_cost)}")
        self._cost_label.setStyleSheet("color: #4caf50;")
        self._set_buttons_done()
        self._worker = None

        # Refresh Anki main window to show new cards
        import contextlib

        with contextlib.suppress(Exception):
            self._mw.reset()

    def _on_generation_error(self, error_msg: str) -> None:
        """Handle generation error."""
        self._progress_widget.hide()
        self._cost_label.setText(f"Error: {error_msg}")
        self._cost_label.setStyleSheet("color: #f44336;")
        self._set_buttons_idle()
        self._worker = None

    def _on_cancel(self) -> None:
        """Cancel current generation."""
        if self._worker is not None:
            self._worker.cancel()

    def _ensure_pricing(self, config: AddonConfig, client: OpenRouterClient) -> None:
        """Fetch model pricing — always updates all three from API."""
        import contextlib

        from ankiforge.config.dialog import _extract_pricing
        from ankiforge.config.manager import save_config

        with contextlib.suppress(Exception):
            models = client.fetch_models()
            # Always update pricing — model may have changed
            if config.text_model:
                config.text_model_pricing = _extract_pricing(config.text_model, models)
            if config.image_model:
                config.image_model_pricing = _extract_pricing(config.image_model, models)
            if config.audio_model:
                config.audio_model_pricing = _extract_pricing(config.audio_model, models)
            save_config(config)

    def _ensure_note_types(self, mode: GenerationMode) -> None:
        """Create required note types for the selected mode."""
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
        """Show dialog modally.

        Returns:
            True если пользователь нажал Назад (для возврата к выбору режима).
        """
        self._dialog.exec()
        return self._go_back
