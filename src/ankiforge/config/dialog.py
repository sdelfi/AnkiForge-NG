"""AnkiForge settings dialog — API key, model selection, language."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ankiforge.config.manager import get_config, save_config, validate_api_key
from ankiforge.models import AddonConfig
from ankiforge.openrouter.models import Modality, Model
from ankiforge.ui.styles import DIALOG_QSS

if TYPE_CHECKING:
    from aqt.main import AnkiQt  # type: ignore[import-not-found]
    from aqt.qt import QComboBox, QLabel  # type: ignore[import-not-found]

    from ankiforge.models import ModelPricingCache


# ---------------------------------------------------------------------------
# Utilities (testable without Qt)
# ---------------------------------------------------------------------------


def _filter_models_by_modality(models: list[Model], modality: Modality) -> list[Model]:
    """Filter models by modality.

    Args:
        models: List of all models.
        modality: Required modality.

    Returns:
        Filtered list of models.
    """
    return [m for m in models if modality in m.modalities]


def _extract_pricing(model_id: str, models: list[Model]) -> ModelPricingCache | None:
    """Extract model pricing by ID.

    Args:
        model_id: Model ID.
        models: List of loaded models.

    Returns:
        ModelPricingCache or None if model not found.
    """
    from ankiforge.models import ModelPricingCache

    if not model_id:
        return None
    for m in models:
        if m.id == model_id:
            return ModelPricingCache(
                prompt=m.pricing.prompt,
                completion=m.pricing.completion,
                image=m.pricing.image,
                request=m.pricing.request,
            )
    return None


def _build_config_from_dialog_state(
    *,
    api_key: str,
    text_model_id: str,
    image_model_id: str,
    audio_model_id: str,
    language: str,
    models: list[Model] | None = None,
) -> AddonConfig:
    """Build AddonConfig from dialog values.

    Args:
        api_key: API key.
        text_model_id: Text model ID.
        image_model_id: Image model ID.
        audio_model_id: Audio model ID.
        language: Card language.
        models: Loaded models for pricing extraction.

    Returns:
        Configured AddonConfig.
    """
    all_models = models or []
    return AddonConfig(
        api_key=api_key.strip(),
        text_model=text_model_id,
        image_model=image_model_id,
        audio_model=audio_model_id,
        language=language,
        text_model_pricing=_extract_pricing(text_model_id, all_models),
        image_model_pricing=_extract_pricing(image_model_id, all_models),
        audio_model_pricing=_extract_pricing(audio_model_id, all_models),
    )


def _make_searchable_combo() -> QComboBox:
    """Create a QComboBox with substring search.

    Returns:
        Editable QComboBox with QCompleter (MatchContains, CaseInsensitive).
    """
    from aqt.qt import QComboBox, QCompleter, QSizePolicy, Qt

    combo = QComboBox()
    combo.setEditable(True)
    combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    combo.setMinimumWidth(300)
    combo.lineEdit().setPlaceholderText("Start typing a model name...")

    completer = QCompleter()
    completer.setFilterMode(Qt.MatchFlag.MatchContains)
    completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
    combo.setCompleter(completer)

    return combo


def _populate_model_combo(combo: QComboBox, models: list[Model], current_id: str) -> None:
    """Populate QComboBox with models.

    Args:
        combo: QComboBox to populate.
        models: List of models.
        current_id: ID of the currently selected model.
    """
    combo.clear()
    combo.addItem("— not selected —", "")

    selected_index = -1
    for i, model in enumerate(models):
        combo.addItem(f"{model.name} ({model.id})", model.id)
        if model.id == current_id:
            selected_index = i + 1  # +1 for the empty element

    if selected_index > 0:
        combo.setCurrentIndex(selected_index)

    # Update completer model if combo is editable
    if combo.isEditable() and combo.completer() is not None:
        combo.completer().setModel(combo.model())


def _get_selected_model_id(combo: QComboBox) -> str:
    """Get the ID of the selected model from QComboBox.

    If the user selected from the list — returns data (ID).
    If the user typed custom text — extracts ID from text.

    Args:
        combo: QComboBox with models.

    Returns:
        Model ID or empty string.
    """
    data = combo.currentData()
    if data is not None:
        return str(data)

    # Custom input — user typed text manually
    text = str(combo.currentText()).strip()
    if not text or text == "— not selected —":
        return ""

    # Text may be in format "ModelName (provider/model-id)" — extract ID from parentheses
    if "(" in text and text.endswith(")"):
        return text[text.rfind("(") + 1 : -1].strip()

    # Or just the model ID directly (e.g. "google/gemini-2.5-flash")
    return text


def _set_status(label: QLabel, text: str, *, ok: bool) -> None:
    """Set text and color for a status label.

    Args:
        label: QLabel for status.
        text: Message text.
        ok: True — green, False — red.
    """
    label.setText(text)
    color = "#4caf50" if ok else "#f44336"
    label.setStyleSheet(f"color: {color}; font-weight: normal;")


def _validate_api_key_action(api_key: str) -> tuple[bool, str | None]:
    """Validate API key (wrapper for testing).

    Args:
        api_key: API key to validate.

    Returns:
        Tuple (is_valid, error_message).
    """
    if not api_key or not api_key.strip():
        return False, "API key is empty"
    return validate_api_key(api_key.strip())


# ---------------------------------------------------------------------------
# SettingsDialog
# ---------------------------------------------------------------------------


class SettingsDialog:
    """AnkiForge settings dialog.

    Fields: API key, text model, image model, audio model, language.
    Models are loaded from OpenRouter and filtered by modality.
    """

    def __init__(self, mw: AnkiQt) -> None:
        """Initialize settings dialog.

        Args:
            mw: Anki main window.
        """
        from aqt.qt import (
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QVBoxLayout,
        )
        from aqt.qt import (
            QWidget as _QWidget,
        )

        self._mw = mw
        self._models: list[Model] = []

        self._dialog = QDialog(mw)
        self._dialog.setWindowTitle("AnkiForge Settings")
        self._dialog.setMinimumWidth(560)
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

        # === Section 1: OpenRouter Connection ===
        api_group = QGroupBox("OpenRouter Connection")
        api_layout = QFormLayout()
        api_layout.setSpacing(8)
        api_group.setLayout(api_layout)

        # API key
        api_key_row = QHBoxLayout()
        self._api_key_input = QLineEdit()
        self._api_key_input.setPlaceholderText("sk-or-...")
        self._api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        api_key_row.addWidget(self._api_key_input)

        connect_btn = QPushButton("Connect")
        connect_btn.setObjectName("connectBtn")
        connect_btn.clicked.connect(self._on_connect)
        api_key_row.addWidget(connect_btn)

        api_layout.addRow("API Key:", api_key_row)

        self._api_status_label = QLabel("")
        self._api_status_label.setWordWrap(True)
        api_layout.addRow("", self._api_status_label)

        layout.addWidget(api_group)

        # === Section 2: Models ===
        models_group = QGroupBox("Models")
        models_layout = QFormLayout()
        models_layout.setSpacing(8)
        models_group.setLayout(models_layout)

        self._text_model_combo = _make_searchable_combo()
        models_layout.addRow("Text model:", self._text_model_combo)

        self._image_model_combo = _make_searchable_combo()
        models_layout.addRow("Image model:", self._image_model_combo)

        self._audio_model_combo = _make_searchable_combo()
        models_layout.addRow("Audio model:", self._audio_model_combo)

        layout.addWidget(models_group)

        # === Section 3: Balance ===
        balance_group = QGroupBox("Balance")
        balance_layout = QFormLayout()
        balance_layout.setSpacing(4)
        balance_layout.setContentsMargins(8, 6, 8, 8)

        self._usage_label = QLabel("—")
        self._usage_label.setStyleSheet("color: palette(text);")
        balance_layout.addRow("Usage:", self._usage_label)

        self._remaining_label = QLabel("—")
        self._remaining_label.setStyleSheet("color: palette(text);")
        balance_layout.addRow("Remaining:", self._remaining_label)

        refresh_balance_btn = QPushButton("Refresh")
        refresh_balance_btn.clicked.connect(self._on_refresh_balance)
        balance_layout.addRow("", refresh_balance_btn)

        balance_group.setLayout(balance_layout)
        layout.addWidget(balance_group)

        # Scroll area with content
        from ankiforge.ui.styles import get_dialog_size, wrap_in_scroll_area

        dialog_layout.addWidget(wrap_in_scroll_area(content))

        # === OK / Cancel (outside scroll, always visible) ===
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(14, 6, 14, 14)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self._dialog.reject)
        btn_layout.addWidget(button_box)
        dialog_layout.addLayout(btn_layout)

        # Scale to screen size
        w, h = get_dialog_size(width_pct=0.35, height_pct=0.55, min_w=560, min_h=400)
        self._dialog.resize(w, h)

        # Load current settings
        self._load_current_config()

    def _load_current_config(self) -> None:
        """Load current configuration into dialog fields."""
        config = get_config()
        self._api_key_input.setText(config.api_key)

        # Show cached values
        if config.cached_usage is not None:
            self._usage_label.setText(f"${config.cached_usage:.2f}")
        if config.cached_balance is not None:
            self._remaining_label.setText(f"${config.cached_balance:.2f}")

        # Models — try to load if API key is present
        if config.api_key:
            self._load_models_silent(config)

    def _load_models_silent(self, config: AddonConfig) -> None:
        """Load models without showing errors (for initialization)."""
        try:
            from ankiforge.openrouter.client import OpenRouterClient

            client = OpenRouterClient(api_key=config.api_key)
            self._models = client.fetch_models()
            self._populate_combos(config)
            count = len(self._models)
            _set_status(self._api_status_label, f"Connected, loaded {count} models", ok=True)
        except Exception:  # noqa: BLE001
            pass

    def _populate_combos(self, config: AddonConfig) -> None:
        """Populate comboboxes with models."""
        text_models = _filter_models_by_modality(self._models, Modality.TEXT)
        image_models = _filter_models_by_modality(self._models, Modality.IMAGE)
        audio_models = _filter_models_by_modality(self._models, Modality.AUDIO)

        _populate_model_combo(self._text_model_combo, text_models, config.text_model)
        _populate_model_combo(self._image_model_combo, image_models, config.image_model)
        _populate_model_combo(self._audio_model_combo, audio_models, config.audio_model)

    def _on_connect(self) -> None:
        """Connect button handler — validate key + load models."""
        api_key = self._api_key_input.text().strip()
        is_valid, error = _validate_api_key_action(api_key)

        if not is_valid:
            _set_status(self._api_status_label, f"Error: {error}", ok=False)
            return

        # Key is valid — load models immediately
        try:
            from ankiforge.openrouter.client import OpenRouterClient

            _set_status(self._api_status_label, "Loading models...", ok=True)
            client = OpenRouterClient(api_key=api_key)
            self._models = client.fetch_models()

            config = get_config()
            self._populate_combos(config)

            count = len(self._models)
            _set_status(self._api_status_label, f"Connected, loaded {count} models", ok=True)
        except Exception as e:  # noqa: BLE001
            _set_status(self._api_status_label, f"Failed to load models: {e}", ok=False)

    def _on_refresh_balance(self) -> None:
        """Fetch balance from OpenRouter API and update UI + cache."""
        api_key = self._api_key_input.text().strip()
        if not api_key:
            self._usage_label.setText("no API key")
            self._remaining_label.setText("no API key")
            return

        self._usage_label.setText("loading...")
        self._remaining_label.setText("loading...")
        try:
            from ankiforge.openrouter.client import OpenRouterClient

            client = OpenRouterClient(api_key=api_key)
            balance = client.fetch_balance()
            usage = balance["usage"]
            remaining = balance["remaining"]
            self._usage_label.setText(f"${usage:.2f}")
            if remaining < 0:
                self._remaining_label.setText("unknown (unlimited key)")
            else:
                self._remaining_label.setText(f"${remaining:.2f}")

            # Cache in config
            config = get_config()
            config.cached_usage = usage
            config.cached_balance = remaining
            save_config(config)
        except Exception as e:  # noqa: BLE001
            self._usage_label.setText(f"error ({e})")
            self._remaining_label.setText("—")

    def _validate_custom_models(self) -> str | None:
        """Validate custom models (entered manually) via API.

        Returns:
            Error message or None if everything is ok.
        """
        combos = {
            "Text": self._text_model_combo,
            "Image": self._image_model_combo,
            "Audio": self._audio_model_combo,
        }
        known_ids = {m.id for m in self._models}

        for label, combo in combos.items():
            model_id = _get_selected_model_id(combo)
            if not model_id or model_id in known_ids:
                continue

            # Custom model — verify existence via API
            api_key = self._api_key_input.text().strip()
            if not api_key:
                return f"{label} model '{model_id}' not found in the list, and no API key provided"

            try:
                from ankiforge.openrouter.client import OpenRouterClient

                client = OpenRouterClient(api_key=api_key)
                # Load up-to-date model list (no cache)
                client._models_cache = None  # noqa: SLF001
                all_models = client.fetch_models()
                found = next((m for m in all_models if m.id == model_id), None)
                if found is None:
                    return f"{label} model '{model_id}' not found on OpenRouter"
                # Add found model to local cache
                if found not in self._models:
                    self._models.append(found)
            except Exception as e:  # noqa: BLE001
                return f"Error validating model '{model_id}': {e}"

        return None

    def _on_accept(self) -> None:
        """OK button handler — validate custom models + save."""
        # Validate custom models
        error = self._validate_custom_models()
        if error:
            from aqt.qt import QMessageBox

            QMessageBox.warning(self._dialog, "Model Error", error)
            return

        config = _build_config_from_dialog_state(
            api_key=self._api_key_input.text(),
            text_model_id=_get_selected_model_id(self._text_model_combo),
            image_model_id=_get_selected_model_id(self._image_model_combo),
            audio_model_id=_get_selected_model_id(self._audio_model_combo),
            language=get_config().language,
            models=self._models,
        )
        save_config(config)
        self._dialog.accept()

    def run(self) -> int:
        """Show dialog modally.

        Returns:
            QDialog.DialogCode (Accepted / Rejected).
        """
        return self._dialog.exec()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Menu registration
# ---------------------------------------------------------------------------


def setup_settings_menu(mw: AnkiQt) -> None:
    """Add 'AnkiForge Settings' item to the Tools menu.

    Args:
        mw: Anki main window.
    """
    action = mw.form.menuTools.addAction("AnkiForge Settings...")
    action.triggered.connect(lambda: _open_settings(mw))


def _open_settings(mw: AnkiQt) -> None:
    """Open the settings dialog."""
    dialog = SettingsDialog(mw)
    dialog.run()
