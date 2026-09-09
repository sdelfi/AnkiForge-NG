"""AnkiForge settings dialog — API key, model selection, language."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ankiforge.config.manager import get_config, save_config, validate_api_key, validate_custom_endpoint
from ankiforge.local_image.client import validate_local_image_backend
from ankiforge.models import AddonConfig
from ankiforge.openrouter.models import Modality, Model
from ankiforge.openrouter.routing_client import LOCAL_IMAGE_MODEL_ID, add_custom_prefix, is_custom_model
from ankiforge.ui.styles import DIALOG_QSS

_LOCAL_IMAGE_BACKENDS = [
    ("", "Disabled"),
    ("automatic1111", "Automatic1111 (stable-diffusion-webui)"),
    ("comfyui", "ComfyUI"),
]
_LOCAL_IMAGE_URL_PLACEHOLDERS = {
    "automatic1111": "http://127.0.0.1:7860",
    "comfyui": "http://127.0.0.1:8188",
}
_LOCAL_IMAGE_RESOLUTIONS = [
    ("auto", "Auto (detect SDXL from checkpoint name — ComfyUI only)"),
    ("512", "512×512 (SD 1.5)"),
    ("768", "768×768"),
    ("1024", "1024×1024 (SDXL)"),
]
_LOCAL_IMAGE_ADVANCED_KEYS = ("steps", "cfg_scale", "sampler_name", "resolution")

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
    custom_base_url: str = "",
    custom_api_key: str = "",
    custom_label: str = "Local",
    local_image_backend: str = "",
    local_image_url: str = "",
    local_image_checkpoint: str = "",
    local_image_resolution: str = "auto",
    local_image_advanced: str = "",
) -> AddonConfig:
    """Build AddonConfig from dialog values.

    Args:
        api_key: OpenRouter API key.
        text_model_id: Text model ID (may be "custom::"-prefixed).
        image_model_id: Image model ID (may be "custom::"- or "local-image::"-prefixed).
        audio_model_id: Audio model ID (may be "custom::"-prefixed).
        language: Card language.
        models: Loaded models (OpenRouter + custom) for pricing extraction.
        custom_base_url: Base URL of a custom OpenAI-compatible endpoint (LM Studio, etc.), if any.
        custom_api_key: API key for the custom endpoint, if it requires one.
        custom_label: Display label for the custom endpoint in model dropdowns.
        local_image_backend: "", "automatic1111", or "comfyui".
        local_image_url: Base URL of the local image generation backend, if any.
        local_image_checkpoint: Checkpoint filename (ComfyUI only).
        local_image_resolution: "auto", "512", "768", or "1024".
        local_image_advanced: Optional raw JSON overrides for steps/cfg_scale/
            sampler_name/resolution — see AddonConfig.local_image_advanced.

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
        custom_base_url=custom_base_url.strip(),
        custom_api_key=custom_api_key.strip(),
        custom_label=custom_label.strip() or "Local",
        local_image_backend=local_image_backend,
        local_image_url=local_image_url.strip(),
        local_image_checkpoint=local_image_checkpoint.strip(),
        local_image_resolution=local_image_resolution,
        local_image_advanced=local_image_advanced.strip(),
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


def _populate_model_combo(
    combo: QComboBox, models: list[Model], current_id: str, *, custom_label: str = "Local"
) -> None:
    """Populate QComboBox with models from OpenRouter and/or a custom endpoint.

    Custom-provider models (model.source == "custom") are visually tagged
    with `custom_label` so they're easy to tell apart from OpenRouter models
    in the same list.

    Args:
        combo: QComboBox to populate.
        models: List of models (OpenRouter and/or custom, already deduplicated).
        current_id: ID of the currently selected model (may be "custom::"-prefixed).
        custom_label: Label shown next to custom-provider models, e.g. "LM Studio".
    """
    combo.clear()
    combo.addItem("— not selected —", "")

    selected_index = -1
    for i, model in enumerate(models):
        if model.source == "custom":
            tag = f" · {custom_label}"
        elif model.source == "local-image":
            tag = " · Local"
        else:
            tag = ""
        combo.addItem(f"{model.name}{tag} ({model.id})", model.id)
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
            QComboBox,
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
        self._models: list[Model] = []  # OpenRouter models
        self._custom_models: list[Model] = []  # Custom-endpoint models (LM Studio, Ollama, ...)

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

        # === Section 1b: Local / Custom API (LM Studio, Ollama, vLLM, ...) ===
        custom_group = QGroupBox("Local / Custom API (LM Studio, Ollama, etc.)")
        custom_layout = QFormLayout()
        custom_layout.setSpacing(8)
        custom_group.setLayout(custom_layout)

        self._custom_base_url_input = QLineEdit()
        self._custom_base_url_input.setPlaceholderText("http://localhost:1234/v1")
        custom_layout.addRow("Base URL:", self._custom_base_url_input)

        self._custom_api_key_input = QLineEdit()
        self._custom_api_key_input.setPlaceholderText("optional — most local servers don't need one")
        self._custom_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        custom_layout.addRow("API Key:", self._custom_api_key_input)

        self._custom_label_input = QLineEdit()
        self._custom_label_input.setPlaceholderText("Local")
        self._custom_label_input.setToolTip("Shown next to this endpoint's models in the dropdowns below.")
        custom_layout.addRow("Label:", self._custom_label_input)

        custom_connect_row = QHBoxLayout()
        custom_connect_btn = QPushButton("Connect")
        custom_connect_btn.clicked.connect(self._on_connect_custom)
        custom_connect_row.addWidget(custom_connect_btn)
        custom_layout.addRow("", custom_connect_row)

        self._custom_status_label = QLabel("")
        self._custom_status_label.setWordWrap(True)
        custom_layout.addRow("", self._custom_status_label)

        hint = QLabel(
            "Connect to pull the endpoint's model list into the dropdowns below. "
            "To reference a model that isn't listed, type its id as custom::model-id."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(placeholderText);")
        custom_layout.addRow("", hint)

        layout.addWidget(custom_group)

        # === Section 1c: Local Image Generation (Automatic1111 / ComfyUI) ===
        local_image_group = QGroupBox("Local Image Generation (Automatic1111 / ComfyUI)")
        local_image_layout = QFormLayout()
        local_image_layout.setSpacing(8)
        local_image_group.setLayout(local_image_layout)

        self._local_image_backend_combo = QComboBox()
        for value, display_name in _LOCAL_IMAGE_BACKENDS:
            self._local_image_backend_combo.addItem(display_name, value)
        self._local_image_backend_combo.currentIndexChanged.connect(self._on_local_image_backend_changed)
        local_image_layout.addRow("Backend:", self._local_image_backend_combo)

        self._local_image_url_input = QLineEdit()
        local_image_layout.addRow("Base URL:", self._local_image_url_input)

        self._local_image_checkpoint_label = QLabel("Checkpoint:")
        self._local_image_checkpoint_input = QLineEdit()
        self._local_image_checkpoint_input.setPlaceholderText("e.g. sd_xl_turbo_1.0.safetensors")
        local_image_layout.addRow(self._local_image_checkpoint_label, self._local_image_checkpoint_input)

        self._local_image_resolution_combo = QComboBox()
        for value, display_name in _LOCAL_IMAGE_RESOLUTIONS:
            self._local_image_resolution_combo.addItem(display_name, value)
        self._local_image_resolution_combo.setToolTip(
            "Generating below a checkpoint's native resolution (SDXL needs ~1024px) "
            "produces a tiled, fractured-looking image instead of just less detail. "
            "'Auto' guesses from the checkpoint filename — set this explicitly if "
            "that guesses wrong, or if you're on Automatic1111 (no checkpoint info "
            "available to guess from there)."
        )
        local_image_layout.addRow("Resolution:", self._local_image_resolution_combo)

        self._local_image_advanced_input = QLineEdit()
        self._local_image_advanced_input.setPlaceholderText(
            '{"steps": 8, "cfg_scale": 2, "sampler_name": "dpmpp_2m_sde"}'
        )
        self._local_image_advanced_input.setToolTip(
            "Optional JSON overrides for steps / cfg_scale / sampler_name / resolution, "
            "applied on top of the auto-picked values above — for a checkpoint or sampler "
            "preference the automatic detection doesn't cover. Leave empty to just use "
            "those. Recognized sampler_name values are whatever your ComfyUI/Automatic1111 "
            "install supports (e.g. 'euler', 'euler_ancestral', 'dpmpp_2m', 'dpmpp_2m_sde')."
        )
        self._local_image_advanced_input.textChanged.connect(self._on_local_image_advanced_changed)
        local_image_layout.addRow("Advanced (JSON):", self._local_image_advanced_input)

        self._local_image_advanced_status_label = QLabel("")
        local_image_layout.addRow("", self._local_image_advanced_status_label)

        local_image_connect_row = QHBoxLayout()
        local_image_connect_btn = QPushButton("Test connection")
        local_image_connect_btn.clicked.connect(self._on_test_local_image)
        local_image_connect_row.addWidget(local_image_connect_btn)
        local_image_layout.addRow("", local_image_connect_row)

        self._local_image_status_label = QLabel("")
        self._local_image_status_label.setWordWrap(True)
        local_image_layout.addRow("", self._local_image_status_label)

        local_image_hint = QLabel(
            "Free, fully offline image generation via a locally running Automatic1111 "
            "(stable-diffusion-webui) or ComfyUI server. When enabled, pick "
            "'Local Stable Diffusion' in the Image model dropdown below."
        )
        local_image_hint.setWordWrap(True)
        local_image_hint.setStyleSheet("color: palette(placeholderText);")
        local_image_layout.addRow("", local_image_hint)

        layout.addWidget(local_image_group)

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
        self._custom_base_url_input.setText(config.custom_base_url)
        self._custom_api_key_input.setText(config.custom_api_key)
        self._custom_label_input.setText(config.custom_label)

        backend_index = next(
            (i for i, (value, _) in enumerate(_LOCAL_IMAGE_BACKENDS) if value == config.local_image_backend),
            0,
        )
        self._local_image_backend_combo.setCurrentIndex(backend_index)
        self._local_image_url_input.setText(config.local_image_url)
        self._local_image_checkpoint_input.setText(config.local_image_checkpoint)
        resolution_index = next(
            (i for i, (value, _) in enumerate(_LOCAL_IMAGE_RESOLUTIONS) if value == config.local_image_resolution),
            0,
        )
        self._local_image_resolution_combo.setCurrentIndex(resolution_index)
        self._local_image_advanced_input.setText(config.local_image_advanced)
        self._update_local_image_visibility()

        # Show cached values
        if config.cached_usage is not None:
            self._usage_label.setText(f"${config.cached_usage:.2f}")
        if config.cached_balance is not None:
            self._remaining_label.setText(f"${config.cached_balance:.2f}")

        # Models — try to load if configured
        if config.api_key:
            self._load_models_silent(config)
        if config.custom_base_url:
            self._load_custom_models_silent(config)
        self._populate_combos(config)

    def _load_models_silent(self, config: AddonConfig) -> None:
        """Load OpenRouter models without showing errors (for initialization)."""
        try:
            from ankiforge.openrouter.client import OpenRouterClient

            client = OpenRouterClient(api_key=config.api_key)
            self._models = client.fetch_models()
            self._populate_combos(config)
            count = len(self._models)
            _set_status(self._api_status_label, f"Connected, loaded {count} models", ok=True)
        except Exception:  # noqa: BLE001
            pass

    def _load_custom_models_silent(self, config: AddonConfig) -> None:
        """Load models from the custom/local endpoint without showing errors (for initialization)."""
        try:
            self._custom_models = self._fetch_custom_models(config.custom_base_url, config.custom_api_key)
            self._populate_combos(config)
            count = len(self._custom_models)
            _set_status(self._custom_status_label, f"Connected, loaded {count} models", ok=True)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _fetch_custom_models(base_url: str, api_key: str) -> list[Model]:
        """Fetch and tag models from a custom OpenAI-compatible endpoint.

        Args:
            base_url: Base URL of the custom endpoint, e.g. 'http://localhost:1234/v1'.
            api_key: API key for the endpoint (may be empty — most local servers don't need one).

        Returns:
            Models with "custom::"-prefixed IDs and source="custom", ready to
            merge into the same dropdowns as OpenRouter models. Modalities
            default to TEXT since most local servers (LM Studio, Ollama)
            only expose text/chat models and don't report modality info.
        """
        from dataclasses import replace

        from ankiforge.openrouter.client import OpenRouterClient

        client = OpenRouterClient(api_key=api_key or "lm-studio", base_url=base_url.strip().rstrip("/"))
        raw_models = client.fetch_models(default_modalities=[Modality.TEXT], source="custom")
        return [replace(m, id=add_custom_prefix(m.id)) for m in raw_models]

    @staticmethod
    def _local_image_model_entry(config: AddonConfig) -> Model | None:
        """Build the synthetic 'Local Stable Diffusion' entry for the Image model dropdown.

        Only shown once a backend + URL are configured (checkpoint required
        too, for ComfyUI) — there's nothing to fetch a model list from since
        Automatic1111/ComfyUI aren't OpenAI-compatible /models endpoints.
        """
        if not config.local_image_backend or not config.local_image_url.strip():
            return None
        if config.local_image_backend == "comfyui" and not config.local_image_checkpoint.strip():
            return None
        return Model(
            id=LOCAL_IMAGE_MODEL_ID,
            name="Local Stable Diffusion",
            modalities=[Modality.IMAGE],
            source="local-image",
        )

    def _populate_combos(self, config: AddonConfig) -> None:
        """Populate comboboxes with OpenRouter + custom-endpoint models, merged by modality."""
        all_models = self._models + self._custom_models
        text_models = _filter_models_by_modality(all_models, Modality.TEXT)
        image_models = _filter_models_by_modality(all_models, Modality.IMAGE)
        audio_models = _filter_models_by_modality(all_models, Modality.AUDIO)

        local_image_entry = self._local_image_model_entry(config)
        if local_image_entry is not None:
            image_models = [local_image_entry, *image_models]

        label = config.custom_label or "Local"
        _populate_model_combo(self._text_model_combo, text_models, config.text_model, custom_label=label)
        _populate_model_combo(self._image_model_combo, image_models, config.image_model, custom_label=label)
        _populate_model_combo(self._audio_model_combo, audio_models, config.audio_model, custom_label=label)

    def _update_local_image_visibility(self) -> None:
        """Show/hide the checkpoint field and set the URL placeholder based on the selected backend."""
        backend = self._local_image_backend_combo.currentData()
        self._local_image_url_input.setPlaceholderText(_LOCAL_IMAGE_URL_PLACEHOLDERS.get(backend, ""))
        is_comfyui = backend == "comfyui"
        self._local_image_checkpoint_label.setVisible(is_comfyui)
        self._local_image_checkpoint_input.setVisible(is_comfyui)

    def _on_local_image_backend_changed(self) -> None:
        """Backend combo change handler — update field visibility/placeholder."""
        self._update_local_image_visibility()

    def _on_local_image_advanced_changed(self) -> None:
        """Live-validate the Advanced (JSON) field as the user types."""
        text = self._local_image_advanced_input.text().strip()
        if not text:
            self._local_image_advanced_status_label.setText("")
            return

        try:
            data = json.loads(text)
        except ValueError as e:
            _set_status(self._local_image_advanced_status_label, f"Invalid JSON: {e}", ok=False)
            return

        if not isinstance(data, dict):
            _set_status(self._local_image_advanced_status_label, "Must be a JSON object", ok=False)
            return

        unknown = sorted(set(data) - set(_LOCAL_IMAGE_ADVANCED_KEYS))
        if unknown:
            _set_status(
                self._local_image_advanced_status_label,
                f"Unrecognized key(s), ignored: {', '.join(unknown)}",
                ok=False,
            )
            return

        _set_status(self._local_image_advanced_status_label, "Valid", ok=True)

    def _on_test_local_image(self) -> None:
        """Test connection button handler for the local image generation backend."""
        backend = self._local_image_backend_combo.currentData()
        base_url = self._local_image_url_input.text().strip()

        if not backend:
            _set_status(self._local_image_status_label, "Select a backend first", ok=False)
            return

        is_reachable, error = validate_local_image_backend(backend, base_url)
        if not is_reachable:
            _set_status(self._local_image_status_label, f"Error: {error}", ok=False)
            return

        _set_status(self._local_image_status_label, "Connected", ok=True)

        from dataclasses import replace

        config = replace(
            get_config(),
            local_image_backend=backend,
            local_image_url=base_url,
            local_image_checkpoint=self._local_image_checkpoint_input.text().strip(),
            text_model=_get_selected_model_id(self._text_model_combo),
            image_model=_get_selected_model_id(self._image_model_combo),
            audio_model=_get_selected_model_id(self._audio_model_combo),
        )
        self._populate_combos(config)

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

    def _on_connect_custom(self) -> None:
        """Connect button handler for the local/custom endpoint — validate + load models."""
        base_url = self._custom_base_url_input.text().strip()
        api_key = self._custom_api_key_input.text().strip()

        is_reachable, error = validate_custom_endpoint(base_url)
        if not is_reachable:
            _set_status(self._custom_status_label, f"Error: {error}", ok=False)
            return

        try:
            _set_status(self._custom_status_label, "Loading models...", ok=True)
            self._custom_models = self._fetch_custom_models(base_url, api_key)

            config = get_config()
            self._populate_combos(config)

            count = len(self._custom_models)
            _set_status(self._custom_status_label, f"Connected, loaded {count} models", ok=True)
        except Exception as e:  # noqa: BLE001
            _set_status(self._custom_status_label, f"Failed to load models: {e}", ok=False)

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

    def _validate_manually_typed_models(self) -> str | None:
        """Validate models that were typed manually rather than picked from a dropdown.

        Two cases:
        - A plain id not in the fetched OpenRouter list — verified against
          OpenRouter (requires an OpenRouter API key).
        - A "custom::"-prefixed id not in the fetched custom-endpoint list —
          verified against the configured local/custom endpoint (no API key
          required, but the endpoint must be configured).

        Returns:
            Error message, or None if everything is ok.
        """
        combos = {
            "Text": self._text_model_combo,
            "Image": self._image_model_combo,
            "Audio": self._audio_model_combo,
        }
        known_openrouter_ids = {m.id for m in self._models}
        known_custom_ids = {m.id for m in self._custom_models}

        for label, combo in combos.items():
            model_id = _get_selected_model_id(combo)
            if not model_id:
                continue

            if model_id == LOCAL_IMAGE_MODEL_ID:
                if not self._local_image_backend_combo.currentData():
                    return f"{label} model is set to local image generation, but no backend is selected"
                continue

            if is_custom_model(model_id):
                if model_id in known_custom_ids:
                    continue

                base_url = self._custom_base_url_input.text().strip()
                if not base_url:
                    return f"{label} model '{model_id}' not found, and no local/custom API endpoint is configured"

                try:
                    api_key = self._custom_api_key_input.text().strip()
                    found_models = self._fetch_custom_models(base_url, api_key)
                    found = next((m for m in found_models if m.id == model_id), None)
                    if found is None:
                        return f"{label} model not found on the local/custom endpoint: '{model_id}'"
                    if found not in self._custom_models:
                        self._custom_models.append(found)
                except Exception as e:  # noqa: BLE001
                    return f"Error validating model '{model_id}': {e}"
                continue

            if model_id in known_openrouter_ids:
                continue

            # Plain OpenRouter model id — verify existence via API
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
        """OK button handler — validate manually-typed models + save."""
        error = self._validate_manually_typed_models()
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
            models=self._models + self._custom_models,
            custom_base_url=self._custom_base_url_input.text(),
            custom_api_key=self._custom_api_key_input.text(),
            custom_label=self._custom_label_input.text(),
            local_image_backend=self._local_image_backend_combo.currentData() or "",
            local_image_url=self._local_image_url_input.text(),
            local_image_checkpoint=self._local_image_checkpoint_input.text(),
            local_image_resolution=self._local_image_resolution_combo.currentData() or "auto",
            local_image_advanced=self._local_image_advanced_input.text().strip(),
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
