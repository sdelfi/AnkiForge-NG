"""Диалог настроек AnkiForge — API-ключ, выбор моделей, язык."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ankiforge.config.manager import get_config, save_config, validate_api_key
from ankiforge.models import AddonConfig
from ankiforge.openrouter.models import Modality, Model

if TYPE_CHECKING:
    from aqt.main import AnkiQt  # type: ignore[import-not-found]
    from aqt.qt import QComboBox  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Утилиты (тестируемые без Qt)
# ---------------------------------------------------------------------------


def _filter_models_by_modality(models: list[Model], modality: Modality) -> list[Model]:
    """Фильтрует модели по модальности.

    Args:
        models: Список всех моделей.
        modality: Нужная модальность.

    Returns:
        Отфильтрованный список моделей.
    """
    return [m for m in models if modality in m.modalities]


def _build_config_from_dialog_state(
    *,
    api_key: str,
    text_model_id: str,
    image_model_id: str,
    audio_model_id: str,
    language: str,
) -> AddonConfig:
    """Собирает AddonConfig из значений диалога.

    Args:
        api_key: API-ключ.
        text_model_id: ID текстовой модели.
        image_model_id: ID image модели.
        audio_model_id: ID audio модели.
        language: Язык карточек.

    Returns:
        Сконфигурированный AddonConfig.
    """
    return AddonConfig(
        api_key=api_key.strip(),
        text_model=text_model_id,
        image_model=image_model_id,
        audio_model=audio_model_id,
        language=language,
    )


def _populate_model_combo(combo: QComboBox, models: list[Model], current_id: str) -> None:
    """Заполняет QComboBox моделями.

    Args:
        combo: QComboBox для заполнения.
        models: Список моделей.
        current_id: ID текущей выбранной модели.
    """
    combo.clear()
    combo.addItem("— не выбрано —", "")

    selected_index = -1
    for i, model in enumerate(models):
        combo.addItem(f"{model.name} ({model.id})", model.id)
        if model.id == current_id:
            selected_index = i + 1  # +1 за пустой элемент

    if selected_index > 0:
        combo.setCurrentIndex(selected_index)


def _get_selected_model_id(combo: QComboBox) -> str:
    """Получает ID выбранной модели из QComboBox.

    Args:
        combo: QComboBox с моделями.

    Returns:
        ID модели или пустая строка.
    """
    data = combo.currentData()
    if data is None:
        return ""
    return str(data)


def _validate_api_key_action(api_key: str) -> tuple[bool, str | None]:
    """Валидация API-ключа (обёртка для тестирования).

    Args:
        api_key: API-ключ для проверки.

    Returns:
        Кортеж (is_valid, error_message).
    """
    if not api_key or not api_key.strip():
        return False, "API-ключ пуст"
    return validate_api_key(api_key.strip())


# ---------------------------------------------------------------------------
# SettingsDialog
# ---------------------------------------------------------------------------


class SettingsDialog:
    """Диалог настроек AnkiForge.

    Поля: API-ключ, текстовая модель, image модель, audio модель, язык.
    Модели загружаются с OpenRouter и фильтруются по модальности.
    """

    def __init__(self, mw: AnkiQt) -> None:
        """Инициализация диалога настроек.

        Args:
            mw: Главное окно Anki.
        """
        from aqt.qt import (
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QVBoxLayout,
        )

        self._mw = mw
        self._models: list[Model] = []

        self._dialog = QDialog(mw)
        self._dialog.setWindowTitle("AnkiForge Settings")
        self._dialog.setMinimumWidth(500)

        layout = QVBoxLayout()
        self._dialog.setLayout(layout)

        form = QFormLayout()

        # --- API-ключ ---
        api_key_layout = QHBoxLayout()
        self._api_key_input = QLineEdit()
        self._api_key_input.setPlaceholderText("sk-or-...")
        self._api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        api_key_layout.addWidget(self._api_key_input)

        validate_btn = QPushButton("Проверить")
        validate_btn.clicked.connect(self._on_validate_key)
        api_key_layout.addWidget(validate_btn)

        form.addRow("API-ключ OpenRouter:", api_key_layout)

        self._validation_label = QLabel("")
        form.addRow("", self._validation_label)

        # --- Модели ---
        self._text_model_combo = QComboBox()
        form.addRow("Текстовая модель:", self._text_model_combo)

        self._image_model_combo = QComboBox()
        form.addRow("Image модель:", self._image_model_combo)

        self._audio_model_combo = QComboBox()
        form.addRow("Audio модель:", self._audio_model_combo)

        # --- Язык ---
        self._language_combo = QComboBox()
        self._language_combo.addItem("English", "en")
        self._language_combo.addItem("Русский", "ru")
        self._language_combo.addItem("Deutsch", "de")
        self._language_combo.addItem("Français", "fr")
        self._language_combo.addItem("Español", "es")
        self._language_combo.addItem("日本語", "ja")
        self._language_combo.addItem("中文", "zh")
        form.addRow("Язык карточек:", self._language_combo)

        layout.addLayout(form)

        # --- Кнопка загрузки моделей ---
        load_models_btn = QPushButton("Загрузить модели с OpenRouter")
        load_models_btn.clicked.connect(self._on_load_models)
        layout.addWidget(load_models_btn)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        # --- OK / Cancel ---
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self._dialog.reject)
        layout.addWidget(button_box)

        # Загружаем текущие настройки
        self._load_current_config()

    def _load_current_config(self) -> None:
        """Загружает текущую конфигурацию в поля диалога."""
        config = get_config()
        self._api_key_input.setText(config.api_key)

        # Устанавливаем язык
        for i in range(self._language_combo.count()):
            if self._language_combo.itemData(i) == config.language:
                self._language_combo.setCurrentIndex(i)
                break

        # Модели — пытаемся загрузить если есть API-ключ
        if config.api_key:
            self._load_models_silent(config)

    def _load_models_silent(self, config: AddonConfig) -> None:
        """Загружает модели без показа ошибок (для инициализации)."""
        try:
            from ankiforge.openrouter.client import OpenRouterClient

            client = OpenRouterClient(api_key=config.api_key)
            self._models = client.fetch_models()
            self._populate_combos(config)
        except Exception:  # noqa: BLE001
            pass

    def _populate_combos(self, config: AddonConfig) -> None:
        """Заполняет combobox'ы моделями."""
        text_models = _filter_models_by_modality(self._models, Modality.TEXT)
        image_models = _filter_models_by_modality(self._models, Modality.IMAGE)
        audio_models = _filter_models_by_modality(self._models, Modality.AUDIO)

        _populate_model_combo(self._text_model_combo, text_models, config.text_model)
        _populate_model_combo(self._image_model_combo, image_models, config.image_model)
        _populate_model_combo(self._audio_model_combo, audio_models, config.audio_model)

    def _on_validate_key(self) -> None:
        """Обработчик кнопки валидации API-ключа."""
        api_key = self._api_key_input.text().strip()
        is_valid, error = _validate_api_key_action(api_key)

        if is_valid:
            self._validation_label.setText("Ключ валиден")
            self._validation_label.setStyleSheet("color: green;")
        else:
            self._validation_label.setText(f"{error}")
            self._validation_label.setStyleSheet("color: red;")

    def _on_load_models(self) -> None:
        """Обработчик кнопки загрузки моделей."""
        api_key = self._api_key_input.text().strip()
        if not api_key:
            self._status_label.setText("Сначала введите API-ключ")
            self._status_label.setStyleSheet("color: red;")
            return

        try:
            from ankiforge.openrouter.client import OpenRouterClient

            self._status_label.setText("Загрузка...")
            client = OpenRouterClient(api_key=api_key)
            self._models = client.fetch_models()

            config = get_config()
            self._populate_combos(config)

            count = len(self._models)
            self._status_label.setText(f"Загружено {count} моделей")
            self._status_label.setStyleSheet("color: green;")
        except Exception as e:  # noqa: BLE001
            self._status_label.setText(f"Ошибка: {e}")
            self._status_label.setStyleSheet("color: red;")

    def _on_accept(self) -> None:
        """Обработчик кнопки OK — сохранение настроек."""
        config = _build_config_from_dialog_state(
            api_key=self._api_key_input.text(),
            text_model_id=_get_selected_model_id(self._text_model_combo),
            image_model_id=_get_selected_model_id(self._image_model_combo),
            audio_model_id=_get_selected_model_id(self._audio_model_combo),
            language=self._language_combo.currentData() or "en",
        )
        save_config(config)
        self._dialog.accept()

    def run(self) -> int:
        """Показывает диалог модально.

        Returns:
            QDialog.DialogCode (Accepted / Rejected).
        """
        return self._dialog.exec()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Регистрация в меню
# ---------------------------------------------------------------------------


def setup_settings_menu(mw: AnkiQt) -> None:
    """Добавляет пункт 'AnkiForge Settings' в меню Tools.

    Args:
        mw: Главное окно Anki.
    """
    action = mw.form.menuTools.addAction("AnkiForge Settings...")
    action.triggered.connect(lambda: _open_settings(mw))


def _open_settings(mw: AnkiQt) -> None:
    """Открывает диалог настроек."""
    dialog = SettingsDialog(mw)
    dialog.run()
