"""Тесты для ankiforge.config.dialog — SettingsDialog."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ankiforge.models import AddonConfig
from ankiforge.openrouter.models import Modality, Model, ModelPricing

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_models() -> list[Model]:
    """Набор моделей разных модальностей для тестов."""
    return [
        Model(
            id="openai/gpt-4o",
            name="GPT-4o",
            pricing=ModelPricing(prompt=0.005, completion=0.015),
            modalities=[Modality.TEXT],
            context_length=128000,
        ),
        Model(
            id="anthropic/claude-3.5-sonnet",
            name="Claude 3.5 Sonnet",
            pricing=ModelPricing(prompt=0.003, completion=0.015),
            modalities=[Modality.TEXT],
            context_length=200000,
        ),
        Model(
            id="openai/dall-e-3",
            name="DALL-E 3",
            pricing=ModelPricing(image=0.04),
            modalities=[Modality.IMAGE],
        ),
        Model(
            id="stability/sdxl",
            name="Stable Diffusion XL",
            pricing=ModelPricing(image=0.002),
            modalities=[Modality.IMAGE],
        ),
        Model(
            id="openai/tts-1",
            name="TTS-1",
            pricing=ModelPricing(prompt=0.000015),
            modalities=[Modality.AUDIO],
        ),
        Model(
            id="openai/tts-1-hd",
            name="TTS-1 HD",
            pricing=ModelPricing(prompt=0.00003),
            modalities=[Modality.AUDIO],
        ),
    ]


@pytest.fixture()
def mock_qt() -> dict[str, Any]:
    """Моки Qt-виджетов для тестирования без PyQt6."""
    # Мок QDialog
    mock_qdialog = MagicMock()
    mock_qdialog.Accepted = 1
    mock_qdialog.Rejected = 0

    # Мок QVBoxLayout, QHBoxLayout, QFormLayout
    mock_qvbox = MagicMock()
    mock_qhbox = MagicMock()
    mock_qform = MagicMock()

    # Мок QLineEdit
    mock_qlineedit_cls = MagicMock()

    # Мок QComboBox
    mock_qcombobox_cls = MagicMock()

    # Мок QPushButton
    mock_qpushbutton_cls = MagicMock()

    # Мок QLabel
    mock_qlabel_cls = MagicMock()

    # Мок QDialogButtonBox
    mock_button_box_cls = MagicMock()
    mock_button_box_cls.Ok = 0x00000400
    mock_button_box_cls.Cancel = 0x00400000

    # Мок QMessageBox
    mock_msgbox = MagicMock()

    return {
        "QDialog": mock_qdialog,
        "QVBoxLayout": mock_qvbox,
        "QHBoxLayout": mock_qhbox,
        "QFormLayout": mock_qform,
        "QLineEdit": mock_qlineedit_cls,
        "QComboBox": mock_qcombobox_cls,
        "QPushButton": mock_qpushbutton_cls,
        "QLabel": mock_qlabel_cls,
        "QDialogButtonBox": mock_button_box_cls,
        "QMessageBox": mock_msgbox,
    }


# ---------------------------------------------------------------------------
# filter_models_by_modality
# ---------------------------------------------------------------------------


class TestFilterModelsByModality:
    """Тесты фильтрации моделей по модальности."""

    def test_filter_text_models(self, sample_models: list[Model]) -> None:
        from ankiforge.config.dialog import _filter_models_by_modality

        text_models = _filter_models_by_modality(sample_models, Modality.TEXT)
        assert len(text_models) == 2
        assert all(Modality.TEXT in m.modalities for m in text_models)

    def test_filter_image_models(self, sample_models: list[Model]) -> None:
        from ankiforge.config.dialog import _filter_models_by_modality

        image_models = _filter_models_by_modality(sample_models, Modality.IMAGE)
        assert len(image_models) == 2
        assert all(Modality.IMAGE in m.modalities for m in image_models)

    def test_filter_audio_models(self, sample_models: list[Model]) -> None:
        from ankiforge.config.dialog import _filter_models_by_modality

        audio_models = _filter_models_by_modality(sample_models, Modality.AUDIO)
        assert len(audio_models) == 2
        assert all(Modality.AUDIO in m.modalities for m in audio_models)

    def test_filter_empty_list(self) -> None:
        from ankiforge.config.dialog import _filter_models_by_modality

        result = _filter_models_by_modality([], Modality.TEXT)
        assert result == []

    def test_filter_no_matches(self, sample_models: list[Model]) -> None:
        from ankiforge.config.dialog import _filter_models_by_modality

        # Убираем все аудио модели
        text_only = [m for m in sample_models if Modality.AUDIO not in m.modalities]
        result = _filter_models_by_modality(text_only, Modality.AUDIO)
        assert result == []


# ---------------------------------------------------------------------------
# build_config_from_dialog_state
# ---------------------------------------------------------------------------


class TestBuildConfigFromDialogState:
    """Тесты сборки AddonConfig из состояния диалога."""

    def test_builds_config_from_state(self) -> None:
        from ankiforge.config.dialog import _build_config_from_dialog_state

        config = _build_config_from_dialog_state(
            api_key="sk-or-test-key",
            text_model_id="openai/gpt-4o",
            image_model_id="openai/dall-e-3",
            audio_model_id="openai/tts-1",
            language="ru",
        )
        assert isinstance(config, AddonConfig)
        assert config.api_key == "sk-or-test-key"
        assert config.text_model == "openai/gpt-4o"
        assert config.image_model == "openai/dall-e-3"
        assert config.audio_model == "openai/tts-1"
        assert config.language == "ru"

    def test_builds_config_with_empty_models(self) -> None:
        from ankiforge.config.dialog import _build_config_from_dialog_state

        config = _build_config_from_dialog_state(
            api_key="sk-or-key",
            text_model_id="",
            image_model_id="",
            audio_model_id="",
            language="en",
        )
        assert config.text_model == ""
        assert config.image_model == ""
        assert config.audio_model == ""

    def test_strips_api_key_whitespace(self) -> None:
        from ankiforge.config.dialog import _build_config_from_dialog_state

        config = _build_config_from_dialog_state(
            api_key="  sk-or-key  ",
            text_model_id="",
            image_model_id="",
            audio_model_id="",
            language="en",
        )
        assert config.api_key == "sk-or-key"


# ---------------------------------------------------------------------------
# populate_model_combo
# ---------------------------------------------------------------------------


class TestPopulateModelCombo:
    """Тесты заполнения combobox моделями."""

    def test_populates_combo_with_models(self, sample_models: list[Model]) -> None:
        from ankiforge.config.dialog import _populate_model_combo

        combo = MagicMock()
        text_models = [m for m in sample_models if Modality.TEXT in m.modalities]
        _populate_model_combo(combo, text_models, current_id="openai/gpt-4o")

        combo.clear.assert_called_once()
        # Пустой элемент + 2 текстовые модели
        assert combo.addItem.call_count == 3

    def test_selects_current_model(self, sample_models: list[Model]) -> None:
        from ankiforge.config.dialog import _populate_model_combo

        combo = MagicMock()
        text_models = [m for m in sample_models if Modality.TEXT in m.modalities]
        _populate_model_combo(combo, text_models, current_id="anthropic/claude-3.5-sonnet")

        # Должен выбрать индекс 2 (0=пустой, 1=gpt-4o, 2=claude)
        combo.setCurrentIndex.assert_called_once_with(2)

    def test_selects_first_when_no_current(self, sample_models: list[Model]) -> None:
        from ankiforge.config.dialog import _populate_model_combo

        combo = MagicMock()
        text_models = [m for m in sample_models if Modality.TEXT in m.modalities]
        _populate_model_combo(combo, text_models, current_id="")

        # Ничего не должно быть выбрано — остаётся на 0 (пустой)
        combo.setCurrentIndex.assert_not_called()

    def test_populates_empty_list(self) -> None:
        from ankiforge.config.dialog import _populate_model_combo

        combo = MagicMock()
        _populate_model_combo(combo, [], current_id="")

        combo.clear.assert_called_once()
        # Только пустой элемент
        assert combo.addItem.call_count == 1


# ---------------------------------------------------------------------------
# get_selected_model_id
# ---------------------------------------------------------------------------


class TestGetSelectedModelId:
    """Тесты получения ID выбранной модели из combobox."""

    def test_returns_model_id_from_data(self) -> None:
        from ankiforge.config.dialog import _get_selected_model_id

        combo = MagicMock()
        combo.currentData.return_value = "openai/gpt-4o"
        assert _get_selected_model_id(combo) == "openai/gpt-4o"

    def test_returns_empty_when_no_text(self) -> None:
        from ankiforge.config.dialog import _get_selected_model_id

        combo = MagicMock()
        combo.currentData.return_value = None
        combo.currentText.return_value = ""
        assert _get_selected_model_id(combo) == ""

    def test_returns_empty_for_placeholder(self) -> None:
        from ankiforge.config.dialog import _get_selected_model_id

        combo = MagicMock()
        combo.currentData.return_value = None
        combo.currentText.return_value = "— не выбрано —"
        assert _get_selected_model_id(combo) == ""

    def test_extracts_id_from_display_format(self) -> None:
        """Извлекает ID из формата 'Model Name (provider/model-id)'."""
        from ankiforge.config.dialog import _get_selected_model_id

        combo = MagicMock()
        combo.currentData.return_value = None
        combo.currentText.return_value = "GPT-4o (openai/gpt-4o)"
        assert _get_selected_model_id(combo) == "openai/gpt-4o"

    def test_returns_raw_text_as_model_id(self) -> None:
        """Возвращает введённый текст как ID если нет скобок."""
        from ankiforge.config.dialog import _get_selected_model_id

        combo = MagicMock()
        combo.currentData.return_value = None
        combo.currentText.return_value = "google/gemini-2.5-flash-lite-preview-09-2025"
        assert _get_selected_model_id(combo) == "google/gemini-2.5-flash-lite-preview-09-2025"


# ---------------------------------------------------------------------------
# SettingsDialog import
# ---------------------------------------------------------------------------


class TestExtractPricing:
    """Тесты извлечения pricing для кастомных и обычных моделей."""

    def test_returns_pricing_for_known_model(self, sample_models: list[Model]) -> None:
        from ankiforge.config.dialog import _extract_pricing

        result = _extract_pricing("openai/gpt-4o", sample_models)
        assert result is not None
        assert result.prompt == 0.005  # type: ignore[union-attr]

    def test_returns_none_for_unknown_model(self, sample_models: list[Model]) -> None:
        from ankiforge.config.dialog import _extract_pricing

        result = _extract_pricing("google/unknown-model", sample_models)
        assert result is None

    def test_returns_none_for_empty_id(self, sample_models: list[Model]) -> None:
        from ankiforge.config.dialog import _extract_pricing

        result = _extract_pricing("", sample_models)
        assert result is None


class TestSettingsDialogImport:
    """Тест импорта SettingsDialog."""

    def test_import(self) -> None:
        from ankiforge.config.dialog import SettingsDialog

        assert SettingsDialog is not None


# ---------------------------------------------------------------------------
# SettingsDialog._validate_key (логика валидации)
# ---------------------------------------------------------------------------


class TestDialogValidateKey:
    """Тесты вызова валидации API-ключа из диалога."""

    def test_validate_calls_validate_api_key(self) -> None:
        with patch("ankiforge.config.dialog.validate_api_key", return_value=(True, None)) as mock_validate:
            from ankiforge.config.dialog import _validate_api_key_action

            result = _validate_api_key_action("sk-or-test-key")

        mock_validate.assert_called_once_with("sk-or-test-key")
        assert result == (True, None)

    def test_validate_empty_key(self) -> None:
        from ankiforge.config.dialog import _validate_api_key_action

        is_valid, error = _validate_api_key_action("")
        assert is_valid is False
        assert error is not None

    def test_validate_returns_error(self) -> None:
        with patch(
            "ankiforge.config.dialog.validate_api_key",
            return_value=(False, "Невалидный API-ключ"),
        ):
            from ankiforge.config.dialog import _validate_api_key_action

            is_valid, error = _validate_api_key_action("sk-or-invalid")

        assert is_valid is False
        assert error == "Невалидный API-ключ"


# ---------------------------------------------------------------------------
# setup_settings_menu
# ---------------------------------------------------------------------------


class TestSetupSettingsMenu:
    """Тесты регистрации пункта меню."""

    def test_setup_adds_menu_action(self) -> None:
        from ankiforge.config.dialog import setup_settings_menu

        mock_mw = MagicMock()
        mock_menu = MagicMock()
        mock_mw.form.menuTools = mock_menu

        setup_settings_menu(mock_mw)

        mock_menu.addAction.assert_called_once()
        call_args = mock_menu.addAction.call_args
        assert "AnkiForge" in call_args[0][0]


# ---------------------------------------------------------------------------
# make_searchable_combo
# ---------------------------------------------------------------------------


class TestMakeSearchableCombo:
    """Тесты создания searchable combobox (мокается Qt)."""

    def test_returns_editable_combo(self) -> None:
        """_make_searchable_combo создаёт editable combo с completer."""
        mock_combo = MagicMock()
        mock_completer = MagicMock()
        mock_qt = MagicMock()

        with patch.dict("sys.modules", {"aqt.qt": mock_qt, "aqt": MagicMock()}):
            mock_qt.QComboBox.return_value = mock_combo
            mock_qt.QCompleter.return_value = mock_completer

            from importlib import reload

            import ankiforge.config.dialog as dialog_mod

            reload(dialog_mod)
            dialog_mod._make_searchable_combo()

            mock_combo.setEditable.assert_called_once_with(True)
            mock_combo.setInsertPolicy.assert_called_once()
            mock_combo.setCompleter.assert_called_once_with(mock_completer)
            mock_completer.setFilterMode.assert_called_once()
            mock_completer.setCaseSensitivity.assert_called_once()


class TestPopulateSearchableCombo:
    """Тесты заполнения searchable combo — используем MagicMock."""

    def test_populates_and_updates_completer(self, sample_models: list[Model]) -> None:
        """populate_model_combo обновляет completer при editable combo."""
        combo = MagicMock()
        combo.isEditable.return_value = True
        mock_completer = MagicMock()
        combo.completer.return_value = mock_completer
        mock_model = MagicMock()
        combo.model.return_value = mock_model

        from ankiforge.config.dialog import _populate_model_combo

        text_models = [m for m in sample_models if Modality.TEXT in m.modalities]
        _populate_model_combo(combo, text_models, current_id="openai/gpt-4o")

        # completer.setModel вызван с моделью combo
        mock_completer.setModel.assert_called_once_with(mock_model)

    def test_skips_completer_for_non_editable(self, sample_models: list[Model]) -> None:
        """populate_model_combo не трогает completer для обычного combo."""
        combo = MagicMock()
        combo.isEditable.return_value = False

        from ankiforge.config.dialog import _populate_model_combo

        text_models = [m for m in sample_models if Modality.TEXT in m.modalities]
        _populate_model_combo(combo, text_models, current_id="")

        combo.completer.assert_not_called()
