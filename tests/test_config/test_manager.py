"""Тесты для ankiforge.config.manager."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from ankiforge.config.manager import get_config, is_configured, save_config, validate_api_key
from ankiforge.models import AddonConfig

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_mw() -> MagicMock:
    """Мок главного окна Anki (mw) с addonManager."""
    mw = MagicMock()
    mw.addonManager.getConfig.return_value = {
        "api_key": "sk-or-test-key-123",
        "text_model": "openai/gpt-4o",
        "image_model": "openai/dall-e-3",
        "audio_model": "openai/tts-1",
        "language": "en",
    }
    mw.addonManager.addonFromModule.return_value = "ankiforge"
    return mw


@pytest.fixture()
def default_config_dict() -> dict[str, str]:
    """Дефолтный конфиг (пустые поля)."""
    return {
        "api_key": "",
        "text_model": "",
        "image_model": "",
        "audio_model": "",
        "language": "en",
    }


# ---------------------------------------------------------------------------
# get_config
# ---------------------------------------------------------------------------


class TestGetConfig:
    """Тесты get_config()."""

    def test_reads_from_anki_addon_manager(self, mock_mw: MagicMock) -> None:
        with patch("ankiforge.config.manager._get_mw", return_value=mock_mw):
            config = get_config()

        assert config.api_key == "sk-or-test-key-123"
        assert config.text_model == "openai/gpt-4o"
        assert config.image_model == "openai/dall-e-3"
        assert config.audio_model == "openai/tts-1"
        assert config.language == "en"

    def test_returns_defaults_when_anki_unavailable(self, tmp_path: Path) -> None:
        """Fallback — читает config.json из пакета."""
        with patch("ankiforge.config.manager._get_mw", side_effect=RuntimeError):
            config = get_config()

        assert isinstance(config, AddonConfig)
        assert config.api_key == ""
        assert config.language == "en"

    def test_handles_partial_config(self, mock_mw: MagicMock) -> None:
        """Если в конфиге нет каких-то полей — используются дефолты."""
        mock_mw.addonManager.getConfig.return_value = {
            "api_key": "sk-or-key",
        }

        with patch("ankiforge.config.manager._get_mw", return_value=mock_mw):
            config = get_config()

        assert config.api_key == "sk-or-key"
        assert config.text_model == ""
        assert config.language == "en"

    def test_handles_none_config(self, mock_mw: MagicMock) -> None:
        """Если getConfig вернул None."""
        mock_mw.addonManager.getConfig.return_value = None

        with patch("ankiforge.config.manager._get_mw", return_value=mock_mw):
            config = get_config()

        assert config.api_key == ""


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------


class TestSaveConfig:
    """Тесты save_config()."""

    def test_writes_to_anki_addon_manager(self, mock_mw: MagicMock) -> None:
        config = AddonConfig(
            api_key="sk-or-new-key",
            text_model="meta/llama-3",
            image_model="",
            audio_model="",
            language="ru",
        )

        with patch("ankiforge.config.manager._get_mw", return_value=mock_mw):
            save_config(config)

        mock_mw.addonManager.writeConfig.assert_called_once()
        call_args = mock_mw.addonManager.writeConfig.call_args
        saved_data = call_args[0][1]  # второй позиционный аргумент
        assert saved_data["api_key"] == "sk-or-new-key"
        assert saved_data["language"] == "ru"

    def test_saves_fallback_to_json_file(self, tmp_path: Path) -> None:
        """Fallback — пишет в JSON-файл если Anki недоступен."""
        config = AddonConfig(api_key="sk-or-fallback", language="de")
        fallback_path = tmp_path / "config.json"

        with (
            patch("ankiforge.config.manager._get_mw", side_effect=RuntimeError),
            patch("ankiforge.config.manager._fallback_config_path", return_value=fallback_path),
        ):
            save_config(config)

        data = json.loads(fallback_path.read_text())
        assert data["api_key"] == "sk-or-fallback"
        assert data["language"] == "de"


# ---------------------------------------------------------------------------
# is_configured
# ---------------------------------------------------------------------------


class TestIsConfigured:
    """Тесты is_configured()."""

    def test_true_when_api_key_present(self, mock_mw: MagicMock) -> None:
        with patch("ankiforge.config.manager._get_mw", return_value=mock_mw):
            assert is_configured() is True

    def test_false_when_api_key_empty(self, mock_mw: MagicMock) -> None:
        mock_mw.addonManager.getConfig.return_value = {"api_key": ""}

        with patch("ankiforge.config.manager._get_mw", return_value=mock_mw):
            assert is_configured() is False

    def test_false_when_api_key_missing(self, mock_mw: MagicMock) -> None:
        mock_mw.addonManager.getConfig.return_value = {}

        with patch("ankiforge.config.manager._get_mw", return_value=mock_mw):
            assert is_configured() is False

    def test_false_when_anki_unavailable(self) -> None:
        with patch("ankiforge.config.manager._get_mw", side_effect=RuntimeError):
            assert is_configured() is False


# ---------------------------------------------------------------------------
# validate_api_key
# ---------------------------------------------------------------------------


class TestValidateApiKey:
    """Тесты validate_api_key()."""

    def test_empty_key_returns_false(self) -> None:
        is_valid, error = validate_api_key("")
        assert is_valid is False
        assert error is not None
        assert "пуст" in error.lower()

    def test_invalid_prefix_returns_false(self) -> None:
        is_valid, error = validate_api_key("invalid-key-format")
        assert is_valid is False
        assert error is not None

    def test_valid_key_with_successful_api_call(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("ankiforge.config.manager.requests.get", return_value=mock_response):
            is_valid, error = validate_api_key("sk-or-v1-validkey123")

        assert is_valid is True
        assert error is None

    def test_valid_key_with_401_response(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("ankiforge.config.manager.requests.get", return_value=mock_response):
            is_valid, error = validate_api_key("sk-or-v1-invalidkey")

        assert is_valid is False
        assert error is not None

    def test_valid_key_with_network_error(self) -> None:
        with patch("ankiforge.config.manager.requests.get", side_effect=Exception("connection error")):
            is_valid, error = validate_api_key("sk-or-v1-somekey")

        assert is_valid is False
        assert error is not None
