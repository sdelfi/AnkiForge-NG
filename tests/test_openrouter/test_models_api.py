"""Tests for fetch_models and estimate_cost in OpenRouterClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from ankiforge.openrouter.client import OpenRouterClient
from ankiforge.openrouter.exceptions import OpenRouterAuthError, OpenRouterError
from ankiforge.openrouter.models import Modality, Model, ModelPricing

API_KEY = "sk-or-v1-test-key"


def _make_get_response(
    status_code: int = 200,
    json_data: dict[str, object] | None = None,
) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = {}
    return resp


MODELS_API_RESPONSE: dict[str, object] = {
    "data": [
        {
            "id": "openai/gpt-4o",
            "name": "GPT-4o",
            "pricing": {
                "prompt": "0.000005",
                "completion": "0.000015",
                "image": "0",
                "request": "0",
            },
            "architecture": {
                "modality": "text->text",
            },
            "context_length": 128000,
        },
        {
            "id": "openai/dall-e-3",
            "name": "DALL-E 3",
            "pricing": {
                "prompt": "0",
                "completion": "0",
                "image": "0.04",
                "request": "0",
            },
            "architecture": {
                "modality": "text->image",
            },
            "context_length": 4096,
        },
        {
            "id": "openai/tts-1",
            "name": "TTS-1",
            "pricing": {
                "prompt": "0.000015",
                "completion": "0",
                "image": "0",
                "request": "0",
            },
            "architecture": {
                "modality": "text->audio",
            },
            "context_length": 4096,
        },
        {
            "id": "anthropic/claude-3.5-sonnet",
            "name": "Claude 3.5 Sonnet",
            "pricing": {
                "prompt": "0.000003",
                "completion": "0.000015",
                "image": "0",
                "request": "0",
            },
            "architecture": {
                "modality": "text+image->text",
            },
            "context_length": 200000,
        },
    ]
}


class TestFetchModels:
    @patch("ankiforge.openrouter.client.requests.get")
    def test_fetch_models_returns_list(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _make_get_response(200, MODELS_API_RESPONSE)
        client = OpenRouterClient(api_key=API_KEY)
        models = client.fetch_models()
        assert isinstance(models, list)
        assert len(models) == 4

    @patch("ankiforge.openrouter.client.requests.get")
    def test_fetch_models_parses_fields(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _make_get_response(200, MODELS_API_RESPONSE)
        client = OpenRouterClient(api_key=API_KEY)
        models = client.fetch_models()
        gpt4o = next(m for m in models if m.id == "openai/gpt-4o")
        assert gpt4o.name == "GPT-4o"
        assert gpt4o.pricing.prompt == 0.000005
        assert gpt4o.pricing.completion == 0.000015
        assert gpt4o.context_length == 128000

    @patch("ankiforge.openrouter.client.requests.get")
    def test_fetch_models_parses_modalities(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _make_get_response(200, MODELS_API_RESPONSE)
        client = OpenRouterClient(api_key=API_KEY)
        models = client.fetch_models()
        gpt4o = next(m for m in models if m.id == "openai/gpt-4o")
        assert Modality.TEXT in gpt4o.modalities
        dalle = next(m for m in models if m.id == "openai/dall-e-3")
        assert Modality.IMAGE in dalle.modalities
        tts = next(m for m in models if m.id == "openai/tts-1")
        assert Modality.AUDIO in tts.modalities

    @patch("ankiforge.openrouter.client.requests.get")
    def test_fetch_models_multimodal(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _make_get_response(200, MODELS_API_RESPONSE)
        client = OpenRouterClient(api_key=API_KEY)
        models = client.fetch_models()
        claude = next(m for m in models if m.id == "anthropic/claude-3.5-sonnet")
        assert Modality.TEXT in claude.modalities

    @patch("ankiforge.openrouter.client.requests.get")
    def test_fetch_models_sends_auth_header(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _make_get_response(200, MODELS_API_RESPONSE)
        client = OpenRouterClient(api_key=API_KEY)
        client.fetch_models()
        headers = mock_get.call_args.kwargs.get("headers", {})
        assert headers["Authorization"] == f"Bearer {API_KEY}"

    @patch("ankiforge.openrouter.client.requests.get")
    def test_fetch_models_auth_error(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _make_get_response(401, {"error": {"message": "Invalid key"}})
        client = OpenRouterClient(api_key="bad-key")
        with pytest.raises(OpenRouterAuthError):
            client.fetch_models()

    @patch("ankiforge.openrouter.client.requests.get")
    def test_fetch_models_empty_data(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _make_get_response(200, {"data": []})
        client = OpenRouterClient(api_key=API_KEY)
        models = client.fetch_models()
        assert models == []

    @patch("ankiforge.openrouter.client.requests.get")
    def test_fetch_models_invalid_json(self, mock_get: MagicMock) -> None:
        resp = _make_get_response(200)
        resp.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = resp
        client = OpenRouterClient(api_key=API_KEY)
        with pytest.raises(OpenRouterError, match="Failed to parse JSON"):
            client.fetch_models()


class TestFilterModels:
    @patch("ankiforge.openrouter.client.requests.get")
    def test_filter_text_models(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _make_get_response(200, MODELS_API_RESPONSE)
        client = OpenRouterClient(api_key=API_KEY)
        models = client.fetch_models()
        text_models = [m for m in models if Modality.TEXT in m.modalities]
        ids = [m.id for m in text_models]
        assert "openai/gpt-4o" in ids
        assert "anthropic/claude-3.5-sonnet" in ids
        assert "openai/dall-e-3" not in ids

    @patch("ankiforge.openrouter.client.requests.get")
    def test_filter_image_models(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _make_get_response(200, MODELS_API_RESPONSE)
        client = OpenRouterClient(api_key=API_KEY)
        models = client.fetch_models()
        image_models = [m for m in models if Modality.IMAGE in m.modalities]
        assert len(image_models) == 1
        assert image_models[0].id == "openai/dall-e-3"

    @patch("ankiforge.openrouter.client.requests.get")
    def test_filter_audio_models(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _make_get_response(200, MODELS_API_RESPONSE)
        client = OpenRouterClient(api_key=API_KEY)
        models = client.fetch_models()
        audio_models = [m for m in models if Modality.AUDIO in m.modalities]
        assert len(audio_models) == 1
        assert audio_models[0].id == "openai/tts-1"


class TestCaching:
    @patch("ankiforge.openrouter.client.requests.get")
    def test_models_cached_on_second_call(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _make_get_response(200, MODELS_API_RESPONSE)
        client = OpenRouterClient(api_key=API_KEY)
        client.fetch_models()
        client.fetch_models()
        assert mock_get.call_count == 1

    @patch("ankiforge.openrouter.client.time.time")
    @patch("ankiforge.openrouter.client.requests.get")
    def test_cache_expires_after_ttl(self, mock_get: MagicMock, mock_time: MagicMock) -> None:
        mock_get.return_value = _make_get_response(200, MODELS_API_RESPONSE)
        mock_time.side_effect = [0.0, 0.0, 400.0, 400.0]
        client = OpenRouterClient(api_key=API_KEY, models_cache_ttl=300)
        client.fetch_models()
        client.fetch_models()  # TTL expired — makes a new request
        assert mock_get.call_count == 2

    @patch("ankiforge.openrouter.client.requests.get")
    def test_cache_not_expired_within_ttl(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _make_get_response(200, MODELS_API_RESPONSE)
        client = OpenRouterClient(api_key=API_KEY, models_cache_ttl=300)
        client.fetch_models()
        client.fetch_models()
        assert mock_get.call_count == 1


class TestParseModalities:
    """Tests for modality string parsing."""

    @patch("ankiforge.openrouter.client.requests.get")
    def test_empty_modality_string(self, mock_get: MagicMock) -> None:
        """Empty modality string results in an empty list."""
        data: dict[str, object] = {
            "data": [
                {
                    "id": "test/model",
                    "name": "Test",
                    "pricing": {"prompt": "0", "completion": "0", "image": "0", "request": "0"},
                    "architecture": {"modality": ""},
                    "context_length": 4096,
                }
            ]
        }
        mock_get.return_value = _make_get_response(200, data)
        client = OpenRouterClient(api_key=API_KEY)
        models = client.fetch_models()
        assert models[0].modalities == []

    @patch("ankiforge.openrouter.client.requests.get")
    def test_compound_output_modality(self, mock_get: MagicMock) -> None:
        """Compound output parses multiple modalities (text+image->text+audio)."""
        data: dict[str, object] = {
            "data": [
                {
                    "id": "test/multi",
                    "name": "Multi",
                    "pricing": {"prompt": "0", "completion": "0", "image": "0", "request": "0"},
                    "architecture": {"modality": "text->text+audio"},
                    "context_length": 4096,
                }
            ]
        }
        mock_get.return_value = _make_get_response(200, data)
        client = OpenRouterClient(api_key=API_KEY)
        models = client.fetch_models()
        assert Modality.TEXT in models[0].modalities
        assert Modality.AUDIO in models[0].modalities


class TestEstimateCost:
    def _make_text_model(self) -> Model:
        return Model(
            id="openai/gpt-4o",
            name="GPT-4o",
            pricing=ModelPricing(prompt=0.000005, completion=0.000015),
            modalities=[Modality.TEXT],
            context_length=128000,
        )

    def _make_image_model(self) -> Model:
        return Model(
            id="openai/dall-e-3",
            name="DALL-E 3",
            pricing=ModelPricing(image=0.04),
            modalities=[Modality.IMAGE],
        )

    def _make_audio_model(self) -> Model:
        return Model(
            id="openai/tts-1",
            name="TTS-1",
            pricing=ModelPricing(prompt=0.000015),
            modalities=[Modality.AUDIO],
        )

    def test_estimate_cost_questions_mode(self) -> None:
        client = OpenRouterClient(api_key=API_KEY)
        text_model = self._make_text_model()
        cost = client.estimate_cost(
            mode="questions",
            card_count=10,
            text_model=text_model,
        )
        assert cost > 0.0

    def test_estimate_cost_language_mode(self) -> None:
        client = OpenRouterClient(api_key=API_KEY)
        cost = client.estimate_cost(
            mode="language",
            card_count=10,
            text_model=self._make_text_model(),
            image_model=self._make_image_model(),
            audio_model=self._make_audio_model(),
        )
        assert cost > 0.0

    def test_estimate_cost_material_mode(self) -> None:
        client = OpenRouterClient(api_key=API_KEY)
        cost = client.estimate_cost(
            mode="material",
            card_count=10,
            text_model=self._make_text_model(),
        )
        assert cost > 0.0

    def test_estimate_cost_image_mode(self) -> None:
        client = OpenRouterClient(api_key=API_KEY)
        cost = client.estimate_cost(
            mode="image",
            card_count=5,
            text_model=self._make_text_model(),
            image_model=self._make_image_model(),
        )
        assert cost > 0.0

    def test_estimate_cost_audio_mode(self) -> None:
        client = OpenRouterClient(api_key=API_KEY)
        cost = client.estimate_cost(
            mode="audio",
            card_count=5,
            text_model=self._make_text_model(),
            audio_model=self._make_audio_model(),
        )
        assert cost > 0.0

    def test_estimate_cost_zero_cards(self) -> None:
        client = OpenRouterClient(api_key=API_KEY)
        cost = client.estimate_cost(
            mode="questions",
            card_count=0,
            text_model=self._make_text_model(),
        )
        assert cost == 0.0

    def test_estimate_cost_scales_with_count(self) -> None:
        client = OpenRouterClient(api_key=API_KEY)
        text_model = self._make_text_model()
        cost_5 = client.estimate_cost(mode="questions", card_count=5, text_model=text_model)
        cost_10 = client.estimate_cost(mode="questions", card_count=10, text_model=text_model)
        assert cost_10 > cost_5

    def test_estimate_cost_negative_cards(self) -> None:
        """Negative card count results in cost 0."""
        client = OpenRouterClient(api_key=API_KEY)
        cost = client.estimate_cost(mode="questions", card_count=-5, text_model=self._make_text_model())
        assert cost == 0.0

    def test_estimate_cost_no_models_provided(self) -> None:
        """No models provided results in cost 0."""
        client = OpenRouterClient(api_key=API_KEY)
        cost = client.estimate_cost(mode="questions", card_count=10)
        assert cost == 0.0

    def test_estimate_cost_language_text_only(self) -> None:
        """Language mode with only text_model counts text cost only."""
        client = OpenRouterClient(api_key=API_KEY)
        cost = client.estimate_cost(mode="language", card_count=10, text_model=self._make_text_model())
        assert cost > 0.0

    def test_estimate_cost_exact_calculation(self) -> None:
        """Verify exact calculation for questions mode."""
        client = OpenRouterClient(api_key=API_KEY)
        text_model = self._make_text_model()
        cost = client.estimate_cost(mode="questions", card_count=1, text_model=text_model)
        # prompt=0.000005 * 200 + completion=0.000015 * 150 = 0.001 + 0.00225 = 0.00325
        assert abs(cost - 0.00325) < 1e-10
