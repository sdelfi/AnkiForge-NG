"""Tests for RoutingClient — dispatches to OpenRouter or a custom endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ankiforge.models import AddonConfig
from ankiforge.openrouter.routing_client import (
    LOCAL_IMAGE_MODEL_ID,
    RoutingClient,
    add_custom_prefix,
    build_client,
    is_custom_model,
    is_local_image_model,
    strip_custom_prefix,
)


class TestPrefixHelpers:
    def test_is_custom_model(self) -> None:
        assert is_custom_model("custom::llama-3.1-8b") is True
        assert is_custom_model("openai/gpt-4o") is False

    def test_strip_custom_prefix(self) -> None:
        assert strip_custom_prefix("custom::llama-3.1-8b") == "llama-3.1-8b"
        assert strip_custom_prefix("openai/gpt-4o") == "openai/gpt-4o"

    def test_add_custom_prefix(self) -> None:
        assert add_custom_prefix("llama-3.1-8b") == "custom::llama-3.1-8b"
        # Idempotent — doesn't double-prefix.
        assert add_custom_prefix("custom::llama-3.1-8b") == "custom::llama-3.1-8b"

    def test_is_local_image_model(self) -> None:
        assert is_local_image_model(LOCAL_IMAGE_MODEL_ID) is True
        assert is_local_image_model("openai/dall-e-3") is False


class TestRoutingClient:
    def test_routes_plain_model_to_openrouter(self) -> None:
        or_client = MagicMock()
        or_client.generate_text.return_value = "hello"
        client = RoutingClient(or_client, None)

        result = client.generate_text("prompt", "openai/gpt-4o")

        assert result == "hello"
        or_client.generate_text.assert_called_once_with("prompt", "openai/gpt-4o", temperature=None, system_prompt=None)

    def test_routes_custom_model_to_custom_client_and_strips_prefix(self) -> None:
        custom_client = MagicMock()
        custom_client.generate_text.return_value = "local response"
        client = RoutingClient(None, custom_client)

        result = client.generate_text("prompt", "custom::llama-3.1-8b")

        assert result == "local response"
        custom_client.generate_text.assert_called_once_with(
            "prompt", "llama-3.1-8b", temperature=None, system_prompt=None
        )

    def test_raises_when_openrouter_not_configured(self) -> None:
        client = RoutingClient(None, MagicMock())
        with pytest.raises(ValueError, match="No OpenRouter API key"):
            client.generate_text("prompt", "openai/gpt-4o")

    def test_raises_when_custom_endpoint_not_configured(self) -> None:
        client = RoutingClient(MagicMock(), None)
        with pytest.raises(ValueError, match="No local/custom API endpoint"):
            client.generate_text("prompt", "custom::llama-3.1-8b")

    def test_generate_audio_routes_and_strips_prefix(self) -> None:
        custom_client = MagicMock()
        custom_client.generate_audio.return_value = b"audio-bytes"
        client = RoutingClient(None, custom_client)

        result = client.generate_audio("hello", "custom::tts-model", voice="nova")

        assert result == b"audio-bytes"
        custom_client.generate_audio.assert_called_once_with("hello", "tts-model", voice="nova")

    def test_generate_image_routes_and_strips_prefix(self) -> None:
        or_client = MagicMock()
        or_client.generate_image.return_value = b"image-bytes"
        client = RoutingClient(or_client, None)

        result = client.generate_image("a cat", "google/gemini-flash-image", size="1K")

        assert result == b"image-bytes"
        or_client.generate_image.assert_called_once_with("a cat", "google/gemini-flash-image", size="1K")

    def test_last_cost_and_usage_reflect_last_used_client(self) -> None:
        or_client = MagicMock()
        or_client.generate_text.return_value = "hi"
        or_client.last_cost = 0.002
        or_client.last_usage = (100, 50)

        custom_client = MagicMock()
        custom_client.generate_text.return_value = "hi local"
        custom_client.last_cost = 0.0
        custom_client.last_usage = (10, 5)

        client = RoutingClient(or_client, custom_client)

        client.generate_text("p", "openai/gpt-4o")
        assert client.last_cost == 0.002
        assert client.last_usage == (100, 50)

        client.generate_text("p", "custom::llama-3.1-8b")
        assert client.last_cost == 0.0
        assert client.last_usage == (10, 5)

    def test_last_cost_and_usage_default_before_any_call(self) -> None:
        client = RoutingClient(None, None)
        assert client.last_cost == 0.0
        assert client.last_usage == (0, 0)

    def test_generate_image_routes_to_local_image_client(self) -> None:
        local_client = MagicMock()
        local_client.generate_image.return_value = b"local-bytes"
        client = RoutingClient(None, None, local_image_client=local_client)

        result = client.generate_image("a cat", LOCAL_IMAGE_MODEL_ID, size="square")

        assert result == b"local-bytes"
        local_client.generate_image.assert_called_once_with("a cat", size="square")

    def test_generate_image_local_call_resets_cost_and_usage(self) -> None:
        or_client = MagicMock()
        or_client.generate_text.return_value = "hi"
        or_client.last_cost = 0.05
        or_client.last_usage = (100, 50)

        local_client = MagicMock()
        local_client.generate_image.return_value = b"local-bytes"

        client = RoutingClient(or_client, None, local_image_client=local_client)
        client.generate_text("p", "openai/gpt-4o")
        assert client.last_cost == 0.05

        client.generate_image("a cat", LOCAL_IMAGE_MODEL_ID)
        assert client.last_cost == 0.0
        assert client.last_usage == (0, 0)

    def test_raises_when_local_image_backend_not_configured(self) -> None:
        client = RoutingClient(None, None)
        with pytest.raises(ValueError, match="No local image backend"):
            client.generate_image("a cat", LOCAL_IMAGE_MODEL_ID)


class TestBuildClient:
    def test_builds_openrouter_only(self) -> None:
        config = AddonConfig(api_key="sk-or-v1-test")
        client = build_client(config)
        assert client._or_client is not None  # noqa: SLF001
        assert client._custom_client is None  # noqa: SLF001

    def test_builds_custom_only(self) -> None:
        config = AddonConfig(custom_base_url="http://localhost:1234/v1")
        client = build_client(config)
        assert client._or_client is None  # noqa: SLF001
        assert client._custom_client is not None  # noqa: SLF001
        assert client._custom_client.base_url == "http://localhost:1234/v1"  # noqa: SLF001

    def test_builds_both(self) -> None:
        config = AddonConfig(api_key="sk-or-v1-test", custom_base_url="http://localhost:1234/v1/")
        client = build_client(config)
        assert client._or_client is not None  # noqa: SLF001
        assert client._custom_client is not None  # noqa: SLF001
        # Trailing slash is stripped for consistent URL joining.
        assert client._custom_client.base_url == "http://localhost:1234/v1"  # noqa: SLF001

    def test_builds_neither(self) -> None:
        config = AddonConfig()
        client = build_client(config)
        assert client._or_client is None  # noqa: SLF001
        assert client._custom_client is None  # noqa: SLF001
        assert client._local_image_client is None  # noqa: SLF001

    def test_builds_local_image_client(self) -> None:
        config = AddonConfig(local_image_backend="automatic1111", local_image_url="http://127.0.0.1:7860")
        client = build_client(config)
        assert client._local_image_client is not None  # noqa: SLF001
