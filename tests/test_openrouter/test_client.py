"""Tests for OpenRouterClient — text, image and audio generation."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest
import requests

from ankiforge.openrouter.client import OpenRouterClient
from ankiforge.openrouter.exceptions import (
    OpenRouterAuthError,
    OpenRouterError,
    OpenRouterRateLimitError,
    OpenRouterTimeoutError,
)

API_KEY = "sk-or-v1-test-key"
MODEL = "openai/gpt-4o"


def _make_response(status_code: int = 200, json_data: dict[str, object] | None = None) -> MagicMock:
    """Create a mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


def _success_response(content: str = "Hello world") -> MagicMock:
    return _make_response(
        200,
        {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        },
    )


class TestOpenRouterClientInit:
    def test_create_with_api_key(self) -> None:
        client = OpenRouterClient(api_key=API_KEY)
        assert client.api_key == API_KEY

    def test_create_with_custom_base_url(self) -> None:
        client = OpenRouterClient(api_key=API_KEY, base_url="https://custom.api.com")
        assert client.base_url == "https://custom.api.com"

    def test_create_with_custom_timeout(self) -> None:
        client = OpenRouterClient(api_key=API_KEY, timeout=60)
        assert client.timeout == 60


class TestGenerateText:
    @patch("ankiforge.openrouter.client.requests.post")
    def test_successful_generation(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _success_response("The answer is 42")
        client = OpenRouterClient(api_key=API_KEY)
        result = client.generate_text("What is the meaning of life?", model=MODEL)
        assert result == "The answer is 42"

    @patch("ankiforge.openrouter.client.requests.post")
    def test_sends_correct_headers(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _success_response()
        client = OpenRouterClient(api_key=API_KEY)
        client.generate_text("test", model=MODEL)
        call_kwargs = mock_post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert headers["Authorization"] == f"Bearer {API_KEY}"
        assert "application/json" in headers["Content-Type"]

    @patch("ankiforge.openrouter.client.requests.post")
    def test_sends_correct_body(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _success_response()
        client = OpenRouterClient(api_key=API_KEY)
        client.generate_text("Hello", model=MODEL)
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json", {})
        assert body["model"] == MODEL
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "Hello"

    @patch("ankiforge.openrouter.client.requests.post")
    def test_temperature_included_when_set(self, mock_post: MagicMock) -> None:
        """temperature is included in body when explicitly set."""
        mock_post.return_value = _success_response()
        client = OpenRouterClient(api_key=API_KEY)
        client.generate_text("Hello", model=MODEL, temperature=0.3)
        body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert body["temperature"] == 0.3

    @patch("ankiforge.openrouter.client.requests.post")
    def test_temperature_not_included_when_none(self, mock_post: MagicMock) -> None:
        """temperature is not included in body when None (default)."""
        mock_post.return_value = _success_response()
        client = OpenRouterClient(api_key=API_KEY)
        client.generate_text("Hello", model=MODEL)
        body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert "temperature" not in body

    @patch("ankiforge.openrouter.client.requests.post")
    def test_temperature_zero_included(self, mock_post: MagicMock) -> None:
        """temperature=0.0 is passed through (not confused with None)."""
        mock_post.return_value = _success_response()
        client = OpenRouterClient(api_key=API_KEY)
        client.generate_text("Hello", model=MODEL, temperature=0.0)
        body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert body["temperature"] == 0.0


class TestErrorHandling:
    @patch("ankiforge.openrouter.client.requests.post")
    def test_auth_error_401(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_response(401, {"error": {"message": "Invalid API key"}})
        client = OpenRouterClient(api_key="invalid-key")
        with pytest.raises(OpenRouterAuthError):
            client.generate_text("test", model=MODEL)

    @patch("ankiforge.openrouter.client.requests.post")
    def test_rate_limit_429(self, mock_post: MagicMock) -> None:
        resp = _make_response(429, {"error": {"message": "Rate limit exceeded"}})
        resp.headers = {"Retry-After": "5"}
        mock_post.return_value = resp
        client = OpenRouterClient(api_key=API_KEY, max_retries=0)
        with pytest.raises(OpenRouterRateLimitError) as exc_info:
            client.generate_text("test", model=MODEL)
        assert exc_info.value.retry_after == 5.0

    @patch("ankiforge.openrouter.client.requests.post")
    def test_timeout_error(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = requests.Timeout("Connection timed out")
        client = OpenRouterClient(api_key=API_KEY, max_retries=0)
        with pytest.raises(OpenRouterTimeoutError):
            client.generate_text("test", model=MODEL)

    @patch("ankiforge.openrouter.client.requests.post")
    def test_connection_error(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = requests.ConnectionError("Failed to connect")
        client = OpenRouterClient(api_key=API_KEY, max_retries=0)
        with pytest.raises(OpenRouterError):
            client.generate_text("test", model=MODEL)

    @patch("ankiforge.openrouter.client.requests.post")
    def test_invalid_json_response(self, mock_post: MagicMock) -> None:
        resp = _make_response(200)
        resp.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = resp
        client = OpenRouterClient(api_key=API_KEY)
        with pytest.raises(OpenRouterError, match="parse JSON"):
            client.generate_text("test", model=MODEL)

    @patch("ankiforge.openrouter.client.requests.post")
    def test_empty_choices(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_response(200, {"choices": []})
        client = OpenRouterClient(api_key=API_KEY)
        with pytest.raises(OpenRouterError, match="empty"):
            client.generate_text("test", model=MODEL)


class TestErrorHandlingExtended:
    """Additional error handling tests."""

    @patch("ankiforge.openrouter.client.requests.post")
    def test_unexpected_status_code(self, mock_post: MagicMock) -> None:
        """Unexpected HTTP status (not 200/401/429/5xx) raises OpenRouterError."""
        mock_post.return_value = _make_response(403, {"error": {"message": "Forbidden"}})
        client = OpenRouterClient(api_key=API_KEY, max_retries=0)
        with pytest.raises(OpenRouterError, match="Status 403: Forbidden"):
            client.generate_text("test", model=MODEL)

    @patch("ankiforge.openrouter.client.requests.post")
    def test_text_null_content(self, mock_post: MagicMock) -> None:
        """API returned choices with content=None raises OpenRouterError."""
        mock_post.return_value = _make_response(200, {"choices": [{"message": {"role": "assistant", "content": None}}]})
        client = OpenRouterClient(api_key=API_KEY)
        with pytest.raises(OpenRouterError, match="empty content"):
            client.generate_text("test", model=MODEL)

    @patch("ankiforge.openrouter.client.requests.post")
    def test_text_missing_content_key(self, mock_post: MagicMock) -> None:
        """API returned message without content raises OpenRouterError."""
        mock_post.return_value = _make_response(200, {"choices": [{"message": {"role": "assistant"}}]})
        client = OpenRouterClient(api_key=API_KEY)
        with pytest.raises(OpenRouterError, match="empty content"):
            client.generate_text("test", model=MODEL)

    @patch("ankiforge.openrouter.client.requests.post")
    def test_no_retry_on_429(self, mock_post: MagicMock) -> None:
        """429 is not retried — raises RateLimitError immediately."""
        mock_post.return_value = _make_response(429, {"error": {"message": "Rate limit"}})
        client = OpenRouterClient(api_key=API_KEY, max_retries=3)
        with pytest.raises(OpenRouterRateLimitError):
            client.generate_text("test", model=MODEL)
        assert mock_post.call_count == 1


class TestRetry:
    @patch("ankiforge.openrouter.client.time.sleep")
    @patch("ankiforge.openrouter.client.requests.post")
    def test_retry_on_500(self, mock_post: MagicMock, mock_sleep: MagicMock) -> None:
        mock_post.side_effect = [
            _make_response(500, {"error": {"message": "Internal server error"}}),
            _success_response("recovered"),
        ]
        client = OpenRouterClient(api_key=API_KEY, max_retries=2)
        result = client.generate_text("test", model=MODEL)
        assert result == "recovered"
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once()

    @patch("ankiforge.openrouter.client.time.sleep")
    @patch("ankiforge.openrouter.client.requests.post")
    def test_retry_on_timeout(self, mock_post: MagicMock, mock_sleep: MagicMock) -> None:
        mock_post.side_effect = [
            requests.Timeout("timeout"),
            _success_response("recovered"),
        ]
        client = OpenRouterClient(api_key=API_KEY, max_retries=2)
        result = client.generate_text("test", model=MODEL)
        assert result == "recovered"

    @patch("ankiforge.openrouter.client.time.sleep")
    @patch("ankiforge.openrouter.client.requests.post")
    def test_retry_exhausted(self, mock_post: MagicMock, mock_sleep: MagicMock) -> None:
        mock_post.return_value = _make_response(502, {"error": {"message": "Bad gateway"}})
        client = OpenRouterClient(api_key=API_KEY, max_retries=2)
        with pytest.raises(OpenRouterError):
            client.generate_text("test", model=MODEL)
        assert mock_post.call_count == 3  # 1 initial + 2 retries

    @patch("ankiforge.openrouter.client.time.sleep")
    @patch("ankiforge.openrouter.client.requests.post")
    def test_no_retry_on_401(self, mock_post: MagicMock, mock_sleep: MagicMock) -> None:
        mock_post.return_value = _make_response(401, {"error": {"message": "Unauthorized"}})
        client = OpenRouterClient(api_key=API_KEY, max_retries=3)
        with pytest.raises(OpenRouterAuthError):
            client.generate_text("test", model=MODEL)
        assert mock_post.call_count == 1  # No retry

    @patch("ankiforge.openrouter.client.time.sleep")
    @patch("ankiforge.openrouter.client.requests.post")
    def test_exponential_backoff(self, mock_post: MagicMock, mock_sleep: MagicMock) -> None:
        mock_post.side_effect = [
            _make_response(503, {"error": {"message": "Service unavailable"}}),
            _make_response(503, {"error": {"message": "Service unavailable"}}),
            _success_response("ok"),
        ]
        client = OpenRouterClient(api_key=API_KEY, max_retries=3, base_delay=1.0)
        client.generate_text("test", model=MODEL)
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays[0] < delays[1]  # exponential growth

    @patch("ankiforge.openrouter.client.time.sleep")
    @patch("ankiforge.openrouter.client.requests.post")
    def test_retry_on_connection_error(self, mock_post: MagicMock, mock_sleep: MagicMock) -> None:
        """ConnectionError is retried and recovers."""
        mock_post.side_effect = [
            requests.ConnectionError("Connection refused"),
            _success_response("recovered"),
        ]
        client = OpenRouterClient(api_key=API_KEY, max_retries=2)
        result = client.generate_text("test", model=MODEL)
        assert result == "recovered"
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once()

    @patch("ankiforge.openrouter.client.time.sleep")
    @patch("ankiforge.openrouter.client.requests.post")
    def test_connection_error_retry_exhausted(self, mock_post: MagicMock, mock_sleep: MagicMock) -> None:
        """ConnectionError after exhausting retries raises OpenRouterError."""
        mock_post.side_effect = requests.ConnectionError("Connection refused")
        client = OpenRouterClient(api_key=API_KEY, max_retries=2)
        with pytest.raises(OpenRouterError, match="Connection error"):
            client.generate_text("test", model=MODEL)
        assert mock_post.call_count == 3


# --- Image generation ---

IMAGE_MODEL = "openai/dall-e-3"
_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _image_success_response(image_b64: str | None = None) -> MagicMock:
    """Mock OpenRouter response with an image."""
    b64 = image_b64 or base64.b64encode(_FAKE_PNG).decode()
    return _make_response(
        200,
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "images": [{"image_url": {"url": f"data:image/png;base64,{b64}"}}],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0},
        },
    )


def _audio_stream_response(audio_b64: str | None = None) -> MagicMock:
    """Mock streaming OpenRouter response with audio chunks."""
    import json

    b64 = audio_b64 or base64.b64encode(b"fake-wav-data").decode()
    # Split base64 into two chunks for realism
    mid = len(b64) // 2
    chunk1 = b64[:mid]
    chunk2 = b64[mid:]

    lines = [
        f"data: {json.dumps({'choices': [{'delta': {'audio': {'data': chunk1}}}]})}",
        f"data: {json.dumps({'choices': [{'delta': {'audio': {'data': chunk2}}}]})}",
        "data: [DONE]",
    ]

    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = iter(lines)
    return resp


class TestGenerateImage:
    @patch("ankiforge.openrouter.client.requests.post")
    def test_successful_image_generation(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _image_success_response()
        client = OpenRouterClient(api_key=API_KEY)
        result = client.generate_image("A cute cat", model=IMAGE_MODEL)
        assert isinstance(result, bytes)
        assert result == _FAKE_PNG

    @patch("ankiforge.openrouter.client.requests.post")
    def test_sends_image_modality(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _image_success_response()
        client = OpenRouterClient(api_key=API_KEY)
        client.generate_image("test", model=IMAGE_MODEL)
        body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert body["model"] == IMAGE_MODEL
        assert body["messages"][0]["content"] == [{"type": "text", "text": "test"}]

    @patch("ankiforge.openrouter.client.requests.post")
    def test_image_config_sent_with_size(self, mock_post: MagicMock) -> None:
        """image_config is sent with image_size when size is specified."""
        mock_post.return_value = _image_success_response()
        client = OpenRouterClient(api_key=API_KEY)
        client.generate_image("test", model=IMAGE_MODEL, size="0.5K")
        body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert body["image_config"] == {"image_size": "0.5K"}

    @patch("ankiforge.openrouter.client.requests.post")
    def test_image_config_not_sent_for_auto(self, mock_post: MagicMock) -> None:
        """image_config is not sent when size='auto' or None."""
        mock_post.return_value = _image_success_response()
        client = OpenRouterClient(api_key=API_KEY)
        client.generate_image("test", model=IMAGE_MODEL, size="auto")
        body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert "image_config" not in body

    @patch("ankiforge.openrouter.client.requests.post")
    def test_image_multimodal_content_format(self, mock_post: MagicMock) -> None:
        """Parses image from multimodal content[] response format."""
        b64 = base64.b64encode(_FAKE_PNG).decode()
        resp = _make_response(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Here is your image"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                                },
                            ],
                        }
                    }
                ],
            },
        )
        mock_post.return_value = resp
        client = OpenRouterClient(api_key=API_KEY)
        result = client.generate_image("test", model=IMAGE_MODEL)
        assert result == _FAKE_PNG

    @patch("ankiforge.openrouter.client.requests.post")
    def test_image_retry_without_image_config(self, mock_post: MagicMock) -> None:
        """On error with image_config — retries without it."""
        mock_post.side_effect = [
            _make_response(400, {"error": {"message": "INVALID_ARGUMENT"}}),
            _image_success_response(),
        ]
        client = OpenRouterClient(api_key=API_KEY, max_retries=0)
        result = client.generate_image("test", model=IMAGE_MODEL, size="0.5K")
        assert isinstance(result, bytes)
        # 2 calls: first with image_config (failed), second without
        assert mock_post.call_count == 2
        # After retry body no longer has image_config (dict is mutable)
        last_body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert "image_config" not in last_body

    @patch("ankiforge.openrouter.client.requests.post")
    def test_image_empty_images_list(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_response(
            200, {"choices": [{"message": {"role": "assistant", "content": "", "images": []}}]}
        )
        client = OpenRouterClient(api_key=API_KEY)
        with pytest.raises(OpenRouterError, match="image"):
            client.generate_image("test", model=IMAGE_MODEL)

    @patch("ankiforge.openrouter.client.requests.post")
    def test_image_no_images_field(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_response(
            200, {"choices": [{"message": {"role": "assistant", "content": "no images"}}]}
        )
        client = OpenRouterClient(api_key=API_KEY)
        with pytest.raises(OpenRouterError, match="image"):
            client.generate_image("test", model=IMAGE_MODEL)

    @patch("ankiforge.openrouter.client.requests.post")
    def test_image_invalid_base64(self, mock_post: MagicMock) -> None:
        resp = _make_response(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "images": [{"image_url": {"url": "data:image/png;base64,!!!invalid!!!"}}],
                        }
                    }
                ]
            },
        )
        mock_post.return_value = resp
        client = OpenRouterClient(api_key=API_KEY)
        with pytest.raises(OpenRouterError, match="base64"):
            client.generate_image("test", model=IMAGE_MODEL)

    @patch("ankiforge.openrouter.client.requests.post")
    def test_image_invalid_json(self, mock_post: MagicMock) -> None:
        """Invalid JSON in image response raises OpenRouterError."""
        resp = _make_response(200)
        resp.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = resp
        client = OpenRouterClient(api_key=API_KEY)
        with pytest.raises(OpenRouterError, match="parse JSON"):
            client.generate_image("test", model=IMAGE_MODEL)

    @patch("ankiforge.openrouter.client.requests.post")
    def test_image_empty_choices(self, mock_post: MagicMock) -> None:
        """Empty choices in image response raises OpenRouterError."""
        mock_post.return_value = _make_response(200, {"choices": []})
        client = OpenRouterClient(api_key=API_KEY)
        with pytest.raises(OpenRouterError, match="empty"):
            client.generate_image("test", model=IMAGE_MODEL)

    @patch("ankiforge.openrouter.client.requests.post")
    def test_image_no_base64_prefix(self, mock_post: MagicMock) -> None:
        """URL without base64 prefix raises OpenRouterError."""
        resp = _make_response(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "images": [{"image_url": {"url": "https://example.com/image.png"}}],
                        }
                    }
                ]
            },
        )
        mock_post.return_value = resp
        client = OpenRouterClient(api_key=API_KEY)
        with pytest.raises(OpenRouterError, match="base64"):
            client.generate_image("test", model=IMAGE_MODEL)

    @patch("ankiforge.openrouter.client.requests.post")
    def test_image_auth_error(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_response(401, {"error": {"message": "Invalid key"}})
        client = OpenRouterClient(api_key="bad-key")
        with pytest.raises(OpenRouterAuthError):
            client.generate_image("test", model=IMAGE_MODEL)

    @patch("ankiforge.openrouter.client.time.sleep")
    @patch("ankiforge.openrouter.client.requests.post")
    def test_image_retry_on_500(self, mock_post: MagicMock, mock_sleep: MagicMock) -> None:
        mock_post.side_effect = [
            _make_response(500, {"error": {"message": "Server error"}}),
            _image_success_response(),
        ]
        client = OpenRouterClient(api_key=API_KEY, max_retries=2)
        result = client.generate_image("test", model=IMAGE_MODEL)
        assert isinstance(result, bytes)
        assert mock_post.call_count == 2


class TestGenerateAudio:
    @patch("ankiforge.openrouter.client.requests.post")
    def test_successful_audio_generation(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _audio_stream_response()
        client = OpenRouterClient(api_key=API_KEY)
        result = client.generate_audio("Hello world", model="openai/gpt-audio-mini")
        assert isinstance(result, bytes)
        # Result is WAV (PCM16 + WAV header)
        assert result[:4] == b"RIFF"
        assert b"WAVE" in result[:12]
        # Contains the original PCM data
        assert b"fake-wav-data" in result

    @patch("ankiforge.openrouter.client.requests.post")
    def test_sends_audio_params(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _audio_stream_response()
        client = OpenRouterClient(api_key=API_KEY)
        client.generate_audio("test", model="openai/gpt-audio-mini")
        body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert body["model"] == "openai/gpt-audio-mini"
        assert body["modalities"] == ["text", "audio"]
        assert body["audio"]["voice"] == "alloy"
        assert body["stream"] is True

    @patch("ankiforge.openrouter.client.requests.post")
    def test_audio_no_chunks(self, mock_post: MagicMock) -> None:
        """Stream without audio chunks raises OpenRouterError."""
        resp = MagicMock()
        resp.status_code = 200
        resp.iter_lines.return_value = iter(["data: [DONE]"])
        mock_post.return_value = resp
        client = OpenRouterClient(api_key=API_KEY)
        with pytest.raises(OpenRouterError, match="audio"):
            client.generate_audio("test", model="openai/gpt-audio-mini")

    @patch("ankiforge.openrouter.client.requests.post")
    def test_audio_invalid_base64(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _audio_stream_response("!!!bad-base64!!!")
        client = OpenRouterClient(api_key=API_KEY)
        with pytest.raises(OpenRouterError, match="base64"):
            client.generate_audio("test", model="openai/gpt-audio-mini")

    @patch("ankiforge.openrouter.client.requests.post")
    def test_audio_auth_error(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_response(401, {"error": {"message": "Invalid key"}})
        client = OpenRouterClient(api_key="bad-key")
        with pytest.raises(OpenRouterAuthError):
            client.generate_audio("test", model="openai/gpt-audio-mini")
