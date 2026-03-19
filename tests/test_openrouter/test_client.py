"""Тесты для OpenRouterClient — текстовая генерация."""

from __future__ import annotations

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
    """Создаёт мок requests.Response."""
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
        with pytest.raises(OpenRouterError, match="парсинг"):
            client.generate_text("test", model=MODEL)

    @patch("ankiforge.openrouter.client.requests.post")
    def test_empty_choices(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_response(200, {"choices": []})
        client = OpenRouterClient(api_key=API_KEY)
        with pytest.raises(OpenRouterError, match="пустой"):
            client.generate_text("test", model=MODEL)


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
        assert mock_post.call_count == 1  # Нет retry

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
        assert delays[0] < delays[1]  # экспоненциальный рост
