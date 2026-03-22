"""Tests for generators._retry — retry_api_call with exponential backoff."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ankiforge.generators._retry import retry_api_call
from ankiforge.openrouter.exceptions import (
    OpenRouterAuthError,
    OpenRouterError,
    OpenRouterInsufficientCreditsError,
    OpenRouterRateLimitError,
    OpenRouterTimeoutError,
)


class TestRetryApiCall:
    def test_success_first_try(self) -> None:
        fn = MagicMock(return_value="ok")
        result = retry_api_call(fn, item_label="test")
        assert result == "ok"
        assert fn.call_count == 1

    @patch("ankiforge.generators._retry.time.sleep")
    def test_retries_on_timeout(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=[OpenRouterTimeoutError("timeout"), "ok"])
        result = retry_api_call(fn, item_label="test")
        assert result == "ok"
        assert fn.call_count == 2
        mock_sleep.assert_called_once()

    @patch("ankiforge.generators._retry.time.sleep")
    def test_retries_on_server_error(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=[OpenRouterError("500", status_code=500), "ok"])
        result = retry_api_call(fn, item_label="test")
        assert result == "ok"
        assert fn.call_count == 2

    @patch("ankiforge.generators._retry.time.sleep")
    def test_retries_on_rate_limit(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=[OpenRouterRateLimitError("429", retry_after=1.0), "ok"])
        result = retry_api_call(fn, item_label="test")
        assert result == "ok"
        assert fn.call_count == 2
        # Should use retry_after from exception
        mock_sleep.assert_called_once_with(1.0)

    @patch("ankiforge.generators._retry.time.sleep")
    def test_rate_limit_uses_backoff_when_no_retry_after(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=[OpenRouterRateLimitError("429", retry_after=None), "ok"])
        result = retry_api_call(fn, item_label="test")
        assert result == "ok"
        # Should use backoff delay, not retry_after
        assert mock_sleep.call_args[0][0] == pytest.approx(2.0)

    def test_raises_immediately_on_auth_error(self) -> None:
        fn = MagicMock(side_effect=OpenRouterAuthError("401", status_code=401))
        with pytest.raises(OpenRouterAuthError):
            retry_api_call(fn, item_label="test")
        assert fn.call_count == 1

    def test_raises_immediately_on_insufficient_credits(self) -> None:
        fn = MagicMock(side_effect=OpenRouterInsufficientCreditsError("402", status_code=402))
        with pytest.raises(OpenRouterInsufficientCreditsError):
            retry_api_call(fn, item_label="test")
        assert fn.call_count == 1

    @patch("ankiforge.generators._retry.time.sleep")
    def test_multiple_retries_then_success(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(
            side_effect=[
                OpenRouterTimeoutError("t1"),
                OpenRouterError("500"),
                OpenRouterRateLimitError("429"),
                "ok",
            ]
        )
        result = retry_api_call(fn, item_label="test")
        assert result == "ok"
        assert fn.call_count == 4

    @patch("ankiforge.generators._retry.time.sleep")
    def test_exhausts_retries_and_raises(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=OpenRouterTimeoutError("timeout"))
        with pytest.raises(OpenRouterTimeoutError):
            retry_api_call(fn, item_label="test")
        # 10 retries + 1 last attempt = 11
        assert fn.call_count == 11

    @patch("ankiforge.generators._retry.time.sleep")
    def test_backoff_increases_exponentially(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(
            side_effect=[
                OpenRouterError("err"),
                OpenRouterError("err"),
                OpenRouterError("err"),
                "ok",
            ]
        )
        retry_api_call(fn, item_label="test")
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert delays == pytest.approx([2.0, 4.0, 8.0])
