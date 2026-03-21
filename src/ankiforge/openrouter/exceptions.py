"""Custom exceptions for the OpenRouter API."""

from __future__ import annotations


class OpenRouterError(Exception):
    """Base OpenRouter exception."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenRouterAuthError(OpenRouterError):
    """Invalid API key (401)."""


class OpenRouterRateLimitError(OpenRouterError):
    """Rate limit exceeded (429)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class OpenRouterInsufficientCreditsError(OpenRouterError):
    """Insufficient account credits (402)."""


class OpenRouterTimeoutError(OpenRouterError):
    """Request timeout."""
