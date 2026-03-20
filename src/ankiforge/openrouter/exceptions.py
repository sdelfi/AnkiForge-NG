"""Кастомные исключения для OpenRouter API."""

from __future__ import annotations


class OpenRouterError(Exception):
    """Базовое исключение OpenRouter."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenRouterAuthError(OpenRouterError):
    """Невалидный API-ключ (401)."""


class OpenRouterRateLimitError(OpenRouterError):
    """Превышен rate limit (429)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class OpenRouterInsufficientCreditsError(OpenRouterError):
    """Недостаточно кредитов на аккаунте (402)."""


class OpenRouterTimeoutError(OpenRouterError):
    """Таймаут запроса."""
