"""Retry helper for API calls in generators."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from ankiforge.openrouter.exceptions import (
    OpenRouterAuthError,
    OpenRouterInsufficientCreditsError,
    OpenRouterRateLimitError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Fatal errors — no point retrying
_FATAL_EXCEPTIONS = (OpenRouterAuthError, OpenRouterInsufficientCreditsError)

_MAX_RETRIES = 10
_BASE_DELAY = 2.0
_MAX_DELAY = 120.0


def retry_api_call[T](fn: Callable[[], T], *, item_label: str = "") -> T:
    """Retry an API call with exponential backoff on transient errors.

    Retries on: timeout, rate limit, server errors (5xx), connection errors.
    Raises immediately on: auth errors (401), insufficient credits (402).

    Args:
        fn: Callable that performs the API call.
        item_label: Label for logging (e.g. word or question text).

    Returns:
        Result of the API call.

    Raises:
        OpenRouterAuthError: Invalid API key — fatal, no retry.
        OpenRouterInsufficientCreditsError: No credits — fatal, no retry.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except _FATAL_EXCEPTIONS:
            raise
        except OpenRouterRateLimitError as e:
            delay = e.retry_after if e.retry_after and e.retry_after > 0 else _delay_for(attempt)
            logger.warning(
                "Rate limit for %s, retry in %.1fs (attempt %d/%d)", item_label, delay, attempt + 1, _MAX_RETRIES
            )
            time.sleep(delay)
        except Exception as e:  # noqa: BLE001
            delay = _delay_for(attempt)
            logger.warning(
                "Error for %s: %s, retry in %.1fs (attempt %d/%d)", item_label, e, delay, attempt + 1, _MAX_RETRIES
            )
            time.sleep(delay)

    # Last attempt — let exception propagate
    return fn()


def _delay_for(attempt: int) -> float:
    """Exponential backoff with cap."""
    return min(_BASE_DELAY * (2**attempt), _MAX_DELAY)
