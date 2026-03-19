"""OpenRouter API клиент — текстовая, image и audio генерация."""

from __future__ import annotations

import base64
import time

import requests

from ankiforge.openrouter.exceptions import (
    OpenRouterAuthError,
    OpenRouterError,
    OpenRouterRateLimitError,
    OpenRouterTimeoutError,
)

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY = 1.0
_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}


class OpenRouterClient:
    """HTTP-клиент для OpenRouter API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        base_delay: float = _DEFAULT_BASE_DELAY,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_text(self, prompt: str, model: str) -> str:
        """Генерация текста через OpenRouter Chat Completions API.

        Args:
            prompt: Текст промпта.
            model: ID модели (например, 'openai/gpt-4o').

        Returns:
            Сгенерированный текст.

        Raises:
            OpenRouterAuthError: Невалидный API-ключ (401).
            OpenRouterRateLimitError: Превышен rate limit (429).
            OpenRouterTimeoutError: Таймаут запроса.
            OpenRouterError: Прочие ошибки API.
        """
        body: dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = self._request_with_retry(body)
        return self._parse_text_response(response)

    def _request_with_retry(self, body: dict[str, object]) -> requests.Response:
        """Выполняет HTTP-запрос с retry при transient-ошибках."""
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=body,
                    timeout=self.timeout,
                )
                self._check_status(resp)
                return resp

            except requests.Timeout as e:
                last_exception = OpenRouterTimeoutError(f"Таймаут запроса: {e}")
                if attempt < self.max_retries:
                    self._sleep_backoff(attempt)
                    continue
                raise last_exception from e

            except requests.ConnectionError as e:
                last_exception = OpenRouterError(f"Ошибка соединения: {e}")
                if attempt < self.max_retries:
                    self._sleep_backoff(attempt)
                    continue
                raise last_exception from e

            except (OpenRouterAuthError, OpenRouterRateLimitError):
                raise

            except OpenRouterError:
                if attempt < self.max_retries:
                    self._sleep_backoff(attempt)
                    last_exception = None
                    continue
                raise

        msg = "Все попытки исчерпаны"
        raise last_exception or OpenRouterError(msg)

    def _check_status(self, resp: requests.Response) -> None:
        """Проверяет HTTP-статус и бросает типизированные исключения."""
        if resp.status_code == 200:
            return

        if resp.status_code == 401:
            raise OpenRouterAuthError("Невалидный API-ключ", status_code=401)

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            raise OpenRouterRateLimitError(
                "Превышен rate limit",
                retry_after=float(retry_after) if retry_after else None,
            )

        if resp.status_code in _RETRYABLE_STATUS_CODES:
            raise OpenRouterError(
                f"Серверная ошибка: {resp.status_code}",
                status_code=resp.status_code,
            )

        raise OpenRouterError(
            f"Неожиданный статус: {resp.status_code}",
            status_code=resp.status_code,
        )

    def generate_image(self, prompt: str, model: str) -> bytes:
        """Генерация изображения через OpenRouter API.

        Args:
            prompt: Описание изображения.
            model: ID модели (например, 'openai/dall-e-3').

        Returns:
            Байты изображения (PNG).

        Raises:
            OpenRouterAuthError: Невалидный API-ключ (401).
            OpenRouterRateLimitError: Превышен rate limit (429).
            OpenRouterTimeoutError: Таймаут запроса.
            OpenRouterError: Прочие ошибки API.
        """
        body: dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = self._request_with_retry(body)
        return self._parse_image_response(response)

    def generate_audio(self, text: str, model: str) -> bytes:
        """Генерация аудио (TTS) через OpenRouter API.

        Args:
            text: Текст для озвучивания.
            model: ID модели (например, 'openai/tts-1').

        Returns:
            Байты аудио (mp3).

        Raises:
            OpenRouterAuthError: Невалидный API-ключ (401).
            OpenRouterRateLimitError: Превышен rate limit (429).
            OpenRouterTimeoutError: Таймаут запроса.
            OpenRouterError: Прочие ошибки API.
        """
        body: dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": text}],
        }
        response = self._request_with_retry(body)
        return self._parse_audio_response(response)

    def _parse_text_response(self, resp: requests.Response) -> str:
        """Извлекает текст из JSON-ответа OpenRouter."""
        try:
            data = resp.json()
        except (ValueError, TypeError) as e:
            raise OpenRouterError("Ошибка парсинг JSON-ответа") from e

        choices = data.get("choices", [])
        if not choices:
            raise OpenRouterError("API вернул пустой choices")

        content: str | None = choices[0].get("message", {}).get("content")
        if content is None:
            raise OpenRouterError("API вернул пустой content")

        return content

    def _parse_image_response(self, resp: requests.Response) -> bytes:
        """Извлекает изображение из JSON-ответа OpenRouter."""
        try:
            data = resp.json()
        except (ValueError, TypeError) as e:
            raise OpenRouterError("Ошибка парсинг JSON-ответа") from e

        choices = data.get("choices", [])
        if not choices:
            raise OpenRouterError("API вернул пустой choices")

        message = choices[0].get("message", {})
        images = message.get("images", [])
        if not images:
            raise OpenRouterError("API не вернул изображение")

        url: str = images[0].get("image_url", {}).get("url", "")
        prefix = "base64,"
        idx = url.find(prefix)
        if idx == -1:
            raise OpenRouterError("Ответ не содержит base64-данных изображения")

        b64_data = url[idx + len(prefix) :]
        try:
            return base64.b64decode(b64_data)
        except Exception as e:
            raise OpenRouterError("Ошибка декодирования base64 изображения") from e

    def _parse_audio_response(self, resp: requests.Response) -> bytes:
        """Извлекает аудио из JSON-ответа OpenRouter."""
        try:
            data = resp.json()
        except (ValueError, TypeError) as e:
            raise OpenRouterError("Ошибка парсинг JSON-ответа") from e

        choices = data.get("choices", [])
        if not choices:
            raise OpenRouterError("API вернул пустой choices")

        message = choices[0].get("message", {})
        audio = message.get("audio")
        if not audio or not audio.get("data"):
            raise OpenRouterError("API не вернул аудио данные")

        b64_data: str = audio["data"]
        try:
            return base64.b64decode(b64_data)
        except Exception as e:
            raise OpenRouterError("Ошибка декодирования base64 аудио") from e

    def _sleep_backoff(self, attempt: int) -> None:
        """Экспоненциальная задержка между retry."""
        delay = self.base_delay * (2**attempt)
        time.sleep(delay)
