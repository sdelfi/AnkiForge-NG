"""OpenRouter API клиент — текстовая, image и audio генерация, список моделей и стоимость."""

from __future__ import annotations

import base64
import time

import requests

from ankiforge.openrouter.exceptions import (
    OpenRouterAuthError,
    OpenRouterError,
    OpenRouterInsufficientCreditsError,
    OpenRouterRateLimitError,
    OpenRouterTimeoutError,
)
from ankiforge.openrouter.models import Modality, Model, ModelPricing

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY = 1.0
_DEFAULT_MODELS_CACHE_TTL = 300.0
_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}

# Примерные оценки токенов на карточку для расчёта стоимости
_AVG_PROMPT_TOKENS = 200
_AVG_COMPLETION_TOKENS = 150
_AVG_AUDIO_CHARS = 100


def _parse_modalities(modality_str: str) -> list[Modality]:
    """Парсит строку модальности OpenRouter в список Modality."""
    modalities: list[Modality] = []
    lower = modality_str.lower()
    # Формат: "input->output", например "text->text", "text->image", "text+image->text"
    parts = lower.split("->")
    output = parts[-1] if len(parts) > 1 else ""
    if "text" in output:
        modalities.append(Modality.TEXT)
    if "image" in output:
        modalities.append(Modality.IMAGE)
    if "audio" in output:
        modalities.append(Modality.AUDIO)
    return modalities


class OpenRouterClient:
    """HTTP-клиент для OpenRouter API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        base_delay: float = _DEFAULT_BASE_DELAY,
        models_cache_ttl: float = _DEFAULT_MODELS_CACHE_TTL,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._models_cache_ttl = models_cache_ttl
        self._models_cache: list[Model] | None = None
        self._models_cache_time: float = 0.0
        self._last_usage: tuple[int, int] = (0, 0)  # (prompt_tokens, completion_tokens)
        self._last_cost: float = 0.0  # стоимость последнего вызова из usage.cost

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_text(self, prompt: str, model: str, *, temperature: float | None = None) -> str:
        """Генерация текста через OpenRouter Chat Completions API.

        Args:
            prompt: Текст промпта.
            model: ID модели (например, 'openai/gpt-4o').
            temperature: Температура генерации (0.0–2.0). None — дефолт модели.

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
        if temperature is not None:
            body["temperature"] = temperature
        response = self._request_with_retry(body)
        return self._parse_text_response(response)

    def _request_with_retry(self, body: dict[str, object], *, timeout: int | None = None) -> requests.Response:
        """Выполняет HTTP-запрос с retry при transient-ошибках."""
        effective_timeout = timeout or self.timeout
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=body,
                    timeout=effective_timeout,
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

            except (OpenRouterAuthError, OpenRouterInsufficientCreditsError, OpenRouterRateLimitError):
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

        if resp.status_code == 402:
            raise OpenRouterInsufficientCreditsError(
                "Недостаточно кредитов на OpenRouter — пополните баланс на openrouter.ai",
                status_code=402,
            )

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

        # Пытаемся достать тело ошибки для диагностики
        try:
            error_body = resp.json()
            error_obj = error_body.get("error", {})
            if isinstance(error_obj, dict):
                error_msg = error_obj.get("message", "")
                # OpenRouter часто прячет детали в metadata
                metadata = error_obj.get("metadata", {})
                if isinstance(metadata, dict) and metadata.get("raw"):
                    error_msg = f"{error_msg} | {str(metadata['raw'])[:300]}"
                elif not error_msg:
                    error_msg = resp.text[:400]
            else:
                error_msg = resp.text[:400]
        except Exception:  # noqa: BLE001
            error_msg = resp.text[:400] if resp.text else "нет тела ответа"

        raise OpenRouterError(
            f"Статус {resp.status_code}: {error_msg}",
            status_code=resp.status_code,
        )

    def generate_image(self, prompt: str, model: str, *, size: str | None = None) -> bytes:
        """Генерация изображения через OpenRouter API.

        Args:
            prompt: Описание изображения.
            model: ID модели (например, 'google/gemini-3.1-flash-image-preview').
            size: Размер изображения ('0.5K', '1K', '2K', '4K'). None или 'auto' — размер по умолчанию модели.

        Returns:
            Байты изображения (PNG).

        Raises:
            OpenRouterAuthError: Невалидный API-ключ (401).
            OpenRouterRateLimitError: Превышен rate limit (429).
            OpenRouterTimeoutError: Таймаут запроса.
            OpenRouterError: Прочие ошибки API.
        """
        # Multimodal content format — как в официальных примерах OpenRouter
        body: dict[str, object] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                },
            ],
            "modalities": ["image", "text"],
        }
        if size and size != "auto":
            body["image_config"] = {"image_size": size}
        try:
            response = self._request_with_retry(body, timeout=60)
            return self._parse_image_response(response)
        except (OpenRouterAuthError, OpenRouterRateLimitError):
            raise
        except OpenRouterError:
            if "image_config" not in body:
                raise
            # Retry без image_config — некоторые модели/провайдеры не поддерживают его
            body.pop("image_config")
            response = self._request_with_retry(body, timeout=60)
            return self._parse_image_response(response)

    def generate_audio(self, text: str, model: str, *, voice: str = "alloy") -> bytes:
        """Генерация аудио (TTS) через OpenRouter API.

        Использует streaming с modalities=["text","audio"] для моделей
        типа gpt-audio-mini. Собирает base64-чанки из delta.audio.data.

        Args:
            text: Текст для озвучивания.
            model: ID модели (например, 'openai/gpt-audio-mini').
            voice: Голос диктора (alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse).

        Returns:
            Байты аудио (wav).

        Raises:
            OpenRouterAuthError: Невалидный API-ключ (401).
            OpenRouterRateLimitError: Превышен rate limit (429).
            OpenRouterTimeoutError: Таймаут запроса.
            OpenRouterError: Прочие ошибки API.
        """
        body: dict[str, object] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Repeat the user's text exactly as audio. Say ONLY those words, nothing else.",
                },
                {"role": "user", "content": f'Say: "{text}"'},
            ],
            "modalities": ["text", "audio"],
            "audio": {"voice": voice, "format": "pcm16"},
            "stream": True,
        }
        return self._stream_audio(body)

    def _stream_audio(self, body: dict[str, object]) -> bytes:
        """Выполняет streaming-запрос и собирает аудио-чанки."""
        import json as _json

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=body,
                timeout=self.timeout,
                stream=True,
            )
        except requests.Timeout as e:
            raise OpenRouterTimeoutError(f"Таймаут запроса: {e}") from e
        except requests.ConnectionError as e:
            raise OpenRouterError(f"Ошибка соединения: {e}") from e

        if resp.status_code != 200:
            self._check_status(resp)

        audio_chunks: list[str] = []
        last_chunk_data: dict[str, object] = {}
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload.strip() == "[DONE]":
                break
            try:
                chunk = _json.loads(payload)
            except (ValueError, TypeError):
                continue
            last_chunk_data = chunk
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            audio_data = delta.get("audio", {})
            if isinstance(audio_data, dict) and audio_data.get("data"):
                audio_chunks.append(audio_data["data"])

        # Usage часто приходит в последнем чанке
        if last_chunk_data:
            self._extract_usage(last_chunk_data)

        if not audio_chunks:
            raise OpenRouterError("API не вернул аудио данные в stream")

        combined_b64 = "".join(audio_chunks)
        try:
            pcm_data = base64.b64decode(combined_b64)
        except Exception as e:
            raise OpenRouterError("Ошибка декодирования base64 аудио") from e

        return self._pcm16_to_wav(pcm_data)

    @staticmethod
    def _pcm16_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
        """Конвертирует raw PCM16 данные в WAV формат."""
        import struct

        bits_per_sample = 16
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        data_size = len(pcm_data)

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,  # chunk size
            1,  # PCM format
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )
        return header + pcm_data

    @property
    def last_usage(self) -> tuple[int, int]:
        """Возвращает (prompt_tokens, completion_tokens) последнего вызова."""
        return self._last_usage

    @property
    def last_cost(self) -> float:
        """Возвращает стоимость последнего вызова (из usage.cost OpenRouter)."""
        return self._last_cost

    def _extract_usage(self, data: dict[str, object]) -> None:
        """Извлекает usage и cost из ответа API."""
        usage = data.get("usage", {})
        if isinstance(usage, dict):
            self._last_usage = (
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)),
            )
            # OpenRouter возвращает реальную стоимость в usage.cost
            cost_raw = usage.get("cost")
            self._last_cost = float(cost_raw) if cost_raw is not None else 0.0
        else:
            self._last_usage = (0, 0)
            self._last_cost = 0.0

    def _parse_text_response(self, resp: requests.Response) -> str:
        """Извлекает текст из JSON-ответа OpenRouter."""
        try:
            data = resp.json()
        except (ValueError, TypeError) as e:
            raise OpenRouterError("Ошибка парсинг JSON-ответа") from e

        self._extract_usage(data)

        choices = data.get("choices", [])
        if not choices:
            raise OpenRouterError("API вернул пустой choices")

        content: str | None = choices[0].get("message", {}).get("content")
        if content is None:
            raise OpenRouterError("API вернул пустой content")

        return content

    def _parse_image_response(self, resp: requests.Response) -> bytes:
        """Извлекает изображение из JSON-ответа OpenRouter.

        Поддерживает два формата:
        1. images[] — стандартный формат OpenRouter
        2. content[] с type=image_url — multimodal формат (некоторые провайдеры)
        """
        try:
            data = resp.json()
        except (ValueError, TypeError) as e:
            raise OpenRouterError("Ошибка парсинг JSON-ответа") from e

        self._extract_usage(data)

        choices = data.get("choices", [])
        if not choices:
            raise OpenRouterError("API вернул пустой choices")

        message = choices[0].get("message", {})

        # Формат 1: images[] (стандартный OpenRouter)
        images = message.get("images", [])
        if images:
            url: str = images[0].get("image_url", {}).get("url", "")
            return self._decode_base64_image(url)

        # Формат 2: content[] с type=image_url (multimodal)
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    return self._decode_base64_image(url)

        raise OpenRouterError("API не вернул изображение")

    @staticmethod
    def _decode_base64_image(url: str) -> bytes:
        """Декодирует base64 data URL в байты изображения."""
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

    def fetch_balance(self) -> dict[str, float]:
        """Запрашивает баланс аккаунта через OpenRouter API.

        Returns:
            Словарь с ключами: 'limit', 'usage', 'remaining' (в долларах).

        Raises:
            OpenRouterError: При ошибке API.
        """
        resp = requests.get(
            f"{self.base_url}/auth/key",
            headers=self._headers(),
            timeout=self.timeout,
        )
        self._check_status(resp)
        try:
            data = resp.json().get("data", {}) or {}
        except (ValueError, TypeError) as e:
            raise OpenRouterError("Ошибка парсинг JSON-ответа") from e

        limit_raw = data.get("limit")
        usage_raw = data.get("usage")
        limit_remaining_raw = data.get("limit_remaining")

        usage = float(usage_raw) if usage_raw is not None else 0.0

        # limit_remaining — реальный остаток кредитов (для prepaid/unlimited ключей)
        if limit_remaining_raw is not None:
            remaining = float(limit_remaining_raw)
        elif limit_raw is not None:
            remaining = float(limit_raw) - usage
        else:
            # unlimited без limit_remaining — не можем определить остаток
            remaining = -1.0  # sentinel: "неизвестно"

        is_unlimited = limit_raw is None

        return {
            "usage": usage,
            "remaining": remaining,
            "is_unlimited": is_unlimited,
        }

    def fetch_models(self) -> list[Model]:
        """Загружает список моделей с OpenRouter API.

        Returns:
            Типизированный список моделей с pricing и модальностями.
            Результат кешируется с TTL.
        """
        now = time.time()
        if self._models_cache is not None and (now - self._models_cache_time) < self._models_cache_ttl:
            return self._models_cache

        resp = requests.get(
            f"{self.base_url}/models",
            headers=self._headers(),
            timeout=self.timeout,
        )
        self._check_status(resp)

        try:
            data = resp.json()
        except (ValueError, TypeError) as e:
            raise OpenRouterError("Ошибка парсинг JSON-ответа") from e

        models: list[Model] = []
        for item in data.get("data", []):
            models.append(self._parse_model_item(item))

        self._models_cache = models
        self._models_cache_time = time.time()
        return models

    @staticmethod
    def _parse_model_item(item: dict[str, object]) -> Model:
        """Парсит один элемент из ответа /models."""

        pricing_raw = item.get("pricing", {})
        assert isinstance(pricing_raw, dict)
        pricing = ModelPricing(
            prompt=float(pricing_raw.get("prompt", 0) or 0),
            completion=float(pricing_raw.get("completion", 0) or 0),
            image=float(pricing_raw.get("image", 0) or 0),
            request=float(pricing_raw.get("request", 0) or 0),
        )

        arch = item.get("architecture", {})
        assert isinstance(arch, dict)
        modality_str = str(arch.get("modality", ""))
        modalities = _parse_modalities(modality_str)

        model_id = str(item.get("id", ""))
        model_name = str(item.get("name", ""))
        raw_ctx = item.get("context_length", 0)
        context_length = int(raw_ctx) if isinstance(raw_ctx, (int, float, str)) else 0

        return Model(
            id=model_id,
            name=model_name,
            pricing=pricing,
            modalities=modalities,
            context_length=context_length,
        )

    def estimate_cost(
        self,
        mode: str,
        card_count: int,
        text_model: Model | None = None,
        image_model: Model | None = None,
        audio_model: Model | None = None,
    ) -> float:
        """Расчёт примерной стоимости генерации.

        Args:
            mode: Режим генерации (questions, language, material, image, audio).
            card_count: Количество карточек.
            text_model: Модель для текста.
            image_model: Модель для изображений.
            audio_model: Модель для аудио.

        Returns:
            Примерная стоимость в долларах.
        """
        if card_count <= 0:
            return 0.0

        cost = 0.0

        # Стоимость текстовой генерации
        if text_model is not None and mode in ("questions", "language", "material", "image", "audio"):
            text_cost_per_card = (
                text_model.pricing.prompt * _AVG_PROMPT_TOKENS + text_model.pricing.completion * _AVG_COMPLETION_TOKENS
            )
            cost += text_cost_per_card * card_count

        # Стоимость генерации изображений (~$0.04 за картинку по умолчанию)
        if image_model is not None and mode in ("language", "image"):
            cost += 0.04 * card_count

        # Стоимость генерации аудио
        if audio_model is not None and mode in ("language", "audio"):
            audio_cost_per_card = audio_model.pricing.prompt * _AVG_AUDIO_CHARS
            cost += audio_cost_per_card * card_count

        return cost

    def _sleep_backoff(self, attempt: int) -> None:
        """Экспоненциальная задержка между retry."""
        delay = self.base_delay * (2**attempt)
        time.sleep(delay)
