"""OpenRouter API client — text, image and audio generation, models list and cost estimation."""

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
_DEFAULT_TIMEOUT = 90
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY = 1.0
_DEFAULT_MODELS_CACHE_TTL = 300.0
_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}

# Approximate token estimates per card for cost calculation
_AVG_PROMPT_TOKENS = 200
_AVG_COMPLETION_TOKENS = 150
_AVG_AUDIO_CHARS = 100


def _parse_modalities(modality_str: str) -> list[Modality]:
    """Parse an OpenRouter modality string into a list of Modality values."""
    modalities: list[Modality] = []
    lower = modality_str.lower()
    # Format: "input->output", e.g. "text->text", "text->image", "text+image->text"
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
    """HTTP client for the OpenRouter API."""

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
        self._last_cost: float = 0.0  # cost of the last call from usage.cost

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_text(
        self,
        prompt: str,
        model: str,
        *,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text via the OpenRouter Chat Completions API.

        Args:
            prompt: The user prompt text.
            model: Model ID (e.g. 'openai/gpt-4o').
            temperature: Generation temperature (0.0–2.0). None uses the model default.
            system_prompt: Optional system message (sent as role=system before the user prompt).

        Returns:
            Generated text.

        Raises:
            OpenRouterAuthError: Invalid API key (401).
            OpenRouterRateLimitError: Rate limit exceeded (429).
            OpenRouterTimeoutError: Request timeout.
            OpenRouterError: Other API errors.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, object] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            body["temperature"] = temperature
        response = self._request_with_retry(body)
        return self._parse_text_response(response)

    def _request_with_retry(self, body: dict[str, object], *, timeout: int | None = None) -> requests.Response:
        """Perform an HTTP request with retries on transient errors."""
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
                last_exception = OpenRouterTimeoutError(f"Request timeout: {e}")
                if attempt < self.max_retries:
                    self._sleep_backoff(attempt)
                    continue
                raise last_exception from e

            except requests.ConnectionError as e:
                last_exception = OpenRouterError(f"Connection error: {e}")
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

        msg = "All retries exhausted"
        raise last_exception or OpenRouterError(msg)

    def _check_status(self, resp: requests.Response) -> None:
        """Check HTTP status and raise typed exceptions."""
        if resp.status_code == 200:
            return

        if resp.status_code == 401:
            raise OpenRouterAuthError("Invalid API key", status_code=401)

        if resp.status_code == 402:
            raise OpenRouterInsufficientCreditsError(
                "Insufficient credits on OpenRouter — top up your balance at openrouter.ai",
                status_code=402,
            )

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            raise OpenRouterRateLimitError(
                "Rate limit exceeded",
                retry_after=float(retry_after) if retry_after else None,
            )

        if resp.status_code in _RETRYABLE_STATUS_CODES:
            raise OpenRouterError(
                f"Server error: {resp.status_code}",
                status_code=resp.status_code,
            )

        # Try to extract error body for diagnostics
        try:
            error_body = resp.json()
            error_obj = error_body.get("error", {})
            if isinstance(error_obj, dict):
                error_msg = error_obj.get("message", "")
                # OpenRouter often hides details in metadata
                metadata = error_obj.get("metadata", {})
                if isinstance(metadata, dict) and metadata.get("raw"):
                    error_msg = f"{error_msg} | {str(metadata['raw'])[:300]}"
                elif not error_msg:
                    error_msg = resp.text[:400]
            else:
                error_msg = resp.text[:400]
        except Exception:  # noqa: BLE001
            error_msg = resp.text[:400] if resp.text else "no response body"

        raise OpenRouterError(
            f"Status {resp.status_code}: {error_msg}",
            status_code=resp.status_code,
        )

    def generate_image(self, prompt: str, model: str, *, size: str | None = None) -> bytes:
        """Generate an image via the OpenRouter API.

        Args:
            prompt: Image description.
            model: Model ID (e.g. 'google/gemini-3.1-flash-image-preview').
            size: Image size ('0.5K', '1K', '2K', '4K'). None or 'auto' uses the model default.

        Returns:
            Image bytes (PNG).

        Raises:
            OpenRouterAuthError: Invalid API key (401).
            OpenRouterRateLimitError: Rate limit exceeded (429).
            OpenRouterTimeoutError: Request timeout.
            OpenRouterError: Other API errors.
        """
        # Multimodal content format — as in official OpenRouter examples
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
            # Retry without image_config — some models/providers don't support it
            body.pop("image_config")
            response = self._request_with_retry(body, timeout=60)
            return self._parse_image_response(response)

    def generate_audio(self, text: str, model: str, *, voice: str = "alloy") -> bytes:
        """Generate audio (TTS) via the OpenRouter API.

        Uses streaming with modalities=["text","audio"] for models
        like gpt-audio-mini. Collects base64 chunks from delta.audio.data.

        Args:
            text: Text to synthesize.
            model: Model ID (e.g. 'openai/gpt-audio-mini').
            voice: Speaker voice (alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse).

        Returns:
            Audio bytes (wav).

        Raises:
            OpenRouterAuthError: Invalid API key (401).
            OpenRouterRateLimitError: Rate limit exceeded (429).
            OpenRouterTimeoutError: Request timeout.
            OpenRouterError: Other API errors.
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
        """Perform a streaming request and collect audio chunks."""
        import json as _json

        # (connect_timeout, read_timeout) — read timeout per chunk, not total
        stream_timeout = (self.timeout, self.timeout)

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=body,
                timeout=stream_timeout,
                stream=True,
            )
        except requests.Timeout as e:
            raise OpenRouterTimeoutError(f"Request timeout: {e}") from e
        except requests.ConnectionError as e:
            raise OpenRouterError(f"Connection error: {e}") from e

        if resp.status_code != 200:
            self._check_status(resp)

        audio_chunks: list[str] = []
        last_chunk_data: dict[str, object] = {}
        try:
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
        except requests.Timeout as e:
            raise OpenRouterTimeoutError(f"Stream read timeout: {e}") from e
        except requests.ConnectionError as e:
            raise OpenRouterError(f"Stream connection lost: {e}") from e

        # Usage often arrives in the last chunk
        if last_chunk_data:
            self._extract_usage(last_chunk_data)

        if not audio_chunks:
            raise OpenRouterError("API returned no audio data in stream")

        combined_b64 = "".join(audio_chunks)
        try:
            pcm_data = base64.b64decode(combined_b64)
        except Exception as e:
            raise OpenRouterError("Failed to decode base64 audio") from e

        # Validate: at least 0.05s of audio (24000 Hz * 2 bytes * 0.05s = 2400 bytes)
        if len(pcm_data) < 2400:
            raise OpenRouterError(f"Audio too short ({len(pcm_data)} bytes PCM), likely corrupted")

        return self._pcm16_to_wav(pcm_data)

    @staticmethod
    def _pcm16_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
        """Convert raw PCM16 data to WAV format."""
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
        """Return (prompt_tokens, completion_tokens) of the last call."""
        return self._last_usage

    @property
    def last_cost(self) -> float:
        """Return the cost of the last call (from OpenRouter usage.cost)."""
        return self._last_cost

    def _extract_usage(self, data: dict[str, object]) -> None:
        """Extract usage and cost from the API response."""
        usage = data.get("usage", {})
        if isinstance(usage, dict):
            self._last_usage = (
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)),
            )
            # OpenRouter returns the actual cost in usage.cost
            cost_raw = usage.get("cost")
            self._last_cost = float(cost_raw) if cost_raw is not None else 0.0
        else:
            self._last_usage = (0, 0)
            self._last_cost = 0.0

    def _parse_text_response(self, resp: requests.Response) -> str:
        """Extract text from the OpenRouter JSON response."""
        try:
            data = resp.json()
        except (ValueError, TypeError) as e:
            raise OpenRouterError("Failed to parse JSON response") from e

        self._extract_usage(data)

        choices = data.get("choices", [])
        if not choices:
            raise OpenRouterError("API returned empty choices")

        content: str | None = choices[0].get("message", {}).get("content")
        if content is None:
            raise OpenRouterError("API returned empty content")

        return content

    def _parse_image_response(self, resp: requests.Response) -> bytes:
        """Extract image from the OpenRouter JSON response.

        Supports two formats:
        1. images[] — standard OpenRouter format
        2. content[] with type=image_url — multimodal format (some providers)
        """
        try:
            data = resp.json()
        except (ValueError, TypeError) as e:
            raise OpenRouterError("Failed to parse JSON response") from e

        self._extract_usage(data)

        choices = data.get("choices", [])
        if not choices:
            raise OpenRouterError("API returned empty choices")

        message = choices[0].get("message", {})

        # Format 1: images[] (standard OpenRouter)
        images = message.get("images", [])
        if images:
            url: str = images[0].get("image_url", {}).get("url", "")
            return self._decode_base64_image(url)

        # Format 2: content[] with type=image_url (multimodal)
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    return self._decode_base64_image(url)

        raise OpenRouterError("API returned no image")

    @staticmethod
    def _decode_base64_image(url: str) -> bytes:
        """Decode a base64 data URL into image bytes."""
        prefix = "base64,"
        idx = url.find(prefix)
        if idx == -1:
            raise OpenRouterError("Response does not contain base64 image data")

        b64_data = url[idx + len(prefix) :]
        try:
            return base64.b64decode(b64_data)
        except Exception as e:
            raise OpenRouterError("Failed to decode base64 image") from e

    def _parse_audio_response(self, resp: requests.Response) -> bytes:
        """Extract audio from the OpenRouter JSON response."""
        try:
            data = resp.json()
        except (ValueError, TypeError) as e:
            raise OpenRouterError("Failed to parse JSON response") from e

        choices = data.get("choices", [])
        if not choices:
            raise OpenRouterError("API returned empty choices")

        message = choices[0].get("message", {})
        audio = message.get("audio")
        if not audio or not audio.get("data"):
            raise OpenRouterError("API returned no audio data")

        b64_data: str = audio["data"]
        try:
            return base64.b64decode(b64_data)
        except Exception as e:
            raise OpenRouterError("Failed to decode base64 audio") from e

    def fetch_balance(self) -> dict[str, float]:
        """Fetch account balance via the OpenRouter API.

        Returns:
            Dict with keys: 'limit', 'usage', 'remaining' (in dollars).

        Raises:
            OpenRouterError: On API error.
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
            raise OpenRouterError("Failed to parse JSON response") from e

        limit_raw = data.get("limit")
        usage_raw = data.get("usage")
        limit_remaining_raw = data.get("limit_remaining")

        usage = float(usage_raw) if usage_raw is not None else 0.0

        # limit_remaining — actual remaining credits (for prepaid/unlimited keys)
        if limit_remaining_raw is not None:
            remaining = float(limit_remaining_raw)
        elif limit_raw is not None:
            remaining = float(limit_raw) - usage
        else:
            # unlimited without limit_remaining — cannot determine the remaining balance
            remaining = -1.0  # sentinel: "unknown"

        is_unlimited = limit_raw is None

        return {
            "usage": usage,
            "remaining": remaining,
            "is_unlimited": is_unlimited,
        }

    def fetch_models(self) -> list[Model]:
        """Fetch the list of models from the OpenRouter API.

        Returns:
            Typed list of models with pricing and modalities.
            Results are cached with TTL.
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
            raise OpenRouterError("Failed to parse JSON response") from e

        models: list[Model] = []
        for item in data.get("data", []):
            models.append(self._parse_model_item(item))

        self._models_cache = models
        self._models_cache_time = time.time()
        return models

    @staticmethod
    def _parse_model_item(item: dict[str, object]) -> Model:
        """Parse a single item from the /models response."""

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
        """Estimate the approximate generation cost.

        Args:
            mode: Generation mode (questions, language, material, image, audio).
            card_count: Number of cards.
            text_model: Model for text.
            image_model: Model for images.
            audio_model: Model for audio.

        Returns:
            Approximate cost in dollars.
        """
        if card_count <= 0:
            return 0.0

        cost = 0.0

        # Text generation cost
        if text_model is not None and mode in ("questions", "language", "material", "image", "audio"):
            text_cost_per_card = (
                text_model.pricing.prompt * _AVG_PROMPT_TOKENS + text_model.pricing.completion * _AVG_COMPLETION_TOKENS
            )
            cost += text_cost_per_card * card_count

        # Image generation cost (~$0.04 per image by default)
        if image_model is not None and mode in ("language", "image"):
            cost += 0.04 * card_count

        # Audio generation cost
        if audio_model is not None and mode in ("language", "audio"):
            audio_cost_per_card = audio_model.pricing.prompt * _AVG_AUDIO_CHARS
            cost += audio_cost_per_card * card_count

        return cost

    def _sleep_backoff(self, attempt: int) -> None:
        """Exponential backoff delay between retries."""
        delay = self.base_delay * (2**attempt)
        time.sleep(delay)
