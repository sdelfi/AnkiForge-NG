"""Routes generation calls to OpenRouter or a custom OpenAI-compatible endpoint.

AnkiForge originally talked to a single OpenRouter client for all three model
slots (text, image, audio). To let users mix OpenRouter models with models
served locally (LM Studio, Ollama, vLLM, or anything else exposing an
OpenAI-compatible /chat/completions + /models API), each stored model ID can
carry a "custom::" prefix. RoutingClient looks at that prefix per-call and
forwards the request to the right underlying OpenRouterClient, stripping the
prefix first since the actual server doesn't know about it.

RoutingClient exposes the same public surface as OpenRouterClient
(generate_text, generate_audio, generate_image, last_cost, last_usage), so it
is a drop-in replacement anywhere a single OpenRouterClient was passed
around (generators, cost tracking, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from ankiforge.openrouter.client import OpenRouterClient

if TYPE_CHECKING:
    from ankiforge.models import AddonConfig

CUSTOM_PREFIX = "custom::"

# Shared alias for generator classes: either a plain OpenRouterClient, or a
# RoutingClient that also dispatches "custom::"-prefixed model ids to a
# local/custom endpoint. Both expose the same generate_text/audio/image +
# last_cost/last_usage surface, so generators can accept either.
GenerationClient = Union["OpenRouterClient", "RoutingClient"]
# LM Studio (and most local OpenAI-compatible servers) ignore the API key
# entirely, but some HTTP clients/proxies choke on a missing Authorization
# header — send a harmless placeholder when the user hasn't set one.
_LOCAL_PLACEHOLDER_KEY = "lm-studio"


def is_custom_model(model_id: str) -> bool:
    """Return True if a stored model id refers to the custom/local provider."""
    return model_id.startswith(CUSTOM_PREFIX)


def strip_custom_prefix(model_id: str) -> str:
    """Strip the 'custom::' prefix from a model id, if present."""
    return model_id[len(CUSTOM_PREFIX) :] if is_custom_model(model_id) else model_id


def add_custom_prefix(model_id: str) -> str:
    """Add the 'custom::' prefix to a raw model id (no-op if already present)."""
    return model_id if is_custom_model(model_id) else f"{CUSTOM_PREFIX}{model_id}"


class RoutingClient:
    """Dispatches text/audio/image generation to OpenRouter or a custom endpoint.

    Which client handles a given call is decided per-call from the model id:
    IDs prefixed with "custom::" go to the custom endpoint (with the prefix
    stripped before the request), everything else goes to OpenRouter.
    """

    def __init__(
        self,
        openrouter_client: OpenRouterClient | None,
        custom_client: OpenRouterClient | None,
    ) -> None:
        """Create a routing client.

        Args:
            openrouter_client: Client for OpenRouter, or None if no
                OpenRouter API key is configured.
            custom_client: Client for the custom/local endpoint, or None if
                none is configured.
        """
        self._or_client = openrouter_client
        self._custom_client = custom_client
        self._last_used: OpenRouterClient | None = None

    def _resolve(self, model: str) -> tuple[OpenRouterClient, str]:
        """Pick the underlying client for a model id and strip its prefix.

        Args:
            model: Stored model id, possibly "custom::"-prefixed.

        Returns:
            Tuple of (client to call, raw model id to send to the API).

        Raises:
            ValueError: If the required client isn't configured.
        """
        if is_custom_model(model):
            if self._custom_client is None:
                msg = "No local/custom API endpoint is configured (set it in AnkiForge Settings)"
                raise ValueError(msg)
            return self._custom_client, strip_custom_prefix(model)

        if self._or_client is None:
            msg = "No OpenRouter API key is configured"
            raise ValueError(msg)
        return self._or_client, model

    def generate_text(
        self,
        prompt: str,
        model: str,
        *,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text, routed to the right endpoint. See OpenRouterClient.generate_text."""
        client, raw_model = self._resolve(model)
        result = client.generate_text(prompt, raw_model, temperature=temperature, system_prompt=system_prompt)
        self._last_used = client
        return result

    def generate_audio(self, text: str, model: str, *, voice: str = "alloy") -> bytes:
        """Generate audio, routed to the right endpoint. See OpenRouterClient.generate_audio."""
        client, raw_model = self._resolve(model)
        result = client.generate_audio(text, raw_model, voice=voice)
        self._last_used = client
        return result

    def generate_image(self, prompt: str, model: str, *, size: str | None = None) -> bytes:
        """Generate an image, routed to the right endpoint. See OpenRouterClient.generate_image."""
        client, raw_model = self._resolve(model)
        result = client.generate_image(prompt, raw_model, size=size)
        self._last_used = client
        return result

    @property
    def last_cost(self) -> float:
        """Cost of the last call (0.0 for local/custom calls — they're free)."""
        return self._last_used.last_cost if self._last_used is not None else 0.0

    @property
    def last_usage(self) -> tuple[int, int]:
        """(prompt_tokens, completion_tokens) of the last call."""
        return self._last_used.last_usage if self._last_used is not None else (0, 0)


def build_client(config: AddonConfig) -> RoutingClient:
    """Build a RoutingClient from the add-on configuration.

    Args:
        config: Current add-on configuration.

    Returns:
        A RoutingClient wired up with whichever of OpenRouter / the custom
        endpoint are configured (either or both may be absent).
    """
    or_client = OpenRouterClient(api_key=config.api_key) if config.api_key else None

    custom_client: OpenRouterClient | None = None
    if config.custom_base_url:
        custom_client = OpenRouterClient(
            api_key=config.custom_api_key or _LOCAL_PLACEHOLDER_KEY,
            base_url=config.custom_base_url.rstrip("/"),
        )

    return RoutingClient(or_client, custom_client)
