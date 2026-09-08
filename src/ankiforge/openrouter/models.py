"""Data models for the OpenRouter API."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Modality(Enum):
    """OpenRouter model modalities."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"


@dataclass
class ModelPricing:
    """Model pricing (cost per token/request)."""

    prompt: float = 0.0
    completion: float = 0.0
    image: float = 0.0
    request: float = 0.0


@dataclass
class Model:
    """A model available for generation, from OpenRouter or a custom endpoint.

    Attributes:
        id: Model ID as sent to the API. For custom-provider models this is
            prefixed (see ankiforge.openrouter.routing_client.CUSTOM_PREFIX)
            so the routing client knows which endpoint to call.
        source: Where the model comes from — "openrouter" or "custom"
            (e.g. LM Studio or another OpenAI-compatible server).
    """

    id: str
    name: str
    pricing: ModelPricing = field(default_factory=ModelPricing)
    modalities: list[Modality] = field(default_factory=list)
    context_length: int = 0
    source: str = "openrouter"
