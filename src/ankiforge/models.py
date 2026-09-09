"""Data models for AnkiForge."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GenerationMode(Enum):
    """Card generation modes."""

    LANGUAGE = "language"
    MATERIAL = "material"
    IMAGE = "image"
    AUDIO = "audio"
    QUESTIONS = "questions"


class AnswerDetail(Enum):
    """Answer detail level."""

    SHORT = "short"
    MEDIUM = "medium"
    DETAILED = "detailed"


@dataclass
class MaterialOptions:
    """Options for generating cards from material."""

    max_cards_per_paragraph: int = 3
    include_images: bool = False
    image_size: str = "auto"
    answer_detail: AnswerDetail = AnswerDetail.SHORT


@dataclass
class CardRequest:
    """Card generation request."""

    mode: GenerationMode
    input_text: str
    target_deck: str
    create_new_deck: bool = False
    include_images: bool = False
    custom_prompt: str | None = None
    language: str = "en"
    language_options: LanguageOptions | None = None
    material_options: MaterialOptions | None = None
    voice: str = "alloy"
    image_size: str = "auto"


@dataclass
class LanguageOptions:
    """Language card generation options."""

    include_photo: bool = False
    include_audio_word: bool = True
    include_audio_definition: bool = True
    include_audio_example: bool = True
    include_transcription: bool = True
    image_size: str = "auto"
    detailed_image: bool = False
    voice: str = "alloy"


@dataclass
class GeneratedCard:
    """Generated card."""

    word: str
    note_type: str
    definition: str | None = None
    example: str | None = None
    answer: str | None = None
    audio_data: bytes | None = None
    image_data: bytes | None = None
    audio_definition: bytes | None = None
    audio_example: bytes | None = None
    audio_silence: bytes | None = None
    transcription: str | None = None


@dataclass
class GenerationProgress:
    """Card generation progress."""

    total_cards: int
    completed_cards: int = 0
    estimated_cost: float = 0.0
    current_cost: float = 0.0
    is_cancelled: bool = False


@dataclass
class ModelPricingCache:
    """Pricing cache for a single model."""

    prompt: float = 0.0
    completion: float = 0.0
    image: float = 0.0
    request: float = 0.0


@dataclass
class AddonConfig:
    """Add-on configuration."""

    api_key: str = ""
    text_model: str = ""
    image_model: str = ""
    audio_model: str = ""
    language: str = "en"
    text_model_pricing: ModelPricingCache | None = None
    image_model_pricing: ModelPricingCache | None = None
    audio_model_pricing: ModelPricingCache | None = None
    cached_balance: float | None = None
    cached_usage: float | None = None
    # Custom OpenAI-compatible endpoint (e.g. LM Studio, Ollama, vLLM).
    # When custom_base_url is set, text/image/audio model dropdowns can mix
    # OpenRouter models with models served locally through this endpoint —
    # see ankiforge.openrouter.routing_client.
    custom_base_url: str = ""
    custom_api_key: str = ""
    custom_label: str = "Local"
    # Local image generation (Automatic1111 / ComfyUI) — a separate route from
    # custom_base_url above, since neither speaks the OpenAI /chat/completions
    # format. See ankiforge.local_image.client and
    # ankiforge.openrouter.routing_client.LOCAL_IMAGE_MODEL_ID.
    local_image_backend: str = ""  # "", "automatic1111", or "comfyui"
    local_image_url: str = ""
    local_image_checkpoint: str = ""  # ComfyUI only — checkpoint (.safetensors) filename
    # Base generation resolution: "auto" (detect from the ComfyUI checkpoint
    # filename, or 512 if that's not possible — e.g. Automatic1111), "512",
    # "768", or "1024". SDXL-family checkpoints need ~1024px to avoid the
    # tiled/fractured look of running them below their native resolution —
    # set this explicitly if a checkpoint doesn't match the auto-detection.
    local_image_resolution: str = "auto"
    # Optional raw JSON overrides for steps/cfg_scale/sampler_name/resolution,
    # applied on top of the auto-picked values — the escape hatch for a
    # checkpoint or setup the auto-detection heuristics get wrong (e.g. a
    # non-distilled SDXL checkpoint, or a preferred sampler), without needing
    # a code change. E.g. '{"steps": 8, "cfg_scale": 2, "sampler_name": "dpmpp_2m_sde"}'.
    # See ankiforge.local_image.client._apply_advanced_overrides.
    local_image_advanced: str = ""
