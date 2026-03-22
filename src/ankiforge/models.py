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
