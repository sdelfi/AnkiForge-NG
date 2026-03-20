"""Data models для AnkiForge."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GenerationMode(Enum):
    """Режимы генерации карточек."""

    LANGUAGE = "language"
    MATERIAL = "material"
    IMAGE = "image"
    AUDIO = "audio"
    QUESTIONS = "questions"


class AnswerDetail(Enum):
    """Уровень детальности ответа."""

    SHORT = "short"
    MEDIUM = "medium"
    DETAILED = "detailed"


@dataclass
class MaterialOptions:
    """Опции генерации карточек из материала."""

    max_cards_per_paragraph: int = 3
    include_images: bool = False
    image_size: str = "auto"
    answer_detail: AnswerDetail = AnswerDetail.SHORT


@dataclass
class CardRequest:
    """Запрос на генерацию карточек."""

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
    """Опции генерации языковых карточек."""

    include_photo: bool = True
    include_audio_word: bool = True
    include_audio_definition: bool = True
    include_audio_example: bool = True
    include_transcription: bool = True
    image_size: str = "auto"
    detailed_image: bool = False
    voice: str = "alloy"


@dataclass
class GeneratedCard:
    """Сгенерированная карточка."""

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
    """Прогресс генерации карточек."""

    total_cards: int
    completed_cards: int = 0
    estimated_cost: float = 0.0
    current_cost: float = 0.0
    is_cancelled: bool = False


@dataclass
class ModelPricingCache:
    """Кэш pricing для одной модели."""

    prompt: float = 0.0
    completion: float = 0.0
    image: float = 0.0
    request: float = 0.0


@dataclass
class AddonConfig:
    """Конфигурация add-on."""

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
