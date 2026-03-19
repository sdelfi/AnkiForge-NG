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


@dataclass
class GenerationProgress:
    """Прогресс генерации карточек."""

    total_cards: int
    completed_cards: int = 0
    estimated_cost: float = 0.0
    current_cost: float = 0.0
    is_cancelled: bool = False


@dataclass
class AddonConfig:
    """Конфигурация add-on."""

    api_key: str = ""
    text_model: str = ""
    image_model: str = ""
    audio_model: str = ""
    language: str = "en"
