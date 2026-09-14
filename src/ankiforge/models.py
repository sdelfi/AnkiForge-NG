"""Data models for AnkiForge."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Appended to every generated image prompt (all generators, all backends) —
# a free-form extra instruction, empty by default. Kept as an escape hatch
# editable in Settings; NOT where content restrictions belong (see
# DEFAULT_LOCAL_IMAGE_NEGATIVE_PROMPT below for why).
DEFAULT_IMAGE_PROMPT_EXTRA = ""

# Negative prompt for local image backends (Automatic1111/Draw Things/
# ComfyUI) — see ankiforge.local_image.client. A negative prompt genuinely
# excludes the listed concepts from generation; unlike restrictive wording
# stuffed into the positive prompt (which CLIP-based text encoders don't
# negate reliably — confirmed empirically: a checkpoint biased toward
# exaggerated proportions kept doing so with "no cleavage, no nudity" in
# the positive prompt), this actually suppresses them. OpenRouter/cloud
# image models have no equivalent field, so this only applies locally.
DEFAULT_LOCAL_IMAGE_NEGATIVE_PROMPT = (
    "text, letters, words, writing, typography, diagram, quiz, watermark, signature, "
    "nsfw, nudity, cleavage, exaggerated breasts, sexualized, revealing clothing, "
    "low quality, blurry"
)

# Editable image prompt templates — Settings lets you rewrite these per
# generation mode without a code change. Available placeholders are called
# out in each one; an unrecognized {placeholder} left in by a typo raises a
# KeyError at generation time rather than silently mangling the prompt.
# Deliberately positive-only descriptions — content restrictions (no text,
# no NSFW, ...) belong in DEFAULT_LOCAL_IMAGE_NEGATIVE_PROMPT above instead.

# Language cards (Language mode) — placeholders: {word}, {example}
DEFAULT_LANGUAGE_IMAGE_PROMPT_TEMPLATE = (
    'An illustration depicting a scene, for the word/phrase "{word}". '
    'The scene depicts: "{example}". '
    'The image must clearly and unambiguously depict "{word}" as the main visual '
    "focus — a viewer should be able to guess the word just by looking at the image, "
    "without reading the sentence. Clean style, just the illustrated scene itself."
)

# Language cards, "Detailed image" option — placeholders: {word}, {example}
DEFAULT_LANGUAGE_IMAGE_PROMPT_TEMPLATE_DETAILED = (
    "A stunning, ultra-high-quality photorealistic image depicting a scene, "
    'for the word/phrase "{word}". '
    'The scene vividly depicts the sentence: "{example}". '
    'The image must clearly and unambiguously depict "{word}" as the main visual '
    "focus — a viewer should be able to guess the word just by looking at the image, "
    "without reading the sentence. "
    "The image should be visually rich with cinematic lighting, vivid colors, and fine details. "
    "Composition: centered subject with a complementary background that reinforces the meaning. "
    "Style: professional photography or digital art, magazine-cover quality. Mood: evocative, memorable."
)

# QA+Image and From Material (with images) modes — placeholder: {topic}
DEFAULT_QA_IMAGE_PROMPT_TEMPLATE = "A simple, clear illustration depicting: '{topic}'. Clean educational style."


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
    # Negative prompt for local image backends only (Automatic1111/Draw
    # Things/ComfyUI have this concept; OpenRouter/cloud image models don't)
    # — see DEFAULT_LOCAL_IMAGE_NEGATIVE_PROMPT above.
    local_image_negative_prompt: str = DEFAULT_LOCAL_IMAGE_NEGATIVE_PROMPT
    # Editable image prompt templates, one per generation mode — see the
    # DEFAULT_*_TEMPLATE constants above for placeholders and defaults.
    image_prompt_template: str = DEFAULT_LANGUAGE_IMAGE_PROMPT_TEMPLATE
    image_prompt_template_detailed: str = DEFAULT_LANGUAGE_IMAGE_PROMPT_TEMPLATE_DETAILED
    qa_image_prompt_template: str = DEFAULT_QA_IMAGE_PROMPT_TEMPLATE
    # Appended to every generated image prompt (all generators, all image
    # backends) — see DEFAULT_IMAGE_PROMPT_EXTRA above for why this exists.
    # Editable in Settings so it can be tuned or cleared per checkpoint
    # without a code change.
    image_prompt_extra: str = DEFAULT_IMAGE_PROMPT_EXTRA
