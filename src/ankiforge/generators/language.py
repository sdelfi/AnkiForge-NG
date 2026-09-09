"""Language card generator — input words -> definition + example + audio + image."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from ankiforge.anki_bridge.note_types import LANGUAGE_NOTE_TYPE_NAME
from ankiforge.generators._retry import retry_api_call
from ankiforge.models import GeneratedCard, GenerationProgress

if TYPE_CHECKING:
    from collections.abc import Callable

    from ankiforge.models import CardRequest, ModelPricingCache
    from ankiforge.openrouter.routing_client import GenerationClient

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "You are a vocabulary coach creating flashcard content for language learners.\n\n"
    "Given a word or phrase, return a JSON object with these fields:\n"
    '  "definition" — a clear, concise explanation that STARTS with the word itself '
    "(like a dictionary entry). Use simple language a beginner can understand. "
    'Format: "Word is/means ..." or "To word is to ...".\n'
    '  "example" — a vivid, memorable sentence that USES THE WORD (or its form) '
    "in context. Use a concrete, visual scenario — not abstract or generic. "
    "The word MUST appear in the sentence so the learner sees how it is used in real speech.\n"
    '  "ipa" — IPA phonetic transcription, as a JSON string wrapped in double quotes, '
    'e.g. "ipa": "/wɜːrd/". The slashes go INSIDE the quotes — never emit /wɜːrd/ '
    "as a bare, unquoted value.\n\n"
    "Respond with ONLY a single valid JSON object, and NOTHING else — "
    "no markdown fences, no arrows, no prose before or after it, no restating the word. "
    "Every field name MUST be a double-quoted string. Do not use symbols like '→' anywhere.\n\n"
    "Examples of GOOD output (this is the ENTIRE response, verbatim):\n"
    '  For "eloquent": '
    '{"definition": "Eloquent means able to express thoughts and feelings clearly and beautifully '
    'using words.", "example": "Her eloquent speech at the wedding painted such a vivid picture '
    'that guests laughed, cried, and sat completely silent."}\n'
    '  For "zoom": '
    '{"definition": "To zoom is to move very quickly or to increase rapidly in size.", '
    '"example": "The cars zoomed along the highway, leaving a trail of dust behind them."}\n'
    '  For "serendipity": '
    '{"definition": "Serendipity is a happy accident — finding something wonderful when you were '
    'not looking for it.", "example": "By pure serendipity, she found a first-edition book at a '
    'garage sale for just one dollar."}\n\n'
    "Examples of BAD output — do NOT do any of this:\n"
    '  `"eloquent" → definition: "..."` ← WRONG, not JSON: unquoted key, uses an arrow, restates the word\n'
    '  `Here is the JSON: {"definition": "..."}` ← WRONG, text before the JSON object\n'
    '  definition without the word: "Moving very quickly" ← WRONG, must start with the word\n'
    '  example without the word: "The cars moved fast along the road" ← WRONG, '
    "the word must appear in the example sentence"
)

_IMAGE_PROMPT_TEMPLATE = (
    "Create an illustration for a language flashcard. "
    'The scene depicts: "{example}". '
    "Clean style, no text, no watermarks. Suitable for a flashcard."
)

_DETAILED_IMAGE_PROMPT_TEMPLATE = (
    "Create a stunning, ultra-high-quality photorealistic image for a language flashcard. "
    'The scene vividly depicts the sentence: "{example}". '
    "The image should be visually rich with cinematic lighting, vivid colors, and fine details. "
    "Composition: centered subject with a complementary background that reinforces the meaning. "
    "Style: professional photography or digital art, magazine-cover quality. "
    "Mood: evocative, memorable. No text, no watermarks, no logos."
)

# WAV parameters for silence generation (must match TTS output)
_WAV_SAMPLE_RATE = 24000
_WAV_CHANNELS = 1
_WAV_BITS_PER_SAMPLE = 16

# Some models emit the IPA transcription as a bare, unquoted value
# (e.g. "ipa": /weɪt ɪn laɪn/ instead of "ipa": "/weɪt ɪn laɪn/"), which breaks
# json.loads. Without a repair step this used to fall through to the crude
# "first sentence / rest" fallback and swallow the whole response into
# `definition`. This regex quotes a bare /.../ value that follows "ipa":.
_IPA_UNQUOTED_RE = re.compile(r'("ipa"\s*:\s*)(/[^",}\]\n]*/)(?!")')

# Field-level fallbacks used when the response isn't valid JSON even after
# repair (e.g. it's missing a comma, or has stray text around the object).
# These let us still recover the individual fields by regex instead of
# falling back to a naive sentence split. The key itself is matched with
# optional quotes ("?) — weaker models (notably local ones) sometimes drop
# the quotes around field names even though the value is still quoted, e.g.
# `definition: "..."` instead of `"definition": "..."`.
_DEFINITION_FIELD_RE = re.compile(r'"?definition"?\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
_EXAMPLE_FIELD_RE = re.compile(r'"?example"?\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
_IPA_FIELD_RE = re.compile(r'"?ipa"?\s*:\s*"?(/[^",}\]\n]*/)"?')


def _repair_json_like(text: str) -> str:
    """Repair common LLM JSON malformations before handing text to json.loads.

    Currently fixes: an unquoted IPA transcription value (bare /.../ instead
    of a quoted JSON string).

    Args:
        text: Raw (possibly malformed) JSON text.

    Returns:
        Text with known malformations fixed. Safe to call on already-valid JSON.
    """
    return _IPA_UNQUOTED_RE.sub(lambda m: f'{m.group(1)}"{m.group(2)}"', text)


def _unescape_json_string(text: str) -> str:
    """Undo basic JSON string escaping (\\", \\n, \\\\) for regex-extracted field values."""
    return text.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _bold_word(text: str, word: str) -> str:
    """Wrap all occurrences of a word (and its forms) in <b>...</b>.

    Case-insensitive, preserves original case. Matches the word as a whole word
    (word boundaries), including forms with common suffixes (s, es, ed, ing, er, est, ly, tion, ment).

    Args:
        text: Text to process.
        word: Word to highlight.

    Returns:
        Text with <b>...</b> around the word.
    """
    if not text or not word:
        return text
    # Escape regex special chars, match word + optional common suffixes
    escaped = re.escape(word.strip())
    pattern = rf"\b({escaped}(?:s|es|ed|ing|er|est|ly|tion|ment|ness)?)\b"
    return re.sub(pattern, r"<b>\1</b>", text, flags=re.IGNORECASE)


def _generate_silence_wav(seconds: float = 2.0) -> bytes:
    """Generate a WAV file with silence of given duration.

    Args:
        seconds: Silence duration in seconds.

    Returns:
        WAV bytes.
    """
    import struct

    silence_samples = int(_WAV_SAMPLE_RATE * seconds * _WAV_CHANNELS)
    silence_bytes = b"\x00\x00" * silence_samples

    data_size = len(silence_bytes)
    byte_rate = _WAV_SAMPLE_RATE * _WAV_CHANNELS * _WAV_BITS_PER_SAMPLE // 8
    block_align = _WAV_CHANNELS * _WAV_BITS_PER_SAMPLE // 8

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        _WAV_CHANNELS,
        _WAV_SAMPLE_RATE,
        byte_rate,
        block_align,
        _WAV_BITS_PER_SAMPLE,
        b"data",
        data_size,
    )
    return header + silence_bytes


class LanguageGenerator:
    """Language card generator from a list of words."""

    def __init__(
        self,
        client: GenerationClient,
        text_model: str,
        audio_model: str,
        image_model: str,
        *,
        text_pricing: ModelPricingCache | None = None,
        image_pricing: ModelPricingCache | None = None,
        audio_pricing: ModelPricingCache | None = None,
    ) -> None:
        self._client = client
        self._text_model = text_model
        self._audio_model = audio_model
        self._image_model = image_model
        self._text_pricing = text_pricing
        self._image_pricing = image_pricing
        self._audio_pricing = audio_pricing

    def generate(
        self,
        request: CardRequest,
        progress_callback: Callable[[GenerationProgress], None],
    ) -> list[GeneratedCard]:
        """Generate Language cards from a list of words.

        Args:
            request: Request with words in input_text (one per line or comma-separated).
            progress_callback: Callback for tracking progress.

        Returns:
            List of generated cards.

        Raises:
            ValueError: If no words found in input text.
        """
        from ankiforge.models import LanguageOptions

        words = self._parse_words(request.input_text)
        opts = request.language_options or LanguageOptions()
        progress = GenerationProgress(total_cards=len(words))
        cards: list[GeneratedCard] = []
        total_cost = 0.0

        for word in words:
            # 1 request: definition + example + IPA (JSON)
            system_prompt, user_prompt = self._build_prompt(
                word, request.language, request.custom_prompt, opts.include_transcription
            )
            response = retry_api_call(
                lambda s=system_prompt, u=user_prompt: self._client.generate_text(
                    u, self._text_model, temperature=0.3, system_prompt=s
                ),
                item_label=word,
            )
            total_cost += self._cost_from_usage(self._text_pricing)
            definition, example, transcription = self._parse_json_response(response)

            if not opts.include_transcription:
                transcription = None

            # Audio: word pronunciation
            audio_data: bytes | None = None
            if opts.include_audio_word:
                audio_data = retry_api_call(
                    lambda w=word: self._client.generate_audio(w, self._audio_model, voice=opts.voice),
                    item_label=f"audio:{word}",
                )
                total_cost += self._cost_from_usage(self._audio_pricing)

            # Audio: definition narration
            audio_definition: bytes | None = None
            if opts.include_audio_definition and definition:
                audio_definition = retry_api_call(
                    lambda d=definition: self._client.generate_audio(d, self._audio_model, voice=opts.voice),
                    item_label=f"audio-def:{word}",
                )
                total_cost += self._cost_from_usage(self._audio_pricing)

            # Audio: example narration (clean, no silence)
            audio_example: bytes | None = None
            if opts.include_audio_example and example:
                audio_example = retry_api_call(
                    lambda e=example: self._client.generate_audio(e, self._audio_model, voice=opts.voice),
                    item_label=f"audio-ex:{word}",
                )
                total_cost += self._cost_from_usage(self._audio_pricing)

            # Image: scene from example
            image_data: bytes | None = None
            if opts.include_photo and example:
                template = _DETAILED_IMAGE_PROMPT_TEMPLATE if opts.detailed_image else _IMAGE_PROMPT_TEMPLATE
                image_prompt = template.format(example=example)
                size = opts.image_size if opts.image_size != "auto" else None
                image_data = retry_api_call(
                    lambda p=image_prompt, sz=size: self._client.generate_image(p, self._image_model, size=sz),
                    item_label=f"image:{word}",
                )
                total_cost += self._cost_from_usage(self._image_pricing, is_image=True)

            # Silence between definition and example audio (separate file)
            audio_silence: bytes | None = None
            if audio_definition and audio_example:
                audio_silence = _generate_silence_wav(seconds=2.0)

            cards.append(
                GeneratedCard(
                    word=word,
                    definition=_bold_word(definition, word),
                    example=_bold_word(example, word),
                    audio_data=audio_data,
                    image_data=image_data,
                    audio_definition=audio_definition,
                    audio_example=audio_example,
                    audio_silence=audio_silence,
                    transcription=transcription,
                    note_type=LANGUAGE_NOTE_TYPE_NAME,
                )
            )

            progress.completed_cards += 1
            progress.current_cost = total_cost
            progress_callback(progress)

            if progress.is_cancelled:
                break

        return cards

    def _cost_from_usage(self, pricing: ModelPricingCache | None, *, is_image: bool = False) -> float:
        """Calculate cost of the last API call.

        Priority: usage.cost from OpenRouter (exact) -> manual calculation from tokens.
        """
        # OpenRouter returns real cost — use if available
        api_cost = self._client.last_cost
        if api_cost > 0:
            return api_cost

        # Fallback: manual calculation from tokens x pricing
        if pricing is None:
            return 0.0
        prompt_tokens, completion_tokens = self._client.last_usage
        token_cost = pricing.prompt * prompt_tokens + pricing.completion * completion_tokens
        if token_cost > 0:
            return token_cost
        return 0.0

    def _build_prompt(
        self, word: str, language: str, custom_prompt: str | None, include_ipa: bool = True
    ) -> tuple[str, str]:
        """Build (system_prompt, user_prompt) for generating definition + example + IPA."""
        if custom_prompt:
            return (custom_prompt, f"Word: {word}\nLanguage: {language}")

        ipa_note = ' Include "ipa" field.' if include_ipa else " Omit the ipa field."
        lang_note = (
            f"\n\nIMPORTANT: The definition and example MUST be written in {language}. "
            f"Do NOT write in English unless Language is 'en'."
            if language != "en"
            else ""
        )
        system = f"{_DEFAULT_SYSTEM_PROMPT}\n{ipa_note}{lang_note}"
        user = f"Language: {language}\nWord: {word}"
        return (system, user)

    def _parse_words(self, text: str) -> list[str]:
        """Parse words from text (by lines or comma-separated)."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        words = [w.strip() for w in lines[0].split(",") if w.strip()] if len(lines) == 1 and "," in lines[0] else lines

        if not words:
            msg = "No words found in input"
            raise ValueError(msg)
        return words

    def _parse_json_response(self, response: str) -> tuple[str, str, str | None]:
        """Parse AI JSON response into definition, example, and IPA.

        Args:
            response: AI response text.

        Returns:
            Tuple (definition, example, ipa).
        """
        if not response.strip():
            return ("", "", None)

        # Try JSON
        try:
            # Strip markdown fences if present
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip())
            cleaned = _repair_json_like(cleaned)
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return (
                    str(data.get("definition", "")).strip(),
                    str(data.get("example", "")).strip(),
                    str(data.get("ipa", "")).strip() or None,
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: response still looks like a JSON object but isn't valid
        # JSON for some other reason (missing comma, stray text, etc.) —
        # recover the individual fields by regex rather than losing them all.
        recovered = self._parse_json_fields_by_regex(response)
        if recovered is not None:
            return recovered

        # Last resort: legacy DEFINITION: / EXAMPLE: text format
        return self._parse_text_response(response)

    def _parse_json_fields_by_regex(self, response: str) -> tuple[str, str, str | None] | None:
        """Recover definition/example/ipa directly by regex from a malformed JSON-like response.

        Args:
            response: Raw AI response text.

        Returns:
            Tuple (definition, example, ipa), or None if it doesn't look like
            a JSON object with the expected fields at all.
        """
        def_match = _DEFINITION_FIELD_RE.search(response)
        ex_match = _EXAMPLE_FIELD_RE.search(response)
        if not def_match or not ex_match:
            return None

        ipa_match = _IPA_FIELD_RE.search(response)
        ipa = ipa_match.group(1).strip() if ipa_match else None

        definition = _unescape_json_string(def_match.group(1)).strip()
        example = _unescape_json_string(ex_match.group(1)).strip()
        return (definition, example, ipa or None)

    def _parse_text_response(self, response: str) -> tuple[str, str, str | None]:
        """Fallback parser for text format."""
        def_match = re.search(r"DEFINITION:\s*(.+?)(?=\nEXAMPLE:|\Z)", response, re.DOTALL)
        ex_match = re.search(r"EXAMPLE:\s*(.+)", response, re.DOTALL)

        if def_match and ex_match:
            return (def_match.group(1).strip(), ex_match.group(1).strip(), None)

        # Fallback: first sentence = definition, rest = example
        sentences = re.split(r"(?<=\.)\s+", response.strip(), maxsplit=1)
        if len(sentences) >= 2:
            return (sentences[0].strip(), sentences[1].strip(), None)

        return (response.strip(), "", None)
