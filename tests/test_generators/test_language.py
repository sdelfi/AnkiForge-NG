"""Tests for LanguageGenerator."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from ankiforge.anki_bridge.note_types import LANGUAGE_NOTE_TYPE_NAME
from ankiforge.generators.language import LanguageGenerator
from ankiforge.models import CardRequest, GeneratedCard, GenerationMode, GenerationProgress, LanguageOptions
from ankiforge.openrouter.client import OpenRouterClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MOCK_JSON_RESPONSE = json.dumps(
    {
        "definition": "An apple is a round fruit that grows on trees.",
        "example": "She picked a ripe apple from the tree and bit into it.",
        "ipa": "/\u02c8\u00e6p\u0259l/",
    }
)


def _make_fake_wav(pcm_size: int = 100) -> bytes:
    """Creates a minimal valid WAV for tests."""
    import struct

    pcm = b"\x01\x00" * pcm_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        24000,
        48000,
        2,
        16,
        b"data",
        len(pcm),
    )
    return header + pcm


@pytest.fixture
def mock_client() -> MagicMock:
    """Mock OpenRouterClient."""
    client = MagicMock(spec=OpenRouterClient)
    client.generate_text.return_value = _MOCK_JSON_RESPONSE
    client.generate_audio.return_value = _make_fake_wav(100)
    client.generate_image.return_value = b"fake-image-bytes"
    client.last_usage = (0, 0)
    client.last_cost = 0.0
    return client


@pytest.fixture
def generator(mock_client: MagicMock) -> LanguageGenerator:
    return LanguageGenerator(
        client=mock_client,
        text_model="openai/gpt-4o",
        audio_model="openai/tts-1",
        image_model="openai/dall-e-3",
    )


@pytest.fixture
def base_request() -> CardRequest:
    return CardRequest(
        mode=GenerationMode.LANGUAGE,
        input_text="apple\nbanana",
        target_deck="Test Deck",
        language="en",
    )


# ---------------------------------------------------------------------------
# Word parsing
# ---------------------------------------------------------------------------


class TestParseWords:
    def test_splits_by_newline(self, generator: LanguageGenerator) -> None:
        result = generator._parse_words("apple\nbanana\ncherry")
        assert result == ["apple", "banana", "cherry"]

    def test_splits_by_comma(self, generator: LanguageGenerator) -> None:
        result = generator._parse_words("apple, banana, cherry")
        assert result == ["apple", "banana", "cherry"]

    def test_strips_whitespace(self, generator: LanguageGenerator) -> None:
        result = generator._parse_words("  apple  \n  banana  ")
        assert result == ["apple", "banana"]

    def test_skips_empty_lines(self, generator: LanguageGenerator) -> None:
        result = generator._parse_words("apple\n\n\nbanana\n")
        assert result == ["apple", "banana"]

    def test_single_word(self, generator: LanguageGenerator) -> None:
        result = generator._parse_words("apple")
        assert result == ["apple"]

    def test_empty_input_raises(self, generator: LanguageGenerator) -> None:
        with pytest.raises(ValueError, match="No words found in input"):
            generator._parse_words("")

    def test_only_whitespace_raises(self, generator: LanguageGenerator) -> None:
        with pytest.raises(ValueError, match="No words found in input"):
            generator._parse_words("   \n  \n  ")

    def test_mixed_newlines_and_commas(self, generator: LanguageGenerator) -> None:
        result = generator._parse_words("apple, pear\nbanana")
        assert result == ["apple, pear", "banana"]


# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    def test_parses_json(self, generator: LanguageGenerator) -> None:
        response = json.dumps(
            {
                "definition": "A round fruit",
                "example": "I eat an apple every day.",
                "ipa": "/\u02c8\u00e6p\u0259l/",
            }
        )
        definition, example, ipa = generator._parse_json_response(response)
        assert definition == "A round fruit"
        assert example == "I eat an apple every day."
        assert ipa == "/\u02c8\u00e6p\u0259l/"

    def test_parses_json_with_markdown_fences(self, generator: LanguageGenerator) -> None:
        response = '```json\n{"definition": "A fruit", "example": "Eat it.", "ipa": "/x/"}\n```'
        definition, example, ipa = generator._parse_json_response(response)
        assert definition == "A fruit"
        assert example == "Eat it."

    def test_fallback_to_text_format(self, generator: LanguageGenerator) -> None:
        response = "DEFINITION: A round fruit\nEXAMPLE: I eat an apple."
        definition, example, ipa = generator._parse_json_response(response)
        assert definition == "A round fruit"
        assert example == "I eat an apple."
        assert ipa is None

    def test_empty_response(self, generator: LanguageGenerator) -> None:
        definition, example, ipa = generator._parse_json_response("")
        assert definition == ""
        assert example == ""
        assert ipa is None

    def test_json_without_ipa(self, generator: LanguageGenerator) -> None:
        response = json.dumps({"definition": "A fruit", "example": "Eat it."})
        definition, example, ipa = generator._parse_json_response(response)
        assert definition == "A fruit"
        assert example == "Eat it."
        assert ipa is None

    def test_repairs_unquoted_ipa(self, generator: LanguageGenerator) -> None:
        """Regression test: some models emit "ipa": /x/ without quotes,
        which used to break json.loads and swallow the whole response into
        `definition` via the naive sentence-split fallback."""
        response = (
            '{ "definition": "To wait in line means to stand in a queue or line, '
            'waiting for your turn to do something.", "example": "We had to wait in '
            "line for an hour to get tickets to the concert, but the excitement made "
            'it worth it.", "ipa": /weɪt ɪn laɪn/ }'
        )
        definition, example, ipa = generator._parse_json_response(response)
        assert definition == "To wait in line means to stand in a queue or line, waiting for your turn to do something."
        assert example == (
            "We had to wait in line for an hour to get tickets to the concert, but the excitement made it worth it."
        )
        assert ipa == "/weɪt ɪn laɪn/"

    def test_recovers_fields_when_json_otherwise_malformed(self, generator: LanguageGenerator) -> None:
        """Even if JSON repair doesn't fully fix the response (e.g. a missing
        comma), individual fields should still be recovered by regex instead
        of falling back to a naive sentence split."""
        response = '{ "definition": "A round fruit" "example": "I eat an apple.", "ipa": "/ˈæpəl/" }'
        definition, example, ipa = generator._parse_json_response(response)
        assert definition == "A round fruit"
        assert example == "I eat an apple."
        assert ipa == "/ˈæpəl/"

    def test_field_regex_fallback_returns_none_without_expected_keys(self, generator: LanguageGenerator) -> None:
        assert generator._parse_json_fields_by_regex("just some plain text, no JSON at all.") is None

    def test_recovers_fields_with_unquoted_keys_and_arrow(self, generator: LanguageGenerator) -> None:
        """Regression test: some weaker local models drop the quotes around
        field names and prepend the word with an arrow instead of emitting a
        JSON object, e.g. `"suspicious" -> definition: "...", example: "..."`.
        The regex fallback should still recover definition/example from this,
        instead of dumping the whole raw response into `definition`."""
        response = (
            '"suspicious" → definition: "Suspicious means having a feeling that something is wrong, '
            'dishonest, or not as it seems.", example: "The detective looked suspiciously at the empty '
            'wallet left on the table, wondering if it was a trap."'
        )
        definition, example, ipa = generator._parse_json_response(response)
        assert definition == "Suspicious means having a feeling that something is wrong, dishonest, or not as it seems."
        assert example == (
            "The detective looked suspiciously at the empty wallet left on the table, wondering if it was a trap."
        )
        assert ipa is None


# ---------------------------------------------------------------------------
# Card generation — one JSON request per text
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_generated_cards(self, generator: LanguageGenerator, base_request: CardRequest) -> None:
        cards = generator.generate(base_request, MagicMock())
        assert len(cards) == 2
        assert all(isinstance(c, GeneratedCard) for c in cards)

    def test_card_fields_all_options(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        """All options enabled: audio_data, audio_definition, audio_example, image_data, transcription."""
        from ankiforge.models import LanguageOptions

        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple\nbanana",
            target_deck="Test Deck",
            language="en",
            language_options=LanguageOptions(include_photo=True),
        )
        cards = generator.generate(request, MagicMock())
        card = cards[0]
        assert card.word == "apple"
        assert card.definition is not None
        assert card.example is not None
        assert card.audio_data is not None
        assert len(card.audio_data) > 44  # valid WAV
        assert card.audio_example is not None
        # Silence — separate file (generated when both def and example audio exist)
        assert card.audio_silence is not None
        assert len(card.audio_silence) > 44
        assert card.image_data == b"fake-image-bytes"
        assert card.transcription == "/\u02c8\u00e6p\u0259l/"
        assert card.note_type == LANGUAGE_NOTE_TYPE_NAME
        # Word is bolded in definition and example
        assert "<b>" in (card.definition or "")
        assert "<b>" in (card.example or "")

    def test_single_text_call_per_word(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        """One text request per word (definition + example + IPA in JSON)."""
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
        )
        generator.generate(request, MagicMock())
        assert mock_client.generate_text.call_count == 1

    def test_three_audio_calls_per_word(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        """3 audio calls: word, definition, example."""
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
        )
        generator.generate(request, MagicMock())
        assert mock_client.generate_audio.call_count == 3

    def test_progress_callback_called(self, generator: LanguageGenerator, base_request: CardRequest) -> None:
        snapshots: list[int] = []

        def capture(progress: GenerationProgress) -> None:
            snapshots.append(progress.completed_cards)

        generator.generate(base_request, capture)
        assert snapshots == [1, 2]

    def test_empty_input_raises(self, generator: LanguageGenerator) -> None:
        request = CardRequest(mode=GenerationMode.LANGUAGE, input_text="", target_deck="Test")
        with pytest.raises(ValueError, match="No words found in input"):
            generator.generate(request, MagicMock())

    def test_cost_from_api_cost(self, mock_client: MagicMock) -> None:
        """Cost is taken from usage.cost if OpenRouter returned it."""
        mock_client.last_cost = 0.05  # API returned cost
        gen = LanguageGenerator(
            client=mock_client,
            text_model="openai/gpt-4o",
            audio_model="openai/tts-1",
            image_model="openai/dall-e-3",
        )
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
        )
        costs: list[float] = []

        def capture(progress: GenerationProgress) -> None:
            costs.append(progress.current_cost)

        gen.generate(request, capture)
        # 4 API calls (text + 3 audio), each $0.05 (include_photo=False by default)
        assert costs[-1] == pytest.approx(0.20, abs=0.01)

    def test_cost_fallback_to_pricing(self, mock_client: MagicMock) -> None:
        """If usage.cost == 0, calculate from tokens x pricing."""
        from ankiforge.models import ModelPricingCache

        mock_client.last_cost = 0.0
        mock_client.last_usage = (100, 50)
        pricing = ModelPricingCache(prompt=0.001, completion=0.002)
        gen = LanguageGenerator(
            client=mock_client,
            text_model="openai/gpt-4o",
            audio_model="openai/tts-1",
            image_model="openai/dall-e-3",
            text_pricing=pricing,
            image_pricing=pricing,
            audio_pricing=pricing,
        )
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
        )
        costs: list[float] = []

        def capture(progress: GenerationProgress) -> None:
            costs.append(progress.current_cost)

        gen.generate(request, capture)
        # Each call: 100*0.001 + 50*0.002 = 0.2; 4 calls = 0.8 (include_photo=False by default)
        assert costs[-1] == pytest.approx(0.8, abs=0.01)


# ---------------------------------------------------------------------------
# LanguageOptions — conditional generation
# ---------------------------------------------------------------------------


class TestLanguageOptions:
    def test_no_photo(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
            language_options=LanguageOptions(include_photo=False),
        )
        cards = generator.generate(request, MagicMock())
        mock_client.generate_image.assert_not_called()
        assert cards[0].image_data is None

    def test_no_audio_word(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
            language_options=LanguageOptions(include_audio_word=False),
        )
        cards = generator.generate(request, MagicMock())
        assert cards[0].audio_data is None
        assert mock_client.generate_audio.call_count == 2  # definition + example

    def test_no_transcription(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
            language_options=LanguageOptions(include_transcription=False),
        )
        cards = generator.generate(request, MagicMock())
        assert cards[0].transcription is None
        # Still 1 text request (JSON includes definition + example)
        assert mock_client.generate_text.call_count == 1

    def test_all_disabled(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
            language_options=LanguageOptions(
                include_photo=False,
                include_audio_word=False,
                include_audio_definition=False,
                include_audio_example=False,
                include_transcription=False,
            ),
        )
        generator.generate(request, MagicMock())
        assert mock_client.generate_text.call_count == 1
        assert mock_client.generate_audio.call_count == 0
        assert mock_client.generate_image.call_count == 0

    def test_detailed_image(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
            language_options=LanguageOptions(include_photo=True, detailed_image=True),
        )
        generator.generate(request, MagicMock())
        image_prompt = mock_client.generate_image.call_args[0][0]
        assert "photorealistic" in image_prompt.lower() or "ultra" in image_prompt.lower()

    def test_voice_passed_to_audio(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
            language_options=LanguageOptions(
                include_photo=False,
                include_audio_word=True,
                include_audio_definition=False,
                include_audio_example=False,
                voice="nova",
            ),
        )
        generator.generate(request, MagicMock())
        # generate_audio called 1 time (word only) with voice="nova"
        assert mock_client.generate_audio.call_count == 1
        _, kwargs = mock_client.generate_audio.call_args
        assert kwargs["voice"] == "nova"

    def test_default_voice_is_alloy(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
            language_options=LanguageOptions(
                include_photo=False,
                include_audio_definition=False,
                include_audio_example=False,
            ),
        )
        generator.generate(request, MagicMock())
        _, kwargs = mock_client.generate_audio.call_args
        assert kwargs["voice"] == "alloy"

    def test_image_size_passed_to_api(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
            language_options=LanguageOptions(include_photo=True, image_size="0.5K"),
        )
        generator.generate(request, MagicMock())
        _, kwargs = mock_client.generate_image.call_args
        assert kwargs["size"] == "0.5K"

    def test_generate_text_called_with_temperature_03(
        self, generator: LanguageGenerator, mock_client: MagicMock
    ) -> None:
        """generate_text is called with temperature=0.3."""
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
            language_options=LanguageOptions(
                include_photo=False,
                include_audio_word=False,
                include_audio_definition=False,
                include_audio_example=False,
            ),
        )
        generator.generate(request, MagicMock())
        call = mock_client.generate_text.call_args
        assert call.kwargs.get("temperature") == 0.3

    def test_image_size_auto_passes_none(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
            language_options=LanguageOptions(include_photo=True, image_size="auto"),
        )
        generator.generate(request, MagicMock())
        _, kwargs = mock_client.generate_image.call_args
        assert kwargs["size"] is None


# ---------------------------------------------------------------------------
# Generation cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_stops_on_cancel(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple\nbanana\ncherry",
            target_deck="Test",
            language="en",
        )

        call_count = 0

        def cancel_on_first(progress: GenerationProgress) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                progress.is_cancelled = True

        cards = generator.generate(request, cancel_on_first)
        assert len(cards) == 1
        assert mock_client.generate_text.call_count == 1


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


class TestPrompt:
    def test_prompt_contains_word(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        request = CardRequest(mode=GenerationMode.LANGUAGE, input_text="apple", target_deck="Test", language="en")
        generator.generate(request, MagicMock())
        prompt = mock_client.generate_text.call_args[0][0]
        assert "apple" in prompt

    def test_prompt_asks_for_json(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        request = CardRequest(mode=GenerationMode.LANGUAGE, input_text="apple", target_deck="Test", language="en")
        generator.generate(request, MagicMock())
        system_prompt = mock_client.generate_text.call_args.kwargs.get("system_prompt", "")
        assert "json" in system_prompt.lower() or "JSON" in system_prompt

    def test_prompt_requires_word_in_definition(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        """Prompt must require using the word in the definition."""
        request = CardRequest(mode=GenerationMode.LANGUAGE, input_text="apple", target_deck="Test", language="en")
        generator.generate(request, MagicMock())
        system_prompt = mock_client.generate_text.call_args.kwargs.get("system_prompt", "")
        assert "starts with the word" in system_prompt.lower() or "STARTS with the word" in system_prompt

    def test_prompt_requires_word_in_example(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        """Prompt must require using the word in the example."""
        request = CardRequest(mode=GenerationMode.LANGUAGE, input_text="apple", target_deck="Test", language="en")
        generator.generate(request, MagicMock())
        system_prompt = mock_client.generate_text.call_args.kwargs.get("system_prompt", "")
        assert "uses the word" in system_prompt.lower() or "USES THE WORD" in system_prompt

    def test_custom_prompt_overrides(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
            custom_prompt="Fun fact about this word",
        )
        generator.generate(request, MagicMock())
        system_prompt = mock_client.generate_text.call_args.kwargs.get("system_prompt", "")
        assert "Fun fact" in system_prompt

    def test_image_prompt_contains_example(self, generator: LanguageGenerator, mock_client: MagicMock) -> None:
        from ankiforge.models import LanguageOptions

        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
            language_options=LanguageOptions(include_photo=True),
        )
        generator.generate(request, MagicMock())
        image_prompt = mock_client.generate_image.call_args[0][0]
        assert "apple" in image_prompt.lower() or "tree" in image_prompt.lower()


# ---------------------------------------------------------------------------
# _generate_silence_wav
# ---------------------------------------------------------------------------


class TestGenerateSilenceWav:
    def test_generates_valid_wav(self) -> None:
        from ankiforge.generators.language import _generate_silence_wav

        result = _generate_silence_wav(seconds=1.0)
        # 44 header + 24000 samples x 2 bytes = 48044
        assert len(result) == 44 + 24000 * 2
        assert result[:4] == b"RIFF"

    def test_silence_is_zeros(self) -> None:
        from ankiforge.generators.language import _generate_silence_wav

        result = _generate_silence_wav(seconds=0.5)
        silence_size = int(24000 * 0.5) * 2
        pcm_data = result[44:]
        assert len(pcm_data) == silence_size
        assert pcm_data == b"\x00" * silence_size

    def test_default_2_seconds(self) -> None:
        from ankiforge.generators.language import _generate_silence_wav

        result = _generate_silence_wav()
        expected_pcm = 24000 * 2 * 2  # 2 sec x 24000 Hz x 2 bytes
        assert len(result) == 44 + expected_pcm


# ---------------------------------------------------------------------------
# _bold_word
# ---------------------------------------------------------------------------


class TestBoldWord:
    def test_bolds_exact_word(self) -> None:
        from ankiforge.generators.language import _bold_word

        result = _bold_word("To fetch is to go and bring something back.", "fetch")
        assert "<b>fetch</b>" in result

    def test_bolds_word_form_with_suffix(self) -> None:
        from ankiforge.generators.language import _bold_word

        result = _bold_word("The cars zoomed along the road.", "zoom")
        assert "<b>zoomed</b>" in result

    def test_preserves_original_case(self) -> None:
        from ankiforge.generators.language import _bold_word

        result = _bold_word("Eloquent means able to express well.", "eloquent")
        assert "<b>Eloquent</b>" in result

    def test_bolds_multiple_occurrences(self) -> None:
        from ankiforge.generators.language import _bold_word

        result = _bold_word("Fetch the ball. She fetched it.", "fetch")
        assert result.count("<b>") == 2

    def test_empty_text(self) -> None:
        from ankiforge.generators.language import _bold_word

        assert _bold_word("", "word") == ""

    def test_empty_word(self) -> None:
        from ankiforge.generators.language import _bold_word

        assert _bold_word("some text", "") == "some text"

    def test_word_not_found(self) -> None:
        from ankiforge.generators.language import _bold_word

        result = _bold_word("The cat sat on the mat.", "fetch")
        assert "<b>" not in result

    def test_does_not_bold_partial_match(self) -> None:
        from ankiforge.generators.language import _bold_word

        result = _bold_word("The sketching was beautiful.", "sketch")
        # "sketching" should match because -ing is a valid suffix
        assert "<b>sketching</b>" in result

    def test_ing_suffix(self) -> None:
        from ankiforge.generators.language import _bold_word

        result = _bold_word("She was fetching the ball.", "fetch")
        assert "<b>fetching</b>" in result
