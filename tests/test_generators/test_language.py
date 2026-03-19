"""Тесты для LanguageGenerator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ankiforge.anki_bridge.note_types import LANGUAGE_NOTE_TYPE_NAME
from ankiforge.generators.language import LanguageGenerator
from ankiforge.models import CardRequest, GeneratedCard, GenerationMode, GenerationProgress
from ankiforge.openrouter.client import OpenRouterClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    """Мок OpenRouterClient."""
    client = MagicMock(spec=OpenRouterClient)
    client.generate_text.return_value = (
        "DEFINITION: A large, typically red fruit\nEXAMPLE: She ate a delicious apple for lunch."
    )
    client.generate_audio.return_value = b"fake-audio-bytes"
    client.generate_image.return_value = b"fake-image-bytes"
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
# Парсинг слов
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
        with pytest.raises(ValueError, match="Не найдено слов"):
            generator._parse_words("")

    def test_only_whitespace_raises(self, generator: LanguageGenerator) -> None:
        with pytest.raises(ValueError, match="Не найдено слов"):
            generator._parse_words("   \n  \n  ")

    def test_mixed_newlines_and_commas(self, generator: LanguageGenerator) -> None:
        """Если есть переводы строк — разделяем по ним, запятые внутри строки не разделяют."""
        result = generator._parse_words("apple, pear\nbanana")
        assert result == ["apple, pear", "banana"]

    def test_comma_only_single_line(self, generator: LanguageGenerator) -> None:
        """Если одна строка — разделяем по запятым."""
        result = generator._parse_words("apple, banana, cherry")
        assert result == ["apple", "banana", "cherry"]


# ---------------------------------------------------------------------------
# Парсинг ответа AI
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_parses_definition_and_example(self, generator: LanguageGenerator) -> None:
        response = "DEFINITION: A round fruit with red skin\nEXAMPLE: I eat an apple every day."
        definition, example = generator._parse_response(response)
        assert definition == "A round fruit with red skin"
        assert example == "I eat an apple every day."

    def test_multiline_definition(self, generator: LanguageGenerator) -> None:
        response = "DEFINITION: A round fruit\nwith red or green skin\nEXAMPLE: She picked an apple from the tree."
        definition, example = generator._parse_response(response)
        assert "round fruit" in definition
        assert example == "She picked an apple from the tree."

    def test_fallback_when_no_markers(self, generator: LanguageGenerator) -> None:
        """Если AI не следует формату — первое предложение = definition, остальное = example."""
        response = "A round fruit. I eat an apple every day."
        definition, example = generator._parse_response(response)
        assert definition != ""
        assert example != ""

    def test_empty_response(self, generator: LanguageGenerator) -> None:
        definition, example = generator._parse_response("")
        assert definition == ""
        assert example == ""


# ---------------------------------------------------------------------------
# Генерация карточек
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_generated_cards(
        self,
        generator: LanguageGenerator,
        base_request: CardRequest,
    ) -> None:
        callback = MagicMock()
        cards = generator.generate(base_request, callback)

        assert len(cards) == 2
        assert all(isinstance(c, GeneratedCard) for c in cards)

    def test_card_fields(
        self,
        generator: LanguageGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        cards = generator.generate(base_request, MagicMock())

        card = cards[0]
        assert card.word == "apple"
        assert card.definition is not None
        assert card.example is not None
        assert card.audio_data == b"fake-audio-bytes"
        assert card.image_data == b"fake-image-bytes"
        assert card.note_type == LANGUAGE_NOTE_TYPE_NAME

    def test_calls_all_three_apis(
        self,
        generator: LanguageGenerator,
        mock_client: MagicMock,
    ) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
        )
        generator.generate(request, MagicMock())

        assert mock_client.generate_text.call_count == 1
        assert mock_client.generate_audio.call_count == 1
        assert mock_client.generate_image.call_count == 1

    def test_uses_correct_models(
        self,
        generator: LanguageGenerator,
        mock_client: MagicMock,
    ) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
        )
        generator.generate(request, MagicMock())

        # text model
        assert mock_client.generate_text.call_args[0][1] == "openai/gpt-4o"
        # audio model
        assert mock_client.generate_audio.call_args[0][1] == "openai/tts-1"
        # image model
        assert mock_client.generate_image.call_args[0][1] == "openai/dall-e-3"

    def test_progress_callback_called(
        self,
        generator: LanguageGenerator,
        base_request: CardRequest,
    ) -> None:
        snapshots: list[int] = []

        def capture_progress(progress: GenerationProgress) -> None:
            snapshots.append(progress.completed_cards)

        generator.generate(base_request, capture_progress)

        assert snapshots == [1, 2]

    def test_empty_input_raises(self, generator: LanguageGenerator) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="",
            target_deck="Test",
        )
        with pytest.raises(ValueError, match="Не найдено слов"):
            generator.generate(request, MagicMock())


# ---------------------------------------------------------------------------
# Отмена генерации
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_stops_on_cancel(
        self,
        generator: LanguageGenerator,
        mock_client: MagicMock,
    ) -> None:
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
# Промпт
# ---------------------------------------------------------------------------


class TestPrompt:
    def test_prompt_contains_word(
        self,
        generator: LanguageGenerator,
        mock_client: MagicMock,
    ) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
        )
        generator.generate(request, MagicMock())

        prompt = mock_client.generate_text.call_args[0][0]
        assert "apple" in prompt

    def test_prompt_contains_language(
        self,
        generator: LanguageGenerator,
        mock_client: MagicMock,
    ) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
        )
        generator.generate(request, MagicMock())

        prompt = mock_client.generate_text.call_args[0][0]
        assert "en" in prompt.lower() or "english" in prompt.lower()

    def test_prompt_asks_for_definition_and_example(
        self,
        generator: LanguageGenerator,
        mock_client: MagicMock,
    ) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
        )
        generator.generate(request, MagicMock())

        prompt = mock_client.generate_text.call_args[0][0]
        assert "DEFINITION" in prompt
        assert "EXAMPLE" in prompt

    def test_custom_prompt_overrides_default(
        self,
        generator: LanguageGenerator,
        mock_client: MagicMock,
    ) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
            custom_prompt="Give me a fun fact about this word",
        )
        generator.generate(request, MagicMock())

        prompt = mock_client.generate_text.call_args[0][0]
        assert "fun fact" in prompt

    def test_image_prompt_contains_word(
        self,
        generator: LanguageGenerator,
        mock_client: MagicMock,
    ) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
        )
        generator.generate(request, MagicMock())

        image_prompt = mock_client.generate_image.call_args[0][0]
        assert "apple" in image_prompt.lower()

    def test_audio_receives_word(
        self,
        generator: LanguageGenerator,
        mock_client: MagicMock,
    ) -> None:
        request = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple",
            target_deck="Test",
            language="en",
        )
        generator.generate(request, MagicMock())

        audio_text = mock_client.generate_audio.call_args[0][0]
        assert "apple" in audio_text
