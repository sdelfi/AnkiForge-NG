"""Tests for ImageGenerator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ankiforge.anki_bridge.note_types import QA_IMAGE_NOTE_TYPE_NAME
from ankiforge.generators.image import ImageGenerator
from ankiforge.models import CardRequest, GenerationMode, GenerationProgress
from ankiforge.openrouter.client import OpenRouterClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    """Mock OpenRouterClient."""
    client = MagicMock(spec=OpenRouterClient)
    client.generate_text.return_value = "Ответ от AI на вопрос"
    client.generate_image.return_value = b"\x89PNG_fake_image_data"
    client.last_cost = 0.0
    return client


@pytest.fixture
def generator(mock_client: MagicMock) -> ImageGenerator:
    return ImageGenerator(
        client=mock_client,
        text_model="openai/gpt-4o",
        image_model="openai/dall-e-3",
    )


@pytest.fixture
def base_request() -> CardRequest:
    return CardRequest(
        mode=GenerationMode.IMAGE,
        input_text="Что такое фотосинтез?\nКак работает DNS?",
        target_deck="Test Deck",
    )


# ---------------------------------------------------------------------------
# Question parsing
# ---------------------------------------------------------------------------


class TestParseQuestions:
    def test_splits_by_newline(self, generator: ImageGenerator) -> None:
        result = generator._parse_questions("Вопрос 1\nВопрос 2\nВопрос 3")
        assert result == ["Вопрос 1", "Вопрос 2", "Вопрос 3"]

    def test_strips_whitespace(self, generator: ImageGenerator) -> None:
        result = generator._parse_questions("  Вопрос 1  \n  Вопрос 2  ")
        assert result == ["Вопрос 1", "Вопрос 2"]

    def test_skips_empty_lines(self, generator: ImageGenerator) -> None:
        result = generator._parse_questions("Вопрос 1\n\n\nВопрос 2")
        assert result == ["Вопрос 1", "Вопрос 2"]

    def test_empty_input_raises(self, generator: ImageGenerator) -> None:
        with pytest.raises(ValueError, match="No questions found"):
            generator._parse_questions("")

    def test_whitespace_only_raises(self, generator: ImageGenerator) -> None:
        with pytest.raises(ValueError, match="No questions found"):
            generator._parse_questions("   \n  \n  ")


# ---------------------------------------------------------------------------
# Card generation
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_generates_cards_with_text_and_image(
        self,
        generator: ImageGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        progress_cb = MagicMock()
        cards = generator.generate(base_request, progress_cb)

        assert len(cards) == 2
        for card in cards:
            assert card.note_type == QA_IMAGE_NOTE_TYPE_NAME
            assert card.answer == "Ответ от AI на вопрос"
            assert card.image_data == b"\x89PNG_fake_image_data"

    def test_card_word_equals_question(
        self,
        generator: ImageGenerator,
        base_request: CardRequest,
    ) -> None:
        cards = generator.generate(base_request, MagicMock())
        assert cards[0].word == "Что такое фотосинтез?"
        assert cards[1].word == "Как работает DNS?"

    def test_calls_generate_text_per_question(
        self,
        generator: ImageGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        generator.generate(base_request, MagicMock())
        assert mock_client.generate_text.call_count == 2

    def test_calls_generate_image_per_question(
        self,
        generator: ImageGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        generator.generate(base_request, MagicMock())
        assert mock_client.generate_image.call_count == 2

    def test_image_prompt_contains_question(
        self,
        generator: ImageGenerator,
        mock_client: MagicMock,
    ) -> None:
        request = CardRequest(
            mode=GenerationMode.IMAGE,
            input_text="Что такое ДНК?",
            target_deck="Deck",
        )
        generator.generate(request, MagicMock())

        image_call_args = mock_client.generate_image.call_args[0][0]
        assert "ДНК" in image_call_args

    def test_text_model_used_for_text(
        self,
        generator: ImageGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        generator.generate(base_request, MagicMock())
        for c in mock_client.generate_text.call_args_list:
            assert c[0][1] == "openai/gpt-4o"

    def test_image_model_used_for_image(
        self,
        generator: ImageGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        generator.generate(base_request, MagicMock())
        for c in mock_client.generate_image.call_args_list:
            assert c[0][1] == "openai/dall-e-3"


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


class TestProgress:
    def test_callback_called_per_card(
        self,
        generator: ImageGenerator,
        base_request: CardRequest,
    ) -> None:
        progress_cb = MagicMock()
        generator.generate(base_request, progress_cb)
        assert progress_cb.call_count == 2

    def test_progress_increments(
        self,
        generator: ImageGenerator,
        base_request: CardRequest,
    ) -> None:
        progress_values: list[int] = []

        def track_progress(p: GenerationProgress) -> None:
            progress_values.append(p.completed_cards)

        generator.generate(base_request, track_progress)
        assert progress_values == [1, 2]

    def test_progress_total_matches_questions(
        self,
        generator: ImageGenerator,
        base_request: CardRequest,
    ) -> None:
        totals: list[int] = []

        def track(p: GenerationProgress) -> None:
            totals.append(p.total_cards)

        generator.generate(base_request, track)
        assert all(t == 2 for t in totals)


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_stops_on_cancel(
        self,
        generator: ImageGenerator,
        mock_client: MagicMock,
    ) -> None:
        request = CardRequest(
            mode=GenerationMode.IMAGE,
            input_text="Q1\nQ2\nQ3\nQ4\nQ5",
            target_deck="Deck",
        )

        def cancel_after_first(p: GenerationProgress) -> None:
            if p.completed_cards >= 1:
                p.is_cancelled = True

        cards = generator.generate(request, cancel_after_first)
        assert len(cards) == 1
        assert mock_client.generate_text.call_count == 1
        assert mock_client.generate_image.call_count == 1


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_input_raises(self, generator: ImageGenerator) -> None:
        request = CardRequest(
            mode=GenerationMode.IMAGE,
            input_text="",
            target_deck="Deck",
        )
        with pytest.raises(ValueError, match="No questions found"):
            generator.generate(request, MagicMock())


# ---------------------------------------------------------------------------
# image_size
# ---------------------------------------------------------------------------


class TestImageSize:
    def test_image_size_from_constructor(self, mock_client: MagicMock) -> None:
        gen = ImageGenerator(client=mock_client, text_model="m1", image_model="m2", image_size="2K")
        request = CardRequest(
            mode=GenerationMode.IMAGE,
            input_text="Что такое ДНК?",
            target_deck="Deck",
        )
        gen.generate(request, MagicMock())
        call = mock_client.generate_image.call_args
        assert call.kwargs.get("size") == "2K"

    def test_image_size_from_request_overrides(self, mock_client: MagicMock) -> None:
        gen = ImageGenerator(client=mock_client, text_model="m1", image_model="m2", image_size="1K")
        request = CardRequest(
            mode=GenerationMode.IMAGE,
            input_text="Что такое ДНК?",
            target_deck="Deck",
            image_size="4K",
        )
        gen.generate(request, MagicMock())
        call = mock_client.generate_image.call_args
        assert call.kwargs.get("size") == "4K"

    def test_default_image_size_auto(self, generator: ImageGenerator, mock_client: MagicMock) -> None:
        request = CardRequest(
            mode=GenerationMode.IMAGE,
            input_text="Что такое ДНК?",
            target_deck="Deck",
        )
        generator.generate(request, MagicMock())
        call = mock_client.generate_image.call_args
        assert call.kwargs.get("size") == "auto"


# ---------------------------------------------------------------------------
# Custom prompt
# ---------------------------------------------------------------------------


class TestCostTracking:
    def test_current_cost_accumulated(
        self,
        mock_client: MagicMock,
    ) -> None:
        """current_cost accumulates from client.last_cost after each call."""
        mock_client.last_cost = 0.02
        generator = ImageGenerator(client=mock_client, text_model="m1", image_model="m2")
        request = CardRequest(
            mode=GenerationMode.IMAGE,
            input_text="Q1\nQ2",
            target_deck="Test",
        )
        costs: list[float] = []

        def capture(progress: GenerationProgress) -> None:
            costs.append(progress.current_cost)

        generator.generate(request, capture)
        # 2 cards x (generate_text + generate_image) x $0.02 = $0.08
        assert len(costs) == 2
        assert costs[-1] == pytest.approx(0.08, abs=0.001)


class TestTemperature:
    def test_generate_text_called_with_temperature_03(
        self,
        generator: ImageGenerator,
        mock_client: MagicMock,
    ) -> None:
        """generate_text is called with temperature=0.3."""
        request = CardRequest(
            mode=GenerationMode.IMAGE,
            input_text="Что такое ДНК?",
            target_deck="Deck",
        )
        generator.generate(request, MagicMock())
        call = mock_client.generate_text.call_args
        assert call.kwargs.get("temperature") == 0.3


class TestCustomPrompt:
    def test_custom_prompt_replaces_default(self, generator: ImageGenerator, mock_client: MagicMock) -> None:
        request = CardRequest(
            mode=GenerationMode.IMAGE,
            input_text="Что такое ДНК?",
            target_deck="Deck",
            custom_prompt="Be very brief.",
        )
        generator.generate(request, MagicMock())
        system_prompt = mock_client.generate_text.call_args.kwargs.get("system_prompt", "")
        assert "Be very brief." in system_prompt
        assert "concise" not in system_prompt.lower()
