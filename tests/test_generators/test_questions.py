"""Tests for QuestionsGenerator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ankiforge.anki_bridge.note_types import QA_NOTE_TYPE_NAME
from ankiforge.generators.questions import QuestionsGenerator
from ankiforge.models import CardRequest, GeneratedCard, GenerationMode, GenerationProgress
from ankiforge.openrouter.client import OpenRouterClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    """Mock OpenRouterClient."""
    client = MagicMock(spec=OpenRouterClient)
    client.generate_text.return_value = "Ответ от AI"
    client.last_cost = 0.0
    return client


@pytest.fixture
def generator(mock_client: MagicMock) -> QuestionsGenerator:
    return QuestionsGenerator(client=mock_client, model="openai/gpt-4o")


@pytest.fixture
def base_request() -> CardRequest:
    return CardRequest(
        mode=GenerationMode.QUESTIONS,
        input_text="Что такое Python?\nЧем отличается list от tuple?",
        target_deck="Test Deck",
    )


# ---------------------------------------------------------------------------
# Question parsing
# ---------------------------------------------------------------------------


class TestParseQuestions:
    def test_splits_by_newline(self, generator: QuestionsGenerator) -> None:
        result = generator._parse_questions("Вопрос 1\nВопрос 2\nВопрос 3")
        assert result == ["Вопрос 1", "Вопрос 2", "Вопрос 3"]

    def test_strips_whitespace(self, generator: QuestionsGenerator) -> None:
        result = generator._parse_questions("  Вопрос 1  \n  Вопрос 2  ")
        assert result == ["Вопрос 1", "Вопрос 2"]

    def test_skips_empty_lines(self, generator: QuestionsGenerator) -> None:
        result = generator._parse_questions("Вопрос 1\n\n\nВопрос 2\n")
        assert result == ["Вопрос 1", "Вопрос 2"]

    def test_single_question(self, generator: QuestionsGenerator) -> None:
        result = generator._parse_questions("Один вопрос")
        assert result == ["Один вопрос"]

    def test_empty_input_raises(self, generator: QuestionsGenerator) -> None:
        with pytest.raises(ValueError, match="No questions found"):
            generator._parse_questions("")

    def test_only_whitespace_raises(self, generator: QuestionsGenerator) -> None:
        with pytest.raises(ValueError, match="No questions found"):
            generator._parse_questions("   \n  \n  ")


# ---------------------------------------------------------------------------
# Card generation
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_generated_cards(
        self,
        generator: QuestionsGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        mock_client.generate_text.side_effect = ["Ответ 1", "Ответ 2"]
        callback = MagicMock()

        cards = generator.generate(base_request, callback)

        assert len(cards) == 2
        assert all(isinstance(c, GeneratedCard) for c in cards)

    def test_card_fields(
        self,
        generator: QuestionsGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        mock_client.generate_text.side_effect = ["Ответ 1", "Ответ 2"]

        cards = generator.generate(base_request, MagicMock())

        assert cards[0].word == "Что такое Python?"
        assert cards[0].answer == "Ответ 1"
        assert cards[0].note_type == QA_NOTE_TYPE_NAME

        assert cards[1].word == "Чем отличается list от tuple?"
        assert cards[1].answer == "Ответ 2"

    def test_calls_generate_text_for_each_question(
        self,
        generator: QuestionsGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        mock_client.generate_text.side_effect = ["Ответ 1", "Ответ 2"]

        generator.generate(base_request, MagicMock())

        assert mock_client.generate_text.call_count == 2
        # Verify that the question is included in the prompt
        first_prompt = mock_client.generate_text.call_args_list[0][0][0]
        assert "Что такое Python?" in first_prompt

    def test_progress_callback_called(
        self,
        generator: QuestionsGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        mock_client.generate_text.side_effect = ["Ответ 1", "Ответ 2"]
        snapshots: list[int] = []

        def capture_progress(progress: GenerationProgress) -> None:
            snapshots.append(progress.completed_cards)

        generator.generate(base_request, capture_progress)

        assert snapshots == [1, 2]

    def test_uses_model_from_constructor(
        self,
        generator: QuestionsGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        mock_client.generate_text.return_value = "Ответ"
        base_request.input_text = "Один вопрос"

        generator.generate(base_request, MagicMock())

        _, kwargs = mock_client.generate_text.call_args
        assert kwargs.get("model") == "openai/gpt-4o" or mock_client.generate_text.call_args[0][1] == "openai/gpt-4o"

    def test_empty_input_raises(
        self,
        generator: QuestionsGenerator,
    ) -> None:
        request = CardRequest(
            mode=GenerationMode.QUESTIONS,
            input_text="",
            target_deck="Test",
        )
        with pytest.raises(ValueError, match="No questions found"):
            generator.generate(request, MagicMock())


# ---------------------------------------------------------------------------
# Generation cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_stops_on_cancel(
        self,
        generator: QuestionsGenerator,
        mock_client: MagicMock,
    ) -> None:
        """Generation stops if progress.is_cancelled = True."""
        request = CardRequest(
            mode=GenerationMode.QUESTIONS,
            input_text="Вопрос 1\nВопрос 2\nВопрос 3",
            target_deck="Test",
        )
        mock_client.generate_text.return_value = "Ответ"

        call_count = 0

        def cancel_on_second(progress: GenerationProgress) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                progress.is_cancelled = True

        cards = generator.generate(request, cancel_on_second)

        # Only 1 card — generation cancelled after first callback
        assert len(cards) == 1
        assert mock_client.generate_text.call_count == 1


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


class TestPrompt:
    def test_prompt_contains_question(
        self,
        generator: QuestionsGenerator,
        mock_client: MagicMock,
    ) -> None:
        request = CardRequest(
            mode=GenerationMode.QUESTIONS,
            input_text="Что такое рекурсия?",
            target_deck="Test",
        )
        mock_client.generate_text.return_value = "Ответ"

        generator.generate(request, MagicMock())

        prompt = mock_client.generate_text.call_args[0][0]
        assert "Что такое рекурсия?" in prompt

    def test_prompt_asks_for_concise_answer(
        self,
        generator: QuestionsGenerator,
        mock_client: MagicMock,
    ) -> None:
        request = CardRequest(
            mode=GenerationMode.QUESTIONS,
            input_text="Что такое рекурсия?",
            target_deck="Test",
        )
        mock_client.generate_text.return_value = "Ответ"

        generator.generate(request, MagicMock())

        prompt = mock_client.generate_text.call_args[0][0]
        # Prompt should instruct AI to give a compact but informative answer
        assert any(word in prompt.lower() for word in ["concise", "compact", "кратк", "компактн"])


# ---------------------------------------------------------------------------
# Custom prompt
# ---------------------------------------------------------------------------


class TestCostTracking:
    def test_current_cost_accumulated(
        self,
        mock_client: MagicMock,
    ) -> None:
        """current_cost accumulates from client.last_cost after each call."""
        mock_client.generate_text.return_value = "Ответ"
        mock_client.last_cost = 0.01
        generator = QuestionsGenerator(client=mock_client, model="m1")
        request = CardRequest(
            mode=GenerationMode.QUESTIONS,
            input_text="Q1\nQ2\nQ3",
            target_deck="Test",
        )
        costs: list[float] = []

        def capture(progress: GenerationProgress) -> None:
            costs.append(progress.current_cost)

        generator.generate(request, capture)
        # 3 generate_text calls x $0.01 = $0.03
        assert len(costs) == 3
        assert costs[-1] == pytest.approx(0.03, abs=0.001)


class TestTemperature:
    def test_generate_text_called_with_temperature_03(
        self,
        generator: QuestionsGenerator,
        mock_client: MagicMock,
    ) -> None:
        """generate_text is called with temperature=0.3."""
        request = CardRequest(
            mode=GenerationMode.QUESTIONS,
            input_text="Что такое Python?",
            target_deck="Test",
        )
        mock_client.generate_text.return_value = "Ответ"
        generator.generate(request, MagicMock())
        call = mock_client.generate_text.call_args
        assert call.kwargs.get("temperature") == 0.3


class TestCustomPrompt:
    def test_custom_prompt_replaces_default(
        self,
        generator: QuestionsGenerator,
        mock_client: MagicMock,
    ) -> None:
        request = CardRequest(
            mode=GenerationMode.QUESTIONS,
            input_text="Что такое рекурсия?",
            target_deck="Test",
            custom_prompt="Answer in one word only.",
        )
        mock_client.generate_text.return_value = "Ответ"

        generator.generate(request, MagicMock())

        prompt = mock_client.generate_text.call_args[0][0]
        assert "Answer in one word only." in prompt
        assert "concise" not in prompt.lower()

    def test_no_custom_prompt_uses_default(
        self,
        generator: QuestionsGenerator,
        mock_client: MagicMock,
    ) -> None:
        request = CardRequest(
            mode=GenerationMode.QUESTIONS,
            input_text="Что такое Python?",
            target_deck="Test",
        )
        mock_client.generate_text.return_value = "Ответ"

        generator.generate(request, MagicMock())

        prompt = mock_client.generate_text.call_args[0][0]
        assert any(word in prompt.lower() for word in ["concise", "compact"])
