"""Тесты для MaterialGenerator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ankiforge.anki_bridge.note_types import QA_IMAGE_NOTE_TYPE_NAME, QA_NOTE_TYPE_NAME
from ankiforge.generators.material import MaterialGenerator
from ankiforge.models import CardRequest, GeneratedCard, GenerationMode, GenerationProgress
from ankiforge.openrouter.client import OpenRouterClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_AI_RESPONSE_TWO_PAIRS = (
    "QUESTION: Что такое фотосинтез?\n"
    "ANSWER: Фотосинтез — процесс преобразования световой энергии в химическую энергию.\n\n"
    "QUESTION: Где происходит фотосинтез?\n"
    "ANSWER: Фотосинтез происходит в хлоропластах растительных клеток."
)

_AI_RESPONSE_ONE_PAIR = "QUESTION: Что такое ДНК?\nANSWER: ДНК — молекула, хранящая генетическую информацию."


@pytest.fixture
def mock_client() -> MagicMock:
    """Мок OpenRouterClient."""
    client = MagicMock(spec=OpenRouterClient)
    client.generate_text.return_value = _AI_RESPONSE_TWO_PAIRS
    client.generate_image.return_value = b"fake-image-data"
    return client


@pytest.fixture
def generator(mock_client: MagicMock) -> MaterialGenerator:
    return MaterialGenerator(client=mock_client, text_model="openai/gpt-4o")


@pytest.fixture
def generator_with_images(mock_client: MagicMock) -> MaterialGenerator:
    return MaterialGenerator(
        client=mock_client,
        text_model="openai/gpt-4o",
        image_model="openai/dall-e-3",
    )


@pytest.fixture
def base_request() -> CardRequest:
    return CardRequest(
        mode=GenerationMode.MATERIAL,
        input_text="Фотосинтез — процесс преобразования световой энергии в химическую.",
        target_deck="Test Deck",
    )


@pytest.fixture
def request_with_images() -> CardRequest:
    return CardRequest(
        mode=GenerationMode.MATERIAL,
        input_text="Фотосинтез — процесс преобразования световой энергии в химическую.",
        target_deck="Test Deck",
        include_images=True,
    )


# ---------------------------------------------------------------------------
# Парсинг QA-пар из ответа AI
# ---------------------------------------------------------------------------


class TestParseQAPairs:
    def test_parses_two_pairs(self, generator: MaterialGenerator) -> None:
        pairs = generator._parse_qa_pairs(_AI_RESPONSE_TWO_PAIRS)
        assert len(pairs) == 2
        assert pairs[0] == (
            "Что такое фотосинтез?",
            "Фотосинтез — процесс преобразования световой энергии в химическую энергию.",
        )
        assert pairs[1] == ("Где происходит фотосинтез?", "Фотосинтез происходит в хлоропластах растительных клеток.")

    def test_parses_single_pair(self, generator: MaterialGenerator) -> None:
        pairs = generator._parse_qa_pairs(_AI_RESPONSE_ONE_PAIR)
        assert len(pairs) == 1
        assert pairs[0][0] == "Что такое ДНК?"
        assert pairs[0][1] == "ДНК — молекула, хранящая генетическую информацию."

    def test_empty_response_returns_empty(self, generator: MaterialGenerator) -> None:
        pairs = generator._parse_qa_pairs("")
        assert pairs == []

    def test_whitespace_only_returns_empty(self, generator: MaterialGenerator) -> None:
        pairs = generator._parse_qa_pairs("   \n  \n  ")
        assert pairs == []

    def test_no_markers_returns_empty(self, generator: MaterialGenerator) -> None:
        pairs = generator._parse_qa_pairs("Просто текст без маркеров.")
        assert pairs == []

    def test_strips_whitespace_from_pairs(self, generator: MaterialGenerator) -> None:
        response = "QUESTION:   Вопрос?  \nANSWER:   Ответ.  "
        pairs = generator._parse_qa_pairs(response)
        assert pairs[0] == ("Вопрос?", "Ответ.")

    def test_multiline_answer(self, generator: MaterialGenerator) -> None:
        response = (
            "QUESTION: Что такое митоз?\n"
            "ANSWER: Митоз — деление клетки.\n"
            "В результате образуются две идентичные клетки.\n\n"
            "QUESTION: Что такое мейоз?\n"
            "ANSWER: Мейоз — деление клетки с уменьшением числа хромосом."
        )
        pairs = generator._parse_qa_pairs(response)
        assert len(pairs) == 2
        assert "две идентичные клетки" in pairs[0][1]


# ---------------------------------------------------------------------------
# Чанкинг текста
# ---------------------------------------------------------------------------


class TestChunking:
    def test_short_text_single_chunk(self, generator: MaterialGenerator) -> None:
        chunks = generator._split_into_chunks("Короткий текст.", max_chars=1000)
        assert len(chunks) == 1
        assert chunks[0] == "Короткий текст."

    def test_long_text_multiple_chunks(self, generator: MaterialGenerator) -> None:
        # 3 абзаца, каждый ~50 символов, лимит 80
        text = (
            "Абзац первый содержит информацию.\n\n"
            "Абзац второй содержит информацию.\n\n"
            "Абзац третий содержит информацию."
        )
        chunks = generator._split_into_chunks(text, max_chars=80)
        assert len(chunks) >= 2

    def test_chunks_contain_all_content(self, generator: MaterialGenerator) -> None:
        text = "Факт один.\n\nФакт два.\n\nФакт три."
        chunks = generator._split_into_chunks(text, max_chars=30)
        joined = " ".join(chunks)
        assert "Факт один" in joined
        assert "Факт два" in joined
        assert "Факт три" in joined

    def test_empty_text_returns_empty(self, generator: MaterialGenerator) -> None:
        chunks = generator._split_into_chunks("", max_chars=1000)
        assert chunks == []

    def test_whitespace_only_returns_empty(self, generator: MaterialGenerator) -> None:
        chunks = generator._split_into_chunks("   \n\n   ", max_chars=1000)
        assert chunks == []


# ---------------------------------------------------------------------------
# Генерация карточек — без картинок
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_generated_cards(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        cards = generator.generate(base_request, MagicMock())
        assert len(cards) == 2
        assert all(isinstance(c, GeneratedCard) for c in cards)

    def test_card_fields_qa(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        cards = generator.generate(base_request, MagicMock())

        assert cards[0].word == "Что такое фотосинтез?"
        assert cards[0].answer == "Фотосинтез — процесс преобразования световой энергии в химическую энергию."
        assert cards[0].note_type == QA_NOTE_TYPE_NAME
        assert cards[0].image_data is None

    def test_calls_generate_text(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        generator.generate(base_request, MagicMock())
        assert mock_client.generate_text.call_count >= 1

    def test_prompt_contains_material(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        generator.generate(base_request, MagicMock())
        prompt = mock_client.generate_text.call_args[0][0]
        assert "Фотосинтез" in prompt

    def test_prompt_requests_qa_format(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        generator.generate(base_request, MagicMock())
        prompt = mock_client.generate_text.call_args[0][0]
        assert "QUESTION:" in prompt
        assert "ANSWER:" in prompt

    def test_no_image_generation_without_flag(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        generator.generate(base_request, MagicMock())
        mock_client.generate_image.assert_not_called()

    def test_empty_input_raises(self, generator: MaterialGenerator) -> None:
        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text="",
            target_deck="Test",
        )
        with pytest.raises(ValueError, match="Не найдено текста"):
            generator.generate(request, MagicMock())

    def test_whitespace_only_raises(self, generator: MaterialGenerator) -> None:
        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text="   \n  \n  ",
            target_deck="Test",
        )
        with pytest.raises(ValueError, match="Не найдено текста"):
            generator.generate(request, MagicMock())


# ---------------------------------------------------------------------------
# Генерация карточек — с картинками
# ---------------------------------------------------------------------------


class TestGenerateWithImages:
    def test_includes_image_data(
        self,
        generator_with_images: MaterialGenerator,
        mock_client: MagicMock,
        request_with_images: CardRequest,
    ) -> None:
        cards = generator_with_images.generate(request_with_images, MagicMock())
        assert all(c.image_data == b"fake-image-data" for c in cards)

    def test_note_type_qa_image(
        self,
        generator_with_images: MaterialGenerator,
        mock_client: MagicMock,
        request_with_images: CardRequest,
    ) -> None:
        cards = generator_with_images.generate(request_with_images, MagicMock())
        assert all(c.note_type == QA_IMAGE_NOTE_TYPE_NAME for c in cards)

    def test_calls_generate_image(
        self,
        generator_with_images: MaterialGenerator,
        mock_client: MagicMock,
        request_with_images: CardRequest,
    ) -> None:
        cards = generator_with_images.generate(request_with_images, MagicMock())
        assert mock_client.generate_image.call_count == len(cards)

    def test_without_image_model_no_images(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
        request_with_images: CardRequest,
    ) -> None:
        """include_images=True но image_model не задана — картинки не генерируются."""
        cards = generator.generate(request_with_images, MagicMock())
        assert all(c.image_data is None for c in cards)
        assert all(c.note_type == QA_NOTE_TYPE_NAME for c in cards)
        mock_client.generate_image.assert_not_called()


# ---------------------------------------------------------------------------
# Progress и отмена
# ---------------------------------------------------------------------------


class TestProgress:
    def test_progress_callback_called(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        snapshots: list[int] = []

        def capture_progress(progress: GenerationProgress) -> None:
            snapshots.append(progress.completed_cards)

        generator.generate(base_request, capture_progress)
        # 2 карточки в ответе → progress.completed_cards должен дойти до 2
        assert snapshots[-1] == 2

    def test_total_cards_updated(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        totals: list[int] = []

        def capture_total(progress: GenerationProgress) -> None:
            totals.append(progress.total_cards)

        generator.generate(base_request, capture_total)
        assert totals[-1] == 2


class TestCancellation:
    def test_stops_on_cancel(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
    ) -> None:
        """Генерация останавливается при is_cancelled."""
        # AI возвращает 3 пары
        response = "QUESTION: Q1?\nANSWER: A1.\n\nQUESTION: Q2?\nANSWER: A2.\n\nQUESTION: Q3?\nANSWER: A3."
        mock_client.generate_text.return_value = response

        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text="Длинный учебный материал с тремя фактами.",
            target_deck="Test",
        )

        call_count = 0

        def cancel_on_first(progress: GenerationProgress) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                progress.is_cancelled = True

        cards = generator.generate(request, cancel_on_first)
        assert len(cards) == 1


# ---------------------------------------------------------------------------
# Чанкинг с несколькими вызовами AI
# ---------------------------------------------------------------------------


class TestMultiChunk:
    def test_long_text_multiple_api_calls(
        self,
        mock_client: MagicMock,
    ) -> None:
        """Длинный текст разбивается на чанки, каждый чанк → отдельный вызов AI."""
        mock_client.generate_text.return_value = _AI_RESPONSE_ONE_PAIR

        generator = MaterialGenerator(
            client=mock_client,
            text_model="openai/gpt-4o",
            chunk_size=50,
        )

        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text="Абзац первый с информацией.\n\nАбзац второй с информацией.\n\nАбзац третий с информацией.",
            target_deck="Test",
        )

        cards = generator.generate(request, MagicMock())
        # Несколько чанков → несколько вызовов
        assert mock_client.generate_text.call_count >= 2
        # Каждый чанк даёт 1 пару → всего >= 2 карточек
        assert len(cards) >= 2

    def test_cards_from_all_chunks_combined(
        self,
        mock_client: MagicMock,
    ) -> None:
        """Карточки из всех чанков объединяются в один список."""
        mock_client.generate_text.side_effect = [
            "QUESTION: Q1?\nANSWER: A1.",
            "QUESTION: Q2?\nANSWER: A2.",
        ]

        generator = MaterialGenerator(
            client=mock_client,
            text_model="openai/gpt-4o",
            chunk_size=50,
        )

        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text="Абзац первый с информацией.\n\nАбзац второй с информацией.",
            target_deck="Test",
        )

        cards = generator.generate(request, MagicMock())
        questions = [c.word for c in cards]
        assert "Q1?" in questions
        assert "Q2?" in questions
