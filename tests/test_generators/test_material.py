"""Тесты для MaterialGenerator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ankiforge.anki_bridge.note_types import QA_IMAGE_NOTE_TYPE_NAME, QA_NOTE_TYPE_NAME
from ankiforge.generators.material import MaterialGenerator
from ankiforge.models import (
    AnswerDetail,
    CardRequest,
    GeneratedCard,
    GenerationMode,
    GenerationProgress,
    MaterialOptions,
)
from ankiforge.openrouter.client import OpenRouterClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_AI_EXTRACT_RESPONSE = "1. Фотосинтез — процесс преобразования световой энергии.\n2. Происходит в хлоропластах."

_AI_GENERATE_RESPONSE = (
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
    # По умолчанию: extract → факты, generate → QA пары
    client.generate_text.side_effect = [_AI_EXTRACT_RESPONSE, _AI_GENERATE_RESPONSE]
    client.generate_image.return_value = b"fake-image-data"
    client.last_cost = 0.0
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
        input_text="Фотосинтез — процесс преобразования световой энергии в химическую. Он происходит в хлоропластах.",
        target_deck="Test Deck",
    )


@pytest.fixture
def request_with_images() -> CardRequest:
    return CardRequest(
        mode=GenerationMode.MATERIAL,
        input_text="Фотосинтез — процесс преобразования световой энергии в химическую. Он происходит в хлоропластах.",
        target_deck="Test Deck",
        include_images=True,
    )


# ---------------------------------------------------------------------------
# Парсинг QA-пар из ответа AI
# ---------------------------------------------------------------------------


class TestParseQAPairs:
    def test_parses_two_pairs(self, generator: MaterialGenerator) -> None:
        pairs = generator._parse_qa_pairs(_AI_GENERATE_RESPONSE)
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
# Подготовка абзацев
# ---------------------------------------------------------------------------


class TestPrepareParapgraphs:
    def test_single_paragraph(self, generator: MaterialGenerator) -> None:
        result = generator._prepare_paragraphs("Один большой абзац с достаточным количеством текста для генерации.")
        assert len(result) == 1

    def test_multiple_paragraphs_merged_when_small(self, generator: MaterialGenerator) -> None:
        """Небольшие соседние абзацы мержатся в один чанк (до 3000 символов)."""
        text = (
            "Абзац первый содержит достаточно информации для генерации карточки. "
            "Фотосинтез — процесс преобразования световой энергии в химическую энергию. "
            "Он происходит в хлоропластах растительных клеток и играет ключевую роль в экосистемах.\n\n"
            "Абзац второй тоже содержит достаточно информации для генерации карточки. "
            "ДНК — молекула, хранящая генетическую информацию организма. "
            "Она состоит из двух полинуклеотидных цепей, образующих двойную спираль.\n\n"
            "Абзац третий содержит информацию для генерации карточек из учебного материала. "
            "РНК выполняет функцию передачи генетической информации от ДНК к рибосомам."
        )
        result = generator._prepare_paragraphs(text)
        # ~600 символов суммарно → всё мержится в 1 чанк
        assert len(result) == 1

    def test_large_text_splits_into_multiple_chunks(self, generator: MaterialGenerator) -> None:
        """Текст >3000 символов разбивается на несколько чанков."""
        para = "Абзац с достаточной длиной. " * 30  # ~800 символов
        text = f"{para}\n\n{para}\n\n{para}\n\n{para}\n\n{para}"  # ~4000+ символов
        result = generator._prepare_paragraphs(text)
        assert len(result) >= 2

    def test_topic_headings_split_into_sections(self, generator: MaterialGenerator) -> None:
        """Заголовки-вопросы создают отдельные секции даже в коротком тексте."""
        text = (
            "Что такое генераторная функция\n\n"
            "Генераторная функция - функция, в теле которой встречается yield.\n\n"
            "Что делает yield\n\n"
            "yield замораживает состояние функции-генератора.\n\n"
            "В чем отличие [x for x in y] от (x for x in y)\n\n"
            "Первое выражение возвращает список, второе – генератор."
        )
        result = generator._prepare_paragraphs(text)
        assert len(result) == 3
        assert "Что такое" in result[0]
        assert "Что делает" in result[1]
        assert "В чем отличие" in result[2]

    def test_heading_merged_with_content(self, generator: MaterialGenerator) -> None:
        """Заголовок мержится с последующим контентом в одну секцию."""
        text = (
            "Что такое итератор\n\n"
            "Итератор — это объект, который представляет поток данных. "
            "Повторяемый вызов метода __next__() возвращает последующие элементы."
        )
        result = generator._prepare_paragraphs(text)
        assert len(result) == 1
        assert "Что такое итератор" in result[0]
        assert "__next__" in result[0]

    def test_non_heading_paragraphs_merge_normally(self, generator: MaterialGenerator) -> None:
        """Абзацы без заголовков мержатся по обычным правилам."""
        text = (
            "Фотосинтез — процесс преобразования световой энергии в химическую.\n\n"
            "Он происходит в хлоропластах растительных клеток.\n\n"
            "Результатом является глюкоза и кислород."
        )
        result = generator._prepare_paragraphs(text)
        # Всё <3000, нет заголовков → 1 чанк
        assert len(result) == 1

    def test_short_paragraphs_filtered(self, generator: MaterialGenerator) -> None:
        text = "Ок\n\nДостаточно длинный абзац для генерации карточек из материала."
        result = generator._prepare_paragraphs(text)
        assert len(result) == 1
        assert "Достаточно длинный" in result[0]

    def test_empty_text_returns_original(self, generator: MaterialGenerator) -> None:
        result = generator._prepare_paragraphs("Текст")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Map-reduce генерация — без картинок
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

    def test_two_phase_calls(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        """Map-reduce: 2 вызова generate_text — extract + generate."""
        generator.generate(base_request, MagicMock())
        assert mock_client.generate_text.call_count == 2

    def test_extract_prompt_contains_paragraph(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        generator.generate(base_request, MagicMock())
        extract_prompt = mock_client.generate_text.call_args_list[0][0][0]
        assert "Фотосинтез" in extract_prompt

    def test_generate_prompt_contains_facts(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        generator.generate(base_request, MagicMock())
        generate_prompt = mock_client.generate_text.call_args_list[1][0][0]
        assert "QUESTION:" in generate_prompt
        assert "ANSWER:" in generate_prompt

    def test_extract_uses_temperature_02(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        """Extract phase вызывается с temperature=0.2."""
        generator.generate(base_request, MagicMock())
        extract_call = mock_client.generate_text.call_args_list[0]
        assert extract_call.kwargs.get("temperature") == 0.2

    def test_generate_uses_temperature_03(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        """Generate phase вызывается с temperature=0.3."""
        generator.generate(base_request, MagicMock())
        generate_call = mock_client.generate_text.call_args_list[1]
        assert generate_call.kwargs.get("temperature") == 0.3

    def test_extract_prompt_contains_type_markers(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        """Extract prompt содержит маркеры типов знаний."""
        generator.generate(base_request, MagicMock())
        extract_prompt = mock_client.generate_text.call_args_list[0][0][0]
        assert "DEFINITION" in extract_prompt
        assert "FORMULA" in extract_prompt
        assert "PROCEDURE" in extract_prompt
        assert "RELATION" in extract_prompt
        assert "INSIGHT" in extract_prompt

    def test_generate_prompt_contains_type_instructions(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        """Generate prompt содержит type-specific инструкции."""
        generator.generate(base_request, MagicMock())
        generate_prompt = mock_client.generate_text.call_args_list[1][0][0]
        assert "DEFINITION" in generate_prompt
        assert "FORMULA" in generate_prompt
        assert "What is X?" in generate_prompt or "Define X" in generate_prompt

    def test_extract_prompt_contains_code_type(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        """Extract prompt содержит тип CODE для кода."""
        generator.generate(base_request, MagicMock())
        extract_prompt = mock_client.generate_text.call_args_list[0][0][0]
        assert "CODE" in extract_prompt

    def test_generate_prompt_includes_code_instructions(
        self,
        generator: MaterialGenerator,
        mock_client: MagicMock,
        base_request: CardRequest,
    ) -> None:
        """Generate prompt инструктирует включать code snippets."""
        generator.generate(base_request, MagicMock())
        generate_prompt = mock_client.generate_text.call_args_list[1][0][0]
        assert "code" in generate_prompt.lower()
        assert "```" in generate_prompt  # markdown code block в примерах

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
        with pytest.raises(ValueError, match="No text found in input"):
            generator.generate(request, MagicMock())

    def test_whitespace_only_raises(self, generator: MaterialGenerator) -> None:
        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text="   \n  \n  ",
            target_deck="Test",
        )
        with pytest.raises(ValueError, match="No text found in input"):
            generator.generate(request, MagicMock())


# ---------------------------------------------------------------------------
# max_cards_per_paragraph
# ---------------------------------------------------------------------------


class TestCostTracking:
    def test_current_cost_accumulated(self, mock_client: MagicMock) -> None:
        """current_cost накапливается из client.last_cost после каждого вызова."""
        mock_client.generate_text.side_effect = [_AI_EXTRACT_RESPONSE, _AI_GENERATE_RESPONSE]
        mock_client.last_cost = 0.005
        generator = MaterialGenerator(client=mock_client, text_model="m1")
        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text="Достаточно длинный текст для генерации карточек из материала.",
            target_deck="Test",
        )
        costs: list[float] = []

        def capture(progress: GenerationProgress) -> None:
            costs.append(progress.current_cost)

        generator.generate(request, capture)
        # extract + generate = 2 вызова × $0.005 = $0.01, + 2 карточки callbacks
        assert costs[-1] == pytest.approx(0.01, abs=0.001)


class TestMaxCards:
    def test_max_cards_passed_to_extract_prompt(self, mock_client: MagicMock) -> None:
        generator = MaterialGenerator(client=mock_client, text_model="m1")
        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text="Достаточно длинный текст для генерации карточек из материала.",
            target_deck="Test",
            material_options=MaterialOptions(max_cards_per_paragraph=5),
        )
        generator.generate(request, MagicMock())
        extract_prompt = mock_client.generate_text.call_args_list[0][0][0]
        assert "5" in extract_prompt

    def test_max_cards_clamped_to_range(self, mock_client: MagicMock) -> None:
        """Значения за пределами 1-5 обрезаются."""
        generator = MaterialGenerator(client=mock_client, text_model="m1")
        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text="Достаточно длинный текст для генерации карточек из материала.",
            target_deck="Test",
            material_options=MaterialOptions(max_cards_per_paragraph=10),
        )
        generator.generate(request, MagicMock())
        extract_prompt = mock_client.generate_text.call_args_list[0][0][0]
        # Clamped to 5
        assert "5" in extract_prompt

    def test_max_cards_min_clamped(self, mock_client: MagicMock) -> None:
        """Значение 0 обрезается до 1."""
        generator = MaterialGenerator(client=mock_client, text_model="m1")
        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text="Достаточно длинный текст для генерации карточек из материала.",
            target_deck="Test",
            material_options=MaterialOptions(max_cards_per_paragraph=0),
        )
        generator.generate(request, MagicMock())
        extract_prompt = mock_client.generate_text.call_args_list[0][0][0]
        assert "1" in extract_prompt


# ---------------------------------------------------------------------------
# Генерация с картинками
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

    def test_image_size_passed_to_client(self, mock_client: MagicMock) -> None:
        """image_size передаётся в generate_image."""
        mock_client.generate_text.side_effect = [_AI_EXTRACT_RESPONSE, _AI_GENERATE_RESPONSE]
        generator = MaterialGenerator(client=mock_client, text_model="m1", image_model="img-model", image_size="2K")
        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text="Достаточно длинный текст для генерации карточек из материала с картинками.",
            target_deck="Test",
            include_images=True,
        )
        generator.generate(request, MagicMock())
        for call in mock_client.generate_image.call_args_list:
            assert call.kwargs.get("size") == "2K"


# ---------------------------------------------------------------------------
# Custom prompt — single-pass
# ---------------------------------------------------------------------------


class TestCustomPrompt:
    def test_custom_prompt_single_pass(self, mock_client: MagicMock) -> None:
        """Custom prompt использует single-pass (один вызов вместо двух)."""
        mock_client.generate_text.side_effect = [_AI_GENERATE_RESPONSE]
        generator = MaterialGenerator(client=mock_client, text_model="m1")
        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text="Текст материала для генерации.",
            target_deck="Test",
            custom_prompt="My custom prompt with QUESTION: and ANSWER: format",
        )
        cards = generator.generate(request, MagicMock())
        # Single-pass: только 1 вызов generate_text
        assert mock_client.generate_text.call_count == 1
        prompt = mock_client.generate_text.call_args[0][0]
        assert "My custom prompt" in prompt
        assert len(cards) == 2

    def test_custom_prompt_no_temperature(self, mock_client: MagicMock) -> None:
        """Custom prompt (single-pass) не передаёт temperature."""
        mock_client.generate_text.side_effect = [_AI_GENERATE_RESPONSE]
        generator = MaterialGenerator(client=mock_client, text_model="m1")
        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text="Текст.",
            target_deck="Test",
            custom_prompt="Custom",
        )
        generator.generate(request, MagicMock())
        call = mock_client.generate_text.call_args
        assert call.kwargs.get("temperature") is None

    def test_custom_prompt_text_in_prompt(self, mock_client: MagicMock) -> None:
        mock_client.generate_text.side_effect = [_AI_GENERATE_RESPONSE]
        generator = MaterialGenerator(client=mock_client, text_model="m1")
        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text="Мой учебный материал.",
            target_deck="Test",
            custom_prompt="Generate cards",
        )
        generator.generate(request, MagicMock())
        prompt = mock_client.generate_text.call_args[0][0]
        assert "Мой учебный материал" in prompt


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
        # 2 карточки → completed_cards должен дойти до 2
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
        mock_client: MagicMock,
    ) -> None:
        """Генерация останавливается при is_cancelled (между чанками)."""
        # Два больших абзаца (>1600 символов каждый → не мержатся), cancel после первого
        mock_client.generate_text.side_effect = [
            "1. Факт 1",
            _AI_RESPONSE_ONE_PAIR,
            # Второй чанк не должен обрабатываться
        ]
        generator = MaterialGenerator(client=mock_client, text_model="m1")

        para1 = "Фотосинтез — процесс преобразования. " * 50  # ~1850 символов
        para2 = "ДНК хранит генетическую информацию. " * 50  # ~1850 символов

        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text=f"{para1}\n\n{para2}",
            target_deck="Test",
        )

        def cancel_after_first_chunk(progress: GenerationProgress) -> None:
            if progress.completed_cards >= 1:
                progress.is_cancelled = True

        cards = generator.generate(request, cancel_after_first_chunk)
        assert len(cards) == 1
        # Только 2 вызова AI (extract + generate для первого чанка)
        assert mock_client.generate_text.call_count == 2


# ---------------------------------------------------------------------------
# Несколько абзацев → несколько extract/generate циклов
# ---------------------------------------------------------------------------


class TestMultiParagraph:
    def test_small_paragraphs_merged_into_one_chunk(
        self,
        mock_client: MagicMock,
    ) -> None:
        """Два небольших абзаца (~600 символов) мержатся → 1 чанк, 2 вызова AI."""
        mock_client.generate_text.side_effect = [
            "1. Факт 1",
            _AI_RESPONSE_ONE_PAIR,
        ]

        generator = MaterialGenerator(client=mock_client, text_model="m1")

        para1 = (
            "Фотосинтез — процесс преобразования световой энергии в химическую энергию. "
            "Он происходит в хлоропластах растительных клеток и является основным источником "
            "органических веществ на Земле. Без фотосинтеза жизнь на планете была бы невозможна."
        )
        para2 = (
            "ДНК — молекула, хранящая генетическую информацию организма. Она состоит из двух "
            "полинуклеотидных цепей, образующих двойную спираль. ДНК содержится в ядре клетки, "
            "передаётся по наследству и определяет все признаки живого организма от рождения."
        )

        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text=f"{para1}\n\n{para2}",
            target_deck="Test",
        )

        cards = generator.generate(request, MagicMock())
        # Мерж в 1 чанк → 2 вызова AI (extract + generate)
        assert mock_client.generate_text.call_count == 2
        assert len(cards) == 1

    def test_large_paragraphs_stay_separate(
        self,
        mock_client: MagicMock,
    ) -> None:
        """Два больших абзаца (>1500 символов каждый) → 2 чанка, 4 вызова AI."""
        mock_client.generate_text.side_effect = [
            "1. Факт 1",
            _AI_RESPONSE_ONE_PAIR,
            "1. Факт 2",
            "QUESTION: Q2?\nANSWER: A2.",
        ]

        generator = MaterialGenerator(client=mock_client, text_model="m1")

        # Каждый абзац ~1850 символов → суммарно >3000, не мержатся
        para1 = "Фотосинтез — процесс преобразования. " * 50
        para2 = "ДНК хранит генетическую информацию. " * 50

        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text=f"{para1}\n\n{para2}",
            target_deck="Test",
        )

        cards = generator.generate(request, MagicMock())
        assert mock_client.generate_text.call_count == 4
        assert len(cards) == 2


# ---------------------------------------------------------------------------
# AnswerDetail — детальность ответа в промпте
# ---------------------------------------------------------------------------


class TestAnswerDetail:
    def test_short_prompt_contains_1_2_sentences(self, mock_client: MagicMock) -> None:
        """SHORT: промпт содержит '1-2 sentences maximum'."""
        generator = MaterialGenerator(client=mock_client, text_model="m1")
        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text="Достаточно длинный текст для генерации карточек из материала.",
            target_deck="Test",
            material_options=MaterialOptions(answer_detail=AnswerDetail.SHORT),
        )
        generator.generate(request, MagicMock())
        generate_prompt = mock_client.generate_text.call_args_list[1][0][0]
        assert "1-2 sentences maximum" in generate_prompt

    def test_medium_prompt_contains_2_4_sentences(self, mock_client: MagicMock) -> None:
        """MEDIUM: промпт содержит '2-4 sentences'."""
        mock_client.generate_text.side_effect = [_AI_EXTRACT_RESPONSE, _AI_GENERATE_RESPONSE]
        generator = MaterialGenerator(client=mock_client, text_model="m1")
        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text="Достаточно длинный текст для генерации карточек из материала.",
            target_deck="Test",
            material_options=MaterialOptions(answer_detail=AnswerDetail.MEDIUM),
        )
        generator.generate(request, MagicMock())
        generate_prompt = mock_client.generate_text.call_args_list[1][0][0]
        assert "2-4 sentences" in generate_prompt

    def test_detailed_prompt_contains_comprehensive(self, mock_client: MagicMock) -> None:
        """DETAILED: промпт содержит 'comprehensive explanation'."""
        mock_client.generate_text.side_effect = [_AI_EXTRACT_RESPONSE, _AI_GENERATE_RESPONSE]
        generator = MaterialGenerator(client=mock_client, text_model="m1")
        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text="Достаточно длинный текст для генерации карточек из материала.",
            target_deck="Test",
            material_options=MaterialOptions(answer_detail=AnswerDetail.DETAILED),
        )
        generator.generate(request, MagicMock())
        generate_prompt = mock_client.generate_text.call_args_list[1][0][0]
        assert "comprehensive explanation" in generate_prompt

    def test_default_without_material_options_uses_short(self, mock_client: MagicMock) -> None:
        """Без material_options — используется SHORT."""
        generator = MaterialGenerator(client=mock_client, text_model="m1")
        request = CardRequest(
            mode=GenerationMode.MATERIAL,
            input_text="Достаточно длинный текст для генерации карточек из материала.",
            target_deck="Test",
        )
        generator.generate(request, MagicMock())
        generate_prompt = mock_client.generate_text.call_args_list[1][0][0]
        assert "1-2 sentences maximum" in generate_prompt
