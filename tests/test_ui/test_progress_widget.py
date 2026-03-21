"""Tests for ankiforge.ui.progress_widget — generation progress utilities (TASK-021)."""

from __future__ import annotations

from unittest.mock import MagicMock

from ankiforge.models import GenerationMode

# ---------------------------------------------------------------------------
# Tests for _count_input_items
# ---------------------------------------------------------------------------


class TestCountInputItems:
    """Counting the number of input items by mode."""

    def test_questions_counts_lines(self) -> None:
        from ankiforge.ui.progress_widget import _count_input_items

        assert _count_input_items("Q1?\nQ2?\nQ3?", GenerationMode.QUESTIONS) == 3

    def test_questions_skips_empty_lines(self) -> None:
        from ankiforge.ui.progress_widget import _count_input_items

        assert _count_input_items("Q1?\n\n  \nQ2?", GenerationMode.QUESTIONS) == 2

    def test_language_counts_lines(self) -> None:
        from ankiforge.ui.progress_widget import _count_input_items

        assert _count_input_items("hello\nworld", GenerationMode.LANGUAGE) == 2

    def test_language_counts_comma_separated(self) -> None:
        from ankiforge.ui.progress_widget import _count_input_items

        assert _count_input_items("hello, world, test", GenerationMode.LANGUAGE) == 3

    def test_language_mixed_lines_and_commas(self) -> None:
        from ankiforge.ui.progress_widget import _count_input_items

        assert _count_input_items("hello, world\ntest", GenerationMode.LANGUAGE) == 3

    def test_material_small_paragraphs_merge(self) -> None:
        from ankiforge.ui.progress_widget import _count_input_items

        # 3 short paragraphs merge into 1 chunk → 1 × 3 = 3
        text = (
            "Абзац первый — достаточно длинный.\n\n"
            "Абзац второй — достаточно длинный.\n\n"
            "Абзац третий — достаточно длинный."
        )
        count = _count_input_items(text, GenerationMode.MATERIAL)
        assert count == 3

    def test_material_topic_headings_split(self) -> None:
        from ankiforge.ui.progress_widget import _count_input_items

        # 3 question headings → 3 chunks × 1 = 3
        text = (
            "Что такое генератор\n\nОбъяснение генератора достаточно подробное.\n\n"
            "Что такое итератор\n\nОбъяснение итератора достаточно подробное.\n\n"
            "Что делает yield from\n\nОбъяснение yield from достаточно подробное."
        )
        count = _count_input_items(text, GenerationMode.MATERIAL, max_cards_per_paragraph=1)
        assert count == 3

    def test_material_single_paragraph(self) -> None:
        from ankiforge.ui.progress_widget import _count_input_items

        # Single paragraph → 1 × 3 = 3
        count = _count_input_items("Достаточно длинный текст без разбиения.", GenerationMode.MATERIAL)
        assert count == 3

    def test_material_minimum_one(self) -> None:
        from ankiforge.ui.progress_widget import _count_input_items

        assert _count_input_items("Short", GenerationMode.MATERIAL) >= 1

    def test_image_counts_lines(self) -> None:
        from ankiforge.ui.progress_widget import _count_input_items

        assert _count_input_items("Q1?\nQ2?", GenerationMode.IMAGE) == 2

    def test_audio_counts_lines(self) -> None:
        from ankiforge.ui.progress_widget import _count_input_items

        assert _count_input_items("Q1?\nQ2?\nQ3?", GenerationMode.AUDIO) == 3

    def test_empty_input_returns_zero(self) -> None:
        from ankiforge.ui.progress_widget import _count_input_items

        assert _count_input_items("", GenerationMode.QUESTIONS) == 0

    def test_whitespace_only_returns_zero(self) -> None:
        from ankiforge.ui.progress_widget import _count_input_items

        assert _count_input_items("   \n  \n  ", GenerationMode.QUESTIONS) == 0


# ---------------------------------------------------------------------------
# Tests for _format_cost
# ---------------------------------------------------------------------------


class TestFormatCost:
    """Cost formatting."""

    def test_zero_cost(self) -> None:
        from ankiforge.ui.progress_widget import _format_cost

        assert _format_cost(0.0) == "< $0.001"

    def test_small_cost(self) -> None:
        from ankiforge.ui.progress_widget import _format_cost

        assert _format_cost(0.0005) == "< $0.001"

    def test_normal_cost(self) -> None:
        from ankiforge.ui.progress_widget import _format_cost

        assert _format_cost(0.15) == "$0.150"

    def test_large_cost(self) -> None:
        from ankiforge.ui.progress_widget import _format_cost

        assert _format_cost(1.234) == "$1.234"


# ---------------------------------------------------------------------------
# Tests for _find_model_by_id
# ---------------------------------------------------------------------------


class TestFindModelById:
    """Finding a model by ID in a list."""

    def test_finds_existing_model(self) -> None:
        from ankiforge.openrouter.models import Model
        from ankiforge.ui.progress_widget import _find_model_by_id

        models = [
            Model(id="openai/gpt-4o", name="GPT-4o"),
            Model(id="anthropic/claude-3", name="Claude 3"),
        ]
        result = _find_model_by_id("anthropic/claude-3", models)
        assert result is not None
        assert result.id == "anthropic/claude-3"

    def test_returns_none_for_missing(self) -> None:
        from ankiforge.openrouter.models import Model
        from ankiforge.ui.progress_widget import _find_model_by_id

        models = [Model(id="openai/gpt-4o", name="GPT-4o")]
        assert _find_model_by_id("nonexistent", models) is None

    def test_returns_none_for_empty_list(self) -> None:
        from ankiforge.ui.progress_widget import _find_model_by_id

        assert _find_model_by_id("any", []) is None

    def test_returns_none_for_empty_id(self) -> None:
        from ankiforge.openrouter.models import Model
        from ankiforge.ui.progress_widget import _find_model_by_id

        models = [Model(id="openai/gpt-4o", name="GPT-4o")]
        assert _find_model_by_id("", models) is None


# ---------------------------------------------------------------------------
# Tests for _estimate_and_format_cost
# ---------------------------------------------------------------------------


class TestEstimateAndFormatCost:
    """Calculation and formatting of cost estimates."""

    def test_returns_formatted_cost(self) -> None:
        from ankiforge.ui.progress_widget import _estimate_and_format_cost

        client = MagicMock()
        client.estimate_cost.return_value = 0.25

        result = _estimate_and_format_cost(
            client=client,
            mode=GenerationMode.QUESTIONS,
            card_count=10,
            models=[],
            text_model_id="m1",
            image_model_id="",
            audio_model_id="",
        )
        assert "$0.250" in result

    def test_zero_cards_returns_zero_cost(self) -> None:
        from ankiforge.ui.progress_widget import _estimate_and_format_cost

        client = MagicMock()
        client.estimate_cost.return_value = 0.0

        result = _estimate_and_format_cost(
            client=client,
            mode=GenerationMode.QUESTIONS,
            card_count=0,
            models=[],
            text_model_id="m1",
            image_model_id="",
            audio_model_id="",
        )
        assert "< $0.001" in result

    def test_passes_model_objects_to_estimate(self) -> None:
        from ankiforge.openrouter.models import Model
        from ankiforge.ui.progress_widget import _estimate_and_format_cost

        text_model = Model(id="t1", name="Text")
        image_model = Model(id="i1", name="Image")
        audio_model = Model(id="a1", name="Audio")
        models = [text_model, image_model, audio_model]

        client = MagicMock()
        client.estimate_cost.return_value = 0.5

        _estimate_and_format_cost(
            client=client,
            mode=GenerationMode.LANGUAGE,
            card_count=5,
            models=models,
            text_model_id="t1",
            image_model_id="i1",
            audio_model_id="a1",
        )

        client.estimate_cost.assert_called_once_with(
            "language",
            5,
            text_model=text_model,
            image_model=image_model,
            audio_model=audio_model,
        )
