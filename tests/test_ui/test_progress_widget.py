"""Тесты для ankiforge.ui.progress_widget — утилиты прогресса генерации (TASK-021)."""

from __future__ import annotations

from unittest.mock import MagicMock

from ankiforge.models import GenerationMode, GenerationProgress

# ---------------------------------------------------------------------------
# Тесты _count_input_items
# ---------------------------------------------------------------------------


class TestCountInputItems:
    """Подсчёт количества элементов ввода по режиму."""

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

    def test_material_estimates_from_length(self) -> None:
        from ankiforge.ui.progress_widget import _count_input_items

        # Длинный текст (~3000 символов) → ~6 карточек
        text = "A" * 3000
        count = _count_input_items(text, GenerationMode.MATERIAL)
        assert count >= 3

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
# Тесты _format_cost
# ---------------------------------------------------------------------------


class TestFormatCost:
    """Форматирование стоимости."""

    def test_zero_cost(self) -> None:
        from ankiforge.ui.progress_widget import _format_cost

        assert _format_cost(0.0) == "< $0.01"

    def test_small_cost(self) -> None:
        from ankiforge.ui.progress_widget import _format_cost

        assert _format_cost(0.005) == "< $0.01"

    def test_normal_cost(self) -> None:
        from ankiforge.ui.progress_widget import _format_cost

        assert _format_cost(0.15) == "$0.15"

    def test_large_cost(self) -> None:
        from ankiforge.ui.progress_widget import _format_cost

        assert _format_cost(1.234) == "$1.23"


# ---------------------------------------------------------------------------
# Тесты _format_summary
# ---------------------------------------------------------------------------


class TestFormatSummary:
    """Форматирование итога генерации."""

    def test_basic_summary(self) -> None:
        from ankiforge.ui.progress_widget import _format_summary

        result = _format_summary(10, 0.25)
        assert "10" in result
        assert "$0.25" in result

    def test_zero_cards(self) -> None:
        from ankiforge.ui.progress_widget import _format_summary

        result = _format_summary(0, 0.0)
        assert "0" in result

    def test_small_cost_in_summary(self) -> None:
        from ankiforge.ui.progress_widget import _format_summary

        result = _format_summary(5, 0.001)
        assert "< $0.01" in result


# ---------------------------------------------------------------------------
# Тесты _format_progress_text
# ---------------------------------------------------------------------------


class TestFormatProgressText:
    """Форматирование текста прогресса."""

    def test_basic_progress(self) -> None:
        from ankiforge.ui.progress_widget import _format_progress_text

        progress = GenerationProgress(total_cards=10, completed_cards=3, current_cost=0.05)
        result = _format_progress_text(progress)
        assert "3" in result
        assert "10" in result

    def test_progress_with_cost(self) -> None:
        from ankiforge.ui.progress_widget import _format_progress_text

        progress = GenerationProgress(total_cards=5, completed_cards=2, current_cost=0.12)
        result = _format_progress_text(progress)
        assert "$0.12" in result


# ---------------------------------------------------------------------------
# Тесты _find_model_by_id
# ---------------------------------------------------------------------------


class TestFindModelById:
    """Поиск модели по ID в списке."""

    def test_finds_existing_model(self) -> None:
        from ankiforge.ui.progress_widget import _find_model_by_id

        from ankiforge.openrouter.models import Model

        models = [
            Model(id="openai/gpt-4o", name="GPT-4o"),
            Model(id="anthropic/claude-3", name="Claude 3"),
        ]
        result = _find_model_by_id("anthropic/claude-3", models)
        assert result is not None
        assert result.id == "anthropic/claude-3"

    def test_returns_none_for_missing(self) -> None:
        from ankiforge.ui.progress_widget import _find_model_by_id

        from ankiforge.openrouter.models import Model

        models = [Model(id="openai/gpt-4o", name="GPT-4o")]
        assert _find_model_by_id("nonexistent", models) is None

    def test_returns_none_for_empty_list(self) -> None:
        from ankiforge.ui.progress_widget import _find_model_by_id

        assert _find_model_by_id("any", []) is None

    def test_returns_none_for_empty_id(self) -> None:
        from ankiforge.ui.progress_widget import _find_model_by_id

        from ankiforge.openrouter.models import Model

        models = [Model(id="openai/gpt-4o", name="GPT-4o")]
        assert _find_model_by_id("", models) is None


# ---------------------------------------------------------------------------
# Тесты _estimate_and_format_cost
# ---------------------------------------------------------------------------


class TestEstimateAndFormatCost:
    """Расчёт и форматирование оценки стоимости."""

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
        assert "$0.25" in result

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
        assert "< $0.01" in result

    def test_passes_model_objects_to_estimate(self) -> None:
        from ankiforge.ui.progress_widget import _estimate_and_format_cost

        from ankiforge.openrouter.models import Model

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
