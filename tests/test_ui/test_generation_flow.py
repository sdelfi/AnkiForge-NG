"""Tests for ankiforge.ui.generate_dialog — card generation utilities (TASK-020)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ankiforge.models import GeneratedCard, GenerationMode

# ---------------------------------------------------------------------------
# Tests for _create_generator
# ---------------------------------------------------------------------------


class TestCreateGenerator:
    """Verifies the generator factory by mode."""

    def _make_client(self) -> MagicMock:
        return MagicMock()

    def test_questions_mode_creates_questions_generator(self) -> None:
        from ankiforge.ui.generate_dialog import _create_generator

        gen = _create_generator(
            mode=GenerationMode.QUESTIONS,
            client=self._make_client(),
            text_model="m1",
            image_model="m2",
            audio_model="m3",
        )
        from ankiforge.generators.questions import QuestionsGenerator

        assert isinstance(gen, QuestionsGenerator)

    def test_language_mode_creates_language_generator(self) -> None:
        from ankiforge.ui.generate_dialog import _create_generator

        gen = _create_generator(
            mode=GenerationMode.LANGUAGE,
            client=self._make_client(),
            text_model="m1",
            image_model="m2",
            audio_model="m3",
        )
        from ankiforge.generators.language import LanguageGenerator

        assert isinstance(gen, LanguageGenerator)

    def test_material_mode_creates_material_generator(self) -> None:
        from ankiforge.ui.generate_dialog import _create_generator

        gen = _create_generator(
            mode=GenerationMode.MATERIAL,
            client=self._make_client(),
            text_model="m1",
            image_model="m2",
            audio_model="m3",
        )
        from ankiforge.generators.material import MaterialGenerator

        assert isinstance(gen, MaterialGenerator)

    def test_image_mode_creates_image_generator(self) -> None:
        from ankiforge.ui.generate_dialog import _create_generator

        gen = _create_generator(
            mode=GenerationMode.IMAGE,
            client=self._make_client(),
            text_model="m1",
            image_model="m2",
            audio_model="m3",
        )
        from ankiforge.generators.image import ImageGenerator

        assert isinstance(gen, ImageGenerator)

    def test_audio_mode_creates_audio_generator(self) -> None:
        from ankiforge.ui.generate_dialog import _create_generator

        gen = _create_generator(
            mode=GenerationMode.AUDIO,
            client=self._make_client(),
            text_model="m1",
            image_model="m2",
            audio_model="m3",
        )
        from ankiforge.generators.audio import AudioGenerator

        assert isinstance(gen, AudioGenerator)

    def test_material_mode_with_include_images(self) -> None:
        """MaterialGenerator receives image_model when include_images=True."""
        from ankiforge.ui.generate_dialog import _create_generator

        gen = _create_generator(
            mode=GenerationMode.MATERIAL,
            client=self._make_client(),
            text_model="m1",
            image_model="img-model",
            audio_model="m3",
            include_images=True,
        )
        assert gen._image_model == "img-model"

    def test_material_mode_without_include_images(self) -> None:
        """MaterialGenerator does NOT receive image_model when include_images=False."""
        from ankiforge.ui.generate_dialog import _create_generator

        gen = _create_generator(
            mode=GenerationMode.MATERIAL,
            client=self._make_client(),
            text_model="m1",
            image_model="img-model",
            audio_model="m3",
            include_images=False,
        )
        assert gen._image_model is None


# ---------------------------------------------------------------------------
# Tests for _build_card_request
# ---------------------------------------------------------------------------


class TestBuildCardRequest:
    """Verifies CardRequest construction from form state."""

    def test_basic_request(self) -> None:
        from ankiforge.ui.generate_dialog import _build_card_request

        req = _build_card_request(
            mode=GenerationMode.QUESTIONS,
            input_text="What is Python?",
            deck_name="Test Deck",
            create_new_deck=False,
            include_images=False,
            language="en",
        )
        assert req.mode == GenerationMode.QUESTIONS
        assert req.input_text == "What is Python?"
        assert req.target_deck == "Test Deck"
        assert req.create_new_deck is False
        assert req.include_images is False
        assert req.language == "en"

    def test_create_new_deck(self) -> None:
        from ankiforge.ui.generate_dialog import _build_card_request

        req = _build_card_request(
            mode=GenerationMode.LANGUAGE,
            input_text="hello",
            deck_name="New Deck",
            create_new_deck=True,
            include_images=False,
            language="ru",
        )
        assert req.create_new_deck is True
        assert req.target_deck == "New Deck"
        assert req.language == "ru"

    def test_include_images_for_material(self) -> None:
        from ankiforge.ui.generate_dialog import _build_card_request

        req = _build_card_request(
            mode=GenerationMode.MATERIAL,
            input_text="Some text",
            deck_name="Deck",
            create_new_deck=False,
            include_images=True,
            language="en",
        )
        assert req.include_images is True

    def test_empty_input_raises(self) -> None:
        from ankiforge.ui.generate_dialog import _build_card_request

        with pytest.raises(ValueError, match="Enter data to generate"):
            _build_card_request(
                mode=GenerationMode.QUESTIONS,
                input_text="   ",
                deck_name="Deck",
                create_new_deck=False,
                include_images=False,
                language="en",
            )

    def test_empty_deck_name_raises(self) -> None:
        from ankiforge.ui.generate_dialog import _build_card_request

        with pytest.raises(ValueError, match="Specify a deck name"):
            _build_card_request(
                mode=GenerationMode.QUESTIONS,
                input_text="Question?",
                deck_name="  ",
                create_new_deck=False,
                include_images=False,
                language="en",
            )


# ---------------------------------------------------------------------------
# Tests for _save_cards_to_deck
# ---------------------------------------------------------------------------


class TestSaveCardsToDeck:
    """Verifies card saving via anki_bridge."""

    def _make_qa_card(self, word: str = "Q1", answer: str = "A1") -> GeneratedCard:
        return GeneratedCard(word=word, answer=answer, note_type="AnkiForge QA")

    def _make_language_card(self, *, with_extras: bool = False) -> GeneratedCard:
        return GeneratedCard(
            word="hello",
            note_type="AnkiForge Language",
            definition="a greeting",
            example="Hello, world!",
            audio_data=b"audio-bytes",
            image_data=b"image-bytes",
            audio_definition=b"audio-def-bytes" if with_extras else None,
            audio_example=b"audio-ex-bytes" if with_extras else None,
            audio_silence=b"silence-bytes" if with_extras else None,
            transcription="/həˈloʊ/" if with_extras else None,
        )

    def _make_qa_image_card(self) -> GeneratedCard:
        return GeneratedCard(
            word="Q1",
            answer="A1",
            note_type="AnkiForge QA+Image",
            image_data=b"img-bytes",
        )

    def _make_qa_audio_card(self) -> GeneratedCard:
        return GeneratedCard(
            word="Q1",
            answer="A1",
            note_type="AnkiForge QA+Audio",
            audio_data=b"audio-bytes",
        )

    @patch("ankiforge.ui.generate_dialog.add_note")
    @patch("ankiforge.ui.generate_dialog.save_media")
    @patch("ankiforge.ui.generate_dialog.create_deck")
    def test_saves_qa_cards(self, mock_create: MagicMock, mock_media: MagicMock, mock_add: MagicMock) -> None:
        from ankiforge.ui.generate_dialog import _save_cards_to_deck

        cards = [self._make_qa_card("Q1", "A1"), self._make_qa_card("Q2", "A2")]
        _save_cards_to_deck(cards, "My Deck", create_new=False)

        mock_create.assert_not_called()
        assert mock_add.call_count == 2
        mock_add.assert_any_call("My Deck", "AnkiForge QA", {"Question": "Q1", "Answer": "A1"})
        mock_add.assert_any_call("My Deck", "AnkiForge QA", {"Question": "Q2", "Answer": "A2"})

    @patch("ankiforge.ui.generate_dialog.add_note")
    @patch("ankiforge.ui.generate_dialog.save_media")
    @patch("ankiforge.ui.generate_dialog.create_deck")
    def test_creates_new_deck_when_requested(
        self, mock_create: MagicMock, mock_media: MagicMock, mock_add: MagicMock
    ) -> None:
        from ankiforge.ui.generate_dialog import _save_cards_to_deck

        _save_cards_to_deck([self._make_qa_card()], "New Deck", create_new=True)
        mock_create.assert_called_once_with("New Deck")

    @patch("ankiforge.ui.generate_dialog.add_note")
    @patch("ankiforge.ui.generate_dialog.save_media", return_value="uuid_hello.mp3")
    @patch("ankiforge.ui.generate_dialog.create_deck")
    def test_saves_language_card_with_media(
        self, mock_create: MagicMock, mock_media: MagicMock, mock_add: MagicMock
    ) -> None:
        from ankiforge.ui.generate_dialog import _save_cards_to_deck

        card = self._make_language_card()
        _save_cards_to_deck([card], "Lang Deck", create_new=False)

        # Should be 2 save_media calls (audio + image)
        assert mock_media.call_count == 2
        mock_add.assert_called_once()
        fields = mock_add.call_args[0][2]
        assert fields["Word"] == "hello"
        assert fields["Definition"] == "a greeting"
        assert fields["Example"] == "Hello, world!"
        assert "[sound:" in fields["Audio"]
        assert "<img src=" in fields["Image"]

    @patch("ankiforge.ui.generate_dialog.add_note")
    @patch("ankiforge.ui.generate_dialog.save_media", return_value="uuid_hello.mp3")
    @patch("ankiforge.ui.generate_dialog.create_deck")
    def test_saves_language_card_with_all_audio(
        self, mock_create: MagicMock, mock_media: MagicMock, mock_add: MagicMock
    ) -> None:
        from ankiforge.ui.generate_dialog import _save_cards_to_deck

        card = self._make_language_card(with_extras=True)
        _save_cards_to_deck([card], "Lang Deck", create_new=False)

        # 5 save_media calls: audio + image + audio_def + silence + audio_ex
        assert mock_media.call_count == 5
        fields = mock_add.call_args[0][2]
        assert fields["Transcription"] == "/həˈloʊ/"
        assert "[sound:" in fields["AudioDefinition"]
        assert "[sound:" in fields["AudioSilence"]
        assert "[sound:" in fields["AudioExample"]

    @patch("ankiforge.ui.generate_dialog.add_note")
    @patch("ankiforge.ui.generate_dialog.save_media", return_value="uuid_hello.mp3")
    @patch("ankiforge.ui.generate_dialog.create_deck")
    def test_language_card_without_extras_has_empty_fields(
        self, mock_create: MagicMock, mock_media: MagicMock, mock_add: MagicMock
    ) -> None:
        from ankiforge.ui.generate_dialog import _save_cards_to_deck

        card = self._make_language_card(with_extras=False)
        _save_cards_to_deck([card], "Lang Deck", create_new=False)

        fields = mock_add.call_args[0][2]
        assert fields["AudioDefinition"] == ""
        assert fields["AudioSilence"] == ""
        assert fields["AudioExample"] == ""
        assert fields["Transcription"] == ""

    @patch("ankiforge.ui.generate_dialog.add_note")
    @patch("ankiforge.ui.generate_dialog.save_media", return_value="uuid_q1.png")
    @patch("ankiforge.ui.generate_dialog.create_deck")
    def test_saves_qa_image_card(self, mock_create: MagicMock, mock_media: MagicMock, mock_add: MagicMock) -> None:
        from ankiforge.ui.generate_dialog import _save_cards_to_deck

        card = self._make_qa_image_card()
        _save_cards_to_deck([card], "Deck", create_new=False)

        mock_media.assert_called_once()
        fields = mock_add.call_args[0][2]
        assert fields["Question"] == "Q1"
        assert fields["Answer"] == "A1"
        assert "<img src=" in fields["Image"]

    @patch("ankiforge.ui.generate_dialog.add_note")
    @patch("ankiforge.ui.generate_dialog.save_media", return_value="uuid_q1.mp3")
    @patch("ankiforge.ui.generate_dialog.create_deck")
    def test_saves_qa_audio_card(self, mock_create: MagicMock, mock_media: MagicMock, mock_add: MagicMock) -> None:
        from ankiforge.ui.generate_dialog import _save_cards_to_deck

        card = self._make_qa_audio_card()
        _save_cards_to_deck([card], "Deck", create_new=False)

        mock_media.assert_called_once()
        fields = mock_add.call_args[0][2]
        assert fields["Question"] == "Q1"
        assert fields["Answer"] == "A1"
        assert "[sound:" in fields["Audio"]

    @patch("ankiforge.ui.generate_dialog.add_note")
    @patch("ankiforge.ui.generate_dialog.save_media")
    @patch("ankiforge.ui.generate_dialog.create_deck")
    def test_empty_cards_list_does_nothing(
        self, mock_create: MagicMock, mock_media: MagicMock, mock_add: MagicMock
    ) -> None:
        from ankiforge.ui.generate_dialog import _save_cards_to_deck

        _save_cards_to_deck([], "Deck", create_new=False)
        mock_add.assert_not_called()

    @patch("ankiforge.ui.generate_dialog.add_note")
    @patch("ankiforge.ui.generate_dialog.save_media")
    @patch("ankiforge.ui.generate_dialog.create_deck")
    def test_qa_card_without_answer_uses_empty_string(
        self, mock_create: MagicMock, mock_media: MagicMock, mock_add: MagicMock
    ) -> None:
        from ankiforge.ui.generate_dialog import _save_cards_to_deck

        card = GeneratedCard(word="Q1", note_type="AnkiForge QA")
        _save_cards_to_deck([card], "Deck", create_new=False)

        fields = mock_add.call_args[0][2]
        assert fields["Answer"] == ""


# ---------------------------------------------------------------------------
# Tests for _get_input_placeholder
# ---------------------------------------------------------------------------


class TestGetDeckNameFromCombo:
    """Verifies deck name resolution from editable combo."""

    def test_existing_deck_returns_not_new(self) -> None:
        from ankiforge.ui.generate_dialog import _get_deck_name_from_combo

        existing = {"Default", "English", "Math"}
        name, is_new = _get_deck_name_from_combo("English", existing)
        assert name == "English"
        assert is_new is False

    def test_new_deck_returns_is_new(self) -> None:
        from ankiforge.ui.generate_dialog import _get_deck_name_from_combo

        existing = {"Default", "English"}
        name, is_new = _get_deck_name_from_combo("My New Deck", existing)
        assert name == "My New Deck"
        assert is_new is True

    def test_empty_input_returns_empty(self) -> None:
        from ankiforge.ui.generate_dialog import _get_deck_name_from_combo

        name, is_new = _get_deck_name_from_combo("   ", set())
        assert name == ""
        assert is_new is True

    def test_strips_whitespace(self) -> None:
        from ankiforge.ui.generate_dialog import _get_deck_name_from_combo

        existing = {"English"}
        name, is_new = _get_deck_name_from_combo("  English  ", existing)
        assert name == "English"
        assert is_new is False


class TestGetInputPlaceholder:
    """Verifies input field placeholders."""

    def test_all_modes_have_placeholders(self) -> None:
        from ankiforge.ui.generate_dialog import _get_input_placeholder

        for mode in GenerationMode:
            placeholder = _get_input_placeholder(mode)
            assert len(placeholder) > 0

    def test_questions_placeholder(self) -> None:
        from ankiforge.ui.generate_dialog import _get_input_placeholder

        text = _get_input_placeholder(GenerationMode.QUESTIONS)
        assert len(text) > 0

    def test_language_placeholder(self) -> None:
        from ankiforge.ui.generate_dialog import _get_input_placeholder

        text = _get_input_placeholder(GenerationMode.LANGUAGE)
        assert len(text) > 0


# ---------------------------------------------------------------------------
# Tests for _should_show_images_checkbox
# ---------------------------------------------------------------------------


class TestShouldShowImagesCheckbox:
    """Verifies when to show the images checkbox."""

    def test_material_mode_shows_checkbox(self) -> None:
        from ankiforge.ui.generate_dialog import _should_show_images_checkbox

        assert _should_show_images_checkbox(GenerationMode.MATERIAL) is True

    def test_other_modes_hide_checkbox(self) -> None:
        from ankiforge.ui.generate_dialog import _should_show_images_checkbox

        for mode in GenerationMode:
            if mode != GenerationMode.MATERIAL:
                assert _should_show_images_checkbox(mode) is False, f"{mode} should not show images checkbox"


# ---------------------------------------------------------------------------
# Tests for _should_show_custom_prompt (TASK-022)
# ---------------------------------------------------------------------------


class TestShouldShowCustomPrompt:
    """Verifies when to show the custom prompt field."""

    def test_all_modes_show_custom_prompt(self) -> None:
        from ankiforge.ui.generate_dialog import _should_show_custom_prompt

        for mode in GenerationMode:
            assert _should_show_custom_prompt(mode) is True, f"{mode} should show custom prompt"


# ---------------------------------------------------------------------------
# Tests for _get_default_custom_prompt (TASK-022)
# ---------------------------------------------------------------------------


class TestGetDefaultCustomPrompt:
    """Verifies the default prompt for all modes."""

    def test_returns_non_empty_string_for_language(self) -> None:
        from ankiforge.ui.generate_dialog import _get_default_custom_prompt

        prompt = _get_default_custom_prompt(GenerationMode.LANGUAGE)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_contains_definition_and_example_for_language(self) -> None:
        from ankiforge.ui.generate_dialog import _get_default_custom_prompt

        prompt = _get_default_custom_prompt(GenerationMode.LANGUAGE)
        assert "definition" in prompt.lower()
        assert "example" in prompt.lower()

    def test_all_modes_return_non_empty(self) -> None:
        from ankiforge.ui.generate_dialog import _get_default_custom_prompt

        for mode in GenerationMode:
            prompt = _get_default_custom_prompt(mode)
            assert isinstance(prompt, str)
            assert len(prompt) > 0, f"{mode} should have a non-empty default prompt"

    def test_questions_prompt_mentions_concise(self) -> None:
        from ankiforge.ui.generate_dialog import _get_default_custom_prompt

        prompt = _get_default_custom_prompt(GenerationMode.QUESTIONS)
        assert "concise" in prompt.lower()

    def test_material_prompt_mentions_question(self) -> None:
        from ankiforge.ui.generate_dialog import _get_default_custom_prompt

        prompt = _get_default_custom_prompt(GenerationMode.MATERIAL)
        assert "question" in prompt.lower() or "QUESTION" in prompt


# ---------------------------------------------------------------------------
# Tests for _build_card_request with custom_prompt (TASK-022)
# ---------------------------------------------------------------------------


class TestBuildCardRequestCustomPrompt:
    """Verifies custom_prompt passing through _build_card_request."""

    def test_custom_prompt_passed_to_request(self) -> None:
        from ankiforge.ui.generate_dialog import _build_card_request

        req = _build_card_request(
            mode=GenerationMode.LANGUAGE,
            input_text="hello",
            deck_name="Deck",
            create_new_deck=False,
            include_images=False,
            language="en",
            custom_prompt="My custom instructions",
        )
        assert req.custom_prompt == "My custom instructions"

    def test_none_custom_prompt_by_default(self) -> None:
        from ankiforge.ui.generate_dialog import _build_card_request

        req = _build_card_request(
            mode=GenerationMode.QUESTIONS,
            input_text="Q?",
            deck_name="Deck",
            create_new_deck=False,
            include_images=False,
            language="en",
        )
        assert req.custom_prompt is None

    def test_empty_custom_prompt_becomes_none(self) -> None:
        from ankiforge.ui.generate_dialog import _build_card_request

        req = _build_card_request(
            mode=GenerationMode.LANGUAGE,
            input_text="hello",
            deck_name="Deck",
            create_new_deck=False,
            include_images=False,
            language="en",
            custom_prompt="   ",
        )
        assert req.custom_prompt is None


# ---------------------------------------------------------------------------
# Tests for _estimate_cost_from_config
# ---------------------------------------------------------------------------


class TestEstimateCostFromConfig:
    """Verifies cost calculation from cached pricing."""

    def _make_config(self) -> object:
        from ankiforge.models import AddonConfig, ModelPricingCache

        return AddonConfig(
            api_key="sk-or-test",
            text_model="openai/gpt-4o",
            text_model_pricing=ModelPricingCache(prompt=0.000005, completion=0.000015),
            image_model="openai/dall-e-3",
            image_model_pricing=ModelPricingCache(image=0.04),
            audio_model="openai/tts-1",
            audio_model_pricing=ModelPricingCache(prompt=0.000015),
        )

    def test_questions_mode_text_only(self) -> None:
        from ankiforge.ui.generate_dialog import _estimate_cost_from_config

        config = self._make_config()
        cost = _estimate_cost_from_config(config, GenerationMode.QUESTIONS, 10)
        assert cost is not None
        assert cost > 0.0

    def test_language_mode_all_three(self) -> None:
        from ankiforge.ui.generate_dialog import _estimate_cost_from_config

        config = self._make_config()
        cost = _estimate_cost_from_config(config, GenerationMode.LANGUAGE, 5)
        assert cost is not None
        cost_questions = _estimate_cost_from_config(config, GenerationMode.QUESTIONS, 5)
        assert cost is not None and cost_questions is not None
        assert cost > cost_questions

    def test_no_pricing_returns_none(self) -> None:
        from ankiforge.models import AddonConfig
        from ankiforge.ui.generate_dialog import _estimate_cost_from_config

        config = AddonConfig(api_key="sk-or-test", text_model="m1")
        cost = _estimate_cost_from_config(config, GenerationMode.QUESTIONS, 10)
        assert cost is None

    def test_zero_cards_returns_zero(self) -> None:
        from ankiforge.ui.generate_dialog import _estimate_cost_from_config

        config = self._make_config()
        cost = _estimate_cost_from_config(config, GenerationMode.QUESTIONS, 0)
        assert cost == 0.0

    def test_whitespace_stripped_from_custom_prompt(self) -> None:
        from ankiforge.ui.generate_dialog import _build_card_request

        req = _build_card_request(
            mode=GenerationMode.LANGUAGE,
            input_text="hello",
            deck_name="Deck",
            create_new_deck=False,
            include_images=False,
            language="en",
            custom_prompt="  My prompt  ",
        )
        assert req.custom_prompt == "My prompt"

    def test_language_options_no_audio_reduces_cost(self) -> None:
        """Disabling all audio should reduce cost."""
        from ankiforge.models import LanguageOptions
        from ankiforge.ui.generate_dialog import _estimate_cost_from_config

        config = self._make_config()
        cost_all = _estimate_cost_from_config(config, GenerationMode.LANGUAGE, 5)
        cost_no_audio = _estimate_cost_from_config(
            config,
            GenerationMode.LANGUAGE,
            5,
            LanguageOptions(include_audio_word=False, include_audio_definition=False, include_audio_example=False),
        )
        assert cost_all is not None and cost_no_audio is not None
        assert cost_no_audio < cost_all

    def test_language_options_no_photo_reduces_cost(self) -> None:
        """Enabling photo should increase cost vs default (no photo)."""
        from ankiforge.models import LanguageOptions
        from ankiforge.ui.generate_dialog import _estimate_cost_from_config

        config = self._make_config()
        cost_default = _estimate_cost_from_config(config, GenerationMode.LANGUAGE, 5)
        cost_with_photo = _estimate_cost_from_config(
            config, GenerationMode.LANGUAGE, 5, LanguageOptions(include_photo=True)
        )
        assert cost_default is not None and cost_with_photo is not None
        assert cost_with_photo > cost_default

    def test_language_single_text_call_same_cost_with_or_without_ipa(self) -> None:
        """Transcription is included in the single JSON request — text cost is the same."""
        from ankiforge.models import LanguageOptions
        from ankiforge.ui.generate_dialog import _estimate_cost_from_config

        config = self._make_config()
        cost_all = _estimate_cost_from_config(config, GenerationMode.LANGUAGE, 5)
        cost_no_ipa = _estimate_cost_from_config(
            config, GenerationMode.LANGUAGE, 5, LanguageOptions(include_transcription=False)
        )
        assert cost_all is not None and cost_no_ipa is not None
        # Text cost is the same — IPA is in the same request
        # Difference can only arise if other options differ
        assert cost_all == cost_no_ipa

    def test_image_size_affects_cost_estimate(self) -> None:
        """Image size affects cost estimate."""
        from ankiforge.models import LanguageOptions
        from ankiforge.ui.generate_dialog import _estimate_cost_from_config

        config = self._make_config()
        cost_05k = _estimate_cost_from_config(
            config, GenerationMode.LANGUAGE, 5, LanguageOptions(include_photo=True, image_size="0.5K")
        )
        cost_1k = _estimate_cost_from_config(
            config, GenerationMode.LANGUAGE, 5, LanguageOptions(include_photo=True, image_size="1K")
        )
        cost_4k = _estimate_cost_from_config(
            config, GenerationMode.LANGUAGE, 5, LanguageOptions(include_photo=True, image_size="4K")
        )
        assert cost_05k is not None and cost_1k is not None and cost_4k is not None
        assert cost_05k < cost_1k < cost_4k


# ---------------------------------------------------------------------------
# Tests for _build_card_request with language_options
# ---------------------------------------------------------------------------


class TestBuildCardRequestLanguageOptions:
    """Verifies language_options passing through _build_card_request."""

    def test_language_options_passed_to_request(self) -> None:
        from ankiforge.models import LanguageOptions
        from ankiforge.ui.generate_dialog import _build_card_request

        opts = LanguageOptions(include_photo=False, include_audio_word=False)
        req = _build_card_request(
            mode=GenerationMode.LANGUAGE,
            input_text="hello",
            deck_name="Deck",
            create_new_deck=False,
            include_images=False,
            language="en",
            language_options=opts,
        )
        assert req.language_options is not None
        assert req.language_options.include_photo is False
        assert req.language_options.include_audio_word is False

    def test_none_language_options_by_default(self) -> None:
        from ankiforge.ui.generate_dialog import _build_card_request

        req = _build_card_request(
            mode=GenerationMode.QUESTIONS,
            input_text="Q?",
            deck_name="Deck",
            create_new_deck=False,
            include_images=False,
            language="en",
        )
        assert req.language_options is None


# ---------------------------------------------------------------------------
# Tests for _build_card_request with material_options, voice, image_size
# ---------------------------------------------------------------------------


class TestBuildCardRequestNewOptions:
    """Verifies material_options, voice, image_size passing through _build_card_request."""

    def test_material_options_passed(self) -> None:
        from ankiforge.models import MaterialOptions
        from ankiforge.ui.generate_dialog import _build_card_request

        opts = MaterialOptions(max_cards_per_paragraph=5, include_images=True, image_size="2K")
        req = _build_card_request(
            mode=GenerationMode.MATERIAL,
            input_text="text",
            deck_name="Deck",
            create_new_deck=False,
            include_images=True,
            language="en",
            material_options=opts,
        )
        assert req.material_options is not None
        assert req.material_options.max_cards_per_paragraph == 5
        assert req.material_options.image_size == "2K"

    def test_voice_passed(self) -> None:
        from ankiforge.ui.generate_dialog import _build_card_request

        req = _build_card_request(
            mode=GenerationMode.AUDIO,
            input_text="Q?",
            deck_name="Deck",
            create_new_deck=False,
            include_images=False,
            language="en",
            voice="nova",
        )
        assert req.voice == "nova"

    def test_image_size_passed(self) -> None:
        from ankiforge.ui.generate_dialog import _build_card_request

        req = _build_card_request(
            mode=GenerationMode.IMAGE,
            input_text="Q?",
            deck_name="Deck",
            create_new_deck=False,
            include_images=False,
            language="en",
            image_size="4K",
        )
        assert req.image_size == "4K"

    def test_defaults(self) -> None:
        from ankiforge.ui.generate_dialog import _build_card_request

        req = _build_card_request(
            mode=GenerationMode.QUESTIONS,
            input_text="Q?",
            deck_name="Deck",
            create_new_deck=False,
            include_images=False,
            language="en",
        )
        assert req.material_options is None
        assert req.voice == "alloy"
        assert req.image_size == "auto"


# ---------------------------------------------------------------------------
# Tests for _create_generator with new parameters
# ---------------------------------------------------------------------------


class TestBuildCardRequestAnswerDetail:
    """Verifies answer_detail passing through material_options."""

    def test_answer_detail_passed_in_material_options(self) -> None:
        from ankiforge.models import AnswerDetail, MaterialOptions
        from ankiforge.ui.generate_dialog import _build_card_request

        opts = MaterialOptions(answer_detail=AnswerDetail.DETAILED)
        req = _build_card_request(
            mode=GenerationMode.MATERIAL,
            input_text="text",
            deck_name="Deck",
            create_new_deck=False,
            include_images=False,
            language="en",
            material_options=opts,
        )
        assert req.material_options is not None
        assert req.material_options.answer_detail == AnswerDetail.DETAILED

    def test_default_answer_detail_is_short(self) -> None:
        from ankiforge.models import AnswerDetail, MaterialOptions
        from ankiforge.ui.generate_dialog import _build_card_request

        opts = MaterialOptions()
        req = _build_card_request(
            mode=GenerationMode.MATERIAL,
            input_text="text",
            deck_name="Deck",
            create_new_deck=False,
            include_images=False,
            language="en",
            material_options=opts,
        )
        assert req.material_options is not None
        assert req.material_options.answer_detail == AnswerDetail.SHORT


class TestCreateGeneratorNewParams:
    """Verifies image_size and voice passing to generators."""

    def _make_client(self) -> MagicMock:
        return MagicMock()

    def test_image_generator_gets_image_size(self) -> None:
        from ankiforge.ui.generate_dialog import _create_generator

        gen = _create_generator(
            mode=GenerationMode.IMAGE,
            client=self._make_client(),
            text_model="m1",
            image_model="m2",
            audio_model="m3",
            image_size="2K",
        )
        assert gen._image_size == "2K"

    def test_audio_generator_gets_voice(self) -> None:
        from ankiforge.ui.generate_dialog import _create_generator

        gen = _create_generator(
            mode=GenerationMode.AUDIO,
            client=self._make_client(),
            text_model="m1",
            image_model="m2",
            audio_model="m3",
            voice="nova",
        )
        assert gen._voice == "nova"

    def test_material_generator_gets_image_size(self) -> None:
        from ankiforge.ui.generate_dialog import _create_generator

        gen = _create_generator(
            mode=GenerationMode.MATERIAL,
            client=self._make_client(),
            text_model="m1",
            image_model="m2",
            audio_model="m3",
            include_images=True,
            image_size="0.5K",
        )
        assert gen._image_size == "0.5K"


# ---------------------------------------------------------------------------
# Tests for _markdown_to_html
# ---------------------------------------------------------------------------


class TestMarkdownToHtml:
    """Markdown code blocks conversion to HTML for Anki."""

    def test_code_block_to_pre(self) -> None:
        from ankiforge.ui.generate_dialog import _markdown_to_html

        text = "Пример:\n```python\nx = 1\n```\nКонец."
        result = _markdown_to_html(text)
        assert "<pre><code" in result
        assert "x = 1" in result
        assert "```" not in result

    def test_code_block_preserves_language(self) -> None:
        from ankiforge.ui.generate_dialog import _markdown_to_html

        text = "```python\nprint('hello')\n```"
        result = _markdown_to_html(text)
        assert 'class="language-python"' in result

    def test_code_block_without_language(self) -> None:
        from ankiforge.ui.generate_dialog import _markdown_to_html

        text = "```\nx = 1\n```"
        result = _markdown_to_html(text)
        assert "<pre><code>" in result

    def test_inline_code(self) -> None:
        from ankiforge.ui.generate_dialog import _markdown_to_html

        text = "Используйте `OrderedDict.fromkeys()` для этого."
        result = _markdown_to_html(text)
        assert "<code>OrderedDict.fromkeys()</code>" in result
        assert "`" not in result

    def test_html_escaping_in_code(self) -> None:
        from ankiforge.ui.generate_dialog import _markdown_to_html

        text = "```python\nif x < 10 and y > 5:\n```"
        result = _markdown_to_html(text)
        assert "&lt;" in result
        assert "&gt;" in result
        assert "<10" not in result  # should not become an HTML tag

    def test_newlines_to_br_outside_code(self) -> None:
        from ankiforge.ui.generate_dialog import _markdown_to_html

        text = "Строка 1\nСтрока 2"
        result = _markdown_to_html(text)
        assert "<br>" in result

    def test_newlines_preserved_inside_pre(self) -> None:
        from ankiforge.ui.generate_dialog import _markdown_to_html

        text = "```python\nline1\nline2\n```"
        result = _markdown_to_html(text)
        # Inside <pre> there should be no <br>
        assert "<br>" not in result

    def test_plain_text_no_code(self) -> None:
        from ankiforge.ui.generate_dialog import _markdown_to_html

        text = "Обычный текст без кода."
        result = _markdown_to_html(text)
        assert result == "Обычный текст без кода."

    def test_multiple_code_blocks(self) -> None:
        from ankiforge.ui.generate_dialog import _markdown_to_html

        text = "A:\n```python\nx = 1\n```\nB:\n```js\ny = 2\n```"
        result = _markdown_to_html(text)
        assert result.count("<pre>") == 2
        assert "language-python" in result
        assert "language-js" in result
