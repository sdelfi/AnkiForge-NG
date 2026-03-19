"""Тесты для ankiforge.ui.generate_dialog — утилиты генерации карточек (TASK-020)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ankiforge.models import GeneratedCard, GenerationMode

# ---------------------------------------------------------------------------
# Тесты _create_generator
# ---------------------------------------------------------------------------


class TestCreateGenerator:
    """Проверяет фабрику генераторов по режиму."""

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
        """MaterialGenerator получает image_model когда include_images=True."""
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
        """MaterialGenerator НЕ получает image_model когда include_images=False."""
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
# Тесты _build_card_request
# ---------------------------------------------------------------------------


class TestBuildCardRequest:
    """Проверяет сборку CardRequest из состояния формы."""

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

        with pytest.raises(ValueError, match="Введите данные"):
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

        with pytest.raises(ValueError, match="колоды"):
            _build_card_request(
                mode=GenerationMode.QUESTIONS,
                input_text="Question?",
                deck_name="  ",
                create_new_deck=False,
                include_images=False,
                language="en",
            )


# ---------------------------------------------------------------------------
# Тесты _save_cards_to_deck
# ---------------------------------------------------------------------------


class TestSaveCardsToDeck:
    """Проверяет сохранение карточек через anki_bridge."""

    def _make_qa_card(self, word: str = "Q1", answer: str = "A1") -> GeneratedCard:
        return GeneratedCard(word=word, answer=answer, note_type="AnkiForge QA")

    def _make_language_card(self) -> GeneratedCard:
        return GeneratedCard(
            word="hello",
            note_type="AnkiForge Language",
            definition="a greeting",
            example="Hello, world!",
            audio_data=b"audio-bytes",
            image_data=b"image-bytes",
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

        # Должно быть 2 вызова save_media (audio + image)
        assert mock_media.call_count == 2
        mock_add.assert_called_once()
        fields = mock_add.call_args[0][2]
        assert fields["Word"] == "hello"
        assert fields["Definition"] == "a greeting"
        assert fields["Example"] == "Hello, world!"
        assert "[sound:" in fields["Audio"]
        assert "<img src=" in fields["Image"]

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
# Тесты _get_input_placeholder
# ---------------------------------------------------------------------------


class TestGetInputPlaceholder:
    """Проверяет подсказки для поля ввода."""

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
# Тесты _should_show_images_checkbox
# ---------------------------------------------------------------------------


class TestShouldShowImagesCheckbox:
    """Проверяет когда показывать чекбокс картинок."""

    def test_material_mode_shows_checkbox(self) -> None:
        from ankiforge.ui.generate_dialog import _should_show_images_checkbox

        assert _should_show_images_checkbox(GenerationMode.MATERIAL) is True

    def test_other_modes_hide_checkbox(self) -> None:
        from ankiforge.ui.generate_dialog import _should_show_images_checkbox

        for mode in GenerationMode:
            if mode != GenerationMode.MATERIAL:
                assert _should_show_images_checkbox(mode) is False, f"{mode} should not show images checkbox"
