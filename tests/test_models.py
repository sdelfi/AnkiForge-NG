"""Tests for AnkiForge data models."""

from ankiforge.models import (
    AddonConfig,
    CardRequest,
    GeneratedCard,
    GenerationMode,
    GenerationProgress,
)


class TestGenerationMode:
    def test_all_modes_exist(self) -> None:
        assert GenerationMode.LANGUAGE.value == "language"
        assert GenerationMode.MATERIAL.value == "material"
        assert GenerationMode.IMAGE.value == "image"
        assert GenerationMode.AUDIO.value == "audio"
        assert GenerationMode.QUESTIONS.value == "questions"

    def test_mode_count(self) -> None:
        assert len(GenerationMode) == 5


class TestCardRequest:
    def test_create_minimal(self) -> None:
        req = CardRequest(
            mode=GenerationMode.QUESTIONS,
            input_text="What is Python?",
            target_deck="Test Deck",
        )
        assert req.mode == GenerationMode.QUESTIONS
        assert req.input_text == "What is Python?"
        assert req.target_deck == "Test Deck"
        assert req.create_new_deck is False
        assert req.include_images is False
        assert req.custom_prompt is None
        assert req.language == "en"

    def test_create_full(self) -> None:
        req = CardRequest(
            mode=GenerationMode.LANGUAGE,
            input_text="apple, banana",
            target_deck="Vocab",
            create_new_deck=True,
            include_images=True,
            custom_prompt="Define in simple terms",
            language="ru",
        )
        assert req.create_new_deck is True
        assert req.include_images is True
        assert req.custom_prompt == "Define in simple terms"
        assert req.language == "ru"


class TestGeneratedCard:
    def test_create_minimal(self) -> None:
        card = GeneratedCard(word="apple", note_type="AnkiForge QA")
        assert card.word == "apple"
        assert card.note_type == "AnkiForge QA"
        assert card.definition is None
        assert card.example is None
        assert card.answer is None
        assert card.audio_data is None
        assert card.image_data is None

    def test_create_language_card(self) -> None:
        card = GeneratedCard(
            word="ephemeral",
            definition="lasting for a very short time",
            example="The ephemeral beauty of cherry blossoms.",
            audio_data=b"\xff\xfb\x90\x00",
            image_data=b"\x89PNG",
            note_type="AnkiForge Language",
        )
        assert card.definition == "lasting for a very short time"
        assert card.example == "The ephemeral beauty of cherry blossoms."
        assert card.audio_data == b"\xff\xfb\x90\x00"
        assert card.image_data == b"\x89PNG"

    def test_create_qa_card(self) -> None:
        card = GeneratedCard(
            word="What is Python?",
            answer="A high-level programming language.",
            note_type="AnkiForge QA",
        )
        assert card.answer == "A high-level programming language."


class TestGenerationProgress:
    def test_create_default(self) -> None:
        progress = GenerationProgress(total_cards=10)
        assert progress.total_cards == 10
        assert progress.completed_cards == 0
        assert progress.estimated_cost == 0.0
        assert progress.current_cost == 0.0
        assert progress.is_cancelled is False

    def test_create_full(self) -> None:
        progress = GenerationProgress(
            total_cards=20,
            completed_cards=5,
            estimated_cost=0.15,
            current_cost=0.04,
            is_cancelled=False,
        )
        assert progress.completed_cards == 5
        assert progress.estimated_cost == 0.15


class TestAddonConfig:
    def test_create_default(self) -> None:
        config = AddonConfig()
        assert config.api_key == ""
        assert config.text_model == ""
        assert config.image_model == ""
        assert config.audio_model == ""
        assert config.language == "en"

    def test_create_full(self) -> None:
        config = AddonConfig(
            api_key="sk-or-v1-test",
            text_model="openai/gpt-4o",
            image_model="openai/dall-e-3",
            audio_model="openai/tts-1",
            language="ru",
        )
        assert config.api_key == "sk-or-v1-test"
        assert config.text_model == "openai/gpt-4o"
        assert config.language == "ru"
