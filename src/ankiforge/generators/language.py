"""Генератор языковых карточек — ввод слов → definition + example + audio + image."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ankiforge.anki_bridge.note_types import LANGUAGE_NOTE_TYPE_NAME
from ankiforge.models import GeneratedCard, GenerationProgress

if TYPE_CHECKING:
    from collections.abc import Callable

    from ankiforge.models import CardRequest
    from ankiforge.openrouter.client import OpenRouterClient

_DEFAULT_SYSTEM_PROMPT = (
    "You are a language learning assistant creating flashcard content. "
    "Given a word or phrase, provide a clear definition and a natural usage example "
    "in the same language as the word. Do NOT translate. "
    "Respond EXACTLY in this format:\n"
    "DEFINITION: <concise definition>\n"
    "EXAMPLE: <one natural sentence using the word>"
)

_IMAGE_PROMPT_TEMPLATE = (
    "A simple, clear illustration representing the word '{word}'. Clean style, no text, suitable for a flashcard."
)


class LanguageGenerator:
    """Генератор Language-карточек из списка слов."""

    def __init__(
        self,
        client: OpenRouterClient,
        text_model: str,
        audio_model: str,
        image_model: str,
    ) -> None:
        self._client = client
        self._text_model = text_model
        self._audio_model = audio_model
        self._image_model = image_model

    def generate(
        self,
        request: CardRequest,
        progress_callback: Callable[[GenerationProgress], None],
    ) -> list[GeneratedCard]:
        """Генерирует Language-карточки из списка слов.

        Args:
            request: Запрос со словами в input_text (по одному на строку или через запятую).
            progress_callback: Callback для отслеживания прогресса.

        Returns:
            Список сгенерированных карточек.

        Raises:
            ValueError: Если не найдено слов во входном тексте.
        """
        words = self._parse_words(request.input_text)
        progress = GenerationProgress(total_cards=len(words))
        cards: list[GeneratedCard] = []

        for word in words:
            # Текст: definition + example
            prompt = self._build_prompt(word, request.language, request.custom_prompt)
            response = self._client.generate_text(prompt, self._text_model)
            definition, example = self._parse_response(response)

            # Аудио: произношение слова
            audio_data = self._client.generate_audio(word, self._audio_model)

            # Изображение: ассоциация
            image_prompt = _IMAGE_PROMPT_TEMPLATE.format(word=word)
            image_data = self._client.generate_image(image_prompt, self._image_model)

            cards.append(
                GeneratedCard(
                    word=word,
                    definition=definition,
                    example=example,
                    audio_data=audio_data,
                    image_data=image_data,
                    note_type=LANGUAGE_NOTE_TYPE_NAME,
                )
            )

            progress.completed_cards += 1
            progress_callback(progress)

            if progress.is_cancelled:
                break

        return cards

    def _build_prompt(self, word: str, language: str, custom_prompt: str | None) -> str:
        """Строит промпт для генерации definition + example.

        Args:
            word: Слово или фраза.
            language: Язык карточек.
            custom_prompt: Кастомный промпт пользователя.

        Returns:
            Полный промпт для AI.
        """
        if custom_prompt:
            return f"{custom_prompt}\n\nWord: {word}\nLanguage: {language}"

        return f"{_DEFAULT_SYSTEM_PROMPT}\n\nLanguage: {language}\nWord: {word}"

    def _parse_words(self, text: str) -> list[str]:
        """Парсит слова из текста (по строкам или через запятую).

        Args:
            text: Текст со словами.

        Returns:
            Список слов.

        Raises:
            ValueError: Если не найдено слов.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # Если одна строка — пробуем разделить по запятым
        words = [w.strip() for w in lines[0].split(",") if w.strip()] if len(lines) == 1 and "," in lines[0] else lines

        if not words:
            msg = "Не найдено слов во входном тексте"
            raise ValueError(msg)
        return words

    def _parse_response(self, response: str) -> tuple[str, str]:
        """Парсит ответ AI в definition и example.

        Args:
            response: Текст ответа от AI.

        Returns:
            Кортеж (definition, example).
        """
        if not response.strip():
            return ("", "")

        # Пробуем найти маркеры DEFINITION: и EXAMPLE:
        def_match = re.search(r"DEFINITION:\s*(.+?)(?=\nEXAMPLE:|\Z)", response, re.DOTALL)
        ex_match = re.search(r"EXAMPLE:\s*(.+)", response, re.DOTALL)

        if def_match and ex_match:
            return (def_match.group(1).strip(), ex_match.group(1).strip())

        # Fallback: первое предложение = definition, остальное = example
        sentences = re.split(r"(?<=\.)\s+", response.strip(), maxsplit=1)
        if len(sentences) >= 2:
            return (sentences[0].strip(), sentences[1].strip())

        return (response.strip(), "")
