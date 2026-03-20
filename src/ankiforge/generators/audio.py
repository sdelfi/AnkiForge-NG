"""Генератор карточек с аудио — ввод вопросов → QA + audio → QA+Audio карточки."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ankiforge.anki_bridge.note_types import QA_AUDIO_NOTE_TYPE_NAME
from ankiforge.models import GeneratedCard, GenerationProgress

if TYPE_CHECKING:
    from collections.abc import Callable

    from ankiforge.models import CardRequest
    from ankiforge.openrouter.client import OpenRouterClient

_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant creating flashcard answers. "
    "Given a question, provide a concise but comprehensive answer suitable for a flashcard. "
    "The answer should be compact (2-4 sentences) yet informative enough to fully answer the question. "
    "Do not repeat the question. Answer directly."
)


class AudioGenerator:
    """Генератор QA+Audio карточек из списка вопросов."""

    def __init__(
        self,
        client: OpenRouterClient,
        text_model: str,
        audio_model: str,
        voice: str = "alloy",
    ) -> None:
        self._client = client
        self._text_model = text_model
        self._audio_model = audio_model
        self._voice = voice

    def generate(
        self,
        request: CardRequest,
        progress_callback: Callable[[GenerationProgress], None],
    ) -> list[GeneratedCard]:
        """Генерирует QA+Audio карточки из списка вопросов.

        Args:
            request: Запрос с вопросами в input_text (по одному на строку).
            progress_callback: Callback для отслеживания прогресса.

        Returns:
            Список сгенерированных карточек.

        Raises:
            ValueError: Если не найдено вопросов во входном тексте.
        """
        questions = self._parse_questions(request.input_text)
        progress = GenerationProgress(total_cards=len(questions))
        cards: list[GeneratedCard] = []

        system_prompt = request.custom_prompt or _SYSTEM_PROMPT
        voice = request.voice if request.voice != "alloy" else self._voice

        for question in questions:
            text_prompt = f"{system_prompt}\n\nQuestion: {question}"
            answer = self._client.generate_text(text_prompt, self._text_model, temperature=0.3)
            progress.current_cost += self._client.last_cost

            audio_data = self._client.generate_audio(answer, self._audio_model, voice=voice)
            progress.current_cost += self._client.last_cost

            cards.append(
                GeneratedCard(
                    word=question,
                    answer=answer,
                    audio_data=audio_data,
                    note_type=QA_AUDIO_NOTE_TYPE_NAME,
                )
            )

            progress.completed_cards += 1
            progress_callback(progress)

            if progress.is_cancelled:
                break

        return cards

    def _parse_questions(self, text: str) -> list[str]:
        """Парсит вопросы из текста (по одному на строку).

        Args:
            text: Текст с вопросами.

        Returns:
            Список вопросов.

        Raises:
            ValueError: Если не найдено вопросов.
        """
        questions = [line.strip() for line in text.splitlines() if line.strip()]
        if not questions:
            msg = "Не найдено вопросов во входном тексте"
            raise ValueError(msg)
        return questions
