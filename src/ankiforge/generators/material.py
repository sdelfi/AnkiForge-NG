"""Генератор карточек по материалам — ввод текста → чанкинг → QA карточки (опц. с картинками)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ankiforge.anki_bridge.note_types import QA_IMAGE_NOTE_TYPE_NAME, QA_NOTE_TYPE_NAME
from ankiforge.models import GeneratedCard, GenerationProgress

if TYPE_CHECKING:
    from collections.abc import Callable

    from ankiforge.models import CardRequest
    from ankiforge.openrouter.client import OpenRouterClient

_DEFAULT_CHUNK_SIZE = 6000

_SYSTEM_PROMPT = (
    "You are an expert at creating flashcards from study material. "
    "Given a text, extract the most important atomic facts and generate question-answer pairs. "
    "Each pair should test one specific concept or fact. "
    "The number of pairs should be proportional to the amount of material — "
    "aim for 1-3 pairs per paragraph or key concept.\n\n"
    "Respond EXACTLY in this format (one pair per block, separated by blank lines):\n\n"
    "QUESTION: <clear, specific question>\n"
    "ANSWER: <concise but complete answer (2-4 sentences)>\n\n"
    "QUESTION: <next question>\n"
    "ANSWER: <next answer>"
)

_IMAGE_PROMPT_TEMPLATE = (
    "A simple, clear illustration for a flashcard about: '{topic}'. "
    "Clean educational style, no text, suitable for studying."
)


class MaterialGenerator:
    """Генератор QA-карточек из учебного материала."""

    def __init__(
        self,
        client: OpenRouterClient,
        text_model: str,
        image_model: str | None = None,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> None:
        self._client = client
        self._text_model = text_model
        self._image_model = image_model
        self._chunk_size = chunk_size

    def generate(
        self,
        request: CardRequest,
        progress_callback: Callable[[GenerationProgress], None],
    ) -> list[GeneratedCard]:
        """Генерирует QA-карточки из учебного материала.

        Args:
            request: Запрос с текстом материала в input_text.
            progress_callback: Callback для отслеживания прогресса.

        Returns:
            Список сгенерированных карточек.

        Raises:
            ValueError: Если не найдено текста во входных данных.
        """
        text = request.input_text.strip()
        if not text:
            msg = "Не найдено текста во входных данных"
            raise ValueError(msg)

        use_images = request.include_images and self._image_model is not None
        note_type = QA_IMAGE_NOTE_TYPE_NAME if use_images else QA_NOTE_TYPE_NAME

        chunks = self._split_into_chunks(text, max_chars=self._chunk_size)
        all_pairs: list[tuple[str, str]] = []

        for chunk in chunks:
            prompt = f"{_SYSTEM_PROMPT}\n\nMaterial:\n{chunk}"
            response = self._client.generate_text(prompt, self._text_model)
            pairs = self._parse_qa_pairs(response)
            all_pairs.extend(pairs)

        progress = GenerationProgress(total_cards=len(all_pairs))
        cards: list[GeneratedCard] = []

        for question, answer in all_pairs:
            image_data: bytes | None = None
            if use_images:
                image_prompt = _IMAGE_PROMPT_TEMPLATE.format(topic=question)
                image_data = self._client.generate_image(image_prompt, self._image_model)  # type: ignore[arg-type]

            cards.append(
                GeneratedCard(
                    word=question,
                    answer=answer,
                    image_data=image_data,
                    note_type=note_type,
                )
            )

            progress.completed_cards += 1
            progress_callback(progress)

            if progress.is_cancelled:
                break

        return cards

    def _split_into_chunks(self, text: str, max_chars: int) -> list[str]:
        """Разбивает текст на чанки по абзацам с учётом max_chars.

        Args:
            text: Исходный текст.
            max_chars: Максимальная длина чанка в символах.

        Returns:
            Список чанков.
        """
        text = text.strip()
        if not text:
            return []

        if len(text) <= max_chars:
            return [text]

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            return [text]

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for paragraph in paragraphs:
            paragraph_len = len(paragraph)

            if current and current_len + paragraph_len + 2 > max_chars:
                chunks.append("\n\n".join(current))
                current = [paragraph]
                current_len = paragraph_len
            else:
                current.append(paragraph)
                current_len += paragraph_len + (2 if current_len > 0 else 0)

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def _parse_qa_pairs(self, response: str) -> list[tuple[str, str]]:
        """Парсит ответ AI в список пар (question, answer).

        Args:
            response: Текст ответа от AI.

        Returns:
            Список кортежей (question, answer).
        """
        if not response.strip():
            return []

        pairs: list[tuple[str, str]] = []
        pattern = re.compile(
            r"QUESTION:\s*(.+?)\s*\nANSWER:\s*(.+?)(?=\n\s*\nQUESTION:|\n\s*QUESTION:|\Z)",
            re.DOTALL,
        )

        for match in pattern.finditer(response):
            question = match.group(1).strip()
            answer = match.group(2).strip()
            if question and answer:
                pairs.append((question, answer))

        return pairs
