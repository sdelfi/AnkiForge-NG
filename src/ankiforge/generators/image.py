"""Image card generator — input questions/topics -> QA + image -> QA+Image cards."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ankiforge.anki_bridge.note_types import QA_IMAGE_NOTE_TYPE_NAME
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

_IMAGE_PROMPT_TEMPLATE = (
    "A simple, clear illustration for a flashcard about: '{topic}'. "
    "Clean educational style, no text, suitable for studying."
)


class ImageGenerator:
    """QA+Image card generator from a list of questions."""

    def __init__(
        self,
        client: OpenRouterClient,
        text_model: str,
        image_model: str,
        image_size: str = "auto",
    ) -> None:
        self._client = client
        self._text_model = text_model
        self._image_model = image_model
        self._image_size = image_size

    def generate(
        self,
        request: CardRequest,
        progress_callback: Callable[[GenerationProgress], None],
    ) -> list[GeneratedCard]:
        """Generate QA+Image cards from a list of questions.

        Args:
            request: Request with questions in input_text (one per line).
            progress_callback: Callback for tracking progress.

        Returns:
            List of generated cards.

        Raises:
            ValueError: If no questions found in input text.
        """
        questions = self._parse_questions(request.input_text)
        progress = GenerationProgress(total_cards=len(questions))
        cards: list[GeneratedCard] = []

        system_prompt = request.custom_prompt or _SYSTEM_PROMPT
        image_size = request.image_size if request.image_size != "auto" else self._image_size

        for question in questions:
            text_prompt = f"{system_prompt}\n\nQuestion: {question}"
            answer = self._client.generate_text(text_prompt, self._text_model, temperature=0.3)
            progress.current_cost += self._client.last_cost

            image_prompt = _IMAGE_PROMPT_TEMPLATE.format(topic=question)
            image_data = self._client.generate_image(image_prompt, self._image_model, size=image_size)
            progress.current_cost += self._client.last_cost

            cards.append(
                GeneratedCard(
                    word=question,
                    answer=answer,
                    image_data=image_data,
                    note_type=QA_IMAGE_NOTE_TYPE_NAME,
                )
            )

            progress.completed_cards += 1
            progress_callback(progress)

            if progress.is_cancelled:
                break

        return cards

    def _parse_questions(self, text: str) -> list[str]:
        """Parse questions from text (one per line).

        Args:
            text: Text with questions.

        Returns:
            List of questions.

        Raises:
            ValueError: If no questions found.
        """
        questions = [line.strip() for line in text.splitlines() if line.strip()]
        if not questions:
            msg = "No questions found in input"
            raise ValueError(msg)
        return questions
