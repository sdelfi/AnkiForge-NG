"""Questions generator — input questions -> AI answers -> QA cards."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ankiforge.anki_bridge.note_types import QA_NOTE_TYPE_NAME
from ankiforge.generators._retry import retry_api_call
from ankiforge.models import GeneratedCard, GenerationProgress

if TYPE_CHECKING:
    from collections.abc import Callable

    from ankiforge.models import CardRequest
    from ankiforge.openrouter.routing_client import GenerationClient

_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant creating flashcard answers. "
    "Given a question, provide a concise but comprehensive answer suitable for a flashcard. "
    "The answer should be compact (2-4 sentences) yet informative enough to fully answer the question. "
    "Do not repeat the question. Answer directly."
)


class QuestionsGenerator:
    """QA card generator from a list of questions."""

    def __init__(self, client: GenerationClient, model: str) -> None:
        self._client = client
        self._model = model

    def generate(
        self,
        request: CardRequest,
        progress_callback: Callable[[GenerationProgress], None],
    ) -> list[GeneratedCard]:
        """Generate QA cards from a list of questions.

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

        for question in questions:
            user_prompt = f"Question: {question}"
            answer = retry_api_call(
                lambda sp=system_prompt, up=user_prompt: self._client.generate_text(
                    up, self._model, temperature=0.3, system_prompt=sp
                ),
                item_label=question[:50],
            )
            progress.current_cost += self._client.last_cost

            cards.append(
                GeneratedCard(
                    word=question,
                    answer=answer,
                    note_type=QA_NOTE_TYPE_NAME,
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
