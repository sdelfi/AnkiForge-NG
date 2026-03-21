"""Material card generator — map-reduce: paragraphs -> facts -> QA cards."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ankiforge.anki_bridge.note_types import QA_IMAGE_NOTE_TYPE_NAME, QA_NOTE_TYPE_NAME
from ankiforge.models import AnswerDetail, GeneratedCard, GenerationProgress

if TYPE_CHECKING:
    from collections.abc import Callable

    from ankiforge.models import CardRequest
    from ankiforge.openrouter.client import OpenRouterClient

_MIN_PARAGRAPH_LEN = 10
_MAX_HEADING_LEN = 150
_MERGE_TARGET = 3000
_MAX_PARAGRAPH_LEN = 4000
_SPLIT_TARGET = 2000

_EXTRACT_PROMPT = (
    "You are analyzing study material to create flashcards.\n"
    "Extract up to {max_facts} key concepts from the paragraph below.\n"
    "For each concept, classify its type and state the fact.\n\n"
    "Types:\n"
    "  DEFINITION — what something is (term → explanation)\n"
    "  FORMULA — equation, theorem, or rule\n"
    "  PROCEDURE — steps or algorithm\n"
    "  RELATION — comparison, difference, or connection between concepts\n"
    "  INSIGHT — intuition, geometric meaning, or deeper understanding\n"
    "  CODE — code snippet, API usage, or implementation pattern\n\n"
    "Rules:\n"
    "- Each fact must be a single atomic concept directly stated in the text\n"
    "- Do NOT invent or infer — only extract what is explicitly written\n"
    "- If the text contains code examples, extract them as [CODE] facts — "
    "reference the COMPLETE code block, never cherry-pick lines\n"
    "- Return as a numbered list in format: N. [TYPE] fact\n\n"
    "Example output:\n"
    "1. [DEFINITION] Photosynthesis is the process of converting light energy into chemical energy\n"
    "2. [PROCEDURE] Photosynthesis occurs in two stages: light reactions and the Calvin cycle\n"
    "3. [RELATION] Unlike respiration, photosynthesis produces oxygen as a byproduct\n"
    "4. [CODE] List comprehension creates a new list: result = [x*2 for x in range(10)]\n\n"
    "BAD examples (do NOT do this):\n"
    '- "Photosynthesis is important" ← too vague, not a specific fact\n'
    '- "Plants need water" ← not stated in the paragraph, inferred\n'
    "- Combining two facts into one line ← each line = one atomic concept\n\n"
    "Paragraph:\n{paragraph}"
)

_GENERATE_PROMPT = (
    "Create one question-answer flashcard for EACH fact below.\n"
    "Use the fact's TYPE to choose the right question format:\n\n"
    '  DEFINITION → "What is X?" or "Define X" → exact definition from the text\n'
    '  FORMULA → "What is the formula for X?" → formula + conditions\n'
    '  PROCEDURE → "What are the steps to X?" → numbered algorithm\n'
    '  RELATION → "What is the difference between X and Y?" or "How are X and Y related?" → key distinctions\n'
    '  INSIGHT → "What is the intuition behind X?" or "Explain the meaning of X" → explanation + example\n'
    '  CODE → "How do you X?" or "What does this code do?" → explanation + code snippet from the text\n\n'
    "Rules:\n"
    "- Base answers ONLY on the provided source text — do not add external knowledge\n"
    "- Each card = one atomic fact (minimum information principle)\n"
    "- The question must have exactly one correct answer\n"
    "- The card must be self-contained — understandable without the source text\n"
    "{answer_instructions}"
    "- If the source contains code examples, include the COMPLETE code block — never cherry-pick individual lines\n"
    "- Code blocks are atomic: include the full example or omit it, never truncate or rearrange\n"
    "- Code MUST be wrapped in triple backticks with language name: ```python\\ncode\\n```\n"
    "- NEVER paste raw code without triple backtick fences\n"
    "- Language: {language}\n\n"
    "GOOD flashcard examples:\n"
    "QUESTION: What is photosynthesis?\n"
    "ANSWER: Photosynthesis is the process by which plants convert light energy into chemical energy, "
    "producing glucose and oxygen from carbon dioxide and water.\n\n"
    "QUESTION: What is the difference between mitosis and meiosis?\n"
    "ANSWER: Mitosis produces two identical diploid cells, while meiosis produces four genetically "
    "unique haploid cells. Mitosis is for growth; meiosis is for reproduction.\n\n"
    "QUESTION: How do you implement an iterable class in Python?\n"
    "ANSWER: A class is iterable if it implements the __iter__ method. "
    "It can inherit from collections.abc.Iterable or just define __iter__ directly:\n"
    "```python\nclass SomeIterable1(collections.abc.Iterable):\n"
    "    def __iter__(self):\n"
    "        pass\n\n"
    "class SomeIterable2:\n"
    "    def __iter__(self):\n"
    "        pass\n\n"
    "print(isinstance(SomeIterable1(), collections.abc.Iterable))  # True\n"
    "print(isinstance(SomeIterable2(), collections.abc.Iterable))  # True\n```\n\n"
    "BAD flashcard examples (do NOT do this):\n"
    '- "Tell me about photosynthesis" ← too vague, no specific answer expected\n'
    "- Question with multiple facts in the answer ← split into separate cards\n"
    "- Answer that adds facts not in the source text ← only use provided material\n"
    "- Describing code without showing it ← always include the actual code snippet\n"
    "- Pasting code without ```language fences ← ALWAYS wrap code in triple backticks\n"
    "- Cherry-picking lines from a code block (e.g. only print() lines without the class definitions above them) "
    "← include the COMPLETE code example\n\n"
    "Source text:\n{paragraph}\n\n"
    "Facts to create cards for:\n{facts}\n\n"
    "Respond EXACTLY in this format (one pair per block, separated by blank lines):\n\n"
    "QUESTION: <specific question>\n"
    "ANSWER: <concise factual answer>\n"
)

_IMAGE_PROMPT_TEMPLATE = (
    "A simple, clear illustration for a flashcard about: '{topic}'. "
    "Clean educational style, no text, suitable for studying."
)

_ANSWER_INSTRUCTIONS: dict[str, str] = {
    "short": (
        "- Answers: 1-2 sentences maximum. Only the core fact, no elaboration\n"
        "- No examples unless the fact IS an example\n"
    ),
    "medium": (
        "- Answers: 2-4 sentences. State the fact, then add brief context or one example\n"
        "- Include a short example if it helps understanding\n"
    ),
    "detailed": (
        "- Answers: comprehensive explanation, 4-8 sentences\n"
        "- Include examples, analogies, edge cases, or practical implications\n"
        "- For CODE facts: explain WHY the code works, not just WHAT it does\n"
        "- Add nuances, common pitfalls, or related concepts when relevant\n"
    ),
}


class MaterialGenerator:
    """QA card generator from study material (map-reduce)."""

    def __init__(
        self,
        client: OpenRouterClient,
        text_model: str,
        image_model: str | None = None,
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
        """Generate QA cards from study material.

        Args:
            request: Request with material text in input_text.
            progress_callback: Callback for tracking progress.

        Returns:
            List of generated cards.

        Raises:
            ValueError: If no text found in input.
        """
        text = request.input_text.strip()
        if not text:
            msg = "No text found in input"
            raise ValueError(msg)

        use_images = request.include_images and self._image_model is not None
        note_type = QA_IMAGE_NOTE_TYPE_NAME if use_images else QA_NOTE_TYPE_NAME

        max_cards = 3
        if request.material_options is not None:
            max_cards = max(1, min(5, request.material_options.max_cards_per_paragraph))

        image_size = self._image_size
        if request.material_options is not None and request.material_options.image_size != "auto":
            image_size = request.material_options.image_size

        language = request.language or "en"

        detail = AnswerDetail.SHORT
        if request.material_options is not None:
            detail = request.material_options.answer_detail
        answer_instructions = _ANSWER_INSTRUCTIONS[detail.value]

        # If user set custom_prompt — single-pass mode
        if request.custom_prompt:
            return self._generate_single_pass(
                text, request.custom_prompt, note_type, use_images, image_size, progress_callback
            )

        paragraphs = self._prepare_paragraphs(text)

        # Phase 1 — Extract + Generate: extract facts and generate QA
        all_pairs: list[tuple[str, str]] = []
        progress = GenerationProgress(total_cards=len(paragraphs) * max_cards)

        for i, paragraph in enumerate(paragraphs):
            if progress.is_cancelled:
                break

            # Extract facts
            extract_prompt = _EXTRACT_PROMPT.format(max_facts=max_cards, paragraph=paragraph)
            facts = self._client.generate_text(extract_prompt, self._text_model, temperature=0.2)
            progress.current_cost += self._client.last_cost

            # Generate QA from facts + original paragraph
            generate_prompt = _GENERATE_PROMPT.format(
                language=language, paragraph=paragraph, facts=facts, answer_instructions=answer_instructions
            )
            response = self._client.generate_text(generate_prompt, self._text_model, temperature=0.3)
            progress.current_cost += self._client.last_cost
            pairs = self._parse_qa_pairs(response)
            all_pairs.extend(pairs)

            # Update completed and total after each paragraph
            progress.completed_cards = len(all_pairs)
            progress.total_cards = len(all_pairs) + (len(paragraphs) - i - 1) * max_cards
            progress_callback(progress)

        # Phase 2 — create cards (+ images if enabled)
        progress.total_cards = len(all_pairs)
        cards: list[GeneratedCard] = []

        for question, answer in all_pairs:
            image_data: bytes | None = None
            if use_images:
                image_prompt = _IMAGE_PROMPT_TEMPLATE.format(topic=question)
                assert self._image_model is not None
                image_data = self._client.generate_image(
                    image_prompt,
                    self._image_model,
                    size=image_size,
                )
                progress.current_cost += self._client.last_cost
                progress_callback(progress)

            cards.append(
                GeneratedCard(
                    word=question,
                    answer=answer,
                    image_data=image_data,
                    note_type=note_type,
                )
            )

            if progress.is_cancelled:
                break

        return cards

    def _generate_single_pass(
        self,
        text: str,
        custom_prompt: str,
        note_type: str,
        use_images: bool,
        image_size: str,
        progress_callback: Callable[[GenerationProgress], None],
    ) -> list[GeneratedCard]:
        """Single-pass generation with custom prompt."""
        prompt = f"{custom_prompt}\n\nMaterial:\n{text}"
        response = self._client.generate_text(prompt, self._text_model)
        text_cost = self._client.last_cost
        pairs = self._parse_qa_pairs(response)

        progress = GenerationProgress(total_cards=len(pairs))
        progress.current_cost = text_cost
        cards: list[GeneratedCard] = []

        for question, answer in pairs:
            image_data: bytes | None = None
            if use_images:
                image_prompt = _IMAGE_PROMPT_TEMPLATE.format(topic=question)
                assert self._image_model is not None
                image_data = self._client.generate_image(
                    image_prompt,
                    self._image_model,
                    size=image_size,
                )
                progress.current_cost += self._client.last_cost

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

    def _prepare_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs with filtering, topic grouping, and merging.

        Logic:
        1. Split by \\n\\n, filter junk (<10 chars)
        2. Group by topic: heading + content = one section
        3. Greedy merge of small sections up to _MERGE_TARGET
        4. Split overly long chunks

        Args:
            text: Source text.

        Returns:
            List of optimally-sized paragraphs.
        """
        raw = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not raw:
            return [text]

        # Split paragraphs where first line is a heading (separated by single \n from content)
        expanded: list[str] = []
        for p in raw:
            lines = p.split("\n", 1)
            if len(lines) == 2 and self._is_topic_heading(lines[0].strip()) and len(lines[1].strip()) > 0:
                expanded.append(lines[0].strip())
                expanded.append(lines[1].strip())
            else:
                expanded.append(p)

        # Filter junk (too short paragraphs)
        filtered = [p for p in expanded if len(p) >= _MIN_PARAGRAPH_LEN]
        if not filtered:
            return [text] if len(text) >= _MIN_PARAGRAPH_LEN else raw[:1] if raw else [text]

        # Mark heading indices: standalone headings + headings from expanded
        heading_indices: set[int] = set()
        for i, p in enumerate(filtered):
            if self._is_topic_heading(p):
                heading_indices.add(i)

        # Group by topic: heading starts a new section
        sections = self._group_by_topic_indexed(filtered, heading_indices)

        # Greedy merge of adjacent sections up to _MERGE_TARGET chars
        # If topic headings detected — don't merge across sections (each is self-contained)
        if heading_indices:
            merged = sections
        else:
            merged = []
            buf = sections[0]
            for s in sections[1:]:
                if len(buf) + len(s) + 2 <= _MERGE_TARGET:
                    buf = f"{buf}\n\n{s}"
                else:
                    merged.append(buf)
                    buf = s
            merged.append(buf)

        # Split long ones
        result: list[str] = []
        for p in merged:
            if len(p) <= _MAX_PARAGRAPH_LEN:
                result.append(p)
            else:
                result.extend(self._split_long_paragraph(p))

        return result if result else [text]

    @staticmethod
    def _is_topic_heading(text: str) -> bool:
        """Determine if a paragraph is a topic heading.

        Heading: single-line text 20-150 chars, doesn't end with period/comma,
        starts with uppercase letter, doesn't look like code.
        """
        stripped = text.strip()
        if not stripped or len(stripped) < 15 or len(stripped) > _MAX_HEADING_LEN:
            return False
        if "\n" in stripped:
            return False
        if stripped[-1] in ".;,":
            return False
        # Code lines are not headings
        if stripped.startswith(("class ", "def ", "import ", "from ", "return ", "@")):
            return False
        # Must start with an uppercase letter
        return stripped[0].isupper()

    @staticmethod
    def _group_by_topic_indexed(paragraphs: list[str], heading_indices: set[int]) -> list[str]:
        """Group paragraphs by topic using known heading indices.

        Each heading starts a new section. Content paragraphs are appended
        to the current section. If no headings — returns paragraphs as-is.

        Args:
            paragraphs: List of paragraphs.
            heading_indices: Indices of heading paragraphs.

        Returns:
            List of sections (each = heading + content joined by \\n\\n).
        """
        if len(paragraphs) <= 1 or not heading_indices:
            return paragraphs

        sections: list[str] = []
        current_parts: list[str] = [paragraphs[0]]

        for i, p in enumerate(paragraphs[1:], start=1):
            if i in heading_indices:
                sections.append("\n\n".join(current_parts))
                current_parts = [p]
            else:
                current_parts.append(p)

        if current_parts:
            sections.append("\n\n".join(current_parts))

        return sections

    def _split_long_paragraph(self, text: str) -> list[str]:
        """Split a long paragraph by sentences into sub-paragraphs of ~2000 chars."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            if current and current_len + len(sentence) > _SPLIT_TARGET:
                chunks.append(" ".join(current))
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len += len(sentence) + 1

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _parse_qa_pairs(self, response: str) -> list[tuple[str, str]]:
        """Parse AI response into a list of (question, answer) pairs.

        Args:
            response: AI response text.

        Returns:
            List of (question, answer) tuples.
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
