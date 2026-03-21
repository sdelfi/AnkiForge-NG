"""Progress bar, cost estimation, and generation cancellation (QThread)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ankiforge.models import GenerationMode, GenerationProgress

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Protocol

    from ankiforge.openrouter.client import OpenRouterClient
    from ankiforge.openrouter.models import Model

    class _CardGenerator(Protocol):
        def generate(
            self,
            request: CardRequest,
            progress_callback: Callable[[GenerationProgress], None],
        ) -> list[GeneratedCard]: ...

    from ankiforge.models import CardRequest, GeneratedCard


# ---------------------------------------------------------------------------
# Utilities (testable without Qt)
# ---------------------------------------------------------------------------


def _count_input_items(input_text: str, mode: GenerationMode, *, max_cards_per_paragraph: int = 3) -> int:
    """Count input items by mode.

    Args:
        input_text: Input text.
        mode: Generation mode.
        max_cards_per_paragraph: Max cards per paragraph (for MATERIAL).

    Returns:
        Expected number of cards.
    """

    text = input_text.strip()
    if not text:
        return 0

    if mode == GenerationMode.LANGUAGE:
        items: list[str] = []
        for line in text.splitlines():
            items.extend(w.strip() for w in line.split(",") if w.strip())
        return len(items)

    if mode == GenerationMode.MATERIAL:
        # Use real splitting logic from MaterialGenerator
        from ankiforge.generators.material import MaterialGenerator

        gen = MaterialGenerator.__new__(MaterialGenerator)
        chunks = gen._prepare_paragraphs(text)
        return len(chunks) * max_cards_per_paragraph

    # QUESTIONS, IMAGE, AUDIO — by lines
    return len([line for line in text.splitlines() if line.strip()])


def _format_cost(cost: float) -> str:
    """Format cost for display.

    Args:
        cost: Cost in USD.

    Returns:
        Formatted string.
    """
    if cost < 0.001:
        return "< $0.001"
    return f"${cost:.3f}"


def _find_model_by_id(model_id: str, models: list[Model]) -> Model | None:
    """Find model by ID in list.

    Args:
        model_id: Model ID.
        models: List of models.

    Returns:
        Model or None.
    """
    if not model_id:
        return None
    for model in models:
        if model.id == model_id:
            return model
    return None


def _estimate_and_format_cost(
    *,
    client: OpenRouterClient,
    mode: GenerationMode,
    card_count: int,
    models: list[Model],
    text_model_id: str,
    image_model_id: str,
    audio_model_id: str,
) -> str:
    """Calculate and format cost estimate.

    Args:
        client: OpenRouter client.
        mode: Generation mode.
        card_count: Number of cards.
        models: List of available models.
        text_model_id: Text model ID.
        image_model_id: Image model ID.
        audio_model_id: Audio model ID.

    Returns:
        Formatted cost string.
    """
    text_model = _find_model_by_id(text_model_id, models)
    image_model = _find_model_by_id(image_model_id, models)
    audio_model = _find_model_by_id(audio_model_id, models)

    cost = client.estimate_cost(
        mode.value,
        card_count,
        text_model=text_model,
        image_model=image_model,
        audio_model=audio_model,
    )
    return f"Estimate: {_format_cost(cost)}"


# ---------------------------------------------------------------------------
# GenerationWorker — QThread for background generation
# ---------------------------------------------------------------------------


class GenerationWorker:
    """QThread wrapper for background card generation.

    Emits signals: progress_updated, finished, error.
    """

    def __init__(
        self,
        generator: _CardGenerator,
        request: CardRequest,
    ) -> None:
        from aqt.qt import QThread, pyqtSignal  # type: ignore[import-not-found]

        class _Worker(QThread):  # type: ignore[misc]
            progress_updated = pyqtSignal(object)
            finished_ok = pyqtSignal(list)
            error_occurred = pyqtSignal(str)

            def __init__(
                inner_self,  # noqa: N805
                gen: _CardGenerator,
                req: CardRequest,
            ) -> None:
                super().__init__()
                inner_self._generator = gen
                inner_self._request = req
                inner_self._progress = GenerationProgress(total_cards=0)

            def run(inner_self) -> None:  # noqa: N805
                try:

                    def progress_cb(progress: GenerationProgress) -> None:
                        inner_self._progress = progress
                        inner_self.progress_updated.emit(progress)

                    cards = inner_self._generator.generate(inner_self._request, progress_cb)
                    inner_self.finished_ok.emit(cards)
                except Exception as e:  # noqa: BLE001
                    inner_self.error_occurred.emit(str(e))

            def cancel(inner_self) -> None:  # noqa: N805
                inner_self._progress.is_cancelled = True

        self._worker = _Worker(generator, request)

    @property
    def worker(self) -> object:
        """Return the internal QThread."""
        return self._worker

    def start(self) -> None:
        """Start generation in background thread."""
        self._worker.start()

    def cancel(self) -> None:
        """Cancel generation."""
        self._worker.cancel()

    def connect_progress(self, callback: Callable[[GenerationProgress], None]) -> None:
        """Connect callback to progress signal."""
        self._worker.progress_updated.connect(callback)

    def connect_finished(self, callback: Callable[[list[GeneratedCard]], None]) -> None:
        """Connect callback to finished signal."""
        self._worker.finished_ok.connect(callback)

    def connect_error(self, callback: Callable[[str], None]) -> None:
        """Connect callback to error signal."""
        self._worker.error_occurred.connect(callback)


# ---------------------------------------------------------------------------
# ProgressWidget — progress widget (bar + Cancel only)
# ---------------------------------------------------------------------------


class ProgressWidget:
    """Generation progress widget: progress bar and cancel button."""

    def __init__(self, parent: object) -> None:
        from aqt.qt import (
            QProgressBar,
            QVBoxLayout,
            QWidget,
        )

        self._widget = QWidget(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 8, 0, 0)
        self._widget.setLayout(layout)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        self._widget.setVisible(False)

    @property
    def widget(self) -> object:
        """Return Qt widget for layout insertion."""
        return self._widget

    def show(self, total_cards: int) -> None:
        """Show widget with initial data.

        Args:
            total_cards: Expected number of cards.
        """
        self._progress_bar.setMaximum(total_cards)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat(f"0/{total_cards} cards")
        self._widget.setVisible(True)

    def update_progress(self, progress: GenerationProgress) -> None:
        """Update progress bar.

        Args:
            progress: Current progress.
        """
        if progress.total_cards != self._progress_bar.maximum():
            self._progress_bar.setMaximum(progress.total_cards)
        self._progress_bar.setValue(progress.completed_cards)
        self._progress_bar.setFormat(f"{progress.completed_cards}/{progress.total_cards} cards")

    def finish(self) -> None:
        """Finalize widget."""

    def hide(self) -> None:
        """Hide widget."""
        self._widget.setVisible(False)
