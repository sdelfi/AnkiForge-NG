"""Прогресс-бар, оценка стоимости и отмена генерации (QThread)."""

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
# Утилиты (тестируемые без Qt)
# ---------------------------------------------------------------------------


def _count_input_items(input_text: str, mode: GenerationMode, *, max_cards_per_paragraph: int = 3) -> int:
    """Подсчитывает количество элементов ввода по режиму.

    Args:
        input_text: Текст ввода.
        mode: Режим генерации.
        max_cards_per_paragraph: Макс. карточек на абзац (для MATERIAL).

    Returns:
        Ожидаемое количество карточек.
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
        # Используем реальную логику разбиения из MaterialGenerator
        from ankiforge.generators.material import MaterialGenerator

        gen = MaterialGenerator.__new__(MaterialGenerator)
        chunks = gen._prepare_paragraphs(text)
        return len(chunks) * max_cards_per_paragraph

    # QUESTIONS, IMAGE, AUDIO — по строкам
    return len([line for line in text.splitlines() if line.strip()])


def _format_cost(cost: float) -> str:
    """Форматирует стоимость для отображения.

    Args:
        cost: Стоимость в долларах.

    Returns:
        Форматированная строка.
    """
    if cost < 0.001:
        return "< $0.001"
    return f"${cost:.3f}"


def _find_model_by_id(model_id: str, models: list[Model]) -> Model | None:
    """Находит модель по ID в списке.

    Args:
        model_id: ID модели.
        models: Список моделей.

    Returns:
        Model или None.
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
    """Рассчитывает и форматирует оценку стоимости.

    Args:
        client: OpenRouter клиент.
        mode: Режим генерации.
        card_count: Количество карточек.
        models: Список доступных моделей.
        text_model_id: ID текстовой модели.
        image_model_id: ID image модели.
        audio_model_id: ID audio модели.

    Returns:
        Форматированная строка стоимости.
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
    return f"Оценка стоимости: {_format_cost(cost)}"


# ---------------------------------------------------------------------------
# GenerationWorker — QThread для генерации в фоне
# ---------------------------------------------------------------------------


class GenerationWorker:
    """QThread-обёртка для фоновой генерации карточек.

    Эмитит сигналы: progress_updated, finished, error.
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
        """Возвращает внутренний QThread."""
        return self._worker

    def start(self) -> None:
        """Запускает генерацию в фоновом потоке."""
        self._worker.start()

    def cancel(self) -> None:
        """Отменяет генерацию."""
        self._worker.cancel()

    def connect_progress(self, callback: Callable[[GenerationProgress], None]) -> None:
        """Подключает callback к сигналу прогресса."""
        self._worker.progress_updated.connect(callback)

    def connect_finished(self, callback: Callable[[list[GeneratedCard]], None]) -> None:
        """Подключает callback к сигналу завершения."""
        self._worker.finished_ok.connect(callback)

    def connect_error(self, callback: Callable[[str], None]) -> None:
        """Подключает callback к сигналу ошибки."""
        self._worker.error_occurred.connect(callback)


# ---------------------------------------------------------------------------
# ProgressWidget — виджет прогресса (только бар + Cancel)
# ---------------------------------------------------------------------------


class ProgressWidget:
    """Виджет прогресса генерации: прогресс-бар и кнопка отмены."""

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
        """Возвращает Qt-виджет для вставки в layout."""
        return self._widget

    def show(self, total_cards: int) -> None:
        """Показывает виджет с начальными данными.

        Args:
            total_cards: Ожидаемое количество карточек.
        """
        self._progress_bar.setMaximum(total_cards)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat(f"0/{total_cards} карточек")
        self._widget.setVisible(True)

    def update_progress(self, progress: GenerationProgress) -> None:
        """Обновляет прогресс-бар.

        Args:
            progress: Текущий прогресс.
        """
        if progress.total_cards != self._progress_bar.maximum():
            self._progress_bar.setMaximum(progress.total_cards)
        self._progress_bar.setValue(progress.completed_cards)
        self._progress_bar.setFormat(f"{progress.completed_cards}/{progress.total_cards} карточек")

    def finish(self) -> None:
        """Финализирует виджет."""

    def hide(self) -> None:
        """Скрывает виджет."""
        self._widget.setVisible(False)
