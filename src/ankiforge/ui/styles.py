"""Shared QSS styles and UI helpers for AnkiForge dialogs."""

from __future__ import annotations


def get_dialog_size(
    width_pct: float = 0.4,
    height_pct: float = 0.6,
    min_w: int = 480,
    min_h: int = 400,
) -> tuple[int, int]:
    """Calculate dialog size as a percentage of available screen.

    Args:
        width_pct: Screen width fraction (0.0–1.0).
        height_pct: Screen height fraction (0.0–1.0).
        min_w: Minimum width.
        min_h: Minimum height.

    Returns:
        (width, height) in pixels.
    """
    from aqt.qt import QApplication  # type: ignore[import-not-found]

    screen = QApplication.primaryScreen()
    if screen is not None:
        geom = screen.availableGeometry()
        w = max(int(geom.width() * width_pct), min_w)
        h = max(int(geom.height() * height_pct), min_h)
    else:
        w, h = min_w, min_h
    return w, h


def wrap_in_scroll_area(content_widget: object) -> object:
    """Wrap widget in QScrollArea with vertical scrolling.

    Args:
        content_widget: QWidget with dialog content.

    Returns:
        QScrollArea ready to be added to dialog layout.
    """
    from aqt.qt import QScrollArea, Qt

    scroll = QScrollArea()
    scroll.setWidget(content_widget)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    return scroll


DIALOG_QSS = """
/* ── GroupBox ── */
QGroupBox {
    font-weight: bold;
    border: 1px solid palette(mid);
    border-radius: 6px;
    margin-top: 16px;
    padding: 14px 10px 10px 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}

/* ── Text inputs ── */
QPlainTextEdit, QTextEdit {
    border: 1px solid palette(dark);
    border-radius: 4px;
    padding: 6px;
}
QPlainTextEdit:focus, QTextEdit:focus {
    border-color: palette(highlight);
}
QLineEdit {
    border: 1px solid palette(dark);
    border-radius: 4px;
    padding: 4px 6px;
}
QLineEdit:focus {
    border-color: palette(highlight);
}

/* ── ComboBox ── */
QComboBox {
    border: 1px solid palette(dark);
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 28px;
}
QComboBox:focus {
    border-color: palette(highlight);
}
QComboBox:editable {
    padding: 4px 6px;
}
QComboBox QAbstractItemView {
    selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
    outline: none;
}

/* ── Primary button (Generate / Connect) ── */
QPushButton#generateBtn, QPushButton#connectBtn {
    background-color: palette(highlight);
    color: palette(highlighted-text);
    border: none;
    border-radius: 5px;
    padding: 8px 28px;
    font-weight: bold;
    font-size: 13px;
    min-width: 130px;
}
QPushButton#generateBtn:hover, QPushButton#connectBtn:hover {
    opacity: 0.85;
}
QPushButton#generateBtn:disabled, QPushButton#connectBtn:disabled {
    background-color: palette(dark);
    color: palette(mid);
}

/* ── Secondary button (Again) ── */
QPushButton#againBtn {
    background-color: palette(dark);
    color: palette(text);
    border: 1px solid palette(mid);
    border-radius: 5px;
    padding: 8px 18px;
    font-size: 13px;
    min-width: 90px;
}
QPushButton#againBtn:hover {
    background-color: palette(mid);
}

/* ── Generic button hover ── */
QPushButton:hover {
    border-color: palette(highlight);
}

/* ── ProgressBar ── */
QProgressBar {
    border: 1px solid palette(dark);
    border-radius: 4px;
    text-align: center;
    min-height: 18px;
}
QProgressBar::chunk {
    background-color: palette(highlight);
    border-radius: 3px;
}

/* ── CheckBox ── */
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid palette(dark);
    border-radius: 3px;
}
QCheckBox::indicator:checked {
    background-color: palette(highlight);
    border-color: palette(highlight);
}

/* ── Status colors (hardcoded by design) ── */
QLabel#statusSuccess {
    color: #4caf50;
}
QLabel#statusError {
    color: #f44336;
}
"""


def make_help_icon(tooltip_text: str) -> object:
    """Create a small circled '?' label with a tooltip.

    Shows tooltip on both hover and click for better discoverability.

    Args:
        tooltip_text: Text shown on hover/click.

    Returns:
        QLabel instance with the help icon.
    """
    from aqt.qt import QCursor, QEvent, QLabel, QObject, Qt, QToolTip

    class _HelpFilter(QObject):  # type: ignore[misc]
        """Event filter that shows tooltip on mouse click."""

        def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
            if event.type() == QEvent.Type.MouseButtonRelease:
                QToolTip.showText(QCursor.pos(), obj.toolTip(), obj)
                return True
            result: bool = super().eventFilter(obj, event)
            return result

    label = QLabel("?")
    label.setFixedSize(18, 18)
    label.setCursor(Qt.CursorShape.PointingHandCursor)
    label.setToolTip(tooltip_text)
    help_filter = _HelpFilter(label)
    label.installEventFilter(help_filter)
    label.setStyleSheet(
        "QLabel {"
        "  color: palette(text);"
        "  border: 1px solid palette(text);"
        "  border-radius: 9px;"
        "  font-size: 11px;"
        "  font-weight: bold;"
        "  qproperty-alignment: AlignCenter;"
        "}"
        "QLabel:hover {"
        "  color: palette(highlight);"
        "  border-color: palette(highlight);"
        "}"
    )
    return label
