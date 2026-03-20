"""Shared QSS styles and UI helpers for AnkiForge dialogs."""

from __future__ import annotations

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

/* ── Select button (mode chooser) ── */
QPushButton#selectBtn {
    background-color: palette(dark);
    color: palette(text);
    border: none;
    border-radius: 4px;
    padding: 6px 18px;
    font-weight: bold;
}
QPushButton#selectBtn:hover {
    background-color: palette(highlight);
    color: palette(highlighted-text);
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

    Args:
        tooltip_text: Text shown on hover.

    Returns:
        QLabel instance with the help icon.
    """
    from aqt.qt import QLabel  # type: ignore[import-not-found]

    label = QLabel("?")
    label.setFixedSize(18, 18)
    label.setToolTip(tooltip_text)
    label.setStyleSheet(
        "QLabel {"
        "  color: palette(mid);"
        "  border: 1px solid palette(mid);"
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
