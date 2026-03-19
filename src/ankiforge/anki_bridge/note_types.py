"""Управление типами заметок (note types) AnkiForge."""

from __future__ import annotations

from typing import Any

from ankiforge.anki_bridge.deck_manager import _get_mw

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

QA_NOTE_TYPE_NAME = "AnkiForge QA"

QA_FRONT_TEMPLATE = """\
<div class="ankiforge-card front">
  <div class="question">{{Question}}</div>
</div>"""

QA_BACK_TEMPLATE = """\
<div class="ankiforge-card back">
  <div class="question">{{Question}}</div>
  <hr id="answer">
  <div class="answer">{{Answer}}</div>
  <div class="branding">AnkiForge</div>
</div>"""

QA_CSS = """\
.ankiforge-card {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  text-align: center;
  padding: 2rem 1.5rem;
  max-width: 600px;
  margin: 0 auto;
  color: #1a1a2e;
  background: #ffffff;
}

.question {
  font-size: 1.6rem;
  font-weight: 600;
  line-height: 1.4;
  margin-bottom: 1rem;
}

hr#answer {
  border: none;
  border-top: 2px solid #e0e0e0;
  margin: 1.2rem 0;
}

.answer {
  font-size: 1.2rem;
  line-height: 1.6;
  text-align: left;
}

.branding {
  margin-top: 2rem;
  font-size: 0.7rem;
  color: #b0b0b0;
  letter-spacing: 0.05em;
}

/* Anki night mode */
.night_mode .ankiforge-card {
  color: #e0e0e0;
  background: #1a1a2e;
}

.night_mode hr#answer {
  border-top-color: #3a3a5e;
}

.night_mode .branding {
  color: #555;
}"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_qa_note_type() -> dict[str, Any]:
    """Создаёт или находит существующий note type 'AnkiForge QA'.

    Returns:
        Словарь note type (Anki model dict).
    """
    mw = _get_mw()
    models = mw.col.models

    existing = models.by_name(QA_NOTE_TYPE_NAME)
    if existing is not None:
        return existing  # type: ignore[no-any-return]

    model: dict[str, Any] = models.new(QA_NOTE_TYPE_NAME)
    model["name"] = QA_NOTE_TYPE_NAME

    # Поля: Question, Answer
    for field_name in ("Question", "Answer"):
        field = models.new_field(field_name)
        models.add_field(model, field)

    # Шаблон
    tmpl = models.new_template("Card 1")
    tmpl["qfmt"] = QA_FRONT_TEMPLATE
    tmpl["afmt"] = QA_BACK_TEMPLATE
    models.add_template(model, tmpl)

    model["css"] = QA_CSS

    models.add(model)
    return model
