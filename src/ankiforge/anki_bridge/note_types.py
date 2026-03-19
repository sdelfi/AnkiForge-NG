"""Управление типами заметок (note types) AnkiForge."""

from __future__ import annotations

from typing import Any

from ankiforge.anki_bridge.deck_manager import _get_mw

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

QA_NOTE_TYPE_NAME = "AnkiForge QA"
QA_IMAGE_NOTE_TYPE_NAME = "AnkiForge QA+Image"
LANGUAGE_NOTE_TYPE_NAME = "AnkiForge Language"

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


# ---------------------------------------------------------------------------
# QA+Image note type
# ---------------------------------------------------------------------------

QA_IMAGE_FRONT_TEMPLATE = """\
<div class="ankiforge-card front">
  <div class="question">{{Question}}</div>
</div>"""

QA_IMAGE_BACK_TEMPLATE = """\
<div class="ankiforge-card back">
  <div class="question">{{Question}}</div>
  <hr id="answer">
  <div class="answer">{{Answer}}</div>
  {{#Image}}
  <div class="image">{{Image}}</div>
  {{/Image}}
  <div class="branding">AnkiForge</div>
</div>"""

QA_IMAGE_CSS = """\
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
  margin-bottom: 1rem;
}

.image img {
  max-width: 300px;
  max-height: 300px;
  border-radius: 12px;
  margin-top: 0.5rem;
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


def ensure_qa_image_note_type() -> dict[str, Any]:
    """Создаёт или находит существующий note type 'AnkiForge QA+Image'.

    Returns:
        Словарь note type (Anki model dict).
    """
    mw = _get_mw()
    models = mw.col.models

    existing = models.by_name(QA_IMAGE_NOTE_TYPE_NAME)
    if existing is not None:
        return existing  # type: ignore[no-any-return]

    model: dict[str, Any] = models.new(QA_IMAGE_NOTE_TYPE_NAME)
    model["name"] = QA_IMAGE_NOTE_TYPE_NAME

    # Поля: Question, Answer, Image
    for field_name in ("Question", "Answer", "Image"):
        field = models.new_field(field_name)
        models.add_field(model, field)

    # Шаблон
    tmpl = models.new_template("Card 1")
    tmpl["qfmt"] = QA_IMAGE_FRONT_TEMPLATE
    tmpl["afmt"] = QA_IMAGE_BACK_TEMPLATE
    models.add_template(model, tmpl)

    model["css"] = QA_IMAGE_CSS

    models.add(model)
    return model


# ---------------------------------------------------------------------------
# Language note type
# ---------------------------------------------------------------------------

LANGUAGE_FRONT_TEMPLATE = """\
<div class="ankiforge-card front">
  <div class="word">{{Word}}</div>
  <div class="audio">{{Audio}}</div>
</div>"""

LANGUAGE_BACK_TEMPLATE = """\
<div class="ankiforge-card back">
  <div class="word">{{Word}}</div>
  <div class="audio">{{Audio}}</div>
  <hr id="answer">
  <div class="definition">{{Definition}}</div>
  <div class="example">{{Example}}</div>
  {{#Image}}
  <div class="image">{{Image}}</div>
  {{/Image}}
  <div class="branding">AnkiForge</div>
</div>"""

LANGUAGE_CSS = """\
.ankiforge-card {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  text-align: center;
  padding: 2rem 1.5rem;
  max-width: 600px;
  margin: 0 auto;
  color: #1a1a2e;
  background: #ffffff;
}

.word {
  font-size: 2rem;
  font-weight: 700;
  line-height: 1.3;
  margin-bottom: 0.5rem;
}

.audio {
  margin-bottom: 0.5rem;
}

hr#answer {
  border: none;
  border-top: 2px solid #e0e0e0;
  margin: 1.2rem 0;
}

.definition {
  font-size: 1.3rem;
  font-weight: 600;
  line-height: 1.5;
  text-align: left;
  margin-bottom: 1rem;
  padding: 0.8rem;
  background: #f0f4ff;
  border-radius: 8px;
}

.example {
  font-size: 1.1rem;
  font-style: italic;
  line-height: 1.5;
  text-align: left;
  color: #5a6a8a;
  margin-bottom: 1rem;
  padding: 0.6rem 0.8rem;
  border-left: 3px solid #c0d0f0;
}

.image img {
  max-width: 300px;
  max-height: 300px;
  border-radius: 12px;
  margin-top: 0.5rem;
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

.night_mode .definition {
  background: #252545;
  color: #d0d8f0;
}

.night_mode .example {
  color: #8a9ac0;
  border-left-color: #4a5a80;
}

.night_mode .branding {
  color: #555;
}"""


def ensure_language_note_type() -> dict[str, Any]:
    """Создаёт или находит существующий note type 'AnkiForge Language'.

    Returns:
        Словарь note type (Anki model dict).
    """
    mw = _get_mw()
    models = mw.col.models

    existing = models.by_name(LANGUAGE_NOTE_TYPE_NAME)
    if existing is not None:
        return existing  # type: ignore[no-any-return]

    model: dict[str, Any] = models.new(LANGUAGE_NOTE_TYPE_NAME)
    model["name"] = LANGUAGE_NOTE_TYPE_NAME

    # Поля: Word, Audio, Definition, Example, Image
    for field_name in ("Word", "Audio", "Definition", "Example", "Image"):
        field = models.new_field(field_name)
        models.add_field(model, field)

    # Шаблон
    tmpl = models.new_template("Card 1")
    tmpl["qfmt"] = LANGUAGE_FRONT_TEMPLATE
    tmpl["afmt"] = LANGUAGE_BACK_TEMPLATE
    models.add_template(model, tmpl)

    model["css"] = LANGUAGE_CSS

    models.add(model)
    return model
