"""Управление типами заметок (note types) AnkiForge."""

from __future__ import annotations

from typing import Any

from ankiforge.anki_bridge.deck_manager import _get_mw

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

QA_NOTE_TYPE_NAME = "AnkiForge QA"
QA_IMAGE_NOTE_TYPE_NAME = "AnkiForge QA+Image"
QA_AUDIO_NOTE_TYPE_NAME = "AnkiForge QA+Audio"
LANGUAGE_NOTE_TYPE_NAME = "AnkiForge Language"

# highlight.js — подсветка синтаксиса в code blocks
_HLJS_SCRIPT = (
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0'
    '/highlight.min.js"></script>\n'
    "<script>hljs.highlightAll();</script>"
)

# ---------------------------------------------------------------------------
# Design tokens & shared CSS
# ---------------------------------------------------------------------------

_BASE_CSS = """\
/* ── Design tokens ── */
:root {
  --af-text-primary: #1a1a2e;
  --af-text-secondary: #6b7280;
  --af-text-muted: #9ca3af;
  --af-border: #e5e7eb;
  --af-surface: transparent;
  --af-surface-accent: #f0f4ff;
  --af-accent-border: #c0d0f0;
  --af-code-bg: #f4f4f8;
  --af-code-inline-bg: #eef0f5;
  --af-branding: #c0c0c0;
}

.night_mode {
  --af-text-primary: #e2e8f0;
  --af-text-secondary: #9ca3af;
  --af-text-muted: #6b7280;
  --af-border: #334155;
  --af-surface: transparent;
  --af-surface-accent: #1e293b;
  --af-accent-border: #475569;
  --af-code-bg: #1e293b;
  --af-code-inline-bg: #1e293b;
  --af-branding: #4b5563;
}

/* ── Base card ── */
.ankiforge-card {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  text-align: center;
  padding: 2rem 1.5rem;
  max-width: 600px;
  margin: 0 auto;
  color: var(--af-text-primary);
  background: var(--af-surface);
  line-height: 1.6;
}

/* ── Question — front face: large & bold ── */
.front .question {
  font-size: 1.4rem;
  font-weight: 600;
  line-height: 1.45;
  margin-bottom: 1rem;
}

/* ── Question — back face: demoted for cognitive load ── */
.back .question {
  font-size: 0.95rem;
  font-weight: 500;
  line-height: 1.5;
  color: var(--af-text-secondary);
  margin-bottom: 0.75rem;
}

hr#answer {
  border: none;
  border-top: 1px solid var(--af-border);
  margin: 1rem 0;
}

/* ── Answer ── */
.answer {
  font-size: 1.1rem;
  line-height: 1.6;
  text-align: left;
}

/* ── Audio player ── */
.audio {
  margin-bottom: 0.5rem;
}

/* ── Image ── */
.image {
  margin-top: 1rem;
  text-align: center;
}

.image img {
  max-width: min(320px, 100%);
  max-height: 280px;
  border-radius: 10px;
  object-fit: contain;
}

/* ── Branding ── */
.branding {
  margin-top: 2rem;
  font-size: 0.65rem;
  color: var(--af-branding);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

/* ── Code blocks ── */
pre {
  text-align: left;
  background: var(--af-code-bg);
  border-radius: 8px;
  padding: 0.8rem 1rem;
  overflow-x: auto;
  margin: 0.8rem 0;
  border: 1px solid var(--af-border);
}

pre code {
  font-family: "SF Mono", "Fira Code", Menlo, Consolas, monospace;
  font-size: 0.9rem;
  line-height: 1.55;
  color: inherit;
  background: none;
  padding: 0;
}

code {
  font-family: "SF Mono", "Fira Code", Menlo, Consolas, monospace;
  background: var(--af-code-inline-bg);
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
  font-size: 0.88em;
}

/* highlight.js — override background */
pre code.hljs {
  background: transparent;
  padding: 0;
}

/* Syntax colors — light mode */
.hljs-keyword, .hljs-selector-tag, .hljs-built_in, .hljs-type { color: #c2410c; }
.hljs-string, .hljs-attr, .hljs-symbol { color: #15803d; }
.hljs-number, .hljs-literal { color: #1d4ed8; }
.hljs-comment, .hljs-doctag { color: #6b7280; font-style: italic; }
.hljs-title.function_, .hljs-title.class_ { color: #7c3aed; }
.hljs-meta, .hljs-name { color: #1d4ed8; }
.hljs-variable, .hljs-params { color: #1a1a2e; }

/* Syntax colors — dark mode */
.night_mode .hljs-keyword, .night_mode .hljs-selector-tag,
.night_mode .hljs-built_in, .night_mode .hljs-type { color: #fb923c; }
.night_mode .hljs-string, .night_mode .hljs-attr,
.night_mode .hljs-symbol { color: #86efac; }
.night_mode .hljs-number, .night_mode .hljs-literal { color: #93c5fd; }
.night_mode .hljs-comment, .night_mode .hljs-doctag { color: #6b7280; }
.night_mode .hljs-title.function_,
.night_mode .hljs-title.class_ { color: #c4b5fd; }
.night_mode .hljs-meta, .night_mode .hljs-name { color: #93c5fd; }
.night_mode .hljs-variable, .night_mode .hljs-params { color: #e2e8f0; }

/* ── Reduced motion ── */
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; }
}"""

# Language-specific CSS additions
_LANGUAGE_EXTRA_CSS = """

/* ── Language card extras ── */
.word {
  font-size: 2.2rem;
  font-weight: 700;
  line-height: 1.3;
  margin-bottom: 0.4rem;
}

.word--back {
  font-size: 1.6rem;
  margin-bottom: 0.3rem;
}

.transcription {
  font-size: 1rem;
  color: var(--af-text-secondary);
  margin-bottom: 0.5rem;
  font-style: italic;
}

.definition {
  font-size: 1.15rem;
  font-weight: 600;
  line-height: 1.55;
  text-align: left;
  margin-bottom: 1rem;
  padding: 0.8rem 1rem;
  background: var(--af-surface-accent);
  border-left: 3px solid var(--af-accent-border);
  border-radius: 0 8px 8px 0;
}

.example {
  font-size: 1.05rem;
  font-style: italic;
  line-height: 1.55;
  text-align: left;
  color: var(--af-text-secondary);
  margin-bottom: 1rem;
  padding: 0.6rem 0.8rem;
  border-left: 3px solid var(--af-border);
  border-radius: 0 6px 6px 0;
}"""

# ---------------------------------------------------------------------------
# QA note type
# ---------------------------------------------------------------------------

QA_FRONT_TEMPLATE = """\
<div class="ankiforge-card front">
  <div class="question">{{Question}}</div>
</div>"""

QA_BACK_TEMPLATE = (
    """\
<div class="ankiforge-card back">
  <div class="question">{{Question}}</div>
  <hr id="answer">
  <div class="answer">{{Answer}}</div>
  <div class="branding">AnkiForge</div>
</div>\n"""
    + _HLJS_SCRIPT
)

QA_CSS = _BASE_CSS

# ---------------------------------------------------------------------------
# QA+Image note type
# ---------------------------------------------------------------------------

QA_IMAGE_FRONT_TEMPLATE = """\
<div class="ankiforge-card front">
  <div class="question">{{Question}}</div>
</div>"""

QA_IMAGE_BACK_TEMPLATE = (
    """\
<div class="ankiforge-card back">
  <div class="question">{{Question}}</div>
  <hr id="answer">
  <div class="answer">{{Answer}}</div>
  {{#Image}}
  <div class="image">{{Image}}</div>
  {{/Image}}
  <div class="branding">AnkiForge</div>
</div>\n"""
    + _HLJS_SCRIPT
)

QA_IMAGE_CSS = _BASE_CSS

# ---------------------------------------------------------------------------
# QA+Audio note type
# ---------------------------------------------------------------------------

QA_AUDIO_FRONT_TEMPLATE = """\
<div class="ankiforge-card front">
  <div class="question">{{Question}}</div>
  <div class="audio">{{Audio}}</div>
</div>"""

QA_AUDIO_BACK_TEMPLATE = (
    """\
<div class="ankiforge-card back">
  <div class="question">{{Question}}</div>
  <hr id="answer">
  <div class="answer">{{Answer}}</div>
  <div class="branding">AnkiForge</div>
</div>\n"""
    + _HLJS_SCRIPT
)

QA_AUDIO_CSS = _BASE_CSS

# ---------------------------------------------------------------------------
# Language note type
# ---------------------------------------------------------------------------

LANGUAGE_FRONT_TEMPLATE = """\
<div class="ankiforge-card front">
  <div class="word">{{Word}}</div>
  {{#Transcription}}<div class="transcription">{{Transcription}}</div>{{/Transcription}}
  {{Audio}}
</div>
<script>var a=document.querySelector("audio");if(a)a.play();</script>"""

LANGUAGE_BACK_TEMPLATE = (
    """\
<div class="ankiforge-card back">
  <div class="word word--back">{{Word}}</div>
  {{#Transcription}}<div class="transcription">{{Transcription}}</div>{{/Transcription}}
  <hr id="answer">
  <div class="definition">{{Definition}}</div>
  <span data-role="def-audio">{{AudioDefinition}}</span>
  <span data-role="silence" style="display:none">{{AudioSilence}}</span>
  <div class="example">{{Example}}</div>
  <span data-role="ex-audio">{{AudioExample}}</span>
  {{#Image}}
  <div class="image">{{Image}}</div>
  {{/Image}}
  <div class="branding">AnkiForge</div>
</div>\n"""
    + _HLJS_SCRIPT
    + """
<script>
(function(){
  var d=document.querySelector('[data-role="def-audio"] audio');
  var s=document.querySelector('[data-role="silence"] audio');
  var e=document.querySelector('[data-role="ex-audio"] audio');
  if(d){
    d.play();
    var next=s||e;
    if(next){d.addEventListener("ended",function(){next.play()})}
    if(s&&e){s.addEventListener("ended",function(){e.play()})}
  }
})();
</script>"""
)

LANGUAGE_CSS = _BASE_CSS + _LANGUAGE_EXTRA_CSS

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_LANGUAGE_FIELDS = (
    "Word",
    "Audio",
    "Definition",
    "Example",
    "Image",
    "AudioDefinition",
    "AudioSilence",
    "AudioExample",
    "Transcription",
)


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

    for field_name in ("Question", "Answer"):
        field = models.new_field(field_name)
        models.add_field(model, field)

    tmpl = models.new_template("Card 1")
    tmpl["qfmt"] = QA_FRONT_TEMPLATE
    tmpl["afmt"] = QA_BACK_TEMPLATE
    models.add_template(model, tmpl)

    model["css"] = QA_CSS

    models.add(model)
    return model


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

    for field_name in ("Question", "Answer", "Image"):
        field = models.new_field(field_name)
        models.add_field(model, field)

    tmpl = models.new_template("Card 1")
    tmpl["qfmt"] = QA_IMAGE_FRONT_TEMPLATE
    tmpl["afmt"] = QA_IMAGE_BACK_TEMPLATE
    models.add_template(model, tmpl)

    model["css"] = QA_IMAGE_CSS

    models.add(model)
    return model


def ensure_qa_audio_note_type() -> dict[str, Any]:
    """Создаёт или находит существующий note type 'AnkiForge QA+Audio'.

    Returns:
        Словарь note type (Anki model dict).
    """
    mw = _get_mw()
    models = mw.col.models

    existing = models.by_name(QA_AUDIO_NOTE_TYPE_NAME)
    if existing is not None:
        return existing  # type: ignore[no-any-return]

    model: dict[str, Any] = models.new(QA_AUDIO_NOTE_TYPE_NAME)
    model["name"] = QA_AUDIO_NOTE_TYPE_NAME

    for field_name in ("Question", "Answer", "Audio"):
        field = models.new_field(field_name)
        models.add_field(model, field)

    tmpl = models.new_template("Card 1")
    tmpl["qfmt"] = QA_AUDIO_FRONT_TEMPLATE
    tmpl["afmt"] = QA_AUDIO_BACK_TEMPLATE
    models.add_template(model, tmpl)

    model["css"] = QA_AUDIO_CSS

    models.add(model)
    return model


def ensure_language_note_type() -> dict[str, Any]:
    """Создаёт или находит существующий note type 'AnkiForge Language'.

    Если note type уже существует — вызывает upgrade для добавления недостающих полей
    и обновления шаблонов.

    Returns:
        Словарь note type (Anki model dict).
    """
    mw = _get_mw()
    models = mw.col.models

    existing = models.by_name(LANGUAGE_NOTE_TYPE_NAME)
    if existing is not None:
        return _upgrade_language_note_type(existing, models)

    model: dict[str, Any] = models.new(LANGUAGE_NOTE_TYPE_NAME)
    model["name"] = LANGUAGE_NOTE_TYPE_NAME

    for field_name in _LANGUAGE_FIELDS:
        field = models.new_field(field_name)
        models.add_field(model, field)

    tmpl = models.new_template("Card 1")
    tmpl["qfmt"] = LANGUAGE_FRONT_TEMPLATE
    tmpl["afmt"] = LANGUAGE_BACK_TEMPLATE
    models.add_template(model, tmpl)

    model["css"] = LANGUAGE_CSS

    models.add(model)
    return model


def _upgrade_language_note_type(model: dict[str, Any], models: Any) -> dict[str, Any]:  # noqa: ANN401
    """Добавляет недостающие поля и обновляет шаблоны существующего Language note type.

    Args:
        model: Существующий note type dict.
        models: Anki ModelManager.

    Returns:
        Обновлённый note type dict.
    """
    existing_field_names = {f["name"] for f in model["flds"]}
    changed = False

    for field_name in _LANGUAGE_FIELDS:
        if field_name not in existing_field_names:
            field = models.new_field(field_name)
            models.add_field(model, field)
            changed = True

    tmpl = model["tmpls"][0]
    if tmpl["qfmt"] != LANGUAGE_FRONT_TEMPLATE:
        tmpl["qfmt"] = LANGUAGE_FRONT_TEMPLATE
        changed = True
    if tmpl["afmt"] != LANGUAGE_BACK_TEMPLATE:
        tmpl["afmt"] = LANGUAGE_BACK_TEMPLATE
        changed = True

    if model.get("css") != LANGUAGE_CSS:
        model["css"] = LANGUAGE_CSS
        changed = True

    if changed:
        models.save(model)

    return model
