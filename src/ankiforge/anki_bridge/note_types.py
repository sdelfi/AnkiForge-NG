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

# CSS для code blocks — общий для всех note types
_CODE_CSS = """
/* Code blocks */
pre {
  text-align: left;
  background: #f4f4f8;
  border-radius: 6px;
  padding: 0.8rem 1rem;
  overflow-x: auto;
  margin: 0.8rem 0;
}

pre code {
  font-family: "SF Mono", "Fira Code", Menlo, Consolas, monospace;
  font-size: 0.95rem;
  line-height: 1.5;
  color: inherit;
  background: none;
  padding: 0;
}

code {
  font-family: "SF Mono", "Fira Code", Menlo, Consolas, monospace;
  background: #eef0f5;
  padding: 0.1rem 0.35rem;
  border-radius: 3px;
  font-size: 0.9em;
}

.night_mode pre {
  background: #252545;
}

.night_mode code {
  background: #303050;
}

/* highlight.js — override background */
pre code.hljs {
  background: transparent;
  padding: 0;
}

/* Syntax colors — light mode */
.hljs-keyword, .hljs-selector-tag, .hljs-built_in, .hljs-type { color: #d73a49; }
.hljs-string, .hljs-attr, .hljs-symbol { color: #032f62; }
.hljs-number, .hljs-literal { color: #005cc5; }
.hljs-comment, .hljs-doctag { color: #6a737d; font-style: italic; }
.hljs-title.function_, .hljs-title.class_ { color: #6f42c1; }
.hljs-meta, .hljs-name { color: #005cc5; }
.hljs-variable, .hljs-params { color: #24292e; }

/* Syntax colors — dark mode */
.night_mode .hljs-keyword, .night_mode .hljs-selector-tag,
.night_mode .hljs-built_in, .night_mode .hljs-type { color: #ff7b72; }
.night_mode .hljs-string, .night_mode .hljs-attr,
.night_mode .hljs-symbol { color: #a5d6ff; }
.night_mode .hljs-number, .night_mode .hljs-literal { color: #79c0ff; }
.night_mode .hljs-comment, .night_mode .hljs-doctag { color: #8b949e; }
.night_mode .hljs-title.function_,
.night_mode .hljs-title.class_ { color: #d2a8ff; }
.night_mode .hljs-meta, .night_mode .hljs-name { color: #79c0ff; }
.night_mode .hljs-variable, .night_mode .hljs-params { color: #c9d1d9; }"""

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

QA_CSS = (
    """\
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
    + _CODE_CSS
)


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

QA_IMAGE_CSS = (
    """\
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
    + _CODE_CSS
)


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

QA_AUDIO_CSS = (
    """\
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

.audio {
  margin-bottom: 0.5rem;
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
    + _CODE_CSS
)


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

    # Поля: Question, Answer, Audio
    for field_name in ("Question", "Answer", "Audio"):
        field = models.new_field(field_name)
        models.add_field(model, field)

    # Шаблон
    tmpl = models.new_template("Card 1")
    tmpl["qfmt"] = QA_AUDIO_FRONT_TEMPLATE
    tmpl["afmt"] = QA_AUDIO_BACK_TEMPLATE
    models.add_template(model, tmpl)

    model["css"] = QA_AUDIO_CSS

    models.add(model)
    return model


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
  <div class="word">{{Word}}</div>
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

LANGUAGE_CSS = (
    """\
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

.transcription {
  font-size: 1.1rem;
  color: #7a8aaa;
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

.night_mode .transcription {
  color: #8a9ac0;
}

.night_mode .example {
  color: #8a9ac0;
  border-left-color: #4a5a80;
}

.night_mode .branding {
  color: #555;
}"""
    + _CODE_CSS
)


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

    # Шаблон
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

    # Обновляем шаблоны
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
