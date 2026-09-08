"""'Regenerate with AnkiForge' button in the note editor.

Lets the user regenerate the Definition/Example/Audio/Image/Transcription
fields of an existing 'AnkiForge Language' note from its Word field, reusing
the same LanguageGenerator used by the main generation dialog — without
creating a new note. Generation runs in a background QThread (via the same
GenerationWorker used by the main dialog) so the editor never freezes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ankiforge.anki_bridge.note_types import LANGUAGE_NOTE_TYPE_NAME
from ankiforge.models import CardRequest, GenerationMode, LanguageOptions

if TYPE_CHECKING:
    from anki.notes import Note  # type: ignore[import-not-found]
    from aqt.editor import Editor  # type: ignore[import-not-found]
    from aqt.qt import QCheckBox, QComboBox  # type: ignore[import-not-found]

    from ankiforge.models import AddonConfig, GeneratedCard

_BUTTON_CMD = "ankiforgeRegenerate"
_BUTTON_ID = "ankiforge_regenerate_btn"

# Mirrors the checkbox order/labels in generate_dialog.InputDialog's
# "Generation options" section for Language mode, so the same choices are
# available when regenerating a single card from the editor.
_CHECKBOX_DEFS = (
    ("include_photo", "Generate photo"),
    ("include_audio_word", "Audio: word pronunciation"),
    ("include_audio_definition", "Audio: definition narration"),
    ("include_audio_example", "Audio: example narration"),
    ("include_transcription", "Phonetic transcription (IPA)"),
    ("detailed_image", "Detailed image (HD prompt)"),
)
_CHECKBOX_TOOLTIPS = {
    "detailed_image": "Uses a richer prompt for higher quality images. Costs more per image",
}
# Unchecked by default — mirrors generate_dialog's Language "Generation options".
_CHECKBOX_DEFAULT_UNCHECKED = {"include_photo", "detailed_image"}


# ---------------------------------------------------------------------------
# Note helpers (testable without Qt/Anki)
# ---------------------------------------------------------------------------


def _note_field_names(note: Note) -> set[str]:
    """Return the set of field names on a note's note type."""
    return {f["name"] for f in note.note_type()["flds"]}


def _get_word(note: Note) -> str:
    """Read and strip the Word field, or '' if the field is missing/empty."""
    if "Word" not in _note_field_names(note):
        return ""
    return str(note["Word"]).strip()


def _has_other_content(note: Note) -> bool:
    """True if any field other than Word already has content."""
    return any(name != "Word" and str(note[name]).strip() for name in _note_field_names(note))


def _build_regenerate_request(word: str, config: AddonConfig, options: LanguageOptions) -> CardRequest:
    """Build the CardRequest for regenerating a single Language card.

    Args:
        word: The word to regenerate the card for.
        config: Current add-on configuration (used for the language setting).
        options: User-chosen generation options (which fields to regenerate).

    Returns:
        A single-word CardRequest for LanguageGenerator.
    """
    return CardRequest(
        mode=GenerationMode.LANGUAGE,
        input_text=word,
        target_deck="",
        language=config.language,
        language_options=options,
    )


def _apply_card_to_note(note: Note, card: GeneratedCard) -> None:
    """Write a regenerated card's fields into an existing note, in place.

    Reuses the same GeneratedCard → field mapping (and media saving) as
    regular generation, restricted to fields that exist on this note type.

    Args:
        note: Note to update (mutated in place).
        card: Freshly generated card.
    """
    from ankiforge.ui.generate_dialog import _build_note_fields

    fields = _build_note_fields(card)
    field_names = _note_field_names(note)
    for name, value in fields.items():
        if name in field_names:
            note[name] = value


# ---------------------------------------------------------------------------
# Options dialog — same choices as the main Language generation dialog
# ---------------------------------------------------------------------------


def _create_voice_combo() -> QComboBox:
    """Create narrator voice combobox (mirrors generate_dialog's)."""
    from aqt.qt import QComboBox

    combo = QComboBox()
    combo.addItem("Alloy (neutral)", "alloy")
    combo.addItem("Ash (male, warm)", "ash")
    combo.addItem("Ballad (male, soft)", "ballad")
    combo.addItem("Coral (female, warm)", "coral")
    combo.addItem("Echo (male, clear)", "echo")
    combo.addItem("Fable (male, British)", "fable")
    combo.addItem("Nova (female, energetic)", "nova")
    combo.addItem("Onyx (male, deep)", "onyx")
    combo.addItem("Sage (female, calm)", "sage")
    combo.addItem("Shimmer (female, bright)", "shimmer")
    combo.addItem("Verse (male, expressive)", "verse")
    return combo


def _create_image_size_combo() -> QComboBox:
    """Create image size combobox (mirrors generate_dialog's)."""
    from aqt.qt import QComboBox

    combo = QComboBox()
    combo.addItem("Auto (1K, default)", "auto")
    combo.addItem("0.5K (cheaper, sufficient for cards)", "0.5K")
    combo.addItem("1K (standard)", "1K")
    combo.addItem("2K (high quality)", "2K")
    combo.addItem("4K (maximum)", "4K")
    return combo


def _build_options_from_checkboxes(
    checkboxes: dict[str, QCheckBox], voice_combo: QComboBox, image_size_combo: QComboBox
) -> LanguageOptions:
    """Build LanguageOptions from the options dialog's widgets."""
    return LanguageOptions(
        include_photo=checkboxes["include_photo"].isChecked(),
        include_audio_word=checkboxes["include_audio_word"].isChecked(),
        include_audio_definition=checkboxes["include_audio_definition"].isChecked(),
        include_audio_example=checkboxes["include_audio_example"].isChecked(),
        include_transcription=checkboxes["include_transcription"].isChecked(),
        detailed_image=checkboxes["detailed_image"].isChecked(),
        voice=voice_combo.currentData() or "alloy",
        image_size=image_size_combo.currentData() or "auto",
    )


class _RegenerateOptionsDialog:
    """Small modal dialog to pick which fields to regenerate.

    Mirrors the "Generation options" section of the main Language
    generation dialog exactly: every checkbox stays togglable regardless of
    which models are configured (checking "Generate photo" without an image
    model configured just fails at generation time, same as in the main
    dialog), and a live cost estimate updates as options change.
    """

    def __init__(self, parent: object, note: Note, config: AddonConfig) -> None:
        from aqt.qt import (
            QCheckBox,
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QLabel,
            QVBoxLayout,
        )

        self._config = config

        self._dialog = QDialog(parent)
        self._dialog.setWindowTitle("Regenerate with AnkiForge")

        layout = QVBoxLayout()
        self._dialog.setLayout(layout)

        if _has_other_content(note):
            warning = QLabel("This will overwrite the existing content of this card.")
            warning.setWordWrap(True)
            warning.setStyleSheet("color: #f44336;")
            layout.addWidget(warning)

        form = QFormLayout()
        layout.addLayout(form)

        self._checkboxes: dict[str, QCheckBox] = {}
        for key, label in _CHECKBOX_DEFS:
            cb = QCheckBox(label)
            cb.setChecked(key not in _CHECKBOX_DEFAULT_UNCHECKED)
            cb.stateChanged.connect(self._update_cost_estimate)
            tip = _CHECKBOX_TOOLTIPS.get(key)
            if tip:
                cb.setToolTip(tip)
            self._checkboxes[key] = cb
            form.addRow(cb)

        self._voice_combo = _create_voice_combo()
        form.addRow("Voice:", self._voice_combo)

        self._image_size_combo = _create_image_size_combo()
        self._image_size_combo.currentIndexChanged.connect(self._update_cost_estimate)
        form.addRow("Image size:", self._image_size_combo)

        self._cost_label = QLabel("")
        layout.addWidget(self._cost_label)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self._dialog.accept)
        button_box.rejected.connect(self._dialog.reject)
        layout.addWidget(button_box)

        self._update_cost_estimate()

    def _update_cost_estimate(self) -> None:
        """Recalculate and display the cost estimate from cached pricing in config."""
        from ankiforge.ui.generate_dialog import _estimate_cost_from_config
        from ankiforge.ui.progress_widget import _format_cost

        options = _build_options_from_checkboxes(self._checkboxes, self._voice_combo, self._image_size_combo)
        cost = _estimate_cost_from_config(self._config, GenerationMode.LANGUAGE, 1, options)
        if cost is None:
            self._cost_label.setText("Pricing not loaded — open AnkiForge Settings")
            return
        self._cost_label.setText(f"Estimate: {_format_cost(cost)}")

    def run(self) -> LanguageOptions | None:
        """Show the dialog modally.

        Returns:
            Chosen LanguageOptions, or None if the user cancelled.
        """
        from aqt.qt import QDialog

        if self._dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        return _build_options_from_checkboxes(self._checkboxes, self._voice_combo, self._image_size_combo)


# ---------------------------------------------------------------------------
# Click handler
# ---------------------------------------------------------------------------


def _set_button_busy(editor: Editor, *, busy: bool) -> None:
    """Enable/disable the regenerate button via the editor's webview."""
    disabled_js = "true" if busy else "false"
    editor.web.eval(f'(function(){{var b=document.getElementById("{_BUTTON_ID}");if(b)b.disabled={disabled_js};}})();')


def _on_regenerate_clicked(editor: Editor) -> None:
    """'Regenerate' button handler — validates state and starts background generation."""
    note = editor.note
    if note is None:
        return

    from aqt.utils import showWarning, tooltip  # type: ignore[import-not-found]

    if note.note_type()["name"] != LANGUAGE_NOTE_TYPE_NAME:
        showWarning("This button only works on 'AnkiForge Language' notes.")
        return

    word = _get_word(note)
    if not word:
        showWarning("Enter a word in the Word field first.")
        return

    from ankiforge.config.manager import get_config

    config = get_config()
    if not (config.api_key or config.custom_base_url):
        showWarning("Configure AnkiForge (API key or a local endpoint) in Settings first.")
        return
    if not config.text_model:
        showWarning("Select a text model in AnkiForge Settings first.")
        return

    options = _RegenerateOptionsDialog(editor.widget, note, config).run()
    if options is None:
        return  # user cancelled

    from ankiforge.anki_bridge.note_types import ensure_language_note_type
    from ankiforge.openrouter.routing_client import build_client
    from ankiforge.ui.generate_dialog import _create_generator
    from ankiforge.ui.progress_widget import GenerationWorker

    ensure_language_note_type()

    request = _build_regenerate_request(word, config, options)
    client = build_client(config)
    generator = _create_generator(
        mode=GenerationMode.LANGUAGE,
        client=client,
        text_model=config.text_model,
        image_model=config.image_model,
        audio_model=config.audio_model,
        config=config,
        voice=options.voice,
        image_size=options.image_size,
    )

    _set_button_busy(editor, busy=True)
    tooltip("AnkiForge: generating…")

    worker = GenerationWorker(generator, request)
    # Keep the worker (and its QThread) alive for the duration of the
    # background generation — otherwise it would be garbage-collected as
    # soon as this handler returns.
    editor._af_worker = worker
    worker.connect_finished(lambda cards: _on_regenerate_finished(editor, note, cards))
    worker.connect_error(lambda msg: _on_regenerate_error(editor, msg))
    worker.start()


def _on_regenerate_finished(editor: Editor, note: Note, cards: list[GeneratedCard]) -> None:
    """Apply the regenerated card to the note and refresh the editor."""
    from aqt.utils import showWarning, tooltip

    _set_button_busy(editor, busy=False)
    editor._af_worker = None

    if not cards:
        showWarning("AnkiForge: generation returned no card.")
        return

    _apply_card_to_note(note, cards[0])
    editor.mw.col.update_note(note)
    editor.loadNoteKeepingFocus()
    tooltip("AnkiForge: card regenerated")


def _on_regenerate_error(editor: Editor, error_msg: str) -> None:
    """Show the error and reset the button."""
    from aqt.utils import showWarning

    _set_button_busy(editor, busy=False)
    editor._af_worker = None
    showWarning(f"AnkiForge generation failed: {error_msg}")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _add_regenerate_button(buttons: list[str], editor: Editor) -> None:
    """editor_did_init_buttons hook — add the 'Regenerate with AnkiForge' button."""
    button_html = editor.addButton(
        icon=None,
        cmd=_BUTTON_CMD,
        func=_on_regenerate_clicked,
        tip="Regenerate this card's fields from the Word field using AnkiForge",
        label="🔄 AnkiForge",
        id=_BUTTON_ID,
    )
    buttons.append(button_html)


def setup_editor_button() -> None:
    """Register the 'Regenerate with AnkiForge' button in the note editor."""
    from aqt import gui_hooks  # type: ignore[import-not-found]

    gui_hooks.editor_did_init_buttons.append(_add_regenerate_button)
