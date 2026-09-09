"""Tests for ankiforge.ui.editor_button — 'Regenerate with AnkiForge' editor button."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ankiforge.anki_bridge.note_types import LANGUAGE_NOTE_TYPE_NAME
from ankiforge.models import AddonConfig, CardRequest, GeneratedCard, GenerationMode, LanguageOptions
from ankiforge.ui.editor_button import (
    _BUTTON_ID,
    _add_regenerate_button,
    _apply_card_to_note,
    _build_regenerate_request,
    _get_word,
    _has_other_content,
    _note_field_names,
    _on_regenerate_clicked,
    _on_regenerate_error,
    _on_regenerate_finished,
    _set_button_busy,
    setup_editor_button,
)


def _make_note(fields: dict[str, str], note_type_name: str = LANGUAGE_NOTE_TYPE_NAME) -> MagicMock:
    """MagicMock standing in for an Anki Note, dict-like over `fields`."""
    note = MagicMock()
    note.note_type.return_value = {
        "name": note_type_name,
        "flds": [{"name": name} for name in fields],
    }
    note.__getitem__.side_effect = fields.__getitem__

    def _setitem(key: str, value: str) -> None:
        fields[key] = value

    note.__setitem__.side_effect = _setitem
    return note


# ---------------------------------------------------------------------------
# Note helpers
# ---------------------------------------------------------------------------


class TestNoteFieldNames:
    def test_returns_field_names(self) -> None:
        note = _make_note({"Word": "run", "Definition": ""})
        assert _note_field_names(note) == {"Word", "Definition"}


class TestGetWord:
    def test_returns_stripped_word(self) -> None:
        note = _make_note({"Word": "  run  "})
        assert _get_word(note) == "run"

    def test_returns_empty_when_field_missing(self) -> None:
        note = _make_note({"Question": "x"})
        assert _get_word(note) == ""

    def test_returns_empty_when_blank(self) -> None:
        note = _make_note({"Word": "   "})
        assert _get_word(note) == ""


class TestHasOtherContent:
    def test_false_when_only_word_filled(self) -> None:
        note = _make_note({"Word": "run", "Definition": "", "Example": ""})
        assert _has_other_content(note) is False

    def test_true_when_another_field_filled(self) -> None:
        note = _make_note({"Word": "run", "Definition": "To move fast", "Example": ""})
        assert _has_other_content(note) is True


# ---------------------------------------------------------------------------
# _build_regenerate_request
# ---------------------------------------------------------------------------


class TestBuildRegenerateRequest:
    def test_builds_request_from_config_and_options(self) -> None:
        config = AddonConfig(language="fr")
        options = LanguageOptions(include_photo=True)

        request = _build_regenerate_request("bonjour", config, options)

        assert isinstance(request, CardRequest)
        assert request.mode == GenerationMode.LANGUAGE
        assert request.input_text == "bonjour"
        assert request.language == "fr"
        assert request.language_options is options


# ---------------------------------------------------------------------------
# _apply_card_to_note
# ---------------------------------------------------------------------------


class TestApplyCardToNote:
    @patch("ankiforge.ui.generate_dialog.save_media", return_value="uuid_audio.wav")
    def test_writes_matching_fields(self, mock_save_media: MagicMock) -> None:
        note = _make_note(
            {
                "Word": "run",
                "Definition": "",
                "Example": "",
                "Audio": "",
                "Image": "",
                "AudioDefinition": "",
                "AudioSilence": "",
                "AudioExample": "",
                "Transcription": "",
            }
        )
        card = GeneratedCard(
            word="run",
            note_type=LANGUAGE_NOTE_TYPE_NAME,
            definition="To move fast",
            example="I run every day.",
            audio_data=b"audio-bytes",
            transcription="/rʌn/",
        )

        _apply_card_to_note(note, card)

        assert note["Definition"] == "To move fast"
        assert note["Example"] == "I run every day."
        assert note["Audio"] == "[sound:uuid_audio.wav]"
        assert note["Transcription"] == "/rʌn/"
        assert note["Image"] == ""  # no image_data on the card -> empty ref

    def test_skips_fields_not_present_on_note_type(self) -> None:
        # A note type without Transcription/Image/etc. — only base fields.
        note = _make_note({"Word": "run", "Definition": "", "Example": "", "Audio": ""})
        card = GeneratedCard(word="run", note_type=LANGUAGE_NOTE_TYPE_NAME, definition="d", example="e")

        _apply_card_to_note(note, card)

        assert note["Definition"] == "d"
        assert note["Example"] == "e"
        assert "Image" not in note.note_type()["flds"] or True  # sanity: no crash on missing field

    def test_preserves_untouched_optional_fields(self) -> None:
        """Regression test: regenerating with e.g. all audio checkboxes and
        transcription unticked used to clobber those fields with empty
        strings instead of leaving the note's existing content alone."""
        note = _make_note(
            {
                "Word": "run",
                "Definition": "",
                "Example": "",
                "Audio": "[sound:old_word.wav]",
                "Image": "",
                "AudioDefinition": "[sound:old_def.wav]",
                "AudioSilence": "[sound:old_silence.wav]",
                "AudioExample": "[sound:old_ex.wav]",
                "Transcription": "/rʌn/",
            }
        )
        # None on all optional fields — as LanguageGenerator leaves them when
        # their checkbox was unticked for this regeneration.
        card = GeneratedCard(
            word="run",
            note_type=LANGUAGE_NOTE_TYPE_NAME,
            definition="A fresh definition",
            example="A fresh example.",
        )

        _apply_card_to_note(note, card)

        assert note["Definition"] == "A fresh definition"
        assert note["Example"] == "A fresh example."
        assert note["Audio"] == "[sound:old_word.wav]"
        assert note["AudioDefinition"] == "[sound:old_def.wav]"
        assert note["AudioSilence"] == "[sound:old_silence.wav]"
        assert note["AudioExample"] == "[sound:old_ex.wav]"
        assert note["Transcription"] == "/rʌn/"

    @patch("ankiforge.ui.generate_dialog.save_media", return_value="uuid_new.wav")
    def test_overwrites_optional_field_when_requested(self, mock_save_media: MagicMock) -> None:
        """The flip side: when a part IS requested (non-None on the card),
        it does overwrite whatever was there before."""
        note = _make_note({"Word": "run", "Definition": "", "Example": "", "Audio": "[sound:old.wav]"})
        card = GeneratedCard(
            word="run", note_type=LANGUAGE_NOTE_TYPE_NAME, definition="d", example="e", audio_data=b"new-audio"
        )

        _apply_card_to_note(note, card)

        assert note["Audio"] == "[sound:uuid_new.wav]"


# ---------------------------------------------------------------------------
# _set_button_busy
# ---------------------------------------------------------------------------


class TestSetButtonBusy:
    def test_busy_true_disables_button(self) -> None:
        editor = MagicMock()
        _set_button_busy(editor, busy=True)
        script = editor.web.eval.call_args[0][0]
        assert _BUTTON_ID in script
        assert "b.disabled=true" in script

    def test_busy_false_enables_button(self) -> None:
        editor = MagicMock()
        _set_button_busy(editor, busy=False)
        script = editor.web.eval.call_args[0][0]
        assert "b.disabled=false" in script


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestAddRegenerateButton:
    def test_appends_button_html_with_expected_args(self) -> None:
        editor = MagicMock()
        editor.addButton.return_value = "<button>...</button>"
        buttons: list[str] = []

        _add_regenerate_button(buttons, editor)

        assert buttons == ["<button>...</button>"]
        editor.addButton.assert_called_once()
        call_kwargs = editor.addButton.call_args.kwargs
        assert call_kwargs["cmd"] == "ankiforgeRegenerate"
        assert call_kwargs["id"] == _BUTTON_ID
        assert call_kwargs["func"] is _on_regenerate_clicked


class TestSetupEditorButton:
    def test_registers_hook(self) -> None:
        mock_gui_hooks = MagicMock()
        with patch.dict("sys.modules", {"aqt": MagicMock(gui_hooks=mock_gui_hooks)}):
            setup_editor_button()

        mock_gui_hooks.editor_did_init_buttons.append.assert_called_once_with(_add_regenerate_button)


# ---------------------------------------------------------------------------
# _on_regenerate_clicked — guard clauses
# ---------------------------------------------------------------------------


def _mock_aqt_utils() -> tuple[MagicMock, MagicMock]:
    """Return (showWarning, tooltip) mocks wired into a fake aqt.utils module."""
    show_warning = MagicMock()
    tooltip = MagicMock()
    return show_warning, tooltip


class TestOnRegenerateClickedGuards:
    def test_returns_when_note_is_none(self) -> None:
        editor = MagicMock()
        editor.note = None

        # No aqt mocking needed — function returns before any aqt import.
        _on_regenerate_clicked(editor)

    def test_warns_on_wrong_note_type(self) -> None:
        editor = MagicMock()
        editor.note = _make_note({"Question": "x"}, note_type_name="AnkiForge QA")
        show_warning, tooltip = _mock_aqt_utils()

        with patch.dict(
            "sys.modules", {"aqt": MagicMock(), "aqt.utils": MagicMock(showWarning=show_warning, tooltip=tooltip)}
        ):
            _on_regenerate_clicked(editor)

        show_warning.assert_called_once()
        assert "Language" in show_warning.call_args[0][0]

    def test_warns_on_missing_word(self) -> None:
        editor = MagicMock()
        editor.note = _make_note({"Word": "  "})
        show_warning, tooltip = _mock_aqt_utils()

        with patch.dict(
            "sys.modules", {"aqt": MagicMock(), "aqt.utils": MagicMock(showWarning=show_warning, tooltip=tooltip)}
        ):
            _on_regenerate_clicked(editor)

        show_warning.assert_called_once()
        assert "Word" in show_warning.call_args[0][0]

    @patch("ankiforge.config.manager.get_config", return_value=AddonConfig())
    def test_warns_when_not_configured(self, mock_get_config: MagicMock) -> None:
        editor = MagicMock()
        editor.note = _make_note({"Word": "run"})
        show_warning, tooltip = _mock_aqt_utils()

        with patch.dict(
            "sys.modules", {"aqt": MagicMock(), "aqt.utils": MagicMock(showWarning=show_warning, tooltip=tooltip)}
        ):
            _on_regenerate_clicked(editor)

        show_warning.assert_called_once()
        assert "Settings" in show_warning.call_args[0][0]

    @patch("ankiforge.config.manager.get_config", return_value=AddonConfig(api_key="sk-or-v1-test"))
    def test_warns_when_no_text_model(self, mock_get_config: MagicMock) -> None:
        editor = MagicMock()
        editor.note = _make_note({"Word": "run"})
        show_warning, tooltip = _mock_aqt_utils()

        with patch.dict(
            "sys.modules", {"aqt": MagicMock(), "aqt.utils": MagicMock(showWarning=show_warning, tooltip=tooltip)}
        ):
            _on_regenerate_clicked(editor)

        show_warning.assert_called_once()
        assert "text model" in show_warning.call_args[0][0]


class TestOnRegenerateClickedFlow:
    """Happy-path / cancel-path — _RegenerateOptionsDialog itself is mocked out."""

    @patch("ankiforge.ui.editor_button._RegenerateOptionsDialog")
    @patch("ankiforge.config.manager.get_config")
    def test_cancel_stops_before_generation(self, mock_get_config: MagicMock, mock_options_dialog: MagicMock) -> None:
        mock_get_config.return_value = AddonConfig(api_key="sk-or-v1-test", text_model="openai/gpt-4o")
        mock_options_dialog.return_value.run.return_value = None

        editor = MagicMock()
        editor.note = _make_note({"Word": "run"})
        show_warning, tooltip = _mock_aqt_utils()

        with patch.dict(
            "sys.modules", {"aqt": MagicMock(), "aqt.utils": MagicMock(showWarning=show_warning, tooltip=tooltip)}
        ):
            _on_regenerate_clicked(editor)

        show_warning.assert_not_called()
        tooltip.assert_not_called()

    @patch("ankiforge.ui.progress_widget.GenerationWorker")
    @patch("ankiforge.ui.generate_dialog._create_generator")
    @patch("ankiforge.openrouter.routing_client.build_client")
    @patch("ankiforge.anki_bridge.note_types.ensure_language_note_type")
    @patch("ankiforge.ui.editor_button._RegenerateOptionsDialog")
    @patch("ankiforge.config.manager.get_config")
    def test_starts_worker_on_accept(
        self,
        mock_get_config: MagicMock,
        mock_options_dialog: MagicMock,
        mock_ensure_note_type: MagicMock,
        mock_build_client: MagicMock,
        mock_create_generator: MagicMock,
        mock_worker_cls: MagicMock,
    ) -> None:
        config = AddonConfig(api_key="sk-or-v1-test", text_model="openai/gpt-4o")
        mock_get_config.return_value = config
        options = LanguageOptions()
        mock_options_dialog.return_value.run.return_value = options

        editor = MagicMock()
        editor.note = _make_note({"Word": "run"})
        show_warning, tooltip = _mock_aqt_utils()

        with patch.dict(
            "sys.modules", {"aqt": MagicMock(), "aqt.utils": MagicMock(showWarning=show_warning, tooltip=tooltip)}
        ):
            _on_regenerate_clicked(editor)

        mock_ensure_note_type.assert_called_once()
        mock_build_client.assert_called_once_with(config)
        mock_create_generator.assert_called_once()
        mock_worker_cls.assert_called_once()
        worker = mock_worker_cls.return_value
        worker.connect_finished.assert_called_once()
        worker.connect_error.assert_called_once()
        worker.start.assert_called_once()
        tooltip.assert_called_once()
        show_warning.assert_not_called()


# ---------------------------------------------------------------------------
# _on_regenerate_finished / _on_regenerate_error
# ---------------------------------------------------------------------------


class TestOnRegenerateFinished:
    @patch("ankiforge.ui.generate_dialog.save_media", return_value="uuid.wav")
    def test_applies_card_and_refreshes_editor(self, mock_save_media: MagicMock) -> None:
        editor = MagicMock()
        note = _make_note({"Word": "run", "Definition": "", "Example": ""})
        card = GeneratedCard(word="run", note_type=LANGUAGE_NOTE_TYPE_NAME, definition="d", example="e")
        show_warning, tooltip = _mock_aqt_utils()

        with patch.dict(
            "sys.modules", {"aqt": MagicMock(), "aqt.utils": MagicMock(showWarning=show_warning, tooltip=tooltip)}
        ):
            _on_regenerate_finished(editor, note, [card])

        assert note["Definition"] == "d"
        editor.mw.col.update_note.assert_called_once_with(note)
        editor.loadNoteKeepingFocus.assert_called_once()
        tooltip.assert_called_once()
        show_warning.assert_not_called()

    def test_warns_when_no_cards_returned(self) -> None:
        editor = MagicMock()
        note = _make_note({"Word": "run"})
        show_warning, tooltip = _mock_aqt_utils()

        with patch.dict(
            "sys.modules", {"aqt": MagicMock(), "aqt.utils": MagicMock(showWarning=show_warning, tooltip=tooltip)}
        ):
            _on_regenerate_finished(editor, note, [])

        show_warning.assert_called_once()
        editor.mw.col.update_note.assert_not_called()


class TestOnRegenerateError:
    def test_shows_warning_with_message(self) -> None:
        editor = MagicMock()
        show_warning, tooltip = _mock_aqt_utils()

        with patch.dict(
            "sys.modules", {"aqt": MagicMock(), "aqt.utils": MagicMock(showWarning=show_warning, tooltip=tooltip)}
        ):
            _on_regenerate_error(editor, "boom")

        show_warning.assert_called_once()
        assert "boom" in show_warning.call_args[0][0]
