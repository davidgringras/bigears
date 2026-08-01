"""Tests for the Hugging Face Space front end (``space/app.py``).

The Space is the same ``run_pipeline`` behind a Gradio page, so these tests are
about the layer around it: what gets refused before the pipeline is ever
called, how the options on the page map onto the pipeline's arguments, and what
comes back when the pipeline raises. ``run_pipeline`` itself is monkeypatched
throughout — the pipeline has its own tests, and a real transcription here
would make this file take minutes.

``space/app.py`` is loaded by path rather than imported: ``space/`` is a
deployable directory, not a package, and giving it an ``__init__.py`` purely to
be importable would put a stray file in the thing being pushed to Hugging Face.

The results the handler returns are ``(gp5_path | None, report_path | None,
html)``. The third slot is never empty: every refusal and every failure has to
arrive as a sentence, because a Gradio page in front of a guitarist has nowhere
else to put one.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `.venv/bin/pytest tests/test_space_app.py`
    sys.path.insert(0, str(REPO_ROOT))

APP_PATH = REPO_ROOT / "space" / "app.py"


def _load_app():
    spec = importlib.util.spec_from_file_location("soloscribe_space_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


app = _load_app()

GP5_BYTES = b"FICHIER GUITAR PRO v5.10\x00fake-but-nonempty"
REPORT_HTML = "<!doctype html><title>Audit</title><p>4 of 5 notes matched.</p>"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _fake_result(out_dir, *, warnings=None, level="high", reasons=(), write_gp5=True):
    """A PipelineResult with real files on disk, as the handler expects."""
    from soloscribe.model import BeatGrid, Score
    from soloscribe.pipeline import PipelineResult

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gp5 = out / "solo.gp5"
    if write_gp5:
        gp5.write_bytes(GP5_BYTES)
    report = out / "report.html"
    report.write_text(REPORT_HTML, encoding="utf-8")
    return PipelineResult(
        gp5_path=str(gp5),
        report_path=str(report),
        score=Score(),
        events=[],
        grid=BeatGrid(beat_times=[0.0, 0.5, 1.0]),
        metrics={"verdict": {"level": level, "reasons": list(reasons)}},
        warnings=list(warnings or []),
    )


@pytest.fixture
def audio(tmp_path):
    """A file that exists. Its contents never matter: probe_duration is stubbed."""
    path = tmp_path / "solo.wav"
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    return str(path)


@pytest.fixture
def calls(monkeypatch):
    """Record every run_pipeline call, and hand back a finished result."""
    seen: list[dict] = []

    def fake(audio_path, out_dir, **kwargs):
        seen.append({"audio_path": audio_path, "out_dir": out_dir, **kwargs})
        return _fake_result(out_dir)

    monkeypatch.setattr(app, "run_pipeline", fake)
    monkeypatch.setattr(app, "probe_duration", lambda path: 90.0)
    return seen


def _call(**overrides):
    """Invoke the handler with sensible defaults for everything unspecified."""
    kwargs = dict(
        audio_path=None,
        key_label=app.KEY_CHOICES[0],
        tempo=None,
        feel_label=app.FEEL_CHOICES[0],
        mode_label=app.MODE_CHOICES[0],
        chords=None,
        title="",
        isolate=False,
        beats_label=app.BEATS_CHOICES[0],
        downbeat=None,
        capo=None,
        start=None,
        end=None,
        progress=None,
    )
    kwargs.update(overrides)
    return app.transcribe_solo(**kwargs)


# --------------------------------------------------------------------------
# import level
# --------------------------------------------------------------------------


def test_module_loads_and_builds_a_page():
    import gradio as gr

    assert isinstance(app.demo, gr.Blocks)
    assert callable(app.transcribe_solo)
    assert callable(app.run_pipeline), "must be a module global so tests can replace it"


def test_the_clip_limit_is_three_minutes():
    assert app.MAX_CLIP_SECONDS == 180.0


def test_every_choice_on_the_page_maps_to_a_pipeline_value():
    # A label with no entry in its map would silently fall back to the default,
    # so the person choosing "Straight" would quietly get "auto".
    assert set(app.FEEL_CHOICES) == set(app.FEEL_VALUES)
    assert set(app.MODE_CHOICES) == set(app.MODE_VALUES)
    assert set(app.BEATS_CHOICES) == set(app.BEATS_VALUES)
    assert set(app.KEY_CHOICES) == set(app.KEY_VALUES)
    assert set(app.FEEL_VALUES.values()) == {"auto", "on", "off"}
    assert set(app.MODE_VALUES.values()) == {"solo", "poly"}


def test_keys_are_spelled_the_way_the_pipeline_expects():
    import re

    key_re = re.compile(r"^[A-G](#|b)?m?$")  # soloscribe/webapp/server.py:60
    for label, value in app.KEY_VALUES.items():
        assert key_re.match(value), f"{label} maps to {value!r}, which Score.key would reject"


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------


def test_a_finished_transcription_returns_both_files(audio, calls):
    gp5, report, html = _call(audio_path=audio, title="Blue Bossa")

    assert gp5 is not None and Path(gp5).read_bytes() == GP5_BYTES
    assert report is not None and Path(report).read_text(encoding="utf-8") == REPORT_HTML
    assert len(calls) == 1
    assert calls[0]["audio_path"] == audio
    assert calls[0]["title"] == "Blue Bossa"
    assert "came out well" in html  # the "high" verdict copy


def test_the_options_reach_the_pipeline(audio, calls):
    _call(
        audio_path=audio,
        key_label="B♭ major",
        tempo="152",
        feel_label="Swing",
        mode_label="Chords as well",
        chords="Fmaj7\nGm7 | C7",
        beats_label=app.BEATS_CHOICES[1],
        downbeat="1.25",
        capo="3",
        start="10",
        end="70",
    )

    got = calls[0]
    assert got["key"] == "Bb"
    assert got["bpm"] == 152.0
    assert got["swing"] == "on"
    assert got["mode"] == "poly"
    assert got["chords"] == "Fmaj7|Gm7|C7"
    assert got["beats_per_bar"] == 3
    assert got["downbeat"] == 1.25
    assert got["capo"] == 3
    assert (got["start"], got["end"]) == (10.0, 70.0)


@pytest.mark.parametrize(
    "isolate, expected",
    [
        # Unticked must mean "do not run Demucs at all" — on two shared cores
        # that is the difference between a wait and an apparent hang.
        (False, "off"),
        # Ticked maps to "auto", not "on": both run separation and cost the same
        # minutes, but "auto" keeps the pipeline's fallback when isolation turns
        # out to have lost notes (pipeline.py:119-136).
        (True, "auto"),
    ],
)
def test_the_isolation_checkbox_maps_honestly(audio, calls, isolate, expected):
    _call(audio_path=audio, isolate=isolate)
    assert calls[0]["separate"] == expected


def test_progress_is_forwarded_and_a_broken_bar_does_not_lose_the_run(audio, monkeypatch):
    reported: list[tuple[float, str]] = []

    def fake(audio_path, out_dir, **kwargs):
        kwargs["progress"]("Working out the notes", 0.5)
        return _fake_result(out_dir)

    monkeypatch.setattr(app, "run_pipeline", fake)
    monkeypatch.setattr(app, "probe_duration", lambda path: 90.0)

    class Bar:
        def __call__(self, frac, desc=None):
            reported.append((frac, desc))
            raise RuntimeError("the queue went away")

    gp5, _, _ = _call(audio_path=audio, progress=Bar())
    assert gp5 is not None, "a failing progress bar must not fail the transcription"
    assert (0.5, "Working out the notes") in reported


def test_warnings_and_reasons_are_shown_and_escaped(audio, monkeypatch):
    monkeypatch.setattr(app, "probe_duration", lambda path: 90.0)
    monkeypatch.setattr(
        app,
        "run_pipeline",
        lambda audio_path, out_dir, **kw: _fake_result(
            out_dir,
            level="low",
            reasons=["only 41% of the attacks were accounted for"],
            warnings=["transcribed in single-note mode <script>alert(1)</script>"],
        ),
    )
    _, _, html = _call(audio_path=audio)

    assert "struggled with this one" in html
    assert "only 41% of the attacks were accounted for" in html
    assert "&lt;script&gt;" in html and "<script>" not in html


def test_an_ungraded_run_says_so_rather_than_claiming_a_verdict(audio, monkeypatch):
    # The pipeline swallows an audit failure and still returns the .gp5
    # (pipeline.py:197-201), so metrics can arrive empty. Reporting that as
    # "high" would be the one lie this project cannot afford.
    monkeypatch.setattr(app, "probe_duration", lambda path: 90.0)
    monkeypatch.setattr(
        app,
        "run_pipeline",
        lambda audio_path, out_dir, **kw: SimpleNamespace(
            gp5_path=str(Path(_fake_result(out_dir).gp5_path)),
            report_path=None,
            warnings=[],
            metrics={},
        ),
    )
    gp5, report, html = _call(audio_path=audio)

    assert gp5 is not None and report is None
    assert "could not grade myself" in html
    assert "unverified" in html


# --------------------------------------------------------------------------
# what gets refused, and how politely
# --------------------------------------------------------------------------


def test_no_recording_asks_for_one(calls):
    gp5, report, html = _call(audio_path=None)

    assert (gp5, report) == (None, None)
    assert "Choose a recording first" in html
    assert not calls


def test_a_clip_over_three_minutes_is_refused_before_the_pipeline_runs(audio, calls, monkeypatch):
    monkeypatch.setattr(app, "probe_duration", lambda path: 252.0)
    gp5, report, html = _call(audio_path=audio)

    assert (gp5, report) == (None, None)
    assert not calls, "the refusal must happen before any CPU is spent"
    assert "4 minutes 12 seconds" in html, "say how long it actually is"
    assert "three minutes" in html
    assert "From and To" in html, "point at the way out, not just the rule"


def test_trimming_rescues_a_long_recording(audio, calls, monkeypatch):
    # The cap is on what will be ANALYSED. Someone who trims a five-minute
    # track down to ninety seconds has done the right thing.
    monkeypatch.setattr(app, "probe_duration", lambda path: 300.0)
    gp5, _, _ = _call(audio_path=audio, start="60", end="150")

    assert gp5 is not None
    assert len(calls) == 1


def test_an_unreadable_number_names_the_box(audio, calls):
    gp5, report, html = _call(audio_path=audio, tempo="about 150ish")

    assert (gp5, report) == (None, None)
    assert "the tempo" in html
    assert not calls


def test_an_end_before_the_start_is_refused(audio, calls):
    gp5, _, html = _call(audio_path=audio, start="90", end="30")

    assert gp5 is None
    assert "after the start time" in html
    assert not calls


# --------------------------------------------------------------------------
# when it goes wrong
# --------------------------------------------------------------------------


def test_a_pipeline_crash_becomes_an_apology_not_a_traceback(audio, monkeypatch):
    monkeypatch.setattr(app, "probe_duration", lambda path: 90.0)

    def boom(audio_path, out_dir, **kwargs):
        raise RuntimeError("tensor shape mismatch at layer 7")

    monkeypatch.setattr(app, "run_pipeline", boom)
    gp5, report, html = _call(audio_path=audio)

    assert (gp5, report) == (None, None)
    assert "did not work" in html
    assert "RuntimeError" in html, "name the error so David has something to go on"
    assert "Traceback" not in html and "line " not in html


def test_a_known_failure_is_translated_out_of_jargon(audio, monkeypatch):
    monkeypatch.setattr(app, "probe_duration", lambda path: 90.0)

    def oom(audio_path, out_dir, **kwargs):
        raise MemoryError()

    monkeypatch.setattr(app, "run_pipeline", oom)
    _, _, html = _call(audio_path=audio)

    assert "too big for me to hold all at once" in html
    assert "MemoryError" in html


def test_a_silent_exception_still_says_something(audio, monkeypatch):
    # audioread's NoBackendError carries no message at all; the class name on
    # its own is not something to hand to a guitarist.
    monkeypatch.setattr(app, "probe_duration", lambda path: 90.0)

    class Odd(Exception):
        pass

    def raise_odd(audio_path, out_dir, **kwargs):
        raise Odd()

    monkeypatch.setattr(app, "run_pipeline", raise_odd)
    _, _, html = _call(audio_path=audio)

    assert "did not tell me what" in html


def test_a_missing_gp5_is_reported_as_not_your_fault(audio, monkeypatch):
    monkeypatch.setattr(app, "probe_duration", lambda path: 90.0)
    monkeypatch.setattr(
        app,
        "run_pipeline",
        lambda audio_path, out_dir, **kw: _fake_result(out_dir, write_gp5=False),
    )
    gp5, report, html = _call(audio_path=audio)

    assert (gp5, report) == (None, None)
    assert "not your fault" in html


def test_an_unreadable_file_does_not_stop_the_run(audio, calls, monkeypatch):
    # probe_duration returns 0.0 when it cannot read the header. That must not
    # be treated as a refusal — let the pipeline be the one to complain, since
    # its error messages are better than a guess made here.
    monkeypatch.setattr(app, "probe_duration", lambda path: 0.0)
    gp5, _, _ = _call(audio_path=audio)

    assert gp5 is not None
    assert len(calls) == 1


# --------------------------------------------------------------------------
# small pieces
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, None),
        ("", None),
        ("   \n  ", None),
        ("Fmaj7", "Fmaj7"),
        ("Fmaj7\nD7", "Fmaj7|D7"),
        ("Gm7 | C7", "Gm7|C7"),
        ("Gm7   C7\nF6", "Gm7 C7|F6"),          # two chords in one bar stay together
        ("Fmaj7\r\nD7\r\n", "Fmaj7|D7"),        # a Windows textarea
        ("|| Fmaj7 ||", "Fmaj7"),               # stray bar lines
    ],
)
def test_chord_charts_are_normalised(raw, expected):
    assert app.normalise_chords(raw) == expected


@pytest.mark.parametrize(
    "seconds, expected",
    [(45, "45 seconds"), (60, "1 minute"), (61, "1 minute 1 second"), (252, "4 minutes 12 seconds")],
)
def test_durations_are_said_the_way_a_person_says_them(seconds, expected):
    assert app._minutes(seconds) == expected


def test_the_transcription_backend_gets_pinned():
    """The pin must actually take, or the Space runs whatever pip happened to install."""
    import basic_pitch.inference as inference

    original = inference.predict
    try:
        line = app.pin_transcription_backend()
        assert line.startswith("transcription backend:")
        # On this Mac only coremltools is installed, so that is what it should
        # land on; on the Space, onnxruntime is present and ONNX wins the
        # preference order. Either way the seam must have been rebound, because
        # transcribe.py imports `predict` from this module at call time
        # (transcribe.py:562) and that is the only way to reach it.
        assert inference.predict is not original
        assert any(name in line for name in app.BACKEND_PREFERENCE)
    finally:
        inference.predict = original
