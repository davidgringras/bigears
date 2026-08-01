"""The sampled-guitar voicing: does it line up, does it breathe, does it fail
quietly when the machine has no fluidsynth on it?

This render exists because a human listener rejected the Karplus-Strong one
outright while every metric in the report was green. So the tests here are
about the things that listener heard — placement, articulation, level — and
not about pitch content, which the audit already measures.

Everything is skipped, not failed, on a machine without fluidsynth or a
soundfont: the feature is optional by construction and CI elsewhere must stay
green. The fallback test is deliberately NOT skipped, because a machine
without fluidsynth is precisely the one whose report has to still exist.
"""
from __future__ import annotations

import os
import sys

import librosa
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soloscribe.synth as synth_mod
from soloscribe.audit import audit
from soloscribe.model import TICKS_PER_QUARTER, NoteEvent, QNote, Score
from soloscribe.synth import (
    DEFAULT_SR,
    _fluidsynth,
    _render_spans_fluid,
    _resolve_soundfont,
    articulated_spans,
    render_events,
    render_events_fluid,
)

SR = DEFAULT_SR
FIRST_START = 0.35          # off zero, so "first audible" is a real measurement

HAVE_FLUID = _fluidsynth() is not None and _resolve_soundfont(None) is not None
needs_fluid = pytest.mark.skipif(
    not HAVE_FLUID, reason="fluidsynth binary or a soundfont is not installed"
)


# --------------------------------------------------------------------------
# fixtures and small measurements
# --------------------------------------------------------------------------

def _line(n: int = 8, spacing: float = 0.30, dur: float = 0.28) -> list[NoteEvent]:
    """An n-note single line, evenly spaced, nothing overlapping."""
    pitches = [64, 67, 71, 72, 69, 65, 62, 60, 63, 67]
    return [
        NoteEvent(
            start=FIRST_START + spacing * i,
            end=FIRST_START + spacing * i + dur,
            pitch=pitches[i % len(pitches)],
            velocity=96,
        )
        for i in range(n)
    ]


def _first_audible(y: np.ndarray, rel: float = 0.02) -> float:
    """Seconds to the first sample above `rel` of the peak."""
    thresh = rel * float(np.max(np.abs(y)))
    above = np.abs(y) > thresh
    return float(np.argmax(above)) / SR if above.any() else float("inf")


def _active_rms(y: np.ndarray) -> float:
    sounding = y[np.abs(y) > 0.001]
    return float(np.sqrt(np.mean(sounding ** 2))) if sounding.size else 0.0


def _duty(y: np.ndarray) -> float:
    """Fraction of frames carrying signal — the "air" in a phrase.

    Same measurement the demo acceptance harness uses (tests/demo_acceptance.py
    :_duty), so a change that makes this test happy and that one unhappy is
    impossible.
    """
    env = np.abs(librosa.util.frame(y, frame_length=512, hop_length=256)).max(0)
    return float((env > 0.06 * env.max()).mean())


@pytest.fixture(scope="module")
def events() -> list[NoteEvent]:
    return _line()


@pytest.fixture(scope="module")
def comparison(events) -> np.ndarray:
    """The stand-in for the recording: a KS render of the same notes.

    Its attacks sit exactly on the event starts, which is what makes it usable
    as a reference for "did the voicing land where the notes are". Against a
    real recording the events sit ~28 ms early of the perceived attacks and
    the calibration absorbs that — tested separately below, by moving this
    comparison deliberately.
    """
    return render_events(events, sr=SR)


# --------------------------------------------------------------------------
# 1. it renders, at the right length and the right level
# --------------------------------------------------------------------------

@needs_fluid
def test_render_spans_the_comparison_and_matches_its_level(events, comparison):
    y = render_events_fluid(events, comparison, sr=SR)
    assert y is not None, "fluidsynth is installed; this must produce audio"
    assert y.size == comparison.size, "both players must cover the same span"

    assert np.max(np.abs(y)) > 0.05, "not silent"
    assert np.max(np.abs(y)) <= 0.97 + 1e-9, "peak limiter"
    # Loudness match is iterated because the active region grows as the render
    # is scaled up; one pass lands ~4 % short (measured 0.958), three converge.
    assert _active_rms(y) == pytest.approx(_active_rms(comparison), rel=0.02)
    # A render that is one long ring-out would also pass the tests above.
    assert 0.05 < _duty(y) < 0.98


@needs_fluid
def test_velocity_reaches_the_synth(events, comparison):
    """Dynamics survive the trip through MIDI: a quiet line renders quieter."""
    quiet = [
        NoteEvent(start=e.start, end=e.end, pitch=e.pitch, velocity=45)
        for e in events
    ]
    loud = render_events_fluid(events, comparison, sr=SR)
    soft = render_events_fluid(quiet, comparison, sr=SR)
    assert loud is not None and soft is not None
    # Both are loudness-matched to the same comparison afterwards, so the
    # dynamic difference has to be read before that: use the raw renders.
    binary, font = _fluidsynth(), _resolve_soundfont(None)
    raw_loud = _render_spans_fluid(articulated_spans(events), SR, font, 26, binary)
    raw_soft = _render_spans_fluid(articulated_spans(quiet), SR, font, 26, binary)
    assert _active_rms(raw_soft) < _active_rms(raw_loud) * 0.75


# --------------------------------------------------------------------------
# 2. alignment — both calibrations, and what each assertion can actually see
# --------------------------------------------------------------------------

@needs_fluid
def test_first_attack_lands_on_the_first_event_after_calibration(
    events, comparison
):
    """The brief's bound: within 25 ms of the first event start.

    Read what this can and cannot see before trusting it. Measured on this
    machine (fluidsynth 2.5.7, MS Basic.sf3): the raw render's attacks sit
    +11.6 ms of their MIDI note starts, so an UNCOMPENSATED render would also
    pass a 25 ms bound. The assertion below is therefore necessary but not
    sufficient, and the test after it is the one that goes red if the
    calibration is deleted.
    """
    y = render_events_fluid(events, comparison, sr=SR)
    assert y is not None
    assert _first_audible(y) == pytest.approx(FIRST_START, abs=0.025)


@needs_fluid
def test_calibration_follows_the_clip_it_is_given(events, comparison):
    """Move the comparison's attacks; the voicing must move with them.

    This is the discriminating half. basic-pitch reports onsets early of where
    a listener hears the attack, by an amount that is a property of the clip,
    so the shift is measured per render rather than pinned to the ~28 ms the
    demo happened to show. Delaying the comparison audio by 30 ms — inside the
    60 ms matching window — must move the render 30 ms later. A hard-coded
    constant, or no calibration at all, leaves it where it was and fails here.
    """
    delay_sec = 0.030
    delayed = np.concatenate(
        [np.zeros(int(delay_sec * SR)), comparison]
    )[: comparison.size]

    on_time = render_events_fluid(events, comparison, sr=SR)
    late = render_events_fluid(events, delayed, sr=SR)
    assert on_time is not None and late is not None

    moved = _first_audible(late) - _first_audible(on_time)
    assert moved == pytest.approx(delay_sec, abs=0.010), (
        f"render moved {moved * 1000:+.1f} ms for a {delay_sec * 1000:.0f} ms "
        "shift in the comparison audio"
    )


# --------------------------------------------------------------------------
# 3. articulation
# --------------------------------------------------------------------------

def test_articulated_spans_cap_against_the_next_attack():
    """The arithmetic, before any audio: ends carry decay tails, so cap them."""
    evs = [
        NoteEvent(start=0.0, end=1.20, pitch=60, velocity=90),   # tail over 5 notes
        NoteEvent(start=0.22, end=1.40, pitch=62, velocity=90),
        NoteEvent(start=0.44, end=0.50, pitch=64, velocity=90),  # already short
    ]
    spans = articulated_spans(evs)
    assert [round(t1 - t0, 4) for t0, t1, _, _ in spans] == [
        round(0.85 * 0.22, 4),   # capped at 85 % of the gap to the next attack
        round(0.85 * 0.22, 4),
        0.06,                    # shorter than the cap: left alone
    ]
    # last note has no successor, so it is capped against its own length
    assert articulated_spans(evs[:1])[0][1] == pytest.approx(0.85 * 1.20)


@needs_fluid
def test_articulation_opens_up_a_line_of_decay_tails():
    """Sound the events at their nominal length and the reading is a smear.

    Ten notes 0.55 s apart with 1.6 s ends — every note's nominal end covers
    the next two attacks, which is what a decay tail looks like in NoteEvent
    terms. The comparison is the same notes through the same voice, so the cap
    is the only variable.

    Measured here: duty 0.854 capped against 0.940 uncapped. What the
    measurement CANNOT see, measured rather than assumed: at 0.22 s spacing
    the two come out 0.867 and 0.871, because the cap then leaves ~33 ms of
    note-off before the next attack and MS Basic's guitar takes ~30 ms to
    release (measured: a 0.15 s note is audible for 0.179 s, a 1.2 s note for
    1.212 s), so the air left over is thinner than the duty measure's 11.6 ms
    frame. The cap still fires there — articulated_spans is tested directly
    above — it just does not show up in this statistic.
    """
    evs = _line(n=10, spacing=0.55, dur=1.60)
    binary, font = _fluidsynth(), _resolve_soundfont(None)
    uncapped_spans = [
        (float(e.start), float(e.end), int(e.pitch), int(e.velocity)) for e in evs
    ]

    capped = _render_spans_fluid(articulated_spans(evs), SR, font, 26, binary)
    uncapped = _render_spans_fluid(uncapped_spans, SR, font, 26, binary)
    assert capped is not None and uncapped is not None

    # Same window for both, or the longer ring-out is compared against silence
    # it was never asked to fill.
    window = int((evs[-1].start + 0.3) * SR)
    d_capped, d_uncapped = _duty(capped[:window]), _duty(uncapped[:window])
    assert d_capped < d_uncapped - 0.04, (d_capped, d_uncapped)


# --------------------------------------------------------------------------
# 4. no fluidsynth, no soundfont: the report still gets written
# --------------------------------------------------------------------------

def _score_and_events() -> tuple[Score, list[NoteEvent], np.ndarray]:
    """Six quarter notes at 120 bpm, with the tempo map the audit needs."""
    quarter = 0.5
    pitches = [64, 67, 71, 72, 69, 65]
    qnotes = [
        QNote(onset=i * TICKS_PER_QUARTER, duration=TICKS_PER_QUARTER,
              pitch=p, velocity=92, confidence=0.8)
        for i, p in enumerate(pitches)
    ]
    score = Score(
        qnotes=qnotes,
        tempo_bpm=120.0,
        tempo_map=[(bar * 4 * TICKS_PER_QUARTER, bar * 4 * quarter)
                   for bar in range(3)],
        title="fluid fallback",
    )
    evs = [
        NoteEvent(start=i * quarter, end=i * quarter + 0.45, pitch=p,
                  velocity=92, confidence=0.8)
        for i, p in enumerate(pitches)
    ]
    return score, evs, render_events(evs, sr=SR)


@needs_fluid
def test_degenerate_comparisons_never_return_a_silent_player(events):
    """None beats a silent buffer labelled "the transcription".

    A comparison shorter than the events' lead-in truncates the render away
    entirely; before the guard this returned 100 samples at peak 0.000, which
    the report would have embedded as the machine's reading. The other rows
    are the cases that must still produce audio: no comparison to align to is
    a reason to skip the calibration, not a reason to refuse.
    """
    assert render_events_fluid(events, np.zeros(100), sr=SR) is None

    for cmp_audio in (np.zeros(0), np.zeros(3 * SR), np.zeros((3 * SR, 2))):
        y = render_events_fluid(events, cmp_audio, sr=SR)
        assert y is not None and np.max(np.abs(y)) > 0.01, cmp_audio.shape


def test_bogus_soundfont_returns_none(events, comparison):
    """An explicit path that is not there resolves to nothing, not to a
    substitute instrument. Runs everywhere: it never reaches fluidsynth."""
    assert render_events_fluid(
        events, comparison, sr=SR, soundfont="/nonexistent/not-a-bank.sf2"
    ) is None
    assert _resolve_soundfont("/nonexistent/not-a-bank.sf2") is None


def test_audit_falls_back_to_the_plucked_string_without_a_soundfont(
    tmp_path, monkeypatch
):
    """No soundfont anywhere: the report is still written, still playable, and
    still says what it is playing.

    NEGATIVE-TESTED, because a fallback that is never entered is unobserved
    (and this one is green on this machine for the opposite reason — fluidsynth
    IS installed here, so without the monkeypatch this branch never runs).
    Broke audit.py to select the voiced player unconditionally and ran it: red
    at `text.count("<audio") == 2` with 1, because the voiced block then
    carries a None path and is dropped entirely. Restored, green. The mirror
    case (forcing the plucked string when a soundfont exists) was broken the
    same way and went red in test_audit_voices_the_player_and_says_so at
    'not the quantized tab'.
    """
    monkeypatch.setattr(synth_mod, "SOUNDFONT_CANDIDATES", ())
    score, evs, y = _score_and_events()
    wav = tmp_path / "recording.wav"
    sf.write(str(wav), y, SR, subtype="PCM_16")

    metrics = audit(score, evs, str(wav), None, str(tmp_path / "out"))

    assert metrics["voiced_render_path"] is None
    assert not os.path.exists(tmp_path / "out" / "render_voiced.wav")
    assert os.path.exists(metrics["report_path"])
    text = open(metrics["report_path"], encoding="utf-8").read()
    assert text.count("<audio") == 2, "original and render"
    assert "the tab played back as a plucked string" in text
    assert "sampled guitar" not in text


@needs_fluid
def test_audit_voices_the_player_and_says_so(tmp_path):
    """The other side of the same branch: with a soundfont, the player is the
    sampled guitar and the report is explicit that it is not the tab and not
    what the numbers measure."""
    score, evs, y = _score_and_events()
    wav = tmp_path / "recording.wav"
    sf.write(str(wav), y, SR, subtype="PCM_16")

    metrics = audit(score, evs, str(wav), None, str(tmp_path / "out"))

    assert metrics["voiced_render_path"] == str(tmp_path / "out" / "render_voiced.wav")
    assert os.path.exists(metrics["voiced_render_path"])
    # The metrics render is untouched and still the one on disk as render.wav.
    assert metrics["render_path"].endswith("render.wav")
    assert os.path.exists(metrics["render_path"])

    text = open(metrics["report_path"], encoding="utf-8").read()
    assert text.count("<audio") == 2, "original and the voiced render"
    assert "voiced with a sampled guitar" in text
    assert "not the quantized tab" in text
    assert any("plucked-string render of the quantized score" in c
               for c in metrics["caveats"])
