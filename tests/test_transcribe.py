"""Transcription tests against synthetic fixtures with known ground truth.

Every audio fixture is generated here rather than checked in, so the expected
pitches and onsets are literals in the test rather than something read back out
of the thing under test. Two of the tests carry an explicit precondition
assertion on raw basic-pitch output: without it a green result cannot be
distinguished from "the defect never occurred in the first place", which is the
failure mode that makes a passing suite worthless.

Whole file runs in a few seconds — basic-pitch inference on these clips is
~0.15s each and the pyin contour ~0.35s, with a one-off numba warm-up.
"""
from __future__ import annotations

import contextlib
import io

import librosa
import numpy as np
import pytest
import soundfile as sf

from soloscribe import transcribe as T
from soloscribe.transcribe import _amp_to_velocity, _bend_curve, _velocity_to_amp, transcribe

SR = 22050
ONSET_TOL = 0.06          # seconds
THREE_PLUCKS = [(52, 0.20), (57, 1.00), (64, 1.90)]   # (midi, onset seconds)


# --- synthesis ------------------------------------------------------------
def _pluck(midi: int, dur: float, seed: int) -> np.ndarray:
    """Karplus-Strong: noise burst through a decaying comb. Harmonic-rich, so
    it exercises the same octave/harmonic confusions a real pickup does."""
    rng = np.random.default_rng(seed)
    n = int(round(SR * dur))
    period = int(round(SR / float(librosa.midi_to_hz(midi))))
    buf = np.convolve(rng.uniform(-1, 1, period), np.ones(3) / 3, mode="same")
    out = np.zeros(n)
    idx = 0
    for i in range(n):
        out[i] = buf[idx]
        buf[idx] = 0.5 * (buf[idx] + buf[(idx + 1) % period]) * 0.996
        idx = (idx + 1) % period
    return out


def _tone(
    midi: int, dur: float, harmonics: list[float], vib_hz: float = 0.0, vib_semi: float = 0.0
) -> np.ndarray:
    """Additive tone with an optional sinusoidal frequency modulation."""
    f0 = float(librosa.midi_to_hz(midi))
    t = np.arange(int(dur * SR)) / SR
    if vib_hz:
        dev = vib_semi * np.sin(2 * np.pi * vib_hz * t)
        phase = 2 * np.pi * np.cumsum(f0 * 2 ** (dev / 12.0)) / SR
    else:
        phase = 2 * np.pi * f0 * t
    y = sum(a * np.sin(k * phase) for k, a in enumerate(harmonics, start=1))
    return y * np.minimum(1.0, t / 0.01) * np.exp(-t / (dur * 0.9))


def _place(total: float, parts: list[tuple[np.ndarray, float]]) -> np.ndarray:
    buf = np.zeros(int(SR * total))
    for y, at in parts:
        s = int(at * SR)
        buf[s : s + len(y)] += y
    return buf / np.max(np.abs(buf)) * 0.85


def _write(path, y: np.ndarray) -> str:
    sf.write(str(path), y.astype(np.float32), SR)
    return str(path)


def _raw_pitches(path: str) -> list[tuple[float, float, int]]:
    """Un-merged, un-refined basic-pitch output, for precondition checks.
    Uses the module's own constants so the two cannot drift apart."""
    from basic_pitch.inference import predict

    with contextlib.redirect_stdout(io.StringIO()):
        _, _, raw = predict(
            path,
            onset_threshold=T.ONSET_THRESHOLD,
            frame_threshold=T.FRAME_THRESHOLD,
            minimum_note_length=T.MIN_NOTE_LENGTH_MS,
            minimum_frequency=T.GUITAR_MIN_HZ,
            maximum_frequency=T.GUITAR_MAX_HZ,
            melodia_trick=True,
        )
    return [(float(a), float(b), int(p)) for a, b, p, _, _ in raw]


def _near(events, pitch: int, onset: float, tol: float = ONSET_TOL):
    return [e for e in events if e.pitch == pitch and abs(e.start - onset) <= tol]


def _describe(events) -> str:
    return ", ".join(f"{e.pitch}@{e.start:.3f}-{e.end:.3f}" for e in events)


# --- fixtures -------------------------------------------------------------
@pytest.fixture(scope="module")
def audio_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("audio")


@pytest.fixture(scope="module")
def plucks_wav(audio_dir):
    parts = [(_pluck(m, 0.7, seed=k), t) for k, (m, t) in enumerate(THREE_PLUCKS)]
    return _write(audio_dir / "plucks.wav", _place(3.0, parts))


@pytest.fixture(scope="module")
def dip_wav(audio_dir):
    """Steady G3 whose level collapses to ~6% for ~40ms at its midpoint."""
    y = _tone(55, 1.6, [1.0, 0.6, 0.35, 0.2, 0.12])
    t = np.arange(len(y)) / SR
    y = y * (1.0 - 0.94 * np.exp(-((t - 0.8) ** 2) / (2 * 0.018**2)))
    return _write(audio_dir / "dip.wav", _place(2.0, [(y, 0.2)]))


@pytest.fixture(scope="module")
def harmonic_wav(audio_dir):
    """E3 whose 2nd harmonic is louder than its fundamental."""
    return _write(
        audio_dir / "harmonic.wav",
        _place(1.6, [(_tone(52, 1.3, [0.35, 1.0, 0.5, 0.3, 0.2]), 0.15)]),
    )


@pytest.fixture(scope="module")
def fifth_wav(audio_dir):
    """E3 + B3 struck together — a perfect fifth double-stop."""
    parts = [(_pluck(m, 1.2, seed=10 + k), 0.2) for k, m in enumerate((52, 59))]
    return _write(audio_dir / "fifth.wav", _place(2.0, parts))


@pytest.fixture(scope="module")
def vibrato_wav(audio_dir):
    """C4 with 5.5 Hz / ±0.4 semitone vibrato."""
    return _write(
        audio_dir / "vibrato.wav",
        _place(1.6, [(_tone(60, 1.3, [1.0, 0.5, 0.25], vib_hz=5.5, vib_semi=0.4), 0.15)]),
    )


# --- pure unit tests ------------------------------------------------------
def test_bend_bins_are_thirds_of_a_semitone():
    """The one fact the whole bend path rests on (note_creation.py:209-211)."""
    assert _bend_curve([1] * 20) == [], "a constant offset is bin quantization, not a bend"
    assert _bend_curve(None) == []
    assert _bend_curve([0]) == []

    curve = _bend_curve(list(range(7)))            # 0..6 bins = 0..2 semitones
    assert curve[0] == (0.0, 0.0)
    assert curve[-1] == (1.0, 2.0)
    assert all(0.0 <= t <= 1.0 for t, _ in curve)

    long_curve = _bend_curve(list(range(200)))
    assert len(long_curve) <= T.BEND_MAX_POINTS
    assert long_curve[-1] == (1.0, pytest.approx(199 / 3.0))


def test_velocity_calibration_puts_strong_notes_in_the_musical_band():
    # basic-pitch amplitudes observed on strong synthetic plucks run 0.75-0.85.
    assert 70 <= _amp_to_velocity(0.75) <= 110
    assert 70 <= _amp_to_velocity(0.85) <= 110
    assert _amp_to_velocity(0.0) >= 1 and _amp_to_velocity(1.0) <= 127
    for amp in (0.30, 0.55, 0.85):
        assert _velocity_to_amp(_amp_to_velocity(amp)) == pytest.approx(amp, abs=0.01)


# --- transcription tests --------------------------------------------------
def test_three_plucks_recovered_with_correct_pitch_and_onset(plucks_wav):
    events = transcribe(plucks_wav, mode="solo")
    for pitch, onset in THREE_PLUCKS:
        hits = _near(events, pitch, onset)
        assert len(hits) == 1, f"expected one {pitch} within {ONSET_TOL}s of {onset}s — got {_describe(events)}"
    assert len(events) == 3, _describe(events)

    velocities = [_near(events, p, o)[0].velocity for p, o in THREE_PLUCKS]
    assert all(70 <= v <= 110 for v in velocities), velocities


def test_octave_ghost_on_a_pluck_is_removed_in_solo_mode(plucks_wav):
    """The E3 pluck's 2nd harmonic reads as a separate E4 starting in the same
    frame. Poly mode leaves it; solo mode must not."""
    poly = transcribe(plucks_wav, mode="poly")
    assert _near(poly, 64, 0.20), f"precondition: poly should carry the E4 ghost — got {_describe(poly)}"

    solo = transcribe(plucks_wav, mode="solo")
    assert not _near(solo, 64, 0.20), _describe(solo)


def test_note_split_by_a_volume_dip_merges_to_one_event(dip_wav):
    raw = _raw_pitches(dip_wav)
    fragments = [r for r in raw if r[2] == 55]
    assert len(fragments) >= 2, f"precondition: basic-pitch must split the dipped tone — got {raw}"

    # Asserted in BOTH modes on purpose. basic-pitch also hears a ghost at MIDI
    # 83 spanning the seam, and `merge_adjacent_events` compares each event only
    # against the one it last emitted — so an interleaved pitch blocks the
    # merge. In solo mode refinement happens to delete that ghost first, which
    # would let a broken merge pass this test; poly mode leaves the ghost in
    # place and holds the merge to its actual job.
    for mode in ("solo", "poly"):
        events = transcribe(dip_wav, mode=mode)
        at_pitch = [e for e in events if e.pitch == 55]
        assert len(at_pitch) == 1, f"{mode}: {_describe(events)}"
        note = at_pitch[0]
        assert note.start == pytest.approx(0.2, abs=ONSET_TOL), mode
        assert note.start < 1.0 < note.end, f"{mode}: merged note must span the dip"


def test_dominant_second_harmonic_lands_in_the_right_octave(harmonic_wav):
    """Fundamental at E3 with a louder 2nd harmonic: one note, at E3."""
    poly = transcribe(harmonic_wav, mode="poly")
    assert any(e.pitch == 52 for e in poly), _describe(poly)
    assert any(e.pitch == 64 for e in poly), f"precondition: poly should carry an octave ghost — got {_describe(poly)}"

    solo = transcribe(harmonic_wav, mode="solo")
    assert len(solo) == 1, _describe(solo)
    assert solo[0].pitch == 52


def test_fifth_double_stop_survives_the_octave_rule(fifth_wav):
    """Regression: pyin locks onto the common subharmonic of a harmonic dyad —
    a 52+59 fifth reads as MIDI 40 — so an unguarded octave rule relabels a
    correct note down an octave. Both voices must come back where they were."""
    events = transcribe(fifth_wav, mode="solo")
    pitches = sorted(e.pitch for e in events)
    assert 52 in pitches and 59 in pitches, _describe(events)
    assert 40 not in pitches, f"subharmonic lock moved a correct note — {_describe(events)}"


def test_vibrato_is_flagged_and_a_steady_tone_is_not(vibrato_wav, dip_wav):
    events = transcribe(vibrato_wav, mode="solo")
    assert len(events) == 1, _describe(events)
    assert events[0].pitch == 60
    assert events[0].vibrato, "5.5 Hz / 0.4 semitone modulation should read as vibrato"

    steady = [e for e in transcribe(dip_wav, mode="solo") if e.pitch == 55][0]
    assert not steady.vibrato, "a steady tone must not be flagged"
    assert steady.bend == [], "a steady tone must not carry a bend contour"


def test_pitch_range_filter_and_mode_validation(plucks_wav):
    narrowed = transcribe(plucks_wav, mode="solo", min_pitch=55, max_pitch=70)
    assert narrowed, "narrowing should not empty the transcription"
    assert all(55 <= e.pitch <= 70 for e in narrowed), _describe(narrowed)
    assert not any(e.pitch == 52 for e in narrowed), _describe(narrowed)

    with pytest.raises(ValueError):
        transcribe(plucks_wav, mode="chords")
