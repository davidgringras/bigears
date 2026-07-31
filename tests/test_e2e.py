"""End-to-end accuracy tests against synthesized licks with known ground truth.

These are the numbers the project is allowed to quote. Each test prints its
actual measurements *before* it asserts, so a failure says how far off it was
rather than only that it failed. Run with ``-s`` to see the tables on a pass:

    .venv/bin/python -m pytest tests/test_e2e.py -s

Stage gating: every test that touches a pipeline module resolves it at run
time and skips with the module name if it is missing, incomplete, or still
raising NotImplementedError. A skip here means "not built yet", never "passed".

The two tests at the top validate the ground truth itself. They are not gated
on anything, because a lick that is internally inconsistent or renders out of
tune would make every number below meaningless.
"""
from __future__ import annotations

import contextlib
import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import licks as L  # noqa: E402
from soloscribe.model import TICKS_PER_QUARTER  # noqa: E402

ONSET_TOL_TIGHT = 0.05
ONSET_TOL_LOOSE = 0.10

F1_BAR_CLEAN = 0.85
F1_BAR_PAD = 0.70
QUANTIZE_ONSET_BAR = 0.95
FULL_CHAIN_BAR = 0.80

_IDS = [lk.name for lk in L.LICKS]


# --------------------------------------------------------------------------
# Stage gating
# --------------------------------------------------------------------------


def _stage(module: str, attr: str):
    """Resolve soloscribe.<module>.<attr>, or skip naming what is missing."""
    try:
        mod = importlib.import_module(f"soloscribe.{module}")
    except Exception as exc:  # ImportError, or a broken dependency underneath
        pytest.skip(
            f"stage not available: soloscribe.{module} could not be imported "
            f"({type(exc).__name__}: {exc})"
        )
    if not hasattr(mod, attr):
        pytest.skip(f"stage not available: soloscribe.{module}.{attr} does not exist yet")
    return getattr(mod, attr)


@contextlib.contextmanager
def _stage_guard(what: str):
    try:
        yield
    except NotImplementedError as exc:
        pytest.skip(f"stage not implemented: {what} raised NotImplementedError ({exc})")


# --------------------------------------------------------------------------
# Fixtures and caching
# --------------------------------------------------------------------------

_TRANSCRIPTIONS: dict[tuple[str, str], list] = {}


@pytest.fixture(scope="session")
def fixture_paths() -> dict[tuple[str, str], str]:
    """Render every lick x backing once per session."""
    L.write_fixtures()
    return {
        (lk.name, backing): L.fixture_path(lk, backing)
        for lk in L.LICKS
        for backing in L.BACKINGS
    }


def _transcribed(lick: L.Lick, backing: str, paths: dict) -> list:
    """Transcribe once per (lick, backing); inference is the expensive part."""
    key = (lick.name, backing)
    if key not in _TRANSCRIPTIONS:
        transcribe = _stage("transcribe", "transcribe")
        with _stage_guard("soloscribe.transcribe.transcribe"):
            _TRANSCRIPTIONS[key] = transcribe(paths[key], mode="solo")
    return _TRANSCRIPTIONS[key]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _midi_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _arrays(events) -> tuple[np.ndarray, np.ndarray]:
    """NoteEvents → (intervals, pitches in Hz), dropping degenerate spans.

    mir_eval requires end > start and pitch > 0; a transcriber emitting a
    zero-length note would otherwise raise instead of scoring.
    """
    good = [e for e in events if e.end > e.start and e.pitch > 0]
    if not good:
        return np.zeros((0, 2)), np.zeros(0)
    return (
        np.array([[e.start, e.end] for e in good], dtype=float),
        np.array([_midi_hz(e.pitch) for e in good], dtype=float),
    )


def _score(ref_events, est_events, tolerance: float) -> tuple[float, float, float]:
    """(precision, recall, F1) at an onset tolerance, ignoring offsets.

    ``offset_ratio=None`` is mir_eval's documented switch for onset+pitch-only
    scoring; verified against the installed signature, which also defaults
    ``pitch_tolerance`` to 50 cents (a quarter tone).
    """
    from mir_eval.transcription import precision_recall_f1_overlap

    ref_int, ref_hz = _arrays(ref_events)
    est_int, est_hz = _arrays(est_events)
    p, r, f, _ = precision_recall_f1_overlap(
        ref_int, ref_hz, est_int, est_hz,
        onset_tolerance=tolerance, offset_ratio=None,
    )
    return float(p), float(r), float(f)


def _row(lick: L.Lick, ref, est, header: bool = True) -> float:
    """Print the accuracy row for one lick; return F1 at the loose tolerance."""
    p50, r50, f50 = _score(ref, est, ONSET_TOL_TIGHT)
    p100, r100, f100 = _score(ref, est, ONSET_TOL_LOOSE)
    if header:
        print(
            f"\n  {'lick':<9} {'ref':>4} {'est':>4} "
            f"{'P@50':>6} {'R@50':>6} {'F1@50':>6} "
            f"{'P@100':>6} {'R@100':>6} {'F1@100':>7}"
        )
    print(
        f"  {lick.name:<9} {len(ref):>4} {len(est):>4} "
        f"{p50:>6.3f} {r50:>6.3f} {f50:>6.3f} "
        f"{p100:>6.3f} {r100:>6.3f} {f100:>7.3f}"
    )
    return f100


# --------------------------------------------------------------------------
# 0. The ground truth itself
# --------------------------------------------------------------------------


def test_swing_map_and_lick_consistency():
    """Performed and notated agree, lines are monophonic, onsets land on ticks."""
    L.check_swing_map()
    for lick in L.LICKS:
        L.check_lick(lick)
        pitches = [p for _, _, p, _ in lick.notes]
        print(
            f"  {lick.name:<9} {lick.bpm:>5.0f} bpm  {lick.bars} bars  "
            f"{'swung' if lick.swing else 'straight':<8} {len(lick.notes):>3} notes  "
            f"MIDI {min(pitches)}-{max(pitches)}  {lick.duration_seconds():.2f}s"
        )


def test_render_pitch_accuracy():
    """The renderer is in tune, so a pitch error downstream is not ours.

    mir_eval's default pitch window is 50 cents, so a renderer drifting by tens
    of cents would still score as correct while quietly being wrong.
    """
    print(f"\n  {'midi':>4} {'target Hz':>10} {'rendered Hz':>12} {'error':>9}")
    worst = 0.0
    for midi, want, got, cents in L.check_render_pitch(tolerance_cents=5.0):
        print(f"  {midi:>4} {want:>10.3f} {got:>12.3f} {cents:>+7.2f} c")
        worst = max(worst, abs(cents))
    print(f"  worst absolute error: {worst:.2f} cents (budget 5.00)")


# --------------------------------------------------------------------------
# 1. Transcription accuracy, clean render
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lick", L.LICKS, ids=_IDS)
def test_transcription_clean(lick: L.Lick, fixture_paths):
    """Note F1 against ground truth on an unaccompanied render."""
    ref = L.lick_events(lick)
    est = _transcribed(lick, "none", fixture_paths)
    print(f"\ntranscription accuracy — backing='none' ({lick.name})")
    f100 = _row(lick, ref, est)
    print(f"  bar: F1@100ms >= {F1_BAR_CLEAN:.2f}   actual: {f100:.3f}")
    assert f100 >= F1_BAR_CLEAN, (
        f"{lick.name}: F1@100ms = {f100:.3f} on a CLEAN synthetic render, below "
        f"the {F1_BAR_CLEAN:.2f} bar. This is the easiest audio the pipeline "
        f"will ever see."
    )


# --------------------------------------------------------------------------
# 2. Transcription accuracy over a backing pad
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lick", L.LICKS, ids=_IDS)
def test_transcription_pad(lick: L.Lick, fixture_paths):
    """Note F1 with sustained chord tones under the solo at -20 dB RMS."""
    ref = L.lick_events(lick)
    est = _transcribed(lick, "pad", fixture_paths)
    print(f"\ntranscription accuracy — backing='pad' ({lick.name})")
    f100 = _row(lick, ref, est)
    print(f"  bar: F1@100ms >= {F1_BAR_PAD:.2f}   actual: {f100:.3f}")
    assert f100 >= F1_BAR_PAD, (
        f"{lick.name}: F1@100ms = {f100:.3f} over a backing pad, below the "
        f"{F1_BAR_PAD:.2f} bar."
    )


# --------------------------------------------------------------------------
# 3. Quantization, isolated from transcription
# --------------------------------------------------------------------------


def _pair(expected: list[tuple[int, int, int]], qnotes) -> tuple[int, int, int, list]:
    """Index-wise pairing of expected vs quantized, both in (onset, pitch) order.

    Pairing by index is only ambiguous where the quantizer collapses two notes
    onto one tick; a pair whose pitches disagree is counted as a miss, which is
    the conservative reading.
    """
    exp = sorted(expected, key=lambda t: (t[0], t[2]))
    got = sorted(((q.onset, q.duration, q.pitch) for q in qnotes),
                 key=lambda t: (t[0], t[2]))
    n = min(len(exp), len(got))
    onset_hits = pitch_misses = dur_hits = 0
    deltas = []
    for (e_on, e_dur, e_p), (g_on, g_dur, g_p) in zip(exp[:n], got[:n]):
        if e_p != g_p:
            pitch_misses += 1
            continue
        if e_on == g_on:
            onset_hits += 1
        if e_dur == g_dur:
            dur_hits += 1
        deltas.append(abs(e_dur - g_dur))
    return onset_hits, dur_hits, pitch_misses, deltas


def _onset_coverage(expected: list[tuple[int, int, int]], onsets) -> float:
    """Fraction of expected notated onsets present among the produced onsets.

    Multiset, pitch-agnostic — this is the "did the rhythm land" question,
    independent of how the two sequences pair up.
    """
    from collections import Counter

    have = Counter(onsets)
    hits = 0
    for onset, _, _ in expected:
        if have[onset] > 0:
            have[onset] -= 1
            hits += 1
    return hits / len(expected) if expected else 0.0


@pytest.mark.parametrize("lick", L.LICKS, ids=_IDS)
def test_quantization(lick: L.Lick):
    """Ground-truth events + a perfect beat grid → the notation we expect.

    Deliberately fed PERFECT input: no transcription, no beat tracking, no
    tempo drift. Anything that misses here is the quantizer's grid decision,
    not upstream noise.
    """
    quantize = _stage("quantize", "quantize")
    detect_swing = _stage("quantize", "detect_swing")

    events = L.lick_events(lick)
    grid = L.perfect_beat_grid(lick)
    expected = L.lick_notated_ticks(lick)

    detected, median_pos = detect_swing(events, grid)
    with _stage_guard("soloscribe.quantize.quantize"):
        score = quantize(
            events, grid,
            swing="on" if lick.swing else "off",
            key=lick.key, chords=list(lick.chords),
        )

    onset_hits, dur_hits, pitch_misses, deltas = _pair(expected, score.qnotes)
    n = len(expected)
    onset_rate = onset_hits / n
    dur_rate = dur_hits / n
    coverage = _onset_coverage(expected, [q.onset for q in score.qnotes])
    median_delta = float(np.median(deltas)) if deltas else 0.0

    print(f"\nquantization — {lick.name} (perfect grid, ground-truth events)")
    print(f"  notes in / qnotes out       : {n} / {len(score.qnotes)}")
    print(f"  swing: declared={lick.swing}  detected={detected} "
          f"(median offbeat position {median_pos:.3f})")
    print(f"  onset exact-match           : {onset_hits}/{n} = {onset_rate:.3f}"
          f"   bar >= {QUANTIZE_ONSET_BAR:.2f}")
    print(f"  onset multiset coverage     : {coverage:.3f}")
    print(f"  index-wise pitch mismatches : {pitch_misses}")
    print(f"  duration exact-match        : {dur_hits}/{n} = {dur_rate:.3f} "
          f"(reported, not asserted)")
    print(f"  median |duration error|     : {median_delta:.0f} ticks "
          f"({median_delta / TICKS_PER_QUARTER:.3f} quarter notes)")

    if onset_rate < QUANTIZE_ONSET_BAR:
        misses = []
        exp = sorted(expected, key=lambda t: (t[0], t[2]))
        got = sorted(((q.onset, q.duration, q.pitch) for q in score.qnotes),
                     key=lambda t: (t[0], t[2]))
        for (e_on, _, e_p), (g_on, _, g_p) in zip(exp, got):
            if e_on != g_on or e_p != g_p:
                misses.append(
                    f"    beat {e_on / TICKS_PER_QUARTER:>7.3f} MIDI {e_p:<3} "
                    f"expected tick {e_on:<6} got tick {g_on:<6} (MIDI {g_p})"
                )
        print(f"  first {min(12, len(misses))} mismatches:")
        print("\n".join(misses[:12]))

    assert onset_rate >= QUANTIZE_ONSET_BAR, (
        f"{lick.name}: notated onset exact-match {onset_rate:.3f} "
        f"({onset_hits}/{n}), below the {QUANTIZE_ONSET_BAR:.2f} bar, on "
        f"PERFECT input — this is a notation-grid decision, not noise."
    )


# --------------------------------------------------------------------------
# 4. Full chain
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lick", L.LICKS, ids=_IDS)
def test_full_chain_smoke(lick: L.Lick, fixture_paths):
    """audio → transcribe → beat grid → quantize, scored against the page."""
    import soundfile as sf

    build_beat_grid = _stage("quantize", "build_beat_grid")
    quantize = _stage("quantize", "quantize")

    path = fixture_paths[(lick.name, "none")]
    events = _transcribed(lick, "none", fixture_paths)
    y, sr = sf.read(path, dtype="float32", always_2d=False)

    with _stage_guard("soloscribe.quantize.build_beat_grid"):
        grid = build_beat_grid(
            y, sr,
            bpm=lick.bpm,
            downbeat=0.0,
            beats_per_bar=lick.beats_per_bar,
            cover_until=len(y) / sr + 2 * lick.seconds_per_beat,
        )
    with _stage_guard("soloscribe.quantize.quantize"):
        score = quantize(
            events, grid,
            swing="on" if lick.swing else "off",
            key=lick.key, chords=list(lick.chords),
        )

    expected = L.lick_notated_ticks(lick)
    onsets = [q.onset for q in score.qnotes]
    coverage = _onset_coverage(expected, onsets)

    # Two things can sink this score, and they belong to different owners, so
    # measure them apart. (a) The beat grid can lock onto the wrong phase, in
    # which case correctly transcribed notes get snapped into the wrong cell —
    # nothing to do with transcription. (b) The notes can be wrong. Beyond an
    # eighth of a beat of grid error, an onset lands in the neighbouring
    # sixteenth cell, so that is the threshold worth counting.
    spb = lick.seconds_per_beat
    origin = grid.first_downbeat
    n_beats = min(len(grid.beat_times) - origin, int(lick.duration_beats()) + 1)
    dev = [(grid.beat_times[origin + i] - i * spb) / spb for i in range(n_beats)]
    worst = max(dev, key=abs) if dev else 0.0
    off_cell = sum(1 for d in dev if abs(d) > 0.125)

    best_shift, best = 0, coverage
    cell = TICKS_PER_QUARTER // 4
    for shift in range(-2 * TICKS_PER_QUARTER, 2 * TICKS_PER_QUARTER + 1, cell):
        if shift:
            c = _onset_coverage(expected, [o + shift for o in onsets])
            if c > best:
                best_shift, best = shift, c

    print(f"\nfull chain — {lick.name} (audio → transcribe → grid → quantize)")
    print(f"  tracked tempo               : {grid.bpm_nominal:.2f} bpm "
          f"(nominal {lick.bpm:.0f}), {len(grid.beat_times)} beats, "
          f"first_downbeat={origin}")
    print(f"  grid phase vs true metronome: worst {worst:+.3f} beats "
          f"({worst * TICKS_PER_QUARTER:+.0f} ticks); {off_cell}/{n_beats} beats "
          f"off by more than an eighth")
    print(f"  events in / qnotes out      : {len(events)} / {len(score.qnotes)} "
          f"(ground truth {len(expected)})")
    print(f"  notated onsets recovered    : {coverage:.3f}   bar >= {FULL_CHAIN_BAR:.2f}")
    print(f"  best constant-shift rescue  : {best:.3f} at {best_shift:+d} ticks "
          f"({best_shift / TICKS_PER_QUARTER:+.2f} beats)")
    if off_cell > n_beats // 4:
        print("  → the grid, not the transcription, is what is losing these notes: "
              f"bpm={lick.bpm:.0f} and downbeat=0.0 were both supplied and the "
              "tracked grid still drifted past a sixteenth cell.")

    assert coverage >= FULL_CHAIN_BAR, (
        f"{lick.name}: only {coverage:.3f} of ground-truth notated onsets "
        f"survive the full chain, below the {FULL_CHAIN_BAR:.2f} bar. "
        f"Grid phase error worst {worst:+.3f} beats with {off_cell}/{n_beats} "
        f"beats beyond an eighth; a constant {best_shift:+d}-tick shift would "
        f"recover {best:.3f}."
    )
