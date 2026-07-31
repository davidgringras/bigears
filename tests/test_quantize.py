"""Quantizer unit tests against synthetic, perfectly-known event streams."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from soloscribe.model import BeatGrid, NoteEvent, TICKS_PER_QUARTER as TPQ
from soloscribe.quantize import detect_swing, parse_chords, quantize


def _grid(bpm=120.0, n_beats=32, beats_per_bar=4):
    period = 60.0 / bpm
    return BeatGrid(
        beat_times=[i * period for i in range(n_beats)],
        beats_per_bar=beats_per_bar,
        bpm_nominal=bpm,
    )


def _ev(onset_beats, dur_beats, pitch, bpm=120.0):
    period = 60.0 / bpm
    return NoteEvent(onset_beats * period, (onset_beats + dur_beats) * period, pitch)


def test_straight_eighths_exact():
    grid = _grid()
    events = [_ev(i * 0.5, 0.45, 60 + i) for i in range(16)]
    score = quantize(events, grid, swing="off")
    assert not score.triplet_feel
    for i, q in enumerate(score.qnotes):
        assert q.onset == i * TPQ // 2, (i, q.onset)
        assert q.duration == TPQ // 2


def test_swing_detected_and_notated_straight():
    grid = _grid(bpm=150.0)
    events = []
    for beat in range(8):
        events.append(_ev(beat, 0.6, 60 + beat, bpm=150.0))
        events.append(_ev(beat + 2 / 3, 0.30, 72 - beat, bpm=150.0))  # swung offbeat
    score = quantize(events, grid, swing="auto")
    assert score.triplet_feel, "swing not detected"
    offbeats = [q for q in score.qnotes if q.onset % TPQ != 0]
    assert offbeats, "no offbeat notes survived"
    for q in offbeats:
        assert q.onset % TPQ == TPQ // 2, f"swung 8th notated at {q.onset % TPQ}, want {TPQ//2}"


def test_genuine_triplet_in_swing_context():
    grid = _grid(bpm=150.0)
    events = []
    for beat in range(6):  # enough swung pairs to trigger swing detection
        events.append(_ev(beat, 0.6, 60, bpm=150.0))
        events.append(_ev(beat + 2 / 3, 0.3, 62, bpm=150.0))
    # beat 6: a real triplet — onset at 1/3 is the tell
    events += [_ev(6.0, 0.3, 65, bpm=150.0), _ev(6 + 1 / 3, 0.3, 67, bpm=150.0),
               _ev(6 + 2 / 3, 0.3, 69, bpm=150.0)]
    score = quantize(events, grid, swing="auto")
    assert score.triplet_feel
    trip = [q for q in score.qnotes if q.onset >= 6 * TPQ and q.onset < 7 * TPQ]
    assert [q.onset - 6 * TPQ for q in trip] == [0, TPQ // 3, 2 * TPQ // 3], (
        [q.onset - 6 * TPQ for q in trip]
    )


def test_sixteenth_syncopation():
    grid = _grid(bpm=104.0)
    onsets = [0.0, 0.25, 0.75, 1.5, 1.75, 2.25, 2.5, 3.0]
    events = [_ev(o, 0.2, 52 + i, bpm=104.0) for i, o in enumerate(onsets)]
    score = quantize(events, grid, swing="off")
    got = [q.onset for q in score.qnotes]
    want = [round(o * TPQ) for o in onsets]
    assert got == want, (got, want)


def test_drifting_tempo_still_snaps():
    # Accelerando: beat period shrinks 500ms → 400ms across 24 beats.
    times, t = [], 0.0
    for i in range(24):
        times.append(t)
        t += 0.5 - 0.1 * i / 24
    grid = BeatGrid(beat_times=times, beats_per_bar=4, bpm_nominal=120)
    events = [NoteEvent(times[i] + (times[i + 1] - times[i]) * 0.5, 0.15, 60)
              for i in range(0, 20, 2)]  # offbeat 8ths in local beat time
    score = quantize(events, grid, swing="off")
    for q in score.qnotes:
        assert q.onset % TPQ == TPQ // 2, q.onset
    # tempo map follows the drift: later beats are closer together in seconds
    early = score.tick_to_seconds(4 * TPQ) - score.tick_to_seconds(0)
    late = score.tick_to_seconds(20 * TPQ) - score.tick_to_seconds(16 * TPQ)
    assert late < early


def test_pickup_shifts_notes_and_tempo_map_together():
    grid = _grid(bpm=120.0)
    grid.first_downbeat = 2  # downbeat at beat index 2 → earlier notes are pickup
    events = [_ev(1.0, 0.4, 55), _ev(1.5, 0.4, 57), _ev(2.0, 0.9, 60)]
    score = quantize(events, grid, swing="off")
    assert all(q.onset >= 0 for q in score.qnotes)
    # The downbeat note (originally beat 2 = 1.0s) must map back to ~1.0s.
    down = [q for q in score.qnotes if q.pitch == 60][0]
    assert abs(score.tick_to_seconds(down.onset) - 1.0) < 1e-6


def test_overlap_trim_preserves_double_stops():
    grid = _grid()
    events = [
        NoteEvent(0.0, 2.0, 60),   # would overlap the next onset
        NoteEvent(0.0, 1.0, 64),   # same onset → double-stop, kept
        NoteEvent(1.0, 1.4, 67),
    ]
    score = quantize(events, grid, swing="off")
    d0 = [q for q in score.qnotes if q.pitch == 60][0]
    assert d0.offset <= [q for q in score.qnotes if q.pitch == 67][0].onset
    assert len([q for q in score.qnotes if q.onset == 0]) == 2


def test_parse_chords():
    assert parse_chords("Gm7|C7|Fmaj7") == ["Gm7", "C7", "Fmaj7"]
    assert parse_chords("Gm7 C7\nFmaj7") == ["Gm7 C7", "Fmaj7"]
    assert parse_chords(None) == []
    assert parse_chords("A||B") == ["A", None, "B"]
