"""Core data model for the soloscribe pipeline.

The pipeline moves through three time domains:

  1. Clock time (seconds) — raw audio and transcribed NoteEvents.
  2. Beat time — a BeatGrid maps beat indices to clock time (jazz recordings
     drift; the grid absorbs the drift so notation stays musical).
  3. Tick time — quantized musical time, TICKS_PER_QUARTER per quarter note.
     QNotes and the Score live here; the Score keeps a tempo_map so audit
     synthesis can be laid back onto the original recording's clock.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

TICKS_PER_QUARTER = 960

# Divisors of TICKS_PER_QUARTER we allow as grid cells. 960 is divisible by
# 2, 3, 4, 6, 8 — eighths (480), triplet eighths (320), sixteenths (240),
# triplet sixteenths (160), thirty-seconds (120).
GRID_EIGHTH = TICKS_PER_QUARTER // 2
GRID_TRIPLET = TICKS_PER_QUARTER // 3
GRID_SIXTEENTH = TICKS_PER_QUARTER // 4

STANDARD_TUNING = (64, 59, 55, 50, 45, 40)  # string 1 (high E) → string 6 (low E)


@dataclass
class NoteEvent:
    """A transcribed note in continuous (seconds) time."""

    start: float
    end: float
    pitch: int                     # MIDI note number
    velocity: int = 96             # 1–127
    confidence: float = 1.0        # transcriber's belief this note is real
    # Pitch-bend contour, times normalized 0..1 across the note, values in
    # semitones relative to the nominal pitch. Empty list = no bend.
    bend: list[tuple[float, float]] = field(default_factory=list)
    vibrato: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class BeatGrid:
    """Tracked beats in seconds; the bridge between clock and musical time."""

    beat_times: list[float]        # ascending, one entry per beat
    beats_per_bar: int = 4
    first_downbeat: int = 0        # index into beat_times that is a bar's beat 1
    bpm_nominal: float = 120.0
    swing: bool = False            # eighths are swung (jazz feel)

    def bars(self) -> int:
        usable = len(self.beat_times) - self.first_downbeat
        return max(1, (usable + self.beats_per_bar - 1) // self.beats_per_bar)


@dataclass
class QNote:
    """A quantized note in tick time."""

    onset: int                     # ticks from score start
    duration: int                  # ticks (> 0)
    pitch: int
    velocity: int = 96
    string: int | None = None      # 1 = high E … 6 = low E (None until fretted)
    fret: int | None = None
    tied_from_prev: bool = False   # continuation of an earlier QNote at this pitch
    vibrato: bool = False
    bend: list[tuple[float, float]] = field(default_factory=list)
    confidence: float = 1.0

    @property
    def offset(self) -> int:
        return self.onset + self.duration


@dataclass
class Score:
    """Quantized musical content plus the map back into the recording's clock."""

    qnotes: list[QNote] = field(default_factory=list)
    beats_per_bar: int = 4
    denominator: int = 4           # time-signature denominator (4 = quarter beat)
    tempo_bpm: float = 120.0       # nominal tempo written into the GP5
    triplet_feel: bool = False     # notate straight, mark "swing 8ths" in GP5
    key: str = "C"                 # tonic + optional 'm', e.g. "F", "Bb", "Em"
    # (tick, seconds) anchors, ascending in both columns. Piecewise-linear
    # interpolation between anchors reconstructs where each tick falls in the
    # original recording — this is what lets the audit resynthesize the score
    # in alignment with the source audio even when the band drifts.
    tempo_map: list[tuple[int, float]] = field(default_factory=list)
    chords: list[str | None] = field(default_factory=list)  # one per measure, optional
    tuning: tuple[int, ...] = STANDARD_TUNING
    capo: int = 0
    title: str = ""
    subtitle: str = ""

    @property
    def ticks_per_beat(self) -> int:
        return TICKS_PER_QUARTER * 4 // self.denominator

    @property
    def ticks_per_bar(self) -> int:
        return self.ticks_per_beat * self.beats_per_bar

    def n_measures(self) -> int:
        if not self.qnotes:
            return max(1, len(self.chords) or 1)
        last = max(q.offset for q in self.qnotes)
        bars = (last + self.ticks_per_bar - 1) // self.ticks_per_bar
        return max(bars, len(self.chords) or 0, 1)

    def tick_to_seconds(self, tick: int | float) -> float:
        """Map a tick to seconds in the original recording via tempo_map."""
        tm = self.tempo_map
        if not tm:
            return float(tick) / TICKS_PER_QUARTER * 60.0 / self.tempo_bpm
        ticks = [a for a, _ in tm]
        i = bisect.bisect_right(ticks, tick) - 1
        if i < 0:
            i = 0
        if i >= len(tm) - 1:
            i = len(tm) - 2 if len(tm) >= 2 else 0
        if len(tm) == 1:
            t0, s0 = tm[0]
            return s0 + (tick - t0) / TICKS_PER_QUARTER * 60.0 / self.tempo_bpm
        t0, s0 = tm[i]
        t1, s1 = tm[i + 1]
        if t1 == t0:
            return s0
        return s0 + (tick - t0) * (s1 - s0) / (t1 - t0)


def merge_adjacent_events(
    events: list[NoteEvent],
    max_gap: float = 0.06,
    max_pitch_diff: int = 0,
    rearticulation_ratio: float = 0.85,
) -> list[NoteEvent]:
    """Merge same-pitch events separated by tiny gaps (transcriber splits on decay).

    Merging runs per pitch chain, not against the last emitted event — an
    interleaved event at another pitch must not block the repair of a
    decay-split note (found by the transcribe module's tests).

    Gap time alone cannot separate a decay split (~40 ms dip) from a genuine
    repeated fast note in the 15–60 ms band, so that band gets an amplitude
    gate: a decay continuation is the quiet tail of a dying note, a re-picked
    note has a fresh attack. Gaps ≤ 15 ms (about one analysis frame) merge
    unconditionally.

    SCOPE, measured (do not overclaim this gate): the E2E harness's residual
    recall misses are repeated notes whose OBSERVED gap is 0.000 s — the
    transcriber receives them already contiguous — which this gate therefore
    cannot address (they take the unconditional path). An earlier version of
    this note credited the gate with those misses; that conflated the licks'
    NOTATED gaps (30–49 ms) with the emitted signal, where the renderer's
    ring-down closes them. Splitting 0-gap fusions needs onset re-detection,
    which was measured (0.03 discrimination margin, n=9) and declined.
    """
    if not events:
        return []
    by_chain: dict[int, list[NoteEvent]] = {}
    for ev in sorted(events, key=lambda e: (e.start, e.pitch)):
        merged = False
        for pitch in range(ev.pitch - max_pitch_diff, ev.pitch + max_pitch_diff + 1):
            chain = by_chain.get(pitch)
            if chain and 0 <= ev.start - chain[-1].end <= max_gap:
                prev = chain[-1]
                gap = ev.start - prev.end
                if gap > 0.015 and ev.velocity > prev.velocity * rearticulation_ratio:
                    continue  # fresh attack → a repeated note, not a split
                prev.end = max(prev.end, ev.end)
                prev.velocity = max(prev.velocity, ev.velocity)
                prev.confidence = max(prev.confidence, ev.confidence)
                prev.vibrato = prev.vibrato or ev.vibrato
                merged = True
                break
        if not merged:
            by_chain.setdefault(ev.pitch, []).append(ev)
    out = [ev for chain in by_chain.values() for ev in chain]
    out.sort(key=lambda e: (e.start, e.pitch))
    return out
