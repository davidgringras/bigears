"""Ground-truth test licks: synthesized guitar lines where we know every note.

The point of this module is that accuracy claims about the pipeline are
*measured*, not asserted. Each lick is written twice:

  * ``notated`` — what a correct transcription should print on the page.
    Offbeat eighths sit at 0.5; triplets sit at thirds of a beat.
  * ``notes`` — what the player actually *did*, in performed beat time.
    In a swung lick the offbeat eighths have moved to the 2/3 point, which
    is where the audio actually puts them and what the transcriber will see.

Both are derived from one source table, so they cannot drift apart (the
displacement is applied by :func:`_swing_pos`, whose whole lookup table is
asserted in ``_self_check``). Everything downstream — the rendered audio, the
:class:`~soloscribe.model.NoteEvent` reference for mir_eval, the tick reference
for the quantizer — comes off those two lists.

Synthesis here is deliberately *not* :mod:`soloscribe.synth`. An end-to-end
test that renders with the pipeline's own synthesizer is testing a round trip
through one set of assumptions; a second, independent Karplus-Strong
implementation makes the transcription bar mean something. Pitch accuracy of
this renderer is verified by FFT (see ``_self_check``), not assumed.

Run directly to write ``tests/fixtures/*.wav``:

    .venv/bin/python tests/licks.py
"""
from __future__ import annotations

import math
import os
import sys
import zlib
from dataclasses import dataclass, field

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from soloscribe.model import BeatGrid, NoteEvent, TICKS_PER_QUARTER  # noqa: E402

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
BACKINGS = ("none", "pad", "pad+noise")

# Grid positions written as exact thirds so the musical intent survives a read.
_T1 = 1.0 / 3.0
_T2 = 2.0 / 3.0

_EPS = 1e-6


# --------------------------------------------------------------------------
# Lick definition
# --------------------------------------------------------------------------


@dataclass
class Lick:
    """A lick we know the truth about.

    ``notes`` are PERFORMED: (onset_beats, duration_beats, midi_pitch, velocity),
    quarter-note beats counted from bar 1 beat 1. In a swung lick the offbeat
    eighths sit at the 2/3 point of the beat (0.667, 1.667, ...), because that
    is where they are in the audio.

    ``notated`` is the same line as it should be *written*:
    (onset_beats, duration_beats, midi_pitch), offbeat eighths back at 0.5.
    This is the target a swing-aware quantizer has to reproduce.
    """

    name: str
    bpm: float
    beats_per_bar: int
    swing: bool
    key: str
    notes: list[tuple[float, float, int, int]]
    notated: list[tuple[float, float, int]] = field(default_factory=list)
    chords: list[str] = field(default_factory=list)  # one symbol per bar
    description: str = ""
    # Semitones the melody is doubled by. 12 = Wes-style octaves, i.e. every
    # entry in the source table sounds as two simultaneous notes. 0 = a single
    # monophonic line.
    doubling: int = 0

    @property
    def bars(self) -> int:
        return len(self.chords) if self.chords else 1

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm

    def duration_beats(self) -> float:
        return max(on + dur for on, dur, _, _ in self.notes)

    def duration_seconds(self) -> float:
        return self.duration_beats() * self.seconds_per_beat


# --------------------------------------------------------------------------
# Swing displacement
# --------------------------------------------------------------------------

# Notated grid position within a beat → where it is actually played under a
# swing feel. Two entries are identities on purpose: a genuine triplet is
# never displaced (it is already where swing is pulling toward), and 2/3 is
# both a triplet position and the landing spot of a swung eighth. That
# collision is the ambiguity the quantizer has to resolve from context — a
# triplet beat carries three onsets, a swung pair carries two.
_SWING_MAP: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),      # on the beat
    (1 / 4, 1 / 3),  # swung sixteenth, first offbeat
    (_T1, _T1),      # genuine triplet — never displaced
    (0.5, _T2),      # the swung eighth: the whole point
    (_T2, _T2),      # genuine triplet, and where the swung eighth lands
    (3 / 4, 5 / 6),  # swung sixteenth, second offbeat
)


def _split_beat(t: float) -> tuple[int, float]:
    """(beat index, fraction within the beat) for a notated position."""
    b = math.floor(t + _EPS)
    return b, max(0.0, t - b)


def _swing_frac(frac: float) -> float:
    """Map one notated within-beat position to its performed position."""
    for notated, performed in _SWING_MAP:
        if abs(frac - notated) < _EPS:
            return performed
    raise ValueError(
        f"position {frac:.6f} of a beat is not on the notation grid "
        f"({', '.join(f'{n:.4f}' for n, _ in _SWING_MAP)}) — a lick was written "
        "off-grid, or the grid needs a new entry"
    )


def _swing_pos(t: float, swing: bool) -> float:
    """Notated beat position → performed beat position."""
    b, frac = _split_beat(t)
    if not swing:
        _swing_frac(frac)  # validate grid legality even when nothing moves
        return t
    return b + _swing_frac(frac)


# --------------------------------------------------------------------------
# Velocity: metric accent, no randomness
# --------------------------------------------------------------------------

# How hard each metric position gets hit. Deliberately a narrow range (86–104,
# under 2 dB once mapped to amplitude) — real dynamics, but not so wide that a
# quiet note becomes a transcription coin-flip and muddies what the F1 measures.
_ACCENT_DOWNBEAT = 104
_ACCENT_BACKBEAT = 98
_ACCENT_BEAT = 94
_ACCENT_SWUNG_AND = 96
_ACCENT_STRAIGHT_AND = 90
_ACCENT_TRIPLET_INNER = 86
_ACCENT_SIXTEENTH = 92


def _velocity(onset: float, swing: bool, beats_per_bar: int) -> int:
    """Metric accent for a notated onset: the beat carries, the offbeat pushes."""
    beat, frac = _split_beat(onset)
    in_bar = beat % beats_per_bar
    if abs(frac) < _EPS:
        if in_bar == 0:
            return _ACCENT_DOWNBEAT
        return _ACCENT_BACKBEAT if in_bar == 2 else _ACCENT_BEAT
    if abs(frac - 0.5) < _EPS:
        return _ACCENT_SWUNG_AND if swing else _ACCENT_STRAIGHT_AND
    if abs(frac - _T1) < _EPS or abs(frac - _T2) < _EPS:
        return _ACCENT_TRIPLET_INNER
    return _ACCENT_SIXTEENTH


# Pick lift before the next note. Real picking never holds a note to the
# following onset; a rendered line without this reads as one smeared legato
# blob and hands the transcriber a much easier problem than a real recording.
_ARTICULATION_GAP_BEATS = 0.09
_ARTICULATION_MAX_FRACTION = 0.30

# How much softer the upper note of an octave sits than the lower one.
_DOUBLING_VELOCITY_TRIM = 6


def _build(
    name: str,
    bpm: float,
    key: str,
    swing: bool,
    chords: list[str],
    table: list[tuple[float, float, int]],
    description: str,
    beats_per_bar: int = 4,
    doubling: int = 0,
) -> Lick:
    """Source table (notated) → Lick carrying both notated and performed lists."""
    notated: list[tuple[float, float, int]] = []
    notes: list[tuple[float, float, int, int]] = []
    for onset, dur, pitch in table:
        vel = _velocity(onset, swing, beats_per_bar)
        p_on = _swing_pos(onset, swing)
        p_end = _swing_pos(onset + dur, swing)
        nominal = p_end - p_on
        gap = min(_ARTICULATION_GAP_BEATS, _ARTICULATION_MAX_FRACTION * nominal)
        # Low voice first, then the doubling, so ordering within a simultaneous
        # group is deterministic. The upper octave sits slightly back: in a
        # thumb-played octave the higher string speaks a little softer.
        for offset, vel_trim in ((0, 0), (doubling, _DOUBLING_VELOCITY_TRIM)):
            # Notated values stay unrounded: they are exact thirds and quarters
            # of a beat, and rounding them independently makes onset+duration
            # miss the next onset by ~1e-9, which reads as an overlap that is
            # not there.
            notated.append((onset, dur, pitch + offset))
            notes.append((round(p_on, 6), round(nominal - gap, 6),
                          pitch + offset, vel - vel_trim))
            if not doubling:
                break
    return Lick(
        name=name,
        bpm=bpm,
        beats_per_bar=beats_per_bar,
        swing=swing,
        key=key,
        notes=notes,
        notated=notated,
        chords=chords,
        description=description,
        doubling=doubling,
    )


# --------------------------------------------------------------------------
# The licks
# --------------------------------------------------------------------------

# 8 bars over ii-V-I in F. Scalar bebop eighths with the major-third passing
# tone through Gm7 (bar 1), an enclosure into the 3rd of F (bar 2), an arrival
# that stops moving (bar 3), a climb to the top of the range (bar 5), and a
# descent that ends where the guitar actually lives (bars 7-8).
_BEBOP_F_TABLE: list[tuple[float, float, int]] = [
    # bar 1 — Gm7, ascending G dorian with the bebop B natural
    (0.0, 0.5, 62), (0.5, 0.5, 64), (1.0, 0.5, 65), (1.5, 0.5, 67),
    (2.0, 0.5, 69), (2.5, 0.5, 70), (3.0, 0.5, 71), (3.5, 0.5, 72),
    # bar 2 — C7, up to G5 then a triplet down; Bb-Ab encloses the A
    (4.0, 0.5, 74), (4.5, 0.5, 76), (5.0, 0.5, 79), (5.5, 0.5, 77),
    (6.0, _T1, 76), (6 + _T1, _T1, 74), (6 + _T2, _T1, 72),
    (7.0, 0.5, 70), (7.5, 0.5, 68),
    # bar 3 — Fmaj7, the line resolves onto the 3rd and rests
    (8.0, 2.5, 69),
    (11.0, 0.5, 72), (11.5, 0.5, 74),
    # bar 4 — Fmaj7, turn, triplet descent, F# leading into Gm7
    (12.0, 0.5, 76), (12.5, 0.5, 77),
    (13.0, _T1, 76), (13 + _T1, _T1, 74), (13 + _T2, _T1, 72),
    (14.0, 0.5, 70), (14.5, 0.5, 69), (15.0, 0.5, 67), (15.5, 0.5, 66),
    # bar 5 — Gm7 arpeggio to the peak
    (16.0, 0.5, 67), (16.5, 0.5, 70), (17.0, 0.5, 74), (17.5, 0.5, 77),
    (18.0, 0.5, 79), (18.5, 0.5, 81), (19.0, 0.5, 79), (19.5, 0.5, 77),
    # bar 6 — C7 descent with a chromatic, triplet on beat 4 resolving up to F
    (20.0, 0.5, 76), (20.5, 0.5, 75), (21.0, 0.5, 74), (21.5, 0.5, 72),
    (22.0, 0.5, 70), (22.5, 0.5, 69),
    (23.0, _T1, 67), (23 + _T1, _T1, 65), (23 + _T2, _T1, 64),
    # bar 7 — Fmaj7, drop an octave, bottom of the range, then a breath
    (24.0, 0.5, 65), (24.5, 0.5, 64), (25.0, 0.5, 62), (25.5, 0.5, 60),
    (26.0, 0.5, 57), (26.5, 0.5, 55), (27.0, 0.5, 57),
    # bar 8 — Fmaj7, walk back up and land
    (28.0, 0.5, 60), (28.5, 0.5, 62), (29.0, 0.5, 64), (29.5, 0.5, 65),
    (30.0, 2.0, 69),
]

# 4 bars of A minor pentatonic over the top of an A blues. Sixteenth pickups,
# a quarter-note triplet across beats 1-2 of bar 2, the b5 as a passing tone,
# and an octave drop to the low root to finish.
_BLUES_A_TABLE: list[tuple[float, float, int]] = [
    # bar 1 — statement, with the A repeated three times
    (0.0, 0.5, 57), (0.5, 0.25, 60), (0.75, 0.75, 62), (1.5, 0.5, 60),
    (2.0, 1.25, 57), (3.5, 0.5, 57),
    # bar 2 — quarter-note triplet (three notes across two beats), then a turn
    (4.0, _T2, 64), (4 + _T2, _T2, 62), (4 + 2 * _T2, _T2, 60),
    (6.0, 0.75, 57), (6.75, 0.25, 55), (7.0, 1.0, 57),
    # bar 3 — climb through the blue note to the top of the range
    (8.0, 0.5, 57), (8.5, 0.5, 60), (9.0, 0.25, 62), (9.25, 0.25, 63),
    (9.5, 0.5, 64), (10.0, 0.5, 67), (10.5, 0.5, 64), (11.0, 1.0, 69),
    # bar 4 — descend the box and drop to the low A
    (12.0, 0.5, 67), (12.5, 0.5, 64), (13.0, 0.25, 62), (13.25, 0.25, 60),
    (13.5, 0.5, 57), (14.0, 0.5, 55), (14.5, 0.5, 52), (15.0, 1.0, 45),
]

# 4 bars of E minor pentatonic sixteenths. Onsets sit on the 'e' and 'a' of the
# beat far more often than on it — the line is defined by where it *isn't*.
_FUNK_E_TABLE: list[tuple[float, float, int]] = [
    # bar 1 — low root stabs, then climb the box
    (0.0, 0.25, 40), (0.75, 0.25, 40), (1.25, 0.25, 43), (1.5, 0.25, 45),
    (1.75, 0.5, 47), (2.5, 0.25, 50), (2.75, 0.75, 52), (3.75, 0.25, 40),
    # bar 2 — the octave-up answer, then walk back down
    (4.0, 0.25, 55), (4.25, 0.25, 52), (4.75, 0.25, 55), (5.25, 0.25, 57),
    (5.5, 0.25, 55), (5.75, 0.5, 52), (6.5, 0.25, 50), (6.75, 0.25, 52),
    (7.25, 0.25, 47), (7.5, 0.25, 45), (7.75, 0.25, 43),
    # bar 3 — through the b5 to the top of the range
    (8.0, 0.25, 52), (8.25, 0.25, 55), (8.75, 0.25, 57), (9.0, 0.25, 58),
    (9.25, 0.5, 59), (9.75, 0.25, 62), (10.25, 0.5, 64), (10.75, 0.25, 62),
    (11.0, 0.25, 59), (11.25, 0.5, 57),
    # bar 4 — back to the bottom, three-stab tag
    (12.0, 0.25, 52), (12.25, 0.25, 50), (12.5, 0.25, 47), (12.75, 0.25, 45),
    (13.25, 0.25, 43), (13.75, 0.25, 45), (14.0, 0.5, 47), (14.75, 0.25, 43),
    (15.0, 0.25, 40), (15.25, 0.25, 40), (15.75, 0.25, 40),
]


BEBOP_F = _build(
    "bebop_f", 150.0, "F", True,
    ["Gm7", "C7", "Fmaj7", "Fmaj7", "Gm7", "C7", "Fmaj7", "Fmaj7"],
    _BEBOP_F_TABLE,
    "8 bars of swung bebop eighths over ii-V-I in F, three triplet figures",
)

BLUES_A = _build(
    "blues_a", 92.0, "Am", False,
    ["A7", "A7", "D7", "A7"],
    _BLUES_A_TABLE,
    "4 bars of A minor pentatonic, straight eighths and sixteenths, "
    "one quarter-note triplet",
)

FUNK_E = _build(
    "funk_e", 104.0, "Em", False,
    ["Em7", "Em7", "Em7", "Em7"],
    _FUNK_E_TABLE,
    "4 bars of syncopated straight-sixteenth E minor pentatonic",
)

# 4 bars of Wes-style octave melody over G7: every note of the line sounds
# twice, an octave apart, played as one gesture. This exists to price a
# specific trade — a transcriber tuned to suppress octave GHOSTS has no way to
# tell those from octave DOUBLE-STOPS by pitch alone, so whatever heuristic
# kills the ghosts is also the thing that can flatten this lick to a single
# line. It is deliberately kept out of LICKS: octave playing in a monophonic
# "solo" mode is a known-hard case, and folding it into the F1 bars would
# assert a capability nobody has claimed.
_OCTAVES_G_TABLE: list[tuple[float, float, int]] = [
    # bar 1 — statement, up the G minor pentatonic
    (0.0, 0.5, 55), (0.5, 0.5, 58), (1.0, 0.5, 60), (1.5, 0.5, 62),
    (2.0, 1.0, 60), (3.0, 0.5, 58), (3.5, 0.5, 55),
    # bar 2 — sit on the 9th, then push back up
    (4.0, 1.5, 57), (5.5, 0.5, 58), (6.0, 0.5, 60), (6.5, 1.5, 58),
    # bar 3 — descend from the top of the phrase
    (8.0, 0.5, 62), (8.5, 0.5, 60), (9.0, 0.5, 58), (9.5, 0.5, 57),
    (10.0, 1.0, 55), (11.0, 0.5, 53), (11.5, 0.5, 55),
    # bar 4 — climb and land
    (12.0, 0.5, 58), (12.5, 0.5, 60), (13.0, 1.0, 62), (14.0, 2.0, 55),
]

OCTAVES_G = _build(
    "octaves_g", 132.0, "G", True,
    ["G7", "G7", "G7", "G7"],
    _OCTAVES_G_TABLE,
    "4 bars of Wes-style swung octave melody in G — 22 melody notes sounding "
    "as 44, every one an octave double-stop",
    doubling=12,
)

# The three licks the accuracy bars are asserted against.
LICKS: list[Lick] = [BEBOP_F, BLUES_A, FUNK_E]
# Everything that gets rendered to a fixture, characterization material included.
ALL_LICKS: list[Lick] = LICKS + [OCTAVES_G]
LICKS_BY_NAME: dict[str, Lick] = {lk.name: lk for lk in ALL_LICKS}


# --------------------------------------------------------------------------
# Reference views: seconds for mir_eval, ticks for the quantizer
# --------------------------------------------------------------------------


def lick_events(lick: Lick) -> list[NoteEvent]:
    """Performed ground truth as NoteEvents in seconds — the mir_eval reference."""
    spb = lick.seconds_per_beat
    return [
        NoteEvent(start=on * spb, end=(on + dur) * spb, pitch=pitch, velocity=vel)
        for on, dur, pitch, vel in lick.notes
    ]


def _to_ticks(beats: float) -> int:
    """Beats → ticks, refusing anything that is not exactly on the tick lattice."""
    exact = beats * TICKS_PER_QUARTER
    ticks = round(exact)
    if abs(exact - ticks) > 1e-6:
        raise ValueError(
            f"{beats} beats = {exact} ticks is not an integer tick at "
            f"TICKS_PER_QUARTER={TICKS_PER_QUARTER}"
        )
    return int(ticks)


def lick_notated_ticks(lick: Lick) -> list[tuple[int, int, int]]:
    """Notated ground truth as (onset_ticks, duration_ticks, pitch)."""
    source = lick.notated or [(on, dur, p) for on, dur, p, _ in lick.notes]
    return [(_to_ticks(on), _to_ticks(dur), pitch) for on, dur, pitch in source]


def perfect_beat_grid(lick: Lick, extra_beats: int = 4) -> BeatGrid:
    """A metronomic BeatGrid at the lick's exact tempo, beat 1 at t=0.

    ``extra_beats`` of headroom past the last note matter: the quantizer's
    ``_beat_position`` clamps to ``len(beat_times) - 2``, so a grid that stops
    at the final note folds the tail of the lick back into the previous beat.
    """
    spb = lick.seconds_per_beat
    n_beats = int(math.ceil(lick.duration_beats())) + extra_beats
    return BeatGrid(
        beat_times=[i * spb for i in range(n_beats + 1)],
        beats_per_bar=lick.beats_per_bar,
        first_downbeat=0,
        bpm_nominal=lick.bpm,
        swing=lick.swing,
    )


# --------------------------------------------------------------------------
# Karplus-Strong synthesis (independent of soloscribe.synth on purpose)
# --------------------------------------------------------------------------

_T60 = 2.0          # seconds for a plucked note to fall 60 dB
_RELEASE = 0.045    # seconds of ring-down once the finger lifts
_ATTACK = 0.003     # seconds of onset ramp, enough to kill the click
_PLUCK_KNEE = 2.5   # harmonic index where the pluck's spectrum is 3 dB down


def _midi_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _ks_pluck(freq: float, n: int, sr: int, seed: int) -> np.ndarray:
    """Extended Karplus-Strong with a fractional-delay loop.

    The loop is ``y[n] = g * (c0*y[n-D] + c1*y[n-D-1] + c2*y[n-D-2])``, where
    the three taps are a linear interpolator (1-frac, frac) cascaded with the
    classic one-zero damping average (0.5, 0.5). That cascade has group delay
    ``D + frac + 0.5`` at DC, so the loop length is set to ``sr/freq - 0.5``
    and the half sample the damping filter costs is paid back — without it
    every note renders flat, badly so up high. Taps sum to 1, so ``g`` alone
    sets decay: amplitude falls by ``g`` once per period.

    Written blockwise (a chunk of D-2 samples depends only on samples already
    written) because a scalar Python loop over a whole lick is not free.
    """
    loop = sr / freq - 0.5
    D = int(loop)
    frac = loop - D
    if D < 3:
        raise ValueError(f"{freq:.1f} Hz is too high for sr={sr}")
    c0, c1, c2 = 0.5 * (1.0 - frac), 0.5, 0.5 * frac
    g = 0.001 ** (1.0 / (_T60 * freq))  # -60 dB in _T60 seconds

    rng = np.random.default_rng(seed)
    pre = D + 3
    y = np.zeros(pre + n)

    # The excitation burst is one loop period long, so bin k of its spectrum is
    # very nearly harmonic k — which makes this the right place to impose a
    # plucked-string envelope. It matters more than it looks: an unshaped white
    # burst leaves the harmonic balance to the seed, and measured on a raw
    # burst, MIDI 40 came out with its 5th harmonic 7.6 dB LOUDER than the
    # fundamental. A ground truth like that turns the transcription score into
    # a lottery over which notes happened to render hollow. Rolling off by
    # harmonic index (not by absolute frequency, which would leave the low
    # strings untouched) gives every note the same pluck-shaped spectrum.
    spec = np.fft.rfft(rng.uniform(-1.0, 1.0, pre))
    k = np.arange(len(spec))
    spec *= 1.0 / (1.0 + (k / _PLUCK_KNEE) ** 2)
    spec[0] = 0.0  # no DC thump in the loop
    y[:pre] = np.fft.irfft(spec, n=pre)

    step = D - 2
    i = pre
    end = pre + n
    while i < end:
        k = min(step, end - i)
        y[i:i + k] = g * (
            c0 * y[i - D:i - D + k]
            + c1 * y[i - D - 1:i - D - 1 + k]
            + c2 * y[i - D - 2:i - D - 2 + k]
        )
        i += k
    return y[pre:]


def _render_note(freq: float, dur_s: float, vel: int, sr: int, seed: int) -> np.ndarray:
    n = max(8, int(round((dur_s + _RELEASE) * sr)))
    y = _ks_pluck(freq, n, sr, seed)
    env = np.ones(n)
    a = min(int(_ATTACK * sr), n // 2)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a)
    r = min(int(_RELEASE * sr), n - a)
    if r > 0:
        env[-r:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, r)))
    return y * env * (vel / 127.0) ** 1.5


# --------------------------------------------------------------------------
# Backing
# --------------------------------------------------------------------------

# Low-register comping voicings. The blues dominants are shells with no third:
# a sustained C# under a line built on C natural is a semitone grinder that
# would make the pad test measure clash tolerance rather than backing
# robustness. Dropping the third is also what a blues rhythm player does.
_PAD_VOICINGS: dict[str, tuple[int, ...]] = {
    "Gm7":   (43, 50, 58, 65),  # G2  D3  Bb3 F4   root 5 b3 b7
    "C7":    (48, 55, 64, 70),  # C3  G3  E4  Bb4  root 5 3 b7
    "Fmaj7": (41, 48, 57, 64),  # F2  C3  A3  E4   root 5 3 7
    "A7":    (45, 52, 57, 67),  # A2  E3  A3  G4   root 5 8 b7  (no 3rd)
    "D7":    (50, 57, 62, 72),  # D3  A3  D4  C5   root 5 8 b7  (no 3rd)
    "Em7":   (40, 47, 55, 62),  # E2  B2  G3  D4   root 5 b3 b7
    "G7":    (43, 50, 59, 65),  # G2  D3  B3  F4   root 5 3 b7
}

PAD_DB = -20.0    # pad level relative to the solo, RMS-referenced
NOISE_DB = -32.0  # pink noise level relative to the solo, RMS-referenced
_PAD_CUTOFF = 1200.0


def _lowpass(x: np.ndarray, sr: int, fc: float) -> np.ndarray:
    """Zero-phase two-pole magnitude rolloff, done in the frequency domain."""
    if len(x) < 4:
        return x
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    return np.fft.irfft(spec / (1.0 + (f / fc) ** 2), n=len(x))


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink (1/f power) noise by shaping white noise in the frequency domain."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0)
    scale = np.zeros_like(f)
    scale[1:] = 1.0 / np.sqrt(f[1:])  # amplitude 1/sqrt(f) → power 1/f
    out = np.fft.irfft(spec * scale, n=n)
    peak = float(np.max(np.abs(out)))
    return out / peak if peak > 0 else out


def _render_pad(lick: Lick, sr: int, n: int) -> np.ndarray:
    """Sustained chord tones, one voicing per bar, following the changes."""
    out = np.zeros(n)
    bar_s = lick.seconds_per_beat * lick.beats_per_bar
    for bar, symbol in enumerate(lick.chords):
        voicing = _PAD_VOICINGS[symbol]
        i0 = int(bar * bar_s * sr)
        i1 = min(int((bar + 1) * bar_s * sr), n)
        if i1 - i0 <= 0:
            continue
        # Absolute time keeps each voice phase-continuous across the barline.
        t = np.arange(i0, i1) / sr
        seg = np.zeros(i1 - i0)
        for k, midi in enumerate(voicing):
            seg += np.sin(2.0 * math.pi * _midi_hz(midi) * t + k * 0.7) / (k + 1.6)
        fade = min(int(0.020 * sr), (i1 - i0) // 2)
        if fade > 0:
            ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, fade)))
            seg[:fade] *= ramp
            seg[-fade:] *= ramp[::-1]
        out[i0:i1] += seg
    return _lowpass(out, sr, _PAD_CUTOFF)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def _scale_to_db(x: np.ndarray, reference_rms: float, db: float) -> np.ndarray:
    r = _rms(x)
    if r <= 0 or reference_rms <= 0:
        return x
    return x * (reference_rms * 10.0 ** (db / 20.0) / r)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

PEAK = 0.89  # leave headroom so nothing clips into the transcriber


def render_lick(lick: Lick, sr: int = 22050, backing: str = "none") -> np.ndarray:
    """Render a lick to mono audio.

    ``backing`` is "none", "pad" (sustained chord tones under the solo at
    -20 dB RMS) or "pad+noise" (that, plus pink noise at -32 dB RMS).
    Deterministic: the noise seeds are derived from the lick's name.
    """
    if backing not in BACKINGS:
        raise ValueError(f"backing must be one of {BACKINGS}, got {backing!r}")

    spb = lick.seconds_per_beat
    total = lick.duration_seconds() + _T60 * 0.5
    n = int(math.ceil(total * sr))
    base_seed = zlib.crc32(lick.name.encode()) & 0x7FFFFFFF

    solo = np.zeros(n)
    for i, (onset, dur, pitch, vel) in enumerate(lick.notes):
        note = _render_note(_midi_hz(pitch), dur * spb, vel, sr, base_seed + i)
        start = int(round(onset * spb * sr))
        stop = min(start + len(note), n)
        if stop > start:
            solo[start:stop] += note[:stop - start]

    mix = solo.copy()
    solo_rms = _rms(solo)
    if backing in ("pad", "pad+noise"):
        mix += _scale_to_db(_render_pad(lick, sr, n), solo_rms, PAD_DB)
    if backing == "pad+noise":
        rng = np.random.default_rng(base_seed + 999_983)
        mix += _scale_to_db(_pink_noise(n, rng), solo_rms, NOISE_DB)

    peak = float(np.max(np.abs(mix)))
    if peak > 0:
        mix *= PEAK / peak
    return mix.astype(np.float32)


def fixture_path(lick: Lick, backing: str) -> str:
    tag = backing.replace("+", "_")
    return os.path.join(FIXTURE_DIR, f"{lick.name}__{tag}.wav")


def write_fixtures(sr: int = 22050) -> list[str]:
    """Render every lick × backing to tests/fixtures/. Returns the paths."""
    import soundfile as sf

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    paths = []
    for lick in ALL_LICKS:
        for backing in BACKINGS:
            path = fixture_path(lick, backing)
            sf.write(path, render_lick(lick, sr=sr, backing=backing), sr,
                     subtype="PCM_16")
            paths.append(path)
    return paths


# --------------------------------------------------------------------------
# Self-checks: the ground truth has to be true before it can measure anything
# --------------------------------------------------------------------------


def check_swing_map() -> None:
    """Every legal grid position maps where the docstring says it does."""
    expected = {0.0: 0.0, 0.25: _T1, _T1: _T1, 0.5: _T2, _T2: _T2, 0.75: 5 / 6}
    for frac, want in expected.items():
        got = _swing_frac(frac)
        assert abs(got - want) < _EPS, f"swing({frac}) = {got}, expected {want}"
    # ...and an off-grid position is refused rather than silently rounded.
    for bad in (0.4, 0.1, 0.6):
        try:
            _swing_frac(bad)
        except ValueError:
            pass
        else:  # pragma: no cover - only reached if the guard regresses
            raise AssertionError(f"_swing_frac accepted off-grid position {bad}")
    # Straight licks pass through untouched; swung offbeats move to 2/3.
    assert _swing_pos(3.5, False) == 3.5
    assert abs(_swing_pos(3.5, True) - (3 + _T2)) < _EPS
    assert abs(_swing_pos(6 + _T1, True) - (6 + _T1)) < _EPS


def check_lick(lick: Lick) -> None:
    """Notated and performed agree, the line is monophonic, ticks are exact."""
    assert len(lick.notes) == len(lick.notated), (
        f"{lick.name}: {len(lick.notes)} performed vs {len(lick.notated)} notated"
    )
    for i, ((p_on, p_dur, p_pitch, vel), (n_on, n_dur, n_pitch)) in enumerate(
        zip(lick.notes, lick.notated)
    ):
        assert p_pitch == n_pitch, f"{lick.name}[{i}]: pitch {p_pitch} vs {n_pitch}"
        assert 1 <= vel <= 127, f"{lick.name}[{i}]: velocity {vel} out of range"
        assert p_dur > 0 and n_dur > 0, f"{lick.name}[{i}]: non-positive duration"
        want = _swing_pos(n_on, lick.swing)
        assert abs(p_on - want) < 1e-5, (
            f"{lick.name}[{i}]: performed onset {p_on} is not the swing image "
            f"of notated {n_on} (expected {want})"
        )

    # Notes sharing an onset are a deliberate double-stop, so the non-overlap
    # rule binds between onset GROUPS rather than between raw notes. Tolerance
    # is 1e-6 beats — well under a microsecond at these tempi, so it cannot
    # hide a real overlap (the smallest deliberate gap here is 0.075 beats) but
    # does absorb float noise in the performed onsets, which are rounded to 6
    # decimals for a readable repr.
    expect_voices = 2 if lick.doubling else 1
    for label, seq in (("performed", lick.notes), ("notated", lick.notated)):
        groups: list[list[float]] = []  # [onset, max_end, count]
        for onset, dur, *_ in seq:
            if groups and abs(onset - groups[-1][0]) < 1e-9:
                groups[-1][1] = max(groups[-1][1], onset + dur)
                groups[-1][2] += 1
            else:
                groups.append([onset, onset + dur, 1])
        for onset, _, count in groups:
            assert count == expect_voices, (
                f"{lick.name}: {label} onset {onset} carries {count} notes, "
                f"expected {expect_voices} (doubling={lick.doubling})"
            )
        for i in range(len(groups) - 1):
            assert groups[i][1] <= groups[i + 1][0] + 1e-6, (
                f"{lick.name}: {label} group at {groups[i][0]} runs to "
                f"{groups[i][1]} past the next onset at {groups[i + 1][0]}"
            )
            assert groups[i + 1][0] > groups[i][0], (
                f"{lick.name}: {label} onsets not increasing at {groups[i][0]}"
            )

    for onset, dur, _ in lick.notated:  # raises if off the tick lattice
        _to_ticks(onset)
        _to_ticks(dur)

    if lick.chords:
        span = lick.duration_beats() / lick.beats_per_bar
        assert span <= len(lick.chords) + 1e-9, (
            f"{lick.name}: line spans {span:.2f} bars but only "
            f"{len(lick.chords)} chords are declared"
        )
        for symbol in lick.chords:
            assert symbol in _PAD_VOICINGS, f"{lick.name}: no voicing for {symbol}"


def measure_pitch_hz(y: np.ndarray, sr: int, near_hz: float | None = None) -> float:
    """FFT peak with parabolic interpolation — sub-bin, good to a fraction of a cent.

    ``near_hz`` restricts the search to +-3 semitones around an expected
    frequency. That is needed to measure a *fundamental*: on a low plucked note
    the loudest bin is routinely a harmonic, and a bare argmax then reports a
    pitch error of thousands of cents that is really a harmonic index error.
    The window is 60x wider than the tolerance it is used to check, so it
    cannot conceal a real mistuning.
    """
    mag = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    if near_hz is None:
        k = int(np.argmax(mag))
    else:
        bin_hz = sr / len(y)
        lo = max(1, int(near_hz * 2 ** (-3 / 12) / bin_hz))
        hi = min(len(mag) - 1, int(near_hz * 2 ** (3 / 12) / bin_hz) + 1)
        k = lo + int(np.argmax(mag[lo:hi])) if hi > lo else lo
    if 0 < k < len(mag) - 1:
        a, b, c = (math.log(max(mag[k + d], 1e-20)) for d in (-1, 0, 1))
        denom = a - 2 * b + c
        if abs(denom) > 1e-12:
            k = k + 0.5 * (a - c) / denom
    return k * sr / len(y)


def check_render_pitch(sr: int = 22050, tolerance_cents: float = 5.0) -> list[tuple]:
    """Render sustained tones across the licks' range and measure the error.

    A renderer whose pitch is off is a ground truth that lies, and mir_eval's
    50-cent pitch window would hide a couple of dozen cents of it.
    """
    rows = []
    for midi in (40, 45, 52, 55, 60, 64, 69, 74, 81):
        want = _midi_hz(midi)
        y = _ks_pluck(want, int(0.7 * sr), sr, seed=17)
        got = measure_pitch_hz(y[int(0.05 * sr):], sr, near_hz=want)
        cents = 1200.0 * math.log2(got / want)
        rows.append((midi, want, got, cents))
        assert abs(cents) < tolerance_cents, (
            f"MIDI {midi}: rendered {got:.3f} Hz vs {want:.3f} Hz "
            f"({cents:+.2f} cents), over the {tolerance_cents} cent budget"
        )
    return rows


def _self_check() -> None:
    check_swing_map()
    for lick in ALL_LICKS:
        check_lick(lick)


if __name__ == "__main__":
    _self_check()
    print("swing map + lick consistency: OK\n")

    print("render pitch accuracy (Karplus-Strong loop, FFT peak):")
    print(f"  {'midi':>4}  {'target Hz':>10}  {'rendered Hz':>11}  {'error':>9}")
    for midi, want, got, cents in check_render_pitch():
        print(f"  {midi:>4}  {want:>10.3f}  {got:>11.3f}  {cents:>+7.2f} c")

    print("\nlicks:")
    for lick in ALL_LICKS:
        pitches = [p for _, _, p, _ in lick.notes]
        print(
            f"  {lick.name:<9} {lick.bpm:>5.0f} bpm  {lick.bars} bars  "
            f"{'swung' if lick.swing else 'straight':<8} "
            f"{len(lick.notes):>3} notes  MIDI {min(pitches)}-{max(pitches)}  "
            f"{lick.duration_seconds():>5.2f}s"
        )

    print("\nfixtures:")
    for path in write_fixtures():
        print(f"  {path}")
