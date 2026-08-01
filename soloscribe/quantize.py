"""Beat tracking and rhythm quantization.

The jazz-critical decision lives here: swung eighths land near the 2/3 point
of the beat, and quantizing them to a straight grid produces garbage. The
convention real transcribers use — and what this module does — is to NOTATE
straight eighths and mark the piece "swing feel" (GP5 TripletFeel.eighth),
reserving actual triplet notation for genuine triplet figures (which betray
themselves by an onset near the 1/3 point).

Each beat independently chooses its subdivision template; a per-beat penalty
keeps the notation as simple as the evidence allows.
"""
from __future__ import annotations

import bisect

import numpy as np

from .model import BeatGrid, NoteEvent, QNote, Score, TICKS_PER_QUARTER


def build_beat_grid(
    y: np.ndarray,
    sr: int,
    *,
    bpm: float | None = None,
    downbeat: float | None = None,
    beats_per_bar: int = 4,
    cover_until: float | None = None,
    onsets: list[float] | None = None,
) -> BeatGrid:
    """Track beats in the audio; extrapolate so the grid covers the whole clip."""
    import librosa

    duration = len(y) / sr

    # A caller who supplies BOTH tempo and downbeat has asserted ground
    # truth — honor it with an exact metronomic grid. Tracking would only
    # re-introduce phase errors (a syncopated line pulls the tracker toward
    # the offbeats; comb-energy correction is circular for the same reason).
    # A user-supplied tempo WITHOUT a downbeat: trust their period, measure
    # the phase from the played notes themselves (circular median of onset
    # positions modulo the beat). The tracker can phase-lock off the grid
    # even with a strong tempo prior — measured on the blues fixture at
    # bpm=92/no-downbeat: 27 of 27 score onsets landed half a cell off the
    # audio and coverage read 0.407 (verdict low) on a correct transcription.
    if (bpm is not None and bpm > 0 and downbeat is None
            and onsets is not None and len(onsets) >= 4):
        period = 60.0 / float(bpm)
        angles = np.array([2 * np.pi * (t % period) / period for t in onsets])
        phase = (np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
                 / (2 * np.pi)) * period % period
        downbeat = float(min(onsets, key=lambda t: abs((t - phase) % period)
                             if abs((t - phase) % period) < period / 2
                             else period - (t - phase) % period))
        # fall through to the metronomic path below with the measured phase
        downbeat = phase + round((downbeat - phase) / period) * period

    if bpm is not None and bpm > 0 and downbeat is not None:
        period = 60.0 / float(bpm)
        horizon = max(duration, cover_until or 0.0)
        t0 = float(downbeat)
        while t0 - period >= -period * 0.5:
            t0 -= period
        times = []
        t = t0
        while t < horizon + period:
            times.append(t)
            t += period
        first = int(np.argmin([abs(t - downbeat) for t in times]))
        return BeatGrid(
            beat_times=times,
            beats_per_bar=beats_per_bar,
            first_downbeat=first,
            bpm_nominal=float(bpm),
        )

    if bpm is not None and bpm > 0:
        tempo_prior = float(bpm)
        tightness = 400  # trust the user's tempo; track drift only
    else:
        tempo_prior = 120.0
        tightness = 100
    tempo, frames = librosa.beat.beat_track(
        y=y, sr=sr, start_bpm=tempo_prior, tightness=tightness, trim=False
    )
    times = librosa.frames_to_time(frames, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    if len(times) < 4:
        # Degenerate tracking (very short or arrhythmic clip): fall back to a
        # metronomic grid at the prior tempo.
        period = 60.0 / (bpm or tempo or 120.0)
        times = np.arange(0.0, duration + period, period)
    times = list(map(float, times))

    period = float(np.median(np.diff(times))) if len(times) > 1 else 60.0 / tempo_prior
    # If the user gave a BPM and the tracker landed on a related-but-wrong
    # metrical level (half/double), rescale toward the user's intent.
    if bpm and period > 0:
        ratio = (60.0 / bpm) / period
        if 1.7 < ratio < 2.3:  # tracker found half-time → insert midpoints
            mids = [(a + b) / 2 for a, b in zip(times, times[1:])]
            times = sorted(times + mids)
            period /= 2
        elif 0.42 < ratio < 0.6:  # tracker found double-time → take every other
            times = times[::2]
            period *= 2

    # Constant phase correction: the tracker can lock onto offbeat energy
    # (syncopated 16th lines drag it ~0.2 beats early). Keep its drift, but
    # slide the whole grid to the phase that maximizes onset-envelope energy
    # at the beat times. Found by the ground-truth E2E harness (funk lick:
    # 14/16 beats off by > an eighth before this correction).
    if len(times) > 3:
        env = librosa.onset.onset_strength(y=y, sr=sr)
        env_t = librosa.times_like(env, sr=sr)
        arr = np.asarray(times)

        def _energy(delta: float) -> float:
            shifted = arr + delta
            ok = (shifted >= env_t[0]) & (shifted <= env_t[-1])
            if not ok.any():
                return -1.0
            return float(np.mean(np.interp(shifted[ok], env_t, env)))

        deltas = np.linspace(-0.35 * period, 0.35 * period, 29)
        best = max(deltas, key=_energy)
        if _energy(best) > _energy(0.0) * 1.02:  # only move on clear evidence
            times = [t + float(best) for t in arr]

    # A user-supplied downbeat is ground truth: shift the grid so the nearest
    # tracked beat lands exactly on it.
    if downbeat is not None and times:
        nearest = min(times, key=lambda t: abs(t - downbeat))
        if abs(nearest - downbeat) <= period * 0.5:
            times = [t + (downbeat - nearest) for t in times]

    # Extrapolate to cover [0, duration] (and any caller-requested horizon).
    horizon = max(duration, cover_until or 0.0)
    while times[0] - period > -period * 0.5:
        times.insert(0, times[0] - period)
        if times[0] <= 0:
            break
    while times[-1] < horizon:
        times.append(times[-1] + period)

    first_downbeat = 0
    if downbeat is not None:
        first_downbeat = int(np.argmin([abs(t - downbeat) for t in times]))

    bpm_nominal = 60.0 / period if period > 0 else (bpm or 120.0)
    return BeatGrid(
        beat_times=times,
        beats_per_bar=beats_per_bar,
        first_downbeat=first_downbeat,
        bpm_nominal=float(bpm or bpm_nominal),
    )


def _beat_position(grid_times: list[float], t: float) -> tuple[int, float]:
    """(beat_index, fractional position within that beat) for time t."""
    i = bisect.bisect_right(grid_times, t) - 1
    i = max(0, min(i, len(grid_times) - 2))
    span = grid_times[i + 1] - grid_times[i]
    frac = (t - grid_times[i]) / span if span > 0 else 0.0
    return i, min(max(frac, 0.0), 1.499)  # allow slight spill past the beat


# Subdivision templates: (name, true_positions, notated_positions, penalty)
_TEMPLATES: list[tuple[str, tuple[float, ...], tuple[float, ...], float]] = [
    ("straight8", (0.0, 0.5), (0.0, 0.5), 0.00),
    ("sixteenth", (0.0, 0.25, 0.5, 0.75), (0.0, 0.25, 0.5, 0.75), 0.030),
    ("triplet", (0.0, 1 / 3, 2 / 3), (0.0, 1 / 3, 2 / 3), 0.035),
    ("swing8", (0.0, 2 / 3), (0.0, 0.5), 0.015),
]


def detect_swing(events: list[NoteEvent], grid: BeatGrid) -> tuple[bool, float]:
    """Do offbeat eighths sit nearer 2/3 than 1/2? Returns (swing, median_pos).

    Beats carrying sixteenth-position evidence (onsets near 0.25/0.75) are
    excluded — a straight 16th line's 0.75 onsets would otherwise drag the
    median into the swing band and stamp a funk line "swing feel". The band
    is also closed below 0.75, where the fourth sixteenth lives.
    """
    by_beat: dict[int, list[float]] = {}
    for ev in events:
        bi, frac = _beat_position(grid.beat_times, ev.start)
        by_beat.setdefault(bi, []).append(frac)

    offbeats = []
    for fracs in by_beat.values():
        sixteenthish = any(
            0.21 <= f <= 0.29 or 0.71 <= f <= 0.79 for f in fracs
        )
        if sixteenthish:
            continue
        offbeats.extend(f for f in fracs if 0.40 <= f <= 0.80)
    if len(offbeats) < 4:
        return False, 0.5
    med = float(np.median(offbeats))
    return 0.58 <= med < 0.75, med


def quantize(
    events: list[NoteEvent],
    grid: BeatGrid,
    *,
    swing: str = "auto",
    key: str = "C",
    chords: list[str | None] | None = None,
    title: str = "",
    min_cell: int = TICKS_PER_QUARTER // 4,
) -> Score:
    """Snap transcribed events onto the notation grid → Score (unfretted)."""
    tpq = TICKS_PER_QUARTER
    if swing == "auto":
        swing_on, _ = detect_swing(events, grid)
    else:
        swing_on = swing == "on"

    times = grid.beat_times

    # Choose one subdivision template per beat from the onsets it contains.
    by_beat: dict[int, list[float]] = {}
    for ev in events:
        bi, frac = _beat_position(times, ev.start)
        by_beat.setdefault(bi, []).append(frac)

    beat_template: dict[int, tuple] = {}
    for bi, fracs in by_beat.items():
        best, best_cost = None, float("inf")
        for name, true_pos, notated, penalty in _TEMPLATES:
            if name == "swing8" and not swing_on:
                continue
            if name == "triplet" and swing_on:
                # In swing, only genuine triplets (evidence: an onset near 1/3)
                # earn triplet notation; otherwise swung pairs win.
                if not any(0.20 <= f <= 0.46 for f in fracs):
                    continue
            # Absolute (not squared) misfit: squared distances between grid
            # points are tiny and would let complexity penalties overpower
            # genuine evidence (a real triplet would flatten to eighths).
            cost = penalty + sum(
                min(abs(f - p) for p in true_pos + (1.0,)) for f in fracs
            ) / len(fracs)
            if cost < best_cost:
                best, best_cost = (name, true_pos, notated), cost
        beat_template[bi] = best

    def snap(t: float, is_offset: bool = False) -> int:
        """Time → global notated tick (relative to first downbeat's beat)."""
        bi, frac = _beat_position(times, t)
        tmpl = beat_template.get(bi)
        if tmpl is None:
            name, true_pos, notated = "straight8", (0.0, 0.5), (0.0, 0.5)
        else:
            name, true_pos, notated = tmpl
        candidates = list(zip(true_pos, notated)) + [(1.0, 1.0)]
        j = int(np.argmin([abs(frac - tp) for tp, _ in candidates]))
        notated_frac = candidates[j][1]
        return round(((bi - grid.first_downbeat) + notated_frac) * tpq)

    qnotes: list[QNote] = []
    for ev in sorted(events, key=lambda e: (e.start, e.pitch)):
        onset = snap(ev.start)
        offset = snap(ev.end, is_offset=True)
        if offset <= onset:
            offset = onset + min_cell
        # Keep triplet-grid durations on the triplet lattice.
        dur = offset - onset
        qnotes.append(
            QNote(
                onset=onset,
                duration=dur,
                pitch=ev.pitch,
                velocity=ev.velocity,
                vibrato=ev.vibrato,
                bend=list(ev.bend),
                confidence=ev.confidence,
            )
        )

    # Shift any pre-downbeat content into a pickup bar rather than negative
    # ticks. The tempo map below must shift identically or audit resynthesis
    # lands notes at the wrong clock times.
    shift_ticks = 0
    if qnotes:
        min_onset = min(q.onset for q in qnotes)
        if min_onset < 0:
            tpb = tpq * grid.beats_per_bar
            shift_ticks = ((-min_onset + tpb - 1) // tpb) * tpb
            for q in qnotes:
                q.onset += shift_ticks

    # Monophonic overlap trim (simultaneous onsets = deliberate double-stops).
    qnotes.sort(key=lambda q: (q.onset, q.pitch))
    for i, q in enumerate(qnotes):
        for later in qnotes[i + 1:]:
            if later.onset == q.onset:
                continue
            if later.onset >= q.offset:
                break
            q.duration = max(min_cell, later.onset - q.onset)
            break

    period = float(np.median(np.diff(times))) if len(times) > 1 else 60.0 / grid.bpm_nominal
    tempo_bpm = 60.0 / period if period > 0 else grid.bpm_nominal

    # Tempo map: one anchor per tracked beat → audit resynthesis follows the
    # band's actual drift, not the nominal tempo.
    tempo_map = [
        (round((i - grid.first_downbeat) * tpq) + shift_ticks, t)
        for i, t in enumerate(times)
    ]

    return Score(
        qnotes=qnotes,
        beats_per_bar=grid.beats_per_bar,
        tempo_bpm=tempo_bpm,
        triplet_feel=swing_on,
        key=key,
        tempo_map=tempo_map,
        chords=list(chords) if chords else [],
        title=title,
    )


def parse_chords(spec: str | None) -> list[str | None]:
    """'Gm7|C7|Fmaj7' or newline-separated → one entry per bar."""
    if not spec:
        return []
    parts = [p.strip() for p in spec.replace("\n", "|").split("|")]
    return [p if p else None for p in parts]
