"""Audio → NoteEvents, via basic-pitch with a monophonic-biased refinement pass.

basic-pitch is a polyphonic transcriber, which is the wrong prior for a jazz
guitar solo: it hears the second and fourth harmonics of a fat neck-pickup tone
as extra notes, and on plucks with a weak fundamental it will sometimes place
the whole note an octave high. "solo" mode therefore runs a pyin f0 contour
alongside it and uses the contour — which is monophonic by construction, so it
locks onto one voice and stays there — to relabel octave errors, drop harmonic
ghosts, and mark vibrato. "poly" mode returns the raw basic-pitch reading.

Two measured limits of that refinement, both worth keeping in view before
anyone "simplifies" the guards around it:

  * pyin does not merely pick one voice on a harmonic dyad, it locks onto the
    voices' common subharmonic — a 52+59 fifth reads as MIDI 40, an octave and
    a fifth below either note. An unguarded octave rule therefore moves a
    *correct* note down an octave on one of the most common shapes in the
    idiom. Octave relabelling is consequently refused on any span that has a
    concurrent note and nothing at the target pitch (`_refine_solo`).
  * OCTAVE DOUBLE-STOPS ARE NOT PRESERVED. GHOST_AMP_RATIO=0.85 buys ghost
    rejection by giving them up entirely, and that is a priced trade rather
    than a free win. Measured over a 22-doubling Wes-style octave lick
    (tests/fixtures/octaves_g), upper-voice recall runs 0.455 at ratio 0.60,
    0.227 at 0.70, 0.136 at 0.75, 0.045 at 0.80 and 0.000 at 0.85 — strictly
    monotonic, no plateau — against funk_e precision of 0.826 → 0.974 across
    the same span. Melody (lower voice) recall is 1.000 at every ratio, so the
    loss is confined to the doubling: the line survives, its octave texture
    does not. An earlier version of this note claimed the capability was
    already dead based on ONE synthetic pluck pair that happened to land at a
    0.52 ratio; across 22 real doublings the distribution straddles 0.60, so
    that generalization from n=1 was simply wrong. To transcribe octave
    repertoire, LOWER GHOST_AMP_RATIO (0.60 restores 0.455 upper-voice recall;
    the constant is read at call time, so assigning to it is enough) — lower
    keeps more concurrent octave notes as double-stops, higher relabels more of
    them as harmonics. mode="poly" preserves every doubling but disables the
    whole refinement, ghost rejection included, so it is the blunter option.
    The default is a judgment call, not a measured optimum: ghosts contaminate
    the line the player reads, doublings only thin a texture while melody
    recall stays 1.000, and strong harmonics are far commoner in this
    repertoire than octave passages. There is no knee in the frontier where
    both are acceptable, which is the real finding — amplitude ratio is one
    knob asked to separate two populations that overlap on it.
    Duration is NOT the discriminator to reach for: measured at ratio 0.60,
    real doublings ran 70-267 ms (median 104) against spurious ghosts at
    81-163 ms (median 122), overlapping heavily with the ghost median the
    HIGHER of the two. The useful comparison is ghost-vs-legitimate-double-
    stop, not ghost-vs-host, and on that axis duration carries no signal.

Verified against the installed basic-pitch 0.4.0 source:

  * `predict(audio_path, model_or_model_path, onset_threshold, frame_threshold,
    minimum_note_length, minimum_frequency, maximum_frequency,
    multiple_pitch_bends, melodia_trick, debug_file, midi_tempo)`
                                              — inference.py:414-425
    NOTE: there is no `include_pitch_bends` parameter on `predict`. It exists
    only on `model_output_to_notes` (note_creation.py:54) where it defaults to
    True, so bends are always computed and passed through; requesting it on
    `predict` would be a TypeError.
  * `predict` returns `(model_output, midi_data, note_events)` with note_events
    typed `List[Tuple[float, float, int, float, Optional[List[int]]]]`
                                              — inference.py:426-430
    unpacked in-package as
    `for start_time, end_time, pitch, amplitude, pitch_bends in note_events`
                                              — inference.py:483
    Confirmed at runtime on a synthetic 3-pluck clip: element types are
    (float64 start_s, float64 end_s, int64 midi_pitch, float32 amplitude,
     list[int64] bends).
  * BEND UNITS: `bends` are per-frame integer offsets "in units of 1/3
    semitones" — note_creation.py:209-211, the divisor being
    `CONTOURS_BINS_PER_SEMITONE = 3` (constants.py:24). Corroborated by the
    package's own MIDI writer, which converts with
    `np.round(np.array(pitch_bend) * 4096 / CONTOURS_BINS_PER_SEMITONE)` where
    4096 ticks is one semitone — note_creation.py:254. So
    semitones = bins / 3. One value per model frame, laid on an evenly spaced
    grid across the note (note_creation.py:253), which is exactly the
    normalized-0..1 convention NoteEvent.bend wants.
  * `amplitude` is `np.mean(frames[start:end, freq_idx])`, a frame-activation
    mean in [0, 1] — note_creation.py:426. The package's own velocity mapping
    is `round(127 * amplitude)` (note_creation.py:245), which sends a typical
    strong guitar note (observed amplitude ≈ 0.75-0.85) to 95-108 but leaves
    quiet notes implausibly low; `_amp_to_velocity` below stretches the useful
    part of that range instead.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import librosa
import numpy as np

from .model import NoteEvent, merge_adjacent_events

# --- basic-pitch decoding -------------------------------------------------
ONSET_THRESHOLD = 0.5
FRAME_THRESHOLD = 0.3
MIN_NOTE_LENGTH_MS = 58.0        # ~1/32 at 130bpm; shorter than a bebop 16th
# basic-pitch's `constrain_frequency` zeroes whole semitone bins either side of
# these (note_creation.py:321-328), so the pair below decodes to exactly MIDI
# 38-89: the bottom of drop-D at one end, three semitones above the 24th fret
# of the high E (MIDI 88) at the other.
GUITAR_MIN_HZ = 72.0
GUITAR_MAX_HZ = 1500.0

# basic-pitch amplitude (mean frame activation) → MIDI velocity. Affine through
# (AMP_LO, VEL_LO) and (AMP_HI, VEL_HI), then clipped: the observed amplitude
# band on real playing is roughly 0.30-0.85, so strong notes land 88-110 and
# quiet ones stay legible instead of collapsing toward silence.
AMP_LO, VEL_LO = 0.25, 45.0
AMP_HI, VEL_HI = 0.85, 110.0

# Bend contour: emit only if the contour actually moves. A constant one-bin
# offset is the argmax sitting off-centre, not a bend.
BEND_MIN_RANGE = 0.5             # semitones, peak-to-peak
BEND_MAX_POINTS = 32             # decimate long notes to keep the contour light

# --- pyin contour (solo mode) --------------------------------------------
PYIN_SR = 22050                  # matches basic-pitch's own rate (constants.py:36)
PYIN_HOP = 256
PYIN_FRAME = 2048
PYIN_FMIN = 70.0
PYIN_FMAX = 1600.0
VOICED_PROB_MIN = 0.5
REGION_INSET = 0.02              # trim attack/release before reading the contour
MIN_REGION_FRAMES = 3
MIN_VOICED_FRAC = 0.5            # below this the contour has no opinion

AGREE_SEMITONES = 0.5            # |contour - pitch| within this = corroborated
CONTRA_SEMITONES = 1.5           # beyond this = contradicted
OCTAVE_TOL = 0.7                 # tolerance around an exact 12 or 24
MAX_OCTAVE_ERROR = 2             # relabel up to two octaves (4th harmonic)
OVERLAP_FRAC = 0.5               # of the shorter note, to count as concurrent
# Quieter than this × a concurrent note = ghost rather than a second voice.
# Swept 0.60→0.90 over six ground-truth fixtures (3 licks × clean/pad): recall
# was flat at every step on every lick, precision rose and plateaued at 0.85
# (funk_e clean 0.826→0.974, funk_e pad 0.755→0.881, blues_a pad 0.833→0.926).
# Recall cannot fall here by construction — raising it only converts "keep as
# double-stop" into "relabel onto the host pitch", and the relabelled note is
# then absorbed by `_dedup_overlaps` into a note that already exists, so no
# pitch is ever removed outright. See the octave double-stop caveat in the
# module docstring for what the sweep could NOT see.
GHOST_AMP_RATIO = 0.85

# --- vibrato --------------------------------------------------------------
VIB_MIN_DUR = 0.25               # need ~1.5 cycles at the low end of the band
VIB_MIN_FRAMES = 12
VIB_MIN_VOICED_FRAC = 0.8        # the FFT needs a near-complete series
VIB_RATE_LO, VIB_RATE_HI = 4.0, 8.0          # Hz
VIB_DEV_LO, VIB_DEV_HI = 0.15, 0.8           # semitones, peak deviation
VIB_SEARCH_LO, VIB_SEARCH_HI = 2.0, 15.0     # Hz, where the peak must win
VIB_PEAK_RATIO = 2.0             # peak must beat the in-search-band median by this


def _amp_to_velocity(amp: float) -> int:
    """basic-pitch amplitude in [0, 1] → MIDI velocity in [1, 127]."""
    v = VEL_LO + (amp - AMP_LO) * (VEL_HI - VEL_LO) / (AMP_HI - AMP_LO)
    return int(np.clip(round(v), 1, 127))


def _velocity_to_amp(vel: int) -> float:
    """Exact affine inverse of `_amp_to_velocity`, lossy only by its rounding.

    Amplitude is not a NoteEvent field, but the ghost/double-stop rules are
    stated in amplitude, so relative loudness is read back through here rather
    than compared in the (offset) velocity domain.
    """
    return AMP_LO + (vel - VEL_LO) * (AMP_HI - AMP_LO) / (VEL_HI - VEL_LO)


def _amp_to_confidence(amp: float) -> float:
    """Prior belief a note is real, from its activation alone."""
    return float(np.clip(0.30 + 0.75 * amp, 0.05, 0.99))


def _bend_curve(bins: list[int] | None) -> list[tuple[float, float]]:
    """Per-frame 1/3-semitone bins → (normalized time, semitones) pairs.

    Returns [] when the contour does not move by at least BEND_MIN_RANGE — a
    flat offset is the model's pitch-bin quantization, not a played bend.
    """
    if not bins or len(bins) < 2:
        return []
    semis = np.asarray(bins, dtype=float) / 3.0     # note_creation.py:209-211
    if float(semis.max() - semis.min()) < BEND_MIN_RANGE:
        return []
    n = len(semis)
    if n > BEND_MAX_POINTS:
        idx = np.unique(np.linspace(0, n - 1, BEND_MAX_POINTS).round().astype(int))
    else:
        idx = np.arange(n)
    times = idx / (n - 1)
    return [(float(t), float(semis[i])) for t, i in zip(times, idx)]


def _events_from_predictions(
    note_events, min_pitch: int, max_pitch: int
) -> list[NoteEvent]:
    """basic-pitch tuples → NoteEvents, filtered to the requested pitch range."""
    out: list[NoteEvent] = []
    for start, end, pitch, amp, bends in note_events:
        pitch = int(pitch)
        if pitch < min_pitch or pitch > max_pitch or end <= start:
            continue
        amp = float(amp)
        out.append(
            NoteEvent(
                start=float(start),
                end=float(end),
                pitch=pitch,
                velocity=_amp_to_velocity(amp),
                confidence=_amp_to_confidence(amp),
                bend=_bend_curve(bends),
            )
        )
    return out


@dataclass
class _Contour:
    """A pyin f0 reading, in MIDI, with unvoiced frames left as NaN."""

    times: np.ndarray
    midi: np.ndarray
    confident: np.ndarray        # bool: voiced, probable, and finite

    def region(self, start: float, end: float) -> slice:
        lo = int(np.searchsorted(self.times, start, side="left"))
        hi = int(np.searchsorted(self.times, end, side="right"))
        return slice(lo, hi)


def _pyin_contour(audio_path: str) -> _Contour:
    """Monophonic f0 track over the whole clip. `librosa.pyin` fills unvoiced
    frames with NaN (fill_na default), so NaN is the unvoiced signal here."""
    y, sr = librosa.load(audio_path, sr=PYIN_SR, mono=True)
    f0, _voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=PYIN_FMIN,
        fmax=PYIN_FMAX,
        sr=sr,
        frame_length=PYIN_FRAME,
        hop_length=PYIN_HOP,
    )
    midi = np.full(f0.shape, np.nan, dtype=float)
    finite = np.isfinite(f0) & (f0 > 0)
    midi[finite] = librosa.hz_to_midi(f0[finite])
    confident = finite & (voiced_prob > VOICED_PROB_MIN)
    times = librosa.times_like(f0, sr=sr, hop_length=PYIN_HOP)
    return _Contour(times=times, midi=midi, confident=confident)


def _inset_region(ev: NoteEvent) -> tuple[float, float]:
    """Note span with attack/release trimmed, for notes long enough to spare it."""
    inset = REGION_INSET if ev.duration > 6 * REGION_INSET else 0.0
    return ev.start + inset, ev.end - inset


def _contour_median(contour: _Contour, ev: NoteEvent) -> float | None:
    """Median contour pitch over a note, or None if the contour is not confident."""
    lo, hi = _inset_region(ev)
    sl = contour.region(lo, hi)
    n = sl.stop - sl.start
    if n < MIN_REGION_FRAMES:
        return None
    conf = contour.confident[sl]
    if conf.sum() < MIN_REGION_FRAMES or conf.mean() < MIN_VOICED_FRAC:
        return None
    return float(np.median(contour.midi[sl][conf]))


def _octave_shift(delta: float) -> int:
    """Semitone offset to the contour's octave, or 0 if `delta` is not octave-ish."""
    for k in range(1, MAX_OCTAVE_ERROR + 1):
        if abs(abs(delta) - 12 * k) <= OCTAVE_TOL:
            return 12 * k * (1 if delta > 0 else -1)
    return 0


def _overlap_frac(a: NoteEvent, b: NoteEvent) -> float:
    """Shared duration as a fraction of the shorter note."""
    ov = min(a.end, b.end) - max(a.start, b.start)
    if ov <= 0:
        return 0.0
    shorter = min(a.duration, b.duration)
    return ov / shorter if shorter > 0 else 0.0


def _detect_vibrato(contour: _Contour, ev: NoteEvent) -> bool:
    """True when the note's f0 oscillates in the 4-8 Hz vibrato band.

    Linearly detrended first so a bend or slide — a monotone ramp, however
    large — cannot present as vibrato.
    """
    if ev.duration < VIB_MIN_DUR:
        return False
    lo, hi = _inset_region(ev)
    sl = contour.region(lo, hi)
    n = sl.stop - sl.start
    if n < VIB_MIN_FRAMES:
        return False
    conf = contour.confident[sl]
    if conf.mean() < VIB_MIN_VOICED_FRAC:
        return False

    # Uniformly sampled series: interpolate across the short unvoiced gaps.
    series = contour.midi[sl].copy()
    idx = np.arange(n)
    series = np.interp(idx, idx[conf], series[conf])
    series = series - np.polyval(np.polyfit(idx, series, 1), idx)

    peak_dev = float(np.max(np.abs(series)))
    if not (VIB_DEV_LO <= peak_dev <= VIB_DEV_HI):
        return False

    mag = np.abs(np.fft.rfft(series * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=PYIN_HOP / PYIN_SR)
    band = (freqs >= VIB_SEARCH_LO) & (freqs <= VIB_SEARCH_HI)
    if band.sum() < 3:
        return False
    peak = int(np.argmax(mag[band]))
    peak_hz = float(freqs[band][peak])
    peak_mag = float(mag[band][peak])
    med = float(np.median(mag[band]))
    if med > 0 and peak_mag < VIB_PEAK_RATIO * med:
        return False
    return VIB_RATE_LO <= peak_hz <= VIB_RATE_HI


def _dedup_overlaps(events: list[NoteEvent]) -> list[NoteEvent]:
    """Fold same-pitch events that overlap in time into one.

    `merge_adjacent_events` only joins events separated by a gap, so octave
    relabelling — which can drop a ghost squarely on top of the note it was a
    harmonic of — needs this second, overlap-aware pass.
    """
    out: list[NoteEvent] = []
    for ev in sorted(events, key=lambda e: (e.pitch, e.start)):
        if out and out[-1].pitch == ev.pitch and ev.start < out[-1].end:
            prev = out[-1]
            prev.end = max(prev.end, ev.end)
            if ev.velocity > prev.velocity:
                prev.velocity = ev.velocity
                prev.bend = ev.bend
            prev.confidence = max(prev.confidence, ev.confidence)
            prev.vibrato = prev.vibrato or ev.vibrato
        else:
            out.append(ev)
    return sorted(out, key=lambda e: (e.start, e.pitch))


def _refine_solo(
    events: list[NoteEvent], audio_path: str, stats: dict | None = None
) -> list[NoteEvent]:
    """Use a monophonic f0 contour to fix octaves, drop ghosts, mark vibrato.

    When `stats` is a dict, records (onset, amp_ratio_to_host) of events
    folded onto a host one octave away under "folded_octaves". MEASURED DEAD
    END, kept as instrumentation only: fold statistics do NOT discriminate a
    suppressed Wes-style octave passage from a harmonic-rich single-note line
    (fold-ratio medians 0.65 on the octave fixture vs 0.63 on funk_e, counts
    15 vs 13), so no per-run "octaves suspected" warning is derivable here —
    it would fire on most straight lines. The user-facing answer is the
    unconditional solo-mode caveat in pipeline.py; do not rebuild the
    warning without a discriminator measured to actually separate.
    """
    if not events:
        return events
    contour = _pyin_contour(audio_path)
    medians = [_contour_median(contour, ev) for ev in events]
    amps = [_velocity_to_amp(ev.velocity) for ev in events]
    # Relabelling mutates pitch in place, so host lookups read a snapshot taken
    # before the loop: "is something else sounding at the target pitch" is a
    # question about what basic-pitch reported, and answering it against
    # already-relabelled neighbours would make the pass order-dependent.
    reported = [ev.pitch for ev in events]
    concurrent = [
        [j for j, o in enumerate(events) if j != i and _overlap_frac(ev, o) >= OVERLAP_FRAC]
        for i, ev in enumerate(events)
    ]

    def masked_by_louder(i: int) -> bool:
        return any(amps[i] < GHOST_AMP_RATIO * amps[j] for j in concurrent[i])

    kept: list[NoteEvent] = []
    for i, ev in enumerate(events):
        med = medians[i]
        if med is None:                       # contour has no opinion — leave it
            kept.append(ev)
            continue
        delta = med - ev.pitch
        shift = _octave_shift(delta)

        if shift:
            # Either a genuine octave error or a harmonic of a note that is
            # already transcribed correctly. The two are distinguished by
            # whether a *comparably loud* note already sits at the target
            # pitch: that is a real octave double-stop and must survive.
            host = next(
                (j for j in concurrent[i] if reported[j] == reported[i] + shift),
                None,
            )
            if host is not None:
                if amps[i] >= GHOST_AMP_RATIO * amps[host]:
                    ev.confidence *= 0.7      # octave double-stop, both real
                else:
                    if stats is not None and abs(shift) == 12:
                        stats.setdefault("folded_octaves", []).append(
                            (ev.start, amps[i] / max(amps[host], 1e-6))
                        )
                    ev.pitch += shift         # harmonic; _dedup_overlaps absorbs it
                    ev.confidence *= 0.85
                kept.append(ev)
                continue
            if not concurrent[i]:
                ev.pitch += shift             # octave error over a monophonic span
                ev.confidence *= 0.85
                kept.append(ev)
                continue
            # Polyphonic span with nothing at the target pitch. pyin is an
            # autocorrelation-family tracker, so on a harmonic dyad it locks
            # onto the common subharmonic rather than either voice — a 52+59
            # fifth reads as MIDI 40, and relabelling on that reading would
            # move a correct note down an octave. Refuse, and let the note
            # take its chances with the masked-ghost test below.
        elif abs(delta) <= AGREE_SEMITONES:
            ev.confidence += (1.0 - ev.confidence) * 0.5
            kept.append(ev)
            continue
        elif abs(delta) <= CONTRA_SEMITONES:
            ev.confidence *= 0.9
            kept.append(ev)
            continue

        # Contradicted by the contour: a ghost only if it is also quiet and
        # buried under a concurrent note. An unmasked contradiction is more
        # likely a second voice the monophonic contour simply isn't following.
        if masked_by_louder(i):
            continue
        ev.confidence *= 0.7
        kept.append(ev)

    kept = _dedup_overlaps(kept)
    for ev in kept:
        ev.vibrato = _detect_vibrato(contour, ev)
        if ev.vibrato:
            # The vibrato flag carries the oscillation; a wobbling bend curve
            # alongside it would be notated twice.
            ev.bend = []
        ev.confidence = float(np.clip(ev.confidence, 0.02, 1.0))
    return kept


# --- re-articulation splitting -------------------------------------------
# Repeated same-pitch fast notes reach basic-pitch's note DECODER as one
# contiguous activation (observed gap 0.000 s), so they emerge fused — the
# residual recall class every earlier fix bounced off. But the model's own
# ONSET head sees every re-attack; the decoder just ignores it while frame
# activation stays high. Measured posteriors at the note's pitch bin (±1):
#   true re-articulations (8 fused sites, sampled-guitar AND KS timbre):
#     0.851–0.973
#   amplitude-dip artifact that must NOT split (94% mid-note dip): 0.703
#   held-note interiors (2 controls, 2 timbres): 0.115–0.162
# 0.80 separates every measured class with margin on both sides. A missed
# split is the status quo ante (bounded-safe); a false split would fabricate
# a repeat, which is why the threshold sits above the dip, not midway.
# Pitch-bin-local, so safe in poly mode: another voice's onset lives in
# another bin.
#
# SECOND, independent gate — audio energy must RISE through the claimed
# re-attack. The posterior alone shipped two audible false splits (attack
# smear 62 ms into a held note; a mid-decay sample artifact), both measured
# at env-rise ratio 0.90 (RMS 5–35 ms after ÷ RMS 5–35 ms before), while
# seven known true re-picks on the same recording measured 1.07–1.22 and a
# re-pick after near-silence measured >>1. Ratio 1.00 is the physical
# boundary: a re-articulation injects energy; decay cannot. Both gates must
# agree before a split happens.
SPLIT_ONSET_POSTERIOR = 0.80
SPLIT_ENV_RISE = 1.00
SPLIT_ENV_PRE = (0.035, 0.005)   # s before candidate: RMS window
SPLIT_ENV_POST = (0.005, 0.035)  # s after candidate
SPLIT_MIN_PIECE = 0.06           # s; also the spacing floor between split points
SPLIT_EDGE_START = 0.05          # s into the note before a split may occur
SPLIT_EDGE_END = 0.04            # s before the note's end
_BP_FRAME = 256 / 22050.0        # constants.py: AUDIO_SAMPLE_RATE 22050, FFT_HOP 256
_BP_MIDI_OFFSET = 21             # onset matrix bin 0 = MIDI 21


def _load_mono(audio_path: str) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    return y.astype(float), int(sr)


def _split_rearticulations(
    events: list[NoteEvent], model_output, y=None, sr: int = PYIN_SR
) -> list[NoteEvent]:
    """Split fused repeated notes at the model's own interior onset peaks."""
    try:
        onset = np.asarray(model_output["onset"])
    except Exception:
        return events

    def energy_rises(t: float) -> bool:
        if y is None:
            return True  # no audio handed in: posterior-only (tested paths pass it)
        pre = y[int((t - SPLIT_ENV_PRE[0]) * sr):int((t - SPLIT_ENV_PRE[1]) * sr)]
        post = y[int((t + SPLIT_ENV_POST[0]) * sr):int((t + SPLIT_ENV_POST[1]) * sr)]
        if len(pre) < 8 or len(post) < 8:
            return False
        rp = float(np.sqrt((pre ** 2).mean()))
        rq = float(np.sqrt((post ** 2).mean()))
        return rq >= SPLIT_ENV_RISE * max(rp, 1e-9)
    if onset.ndim == 3:
        onset = onset[0]
    note_act = None
    try:
        note_act = np.asarray(model_output["note"])
        if note_act.ndim == 3:
            note_act = note_act[0]
    except Exception:
        pass

    out: list[NoteEvent] = []
    for ev in events:
        b = ev.pitch - _BP_MIDI_OFFSET
        if not (0 <= b < onset.shape[1]) or ev.duration < 2 * SPLIT_MIN_PIECE:
            out.append(ev)
            continue
        f0 = int(round((ev.start + SPLIT_EDGE_START) / _BP_FRAME))
        f1 = int(round((ev.end - SPLIT_EDGE_END) / _BP_FRAME))
        if f1 - f0 < 2:
            out.append(ev)
            continue
        lane = onset[f0:f1, max(0, b - 1):b + 2].max(axis=1)
        cuts: list[float] = []
        for i in range(1, len(lane) - 1):
            if (
                lane[i] >= SPLIT_ONSET_POSTERIOR
                and lane[i] >= lane[i - 1]
                and lane[i] >= lane[i + 1]
            ):
                t = (f0 + i) * _BP_FRAME
                if (
                    t - ev.start >= SPLIT_MIN_PIECE
                    and ev.end - t >= SPLIT_MIN_PIECE
                    and (not cuts or t - cuts[-1] >= SPLIT_MIN_PIECE)
                    and energy_rises(t)
                ):
                    cuts.append(t)
        if not cuts:
            out.append(ev)
            continue
        bounds = [ev.start] + cuts + [ev.end]
        for j in range(len(bounds) - 1):
            piece = NoteEvent(
                start=bounds[j],
                end=bounds[j + 1],
                pitch=ev.pitch,
                velocity=ev.velocity,
                confidence=ev.confidence,
                bend=list(ev.bend) if j == 0 else [],
                vibrato=ev.vibrato if j == 0 else False,
            )
            if note_act is not None and 0 <= b < note_act.shape[1]:
                pf0 = int(round(bounds[j] / _BP_FRAME))
                pf1 = max(pf0 + 1, int(round(bounds[j + 1] / _BP_FRAME)))
                amp = float(note_act[pf0:pf1, b].mean())
                piece.velocity = _amp_to_velocity(amp)
            out.append(piece)
    return out


def transcribe(
    audio_path: str,
    mode: str = "solo",
    min_pitch: int = 38,
    max_pitch: int = 92,
    stats: dict | None = None,
) -> list[NoteEvent]:
    """Transcribe `audio_path` to NoteEvents, ascending by (start, pitch).

    `mode` is "solo" (basic-pitch plus the pyin refinement described in the
    module docstring) or "poly" (basic-pitch only). Pitches outside
    [min_pitch, max_pitch] are dropped; the frequency gate handed to
    basic-pitch is the intersection of that range with the guitar band
    (GUITAR_MIN_HZ..GUITAR_MAX_HZ), so at the default range the decoder sees
    MIDI 38-89 and raising max_pitch past 89 does not by itself admit higher
    notes. Narrowing either bound does tighten the gate.
    """
    if mode not in ("solo", "poly"):
        raise ValueError(f"mode must be 'solo' or 'poly', got {mode!r}")

    from basic_pitch.inference import predict

    min_hz = max(GUITAR_MIN_HZ, float(librosa.midi_to_hz(min_pitch - 0.5)))
    max_hz = min(GUITAR_MAX_HZ, float(librosa.midi_to_hz(max_pitch + 0.5)))

    # The CoreML backend prints per-window `isfinite/shape/dtype` debug lines
    # (inference.py:156-158) plus a banner (inference.py:449); on a 2-minute
    # clip that is a few hundred lines of noise through the pipeline's stdout.
    with contextlib.redirect_stdout(io.StringIO()):
        _model_output, _midi, note_events = predict(
            audio_path,
            onset_threshold=ONSET_THRESHOLD,
            frame_threshold=FRAME_THRESHOLD,
            minimum_note_length=MIN_NOTE_LENGTH_MS,
            minimum_frequency=min_hz,
            maximum_frequency=max_hz,
            melodia_trick=True,
        )

    events = _events_from_predictions(note_events, min_pitch, max_pitch)
    events = merge_adjacent_events(events)
    if mode == "solo":
        events = _refine_solo(events, audio_path, stats=stats)
        # Octave relabelling can carry a note out of the requested window: with
        # min_pitch above the true fundamental, basic-pitch sees only the
        # harmonic and the contour then correctly places it an octave below the
        # floor. The note is real but out of range, so the range wins — the
        # alternative is emitting it at a pitch the contour contradicts.
        events = [e for e in events if min_pitch <= e.pitch <= max_pitch]
        events = merge_adjacent_events(events)
    # LAST, after every merge pass (splits create 0-gap neighbours the
    # unconditional <=15ms merge would immediately re-fuse):
    _y, _sr = _load_mono(audio_path)
    events = _split_rearticulations(events, _model_output, y=_y, sr=_sr)
    return sorted(events, key=lambda e: (e.start, e.pitch))
