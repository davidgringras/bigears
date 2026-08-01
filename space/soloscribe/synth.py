"""Karplus-Strong resynthesis of a Score (or of raw NoteEvents), plus a
sampled-guitar voicing of the raw events for people to listen to.

The audit needs to *hear* what it wrote. Resynthesizing the quantized Score
onto the original recording's clock — via Score.tick_to_seconds, so the render
drifts exactly where the band drifted — is what makes a frame-by-frame
comparison against the source audio meaningful rather than decorative.

The measured render is deliberately a plucked-string model and not a sampler:
the point is to check *what notes were written and when*, and a synthetic
pluck with correct pitch and attack answers that question without dragging in
a sample library. render_events_fluid (bottom of this file) is a SECOND,
additive render that exists for the opposite reason — a human listener judged
the KS render "sounds nothing like the original" while every metric was green,
so the report needs something a guitarist can actually A/B. It changes nothing
the audit measures.

Not modelled in v1 (each of these makes the audit pessimistic rather than
optimistic, which is the direction we want to err):
  * QNote.bend / NoteEvent.bend — bend contours are carried through the data
    model and written to the GP5, but the render plays the nominal pitch flat.
    A heavily bent phrase will therefore score worse in the chroma comparison
    than the transcription deserves. audit.py states this in the caveats.
  * vibrato — same treatment, ignored here.
  * Articulation (slide / hammer-on / pull-off / palm mute) — every note is
    picked from silence.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np

from .model import TICKS_PER_QUARTER, NoteEvent, Score

logger = logging.getLogger(__name__)

DEFAULT_SR = 22050

# How long a note is allowed to ring past its notated end. Real guitars do not
# stop dead on the barline; without this the render sounds gated and its chroma
# has holes the source audio does not.
RING_OUT_SEC = 0.30

_ATTACK_SEC = 0.003          # click guard on the pluck
_MIN_NOTE_SEC = 0.02         # anything shorter is a grace note / data glitch
_TARGET_PEAK = 0.9

# Fixed excitation seed. The audit compares a render against the recording and
# reports a number; that number must not move because the noise burst did.
_NOISE_SEED = 20502


@dataclass
class MergedNote:
    """A tie chain collapsed into a single sounding note, still in tick time.

    Both synth.py and audit.py need the same answer to "what notes does this
    Score actually sound?", so the merge lives here and audit.py imports it.
    """

    onset: int
    offset: int
    pitch: int
    velocity: int = 96
    confidence: float = 1.0

    @property
    def duration(self) -> int:
        return self.offset - self.onset


# A tie continuation should start exactly where its predecessor ended. Allow a
# 32nd note of slop so a rounding artifact upstream does not silently split a
# chain into two re-picked notes.
_TIE_SLOP_TICKS = TICKS_PER_QUARTER // 8


def merge_tied(score: Score) -> list[MergedNote]:
    """Collapse tied QNote chains into single sustained notes.

    A QNote with tied_from_prev=True is a continuation, not a new attack. If no
    open chain at that pitch is found the note is kept as its own attack rather
    than dropped — a mislabelled tie should degrade the render, not delete a note
    from the audit's view of the score.
    """
    out: list[MergedNote] = []
    open_at_pitch: dict[int, int] = {}   # pitch -> index into out
    for q in sorted(score.qnotes, key=lambda n: (n.onset, n.pitch)):
        dur = max(1, int(q.duration))
        if q.tied_from_prev:
            i = open_at_pitch.get(q.pitch)
            if i is not None and abs(out[i].offset - q.onset) <= _TIE_SLOP_TICKS:
                prev = out[i]
                prev.offset = max(prev.offset, q.onset + dur)
                prev.velocity = max(prev.velocity, q.velocity)
                # A chain is only as trustworthy as its weakest link.
                prev.confidence = min(prev.confidence, q.confidence)
                continue
        out.append(
            MergedNote(
                onset=int(q.onset),
                offset=int(q.onset) + dur,
                pitch=int(q.pitch),
                velocity=int(q.velocity),
                confidence=float(q.confidence),
            )
        )
        open_at_pitch[q.pitch] = len(out) - 1
    out.sort(key=lambda n: (n.onset, n.pitch))
    return out


def velocity_to_amplitude(velocity: int) -> float:
    """Map MIDI velocity 1–127 to a linear amplitude in ~[0.15, 1.0].

    Loudness is roughly a power law in amplitude, so a linear velocity ramp
    sounds top-heavy. The 1.6 exponent opens up the quiet half of the range
    where a solo's dynamics actually live; the 0.15 floor keeps a ghost note
    audible instead of inaudible.
    """
    v = float(np.clip(velocity, 1, 127))
    return 0.15 + 0.85 * (v / 127.0) ** 1.6


def _t60_for(f0: float) -> float:
    """Guitar-ish decay time (seconds to -60 dB) for a given fundamental.

    Wound low strings ring appreciably longer than the plain high ones, so the
    decay is pitch-dependent. Anchored at 2.6 s near A3 (220 Hz) and clamped so
    neither extreme goes silly: ~3.7 s at low E, ~1.4 s two octaves up.
    """
    return float(np.clip(2.6 * (220.0 / max(f0, 1e-6)) ** 0.35, 0.7, 4.0))


def _pluck(
    f0: float,
    n_samples: int,
    sr: int,
    amp: float,
    rng: np.random.Generator,
    brightness: float = 0.5,
) -> np.ndarray:
    """One Karplus-Strong plucked string.

    Classic extended KS: a noise burst circulating through a delay line with a
    two-point averaging lowpass, plus a scalar loop gain that sets the decay.

        d[n] = (1-frac)*y[n-L] + frac*y[n-L-1]      (fractional delay)
        y[n] = damping * 0.5 * (d[n] + d[n-1])      (averager + loop gain)

    TUNING — why fractional delay is not optional here.
    The averaging filter contributes ~0.5 sample of phase delay at DC, so the
    delay line must supply sr/f0 - 0.5 samples. Rounding that to an integer
    costs up to half a sample, and half a sample is a large fraction of a short
    period. Measured at sr=22050 (see the sweep in the module's dev notes):

        f0 =   82 Hz (low E)     ->   0.4 cents
        f0 =  110 Hz             ->   0.4 cents
        f0 =  262 Hz (middle C)  ->   4.5 cents
        f0 =  330 Hz (open high E) -> 10.2 cents   <- budget blown here
        f0 =  880 Hz             ->  30.4 cents
        f0 = 1319 Hz (24th fret) ->  23.3 cents
        worst case over 80-1400 Hz: 54.9 cents, at ~1378 Hz

    The 10-cent budget is first exceeded at ~259 Hz — middle of the range a
    solo lives in, not some exotic corner — and 55 cents is more than a quarter
    tone, which would put the render outside mir_eval's own pitch tolerance
    against the score it is supposed to be reproducing. So: linear-interpolated
    fractional delay, which drops the residual tuning error to well under a
    cent across the whole range. (Linear interpolation also lowpasses slightly,
    which is inaudible next to the averaging filter already in the loop.)
    """
    if n_samples <= 0 or f0 <= 0.0:
        return np.zeros(max(0, n_samples), dtype=np.float64)

    total_delay = sr / f0
    line = total_delay - 0.5          # averager supplies the other half sample
    if line < 2.0:                     # above ~sr/2.5; nothing musical up there
        return np.zeros(n_samples, dtype=np.float64)
    L = int(np.floor(line))
    frac = line - L

    # Loop gain: amplitude is multiplied by `damping` once per period, and we
    # want -60 dB after t60 seconds, i.e. f0*t60 periods.
    t60 = _t60_for(f0)
    damping = 10.0 ** (-3.0 / (t60 * f0))

    # Excitation. White noise is the textbook burst but is harsher than a pick;
    # one pole of lowpass takes the edge off without dulling the attack.
    burst = rng.uniform(-1.0, 1.0, size=L + 1)
    if brightness < 1.0:
        a = 1.0 - float(np.clip(brightness, 0.05, 1.0))
        for i in range(1, burst.size):
            burst[i] = (1.0 - a) * burst[i] + a * burst[i - 1]
    peak = np.max(np.abs(burst))
    if peak > 0:
        burst /= peak

    y = np.zeros(L + 1 + n_samples, dtype=np.float64)
    d = np.zeros_like(y)
    y[: L + 1] = burst
    d[: L + 1] = burst

    # The recursion reaches back at least L samples, so blocks of L samples can
    # be evaluated with vectorized numpy instead of a Python loop per sample.
    # d[] for the block is assigned before y[] reads d[i-1:m-1], so the one-tap
    # dependency inside the block sees fresh values.
    step = max(1, L)
    i = L + 1
    end = y.size
    while i < end:
        m = min(i + step, end)
        d[i:m] = (1.0 - frac) * y[i - L : m - L] + frac * y[i - L - 1 : m - L - 1]
        y[i:m] = damping * 0.5 * (d[i:m] + d[i - 1 : m - 1])
        i = m

    sig = y[L + 1 : L + 1 + n_samples] * amp

    # Attack ramp only; the loop gain already handles the decay, and the caller
    # applies the ring-out fade so it can size it against the notated duration.
    n_atk = min(int(_ATTACK_SEC * sr), n_samples)
    if n_atk > 1:
        sig[:n_atk] *= np.linspace(0.0, 1.0, n_atk)
    return sig


def _mixdown(buf: np.ndarray) -> np.ndarray:
    """Normalize to a fixed peak, soft-clipping first if the sum ran hot.

    Overlapping ring-outs in a fast passage can push the sum well past 1.0;
    tanh bends those peaks instead of shearing them, and the final scale puts
    every render at the same level so the report's three players are
    comparable by ear.
    """
    if buf.size == 0:
        return buf
    peak = float(np.max(np.abs(buf)))
    if peak <= 0.0:
        return buf
    if peak > 1.0:
        buf = np.tanh(buf)
    peak = float(np.max(np.abs(buf)))
    if peak > 0.0:
        buf = buf * (_TARGET_PEAK / peak)
    return buf


def _render_spans(
    spans: list[tuple[float, float, int, int]],
    sr: int,
    total_dur: float | None,
    seed: int,
) -> np.ndarray:
    """Render (start_s, end_s, midi_pitch, velocity) tuples into one buffer."""
    rng = np.random.default_rng(seed)
    if total_dur is None:
        last = max((e for _, e, _, _ in spans), default=0.0)
        total_dur = last + RING_OUT_SEC
    n_total = max(1, int(np.ceil(total_dur * sr)))
    buf = np.zeros(n_total, dtype=np.float64)

    for t0, t1, pitch, velocity in spans:
        if t1 <= t0:
            continue
        body = max(t1 - t0, _MIN_NOTE_SEC)
        # Short notes get a proportionally short ring-out: a 16th note trailing
        # 300 ms of tail would smear the chroma across the next two notes.
        tail = min(RING_OUT_SEC, body)
        i0 = int(round(t0 * sr))
        if i0 >= n_total:
            continue
        n_body = max(1, int(round(body * sr)))
        n_tail = max(1, int(round(tail * sr)))
        n = n_body + n_tail

        f0 = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
        sig = _pluck(f0, n, sr, velocity_to_amplitude(velocity), rng)
        # Raised-cosine fade across the ring-out so the tail dies smoothly.
        fade = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n_tail)))
        sig[n_body:] *= fade

        if i0 < 0:
            sig = sig[-i0:]
            i0 = 0
        room = n_total - i0
        if room <= 0:
            continue
        buf[i0 : i0 + min(room, sig.size)] += sig[:room]

    return _mixdown(buf)


def render_events(
    events: list[NoteEvent],
    sr: int = DEFAULT_SR,
    total_dur: float | None = None,
) -> np.ndarray:
    """Resynthesize raw transcribed NoteEvents (already in seconds).

    This renders the transcriber's output *before* quantization, which is what
    you want when the question is "did the transcriber hear the right notes?"
    as opposed to "did quantization keep them?".
    """
    spans = [
        (float(ev.start), float(ev.end), int(ev.pitch), int(ev.velocity))
        for ev in sorted(events, key=lambda e: e.start)
        if ev.end > ev.start
    ]
    return _render_spans(spans, sr, total_dur, seed=_NOISE_SEED)


def score_note_seconds(
    score: Score, use_tempo_map: bool = True
) -> list[tuple[MergedNote, float, float]]:
    """Merged score notes paired with their (start, end) in recording seconds.

    audit.py needs exactly this list to compare the Score against the raw
    events and against the audio, so the tick->seconds convention has one
    implementation rather than two that drift apart.
    """
    out: list[tuple[MergedNote, float, float]] = []
    for n in merge_tied(score):
        if use_tempo_map:
            t0 = score.tick_to_seconds(n.onset)
            t1 = score.tick_to_seconds(n.offset)
        else:
            q = 60.0 / max(score.tempo_bpm, 1e-6)
            t0 = n.onset / TICKS_PER_QUARTER * q
            t1 = n.offset / TICKS_PER_QUARTER * q
        t0 = max(0.0, float(t0))
        # A non-monotonic or degenerate tempo_map segment must not produce a
        # zero-length note; mir_eval rejects those outright.
        t1 = max(float(t1), t0 + _MIN_NOTE_SEC)
        out.append((n, t0, t1))
    return out


def render_score(
    score: Score, sr: int = DEFAULT_SR, use_tempo_map: bool = True
) -> np.ndarray:
    """Resynthesize a quantized Score.

    With use_tempo_map=True (the default, and the only setting the audit should
    use) note times come from Score.tick_to_seconds, so the render sits on the
    original recording's clock and can be compared to it frame by frame. With
    it False the render uses strict nominal-tempo arithmetic — what the .gp5
    will sound like played back in Guitar Pro at a fixed tempo, which is a
    different and much less forgiving thing to listen to.
    """
    spans = [
        (t0, t1, n.pitch, n.velocity)
        for n, t0, t1 in score_note_seconds(score, use_tempo_map=use_tempo_map)
    ]
    return _render_spans(spans, sr, None, seed=_NOISE_SEED)


# --------------------------------------------------------------------------
# Sampled-guitar voicing — the LISTENING artifact, not the measured one
# --------------------------------------------------------------------------
#
# Everything below is additive. render_score and render_events are what the
# audit measures against the recording, and they are untouched: changing them
# would change what the chroma comparison means.
#
# Three decisions make this sound like a reading of the take rather than a
# rendering of a spreadsheet. Each was measured, and each is re-measured per
# render rather than pinned to a constant that was true of one clip once:
#
#   1. It plays the EVENTS, not the Score. Quantization is where microtiming
#      goes to die; played back, a quantized score sounds mechanical because
#      it is. The events are the transcriber's actual reading.
#   2. Alignment. fluidsynth emits its first sample late, and the
#      transcriber's onset convention sits early of where a listener hears
#      the attack. Both are measured on the audio in play, in render_events_fluid.
#   3. Articulation. NoteEvent.end includes the decay tail, so playing events
#      at their nominal length is more legato than the performance was.

# Homebrew's install location, mirroring audit._ffmpeg (audit.py:129-134):
# the app is launched from Finder, where /opt/homebrew/bin is not on PATH.
_FLUIDSYNTH_FALLBACK = "/opt/homebrew/bin/fluidsynth"

# Resolution order after an explicit argument. MuseScore ships MS Basic with
# every install and is already a dependency of the PDF export path; the second
# is the package-managed General MIDI bank on Linux.
SOUNDFONT_CANDIDATES = (
    "/Applications/MuseScore 4.app/Contents/Resources/sound/MS Basic.sf3",
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
)

GM_JAZZ_GUITAR = 26          # General MIDI program 26, "Electric Guitar (jazz)"

_FLUID_GAIN = "0.9"          # fluidsynth's own output gain, pre-loudness-match
_FLUID_TIMEOUT_SEC = 180.0

# MS Basic.sf3 is 51 MB of Vorbis-compressed samples and fluidsynth decodes the
# whole bank at startup: measured 5.4 s per render regardless of note count
# (1 note and 64 notes both 5.4 s), against 1.2 s with dynamic loading on, and
# the two renders are bit-identical (max sample difference 0.0). This is a
# per-render cost paid on every audit, so it is worth the retry below.
#
# fluidsynth treats an unrecognised -o setting as FATAL (rc 255, no output),
# so an old build must not be assumed to have it — hence try-then-retry rather
# than a version check, which would mean asserting a version history I have
# not verified. The rejection is immediate, well before any soundfont load.
_FLUID_TUNING = ("-o", "synth.dynamic-sample-loading=1")

# pretty_midi writes seconds as ticks, so the file's time resolution is a real
# quantization of the event times: 960 ticks/quarter at 120 bpm is 0.52 ms,
# two orders below the alignment budget this whole function is defending.
_MIDI_RESOLUTION = 960
_MIDI_TEMPO = 120.0

# MIDI velocity 0 is a note-off, so the floor has to be at least 1 or a ghost
# note vanishes. 20 is what the validated demo chain used; measured in MS
# Basic, velocities 1, 8 and 20 all render to the same peak (0.0056), so the
# exact floor is inaudible there and the clip is kept for banks where it isn't.
_VELOCITY_FLOOR = 20

# Articulation cap: a note may not ring past 85 % of the gap to the next
# attack. Without it the reading is audibly more legato than the playing. The
# 70 ms floor only engages when two attacks are within 82 ms of each other —
# faster than sixteenths at 180 bpm (60/180/4 = 83 ms) — where the alternative
# is a note too short to speak at all.
_ARTIC_GAP_FRACTION = 0.85
_ARTIC_MIN_SEC = 0.07

# Matching windows for the two calibrations. 60 ms for the onset convention;
# 150 ms for the render lag is loose enough to survive a synth latency several
# times the 28.0–33.5 ms measured here while still throwing out a note that
# produced no onset at all and matched something a bar away.
_CONVENTION_WINDOW_SEC = 0.06
_LAG_WINDOW_SEC = 0.15
_MIN_CALIBRATION_PAIRS = 3

_ACTIVE_FLOOR = 0.001        # |x| above this counts as sounding, for RMS
_LEVEL_PASSES = 3            # see the loudness block: the region moves as it scales
_LEVEL_TOLERANCE = 0.005
_LIMIT_PEAK = 0.97


def _fluidsynth() -> str | None:
    found = shutil.which("fluidsynth")
    if found:
        return found
    return _FLUIDSYNTH_FALLBACK if os.path.exists(_FLUIDSYNTH_FALLBACK) else None


def _resolve_soundfont(explicit: str | None) -> str | None:
    """Explicit argument, then the known banks, then nothing.

    An explicit path that does not exist resolves to None rather than falling
    through: a caller who names a file means that file, and silently voicing
    with a different instrument than the one asked for is the kind of quiet
    substitution this report exists to not do.
    """
    if explicit is not None:
        return explicit if os.path.exists(explicit) else None
    for path in SOUNDFONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def articulated_spans(
    events: list[NoteEvent],
) -> list[tuple[float, float, int, int]]:
    """(start, end, pitch, velocity) with decay tails trimmed off the ends.

    NoteEvent.end is where the note stopped being *detectable*, which includes
    the decay of a note the player already let go. Sounding that length on a
    sampled guitar re-triggers the next note on top of a still-ringing one and
    the reading comes out legato where the playing was not. Each note is
    therefore capped at 85 % of the gap to the next attack.

    The last note has no next attack, so it is capped against its own length —
    same 85 %, which keeps the ending from hanging on past the recording.
    """
    ordered = sorted(
        (ev for ev in events if ev.end > ev.start), key=lambda e: e.start
    )
    spans: list[tuple[float, float, int, int]] = []
    for i, ev in enumerate(ordered):
        nxt = ordered[i + 1].start if i + 1 < len(ordered) else None
        gap = (nxt - ev.start) if nxt is not None else (ev.end - ev.start)
        dur = min(
            float(ev.end - ev.start),
            max(_ARTIC_MIN_SEC, _ARTIC_GAP_FRACTION * float(gap)),
        )
        spans.append(
            (float(ev.start), float(ev.start) + dur, int(ev.pitch),
             int(ev.velocity))
        )
    return spans


def _write_midi(
    spans: list[tuple[float, float, int, int]], path: str, program: int
) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(
        resolution=_MIDI_RESOLUTION, initial_tempo=_MIDI_TEMPO
    )
    inst = pretty_midi.Instrument(program=int(program))
    for t0, t1, pitch, velocity in spans:
        # end is derived from the CLAMPED start: a negative t0 would otherwise
        # leave end < start, which is not a note.
        start = float(max(0.0, t0))
        inst.notes.append(
            pretty_midi.Note(
                velocity=int(np.clip(velocity, _VELOCITY_FLOOR, 127)),
                pitch=int(np.clip(pitch, 0, 127)),
                start=start,
                end=float(max(t1, start + 0.02)),
            )
        )
    pm.instruments.append(inst)
    pm.write(path)


def _render_spans_fluid(
    spans: list[tuple[float, float, int, int]],
    sr: int,
    soundfont: str,
    program: int,
    binary: str,
) -> np.ndarray | None:
    """Spans -> temp MIDI -> fluidsynth -> mono float array. No calibration.

    Separate from render_events_fluid so the articulation stage can be tested
    against an uncapped render of the same notes through the same voice, which
    is the only way to see the cap do anything.
    """
    import soundfile as sf

    with tempfile.TemporaryDirectory() as work:
        mid = os.path.join(work, "voice.mid")
        wav = os.path.join(work, "voice.wav")
        try:
            _write_midi(spans, mid, program)
        except Exception as exc:                       # pragma: no cover
            logger.info("sampled-guitar voice: could not write MIDI (%s)", exc)
            return None
        proc = None
        for tuning in (_FLUID_TUNING, ()):
            try:
                proc = subprocess.run(
                    [binary, "-ni", *tuning, "-F", wav, "-r", str(int(sr)),
                     "-g", _FLUID_GAIN, soundfont, mid],
                    capture_output=True, timeout=_FLUID_TIMEOUT_SEC,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                logger.info("sampled-guitar voice: fluidsynth failed (%s)", exc)
                return None
            if proc.returncode == 0 and os.path.exists(wav) and os.path.getsize(wav):
                break
            if tuning:
                logger.debug(
                    "sampled-guitar voice: retrying without %s (%s)",
                    " ".join(tuning),
                    proc.stderr.decode("utf-8", "replace").strip()[-120:],
                )
        if proc is None or proc.returncode != 0 or not os.path.exists(wav) \
                or not os.path.getsize(wav):
            logger.info(
                "sampled-guitar voice: fluidsynth exited %s (%s)",
                None if proc is None else proc.returncode,
                "" if proc is None else
                proc.stderr.decode("utf-8", "replace").strip()[-200:] or "no stderr",
            )
            return None
        y, _ = sf.read(wav, dtype="float64", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.asarray(y, dtype=np.float64)


def _detect_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    """Onsets in seconds, backtrack=False.

    Both calibrations below run this same detector over both signals, which is
    what makes them subtractable: the detector's own bias (measured here at
    +16.4 ms against a KS render whose attacks sit exactly on the note starts)
    is present in both terms and cancels. Backtracking to the preceding energy
    minimum would break that — it moves each side by a different amount.
    """
    import librosa

    return np.asarray(
        librosa.onset.onset_detect(
            y=np.asarray(y, dtype=float), sr=sr, units="time", backtrack=False
        ),
        dtype=float,
    )


def _nearest_deltas(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each t in a, t minus the nearest element of b."""
    if a.size == 0 or b.size == 0:
        return np.zeros(0, dtype=float)
    return np.array([t - b[int(np.argmin(np.abs(b - t)))] for t in a], dtype=float)


def _median_within(deltas: np.ndarray, window: float) -> tuple[float, int]:
    """Median of the deltas inside ±window, and how many there were."""
    kept = deltas[np.abs(deltas) <= window] if deltas.size else deltas
    if kept.size < _MIN_CALIBRATION_PAIRS:
        return 0.0, int(kept.size)
    return float(np.median(kept)), int(kept.size)


def render_events_fluid(
    events: list[NoteEvent],
    comparison_audio: np.ndarray,
    sr: int = DEFAULT_SR,
    soundfont: str | None = None,
    program: int = GM_JAZZ_GUITAR,
) -> np.ndarray | None:
    """Voice the raw NoteEvents with a sampled guitar, aligned to `comparison_audio`.

    Returns a mono float array the same length as `comparison_audio` (which
    must already be mono at `sr`), or None — never a silent or unaligned
    array — if fluidsynth, a soundfont, pretty_midi or the events themselves
    are missing. Every None is logged with its reason at INFO.

    The calibration chain, in order, all of it measured on this render and this
    clip rather than assumed:

      articulation  Event ends carry decay tails; see articulated_spans.
      render lag    fluidsynth emits late. Median over (nearest detected onset
                    in the render − its MIDI note start), matched within
                    150 ms. Measured on this machine (fluidsynth 2.5.7, MS
                    Basic.sf3): +28.0 ms on an 8-note synthetic line, identical
                    across three renders, and +33.5 ms (n=57) on
                    tests/fixtures/bebop_f__none.wav through the pipeline. It
                    is a property of the render, not a constant — the demo that
                    motivated this feature saw 32–36 ms moving between renders.
      convention    basic-pitch reports onsets early of where a listener hears
                    the attack. Median over (detected onset in the comparison
                    audio − nearest event start), matched within 60 ms:
                    measured +21.9 ms (n=56) on the bebop fixture, and
                    +16.4 ms against a KS comparison whose attacks sit exactly
                    on the event starts. That second figure is the detector's
                    own bias and is supposed to survive — it is present in the
                    render lag too, and the subtraction cancels it.
      shift         Trim (lag − convention) off the front, or pad if negative,
                    so the render's attacks land where the recording's
                    attacks land rather than where its MIDI says. Measured
                    end to end: first audible sample within 6 ms of the first
                    event start.
      level         Scaled to the comparison's RMS over its sounding region
                    (|x| > 0.001; silence must not drag the target down),
                    iterated because that region grows as the render is scaled
                    up, then limited to a 0.97 peak.

    Both calibrations degrade to 0.0 — no shift — when fewer than three pairs
    survive their window, and say so in the log. A calibration measured on two
    points is not a measurement.
    """
    binary = _fluidsynth()
    if binary is None:
        logger.info("no sampled-guitar voice: fluidsynth not on PATH or at %s",
                    _FLUIDSYNTH_FALLBACK)
        return None
    sf_path = _resolve_soundfont(soundfont)
    if sf_path is None:
        logger.info(
            "no sampled-guitar voice: no soundfont at %s",
            soundfont if soundfont else " or ".join(SOUNDFONT_CANDIDATES),
        )
        return None
    spans = articulated_spans(events)
    if not spans:
        logger.info("no sampled-guitar voice: no sounding events to render")
        return None
    try:
        import librosa  # noqa: F401  (used via _detect_onsets)
        import pretty_midi  # noqa: F401  (used via _write_midi)
        import soundfile  # noqa: F401  (used via _render_spans_fluid)
    except ImportError as exc:
        logger.info("no sampled-guitar voice: %s", exc)
        return None

    y = _render_spans_fluid(spans, sr, sf_path, program, binary)
    if y is None or y.size == 0 or not np.any(y):
        logger.info("no sampled-guitar voice: fluidsynth produced no audio")
        return None

    cmp_audio = np.asarray(comparison_audio, dtype=np.float64)
    if cmp_audio.ndim > 1:
        cmp_audio = cmp_audio.mean(axis=1)

    starts = np.array([s for s, _, _, _ in spans], dtype=float)
    lag, n_lag = _median_within(
        -_nearest_deltas(starts, _detect_onsets(y, sr)), _LAG_WINDOW_SEC
    )
    convention, n_conv = _median_within(
        _nearest_deltas(_detect_onsets(cmp_audio, sr), starts),
        _CONVENTION_WINDOW_SEC,
    )
    logger.info(
        "sampled-guitar voice: render lag %+.1f ms (n=%d), onset convention "
        "%+.1f ms (n=%d), shifting by %+.1f ms",
        lag * 1000, n_lag, convention * 1000, n_conv, (convention - lag) * 1000,
    )

    trim = lag - convention
    n_trim = int(round(abs(trim) * sr))
    if trim > 0:
        y = y[min(n_trim, y.size):]
    elif trim < 0:
        y = np.concatenate([np.zeros(n_trim, dtype=np.float64), y])
    if y.size == 0 or not np.any(y):
        logger.info("no sampled-guitar voice: alignment left nothing audible")
        return None

    # Same span as the comparison, so the two players in the report are the
    # same length and a listener can switch between them without hunting.
    if cmp_audio.size:
        y = np.pad(y, (0, max(0, cmp_audio.size - y.size)))[: cmp_audio.size]
    # Truncating to a comparison shorter than the events' lead-in leaves a
    # silent buffer, and a silent player labelled "the transcription" reads as
    # "the machine heard nothing". Measured: a 100-sample comparison against
    # notes starting at 0.35 s returned 100 samples at peak 0.000 before this
    # check existed. Say nothing rather than say that.
    if float(np.max(np.abs(y))) <= _ACTIVE_FLOOR:
        logger.info(
            "no sampled-guitar voice: nothing audible survived trimming to the "
            "comparison's %.2f s", cmp_audio.size / float(sr),
        )
        return None

    def _active_rms(x: np.ndarray) -> float:
        sounding = x[np.abs(x) > _ACTIVE_FLOOR]
        return float(np.sqrt(np.mean(sounding ** 2))) if sounding.size else 0.0

    # The active region is defined by an absolute floor, so scaling the render
    # moves the region as well as the level and one division lands short.
    # Measured on the test line: 0.328 -> 0.958 -> 0.999 -> 1.000 of target.
    target = _active_rms(cmp_audio)
    here = _active_rms(y)
    if target > 0.0 and here > 0.0:
        for _ in range(_LEVEL_PASSES):
            correction = target / here
            y = y * correction
            here = _active_rms(y)
            if abs(correction - 1.0) < _LEVEL_TOLERANCE:
                break
    else:
        logger.info(
            "sampled-guitar voice: no active region to match level against "
            "(comparison rms %.5f, render rms %.5f) — left at synth level",
            target, here,
        )
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > _LIMIT_PEAK:
        y = y * (_LIMIT_PEAK / peak)
    return y
