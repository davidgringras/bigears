"""Karplus-Strong resynthesis of a Score (or of raw NoteEvents).

The audit needs to *hear* what it wrote. Resynthesizing the quantized Score
onto the original recording's clock — via Score.tick_to_seconds, so the render
drifts exactly where the band drifted — is what makes a frame-by-frame
comparison against the source audio meaningful rather than decorative.

This is deliberately a plucked-string model and not a sampler: the point is to
check *what notes were written and when*, and a synthetic pluck with correct
pitch and attack answers that question without dragging in a sample library.

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

from dataclasses import dataclass

import numpy as np

from .model import TICKS_PER_QUARTER, NoteEvent, Score

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
