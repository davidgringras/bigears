"""Perceptual acceptance harness for the demo's A/B audio pair.

Exists because metrics were deaf: F1 compares MIDI numbers, chroma folds
octaves, and the first shipped demo audibly failed a musician while every
number was green. This harness compares the two AUDIO streams the listener
actually hears, on the axes the listener actually judged:

  extra sounds | missed notes | onset tightness | swing feel | air (duty) | octaves

Run: .venv/bin/python tests/demo_acceptance.py A.wav B.wav
(B is the machine's reading; truth defaults to the bebop lick.)

EDGE EXCLUSION (first 150 ms), with evidence: librosa's onset detector needs
preceding spectral context, so at the file edge it either misses the opening
note entirely (measured on A: first true note at 0.000 s produced no onset)
or reports it displaced (measured on B: +69 ms deviation for a note whose
underlying MIDI sits 12 ms from truth at confidence 0.91). Both streams
carry the note; the detector cannot testify there. Excluding the edge from
the extra/missed tests removes measurement blindness, not audio defects.
"""
import sys

import librosa
import numpy as np
import soundfile as sf

EDGE = 0.15
BEAT = 60 / 150.0


def _mono(path):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(1)
    return y.astype(float), sr


def _onsets(y, sr):
    return librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)


def _offpos(os_):
    ps = [(t / BEAT) % 1 for t in os_ if 0.5 < (t / BEAT) % 1 < 0.85]
    return float(np.mean(ps)) if ps else 0.5


def _duty(y):
    env = np.abs(librosa.util.frame(y, frame_length=512, hop_length=256)).max(0)
    return float((env > 0.06 * env.max()).mean())


def run(a_path, b_path, truth_starts, events):
    a, sr = _mono(a_path)
    b, _ = _mono(b_path)
    oa, ob = _onsets(a, sr), _onsets(b, sr)
    truth = np.asarray(sorted(truth_starts))

    pair = [(t, truth[np.argmin(np.abs(truth - t))]) for t in ob]
    off = np.median([t - u for t, u in pair if abs(t - u) < 0.15])
    extra = [t for t, u in pair
             if t > EDGE and abs(t - u - off) > 0.05
             and np.min(np.abs(oa - t)) > 0.05]

    real_a = [t for t in oa
              if t > EDGE and np.min(np.abs(truth - t)) <= 0.05]
    missed = sum(1 for t in real_a if np.min(np.abs(ob - t)) > 0.05)
    matched = np.array([np.min(np.abs(ob - t)) for t in real_a
                        if np.min(np.abs(ob - t)) <= 0.05])

    f0, _, _ = librosa.pyin(b, fmin=60, fmax=1400, sr=sr)
    tt = librosa.times_like(f0, sr=sr)
    octerr = 0
    for e in events:
        sel = (tt >= e.start + 0.03) & (tt <= e.start + max(0.09, (e.end - e.start) * 0.6))
        fs = f0[sel]
        fs = fs[~np.isnan(fs)]
        if len(fs) >= 2 and abs(abs(librosa.hz_to_midi(np.median(fs)) - e.pitch) - 12) < 0.7:
            octerr += 1

    checks = {
        "genuinely extra sounds in B == 0": (len(extra), len(extra) == 0),
        "missed real notes <= 4": (missed, missed <= 4),
        "matched onset median <= 15ms": (round(float(np.median(matched)) * 1000, 1),
                                         np.median(matched) <= 0.015),
        "matched onset p90 <= 35ms": (round(float(np.percentile(matched, 90)) * 1000, 1),
                                      np.percentile(matched, 90) <= 0.035),
        "swing |A-B| <= 0.02 beat": (round(abs(_offpos(oa) - _offpos(ob)), 3),
                                     abs(_offpos(oa) - _offpos(ob)) <= 0.02),
        "duty |A-B| <= 0.06": (round(abs(_duty(a) - _duty(b)), 3),
                               abs(_duty(a) - _duty(b)) <= 0.06),
        "octave errors == 0": (octerr, octerr == 0),
    }
    ok = True
    for k, (v, p) in checks.items():
        print(f"  [{'PASS' if p else 'FAIL'}] {k}: {v}")
        ok &= p
    print("ACCEPTANCE:", "ALL PASS" if ok else "FAILING")
    return ok


if __name__ == "__main__":
    import os
    import pickle

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import tests.licks as TL

    a_path = sys.argv[1] if len(sys.argv) > 2 else "output/demo4/recording.wav"
    b_path = sys.argv[2] if len(sys.argv) > 2 else "output/demo4/heard.wav"
    truth = [e.start for e in TL.lick_events(TL.LICKS[0])]
    events = pickle.load(open("output/demo4/events.pkl", "rb"))
    sys.exit(0 if run(a_path, b_path, truth, events) else 1)
