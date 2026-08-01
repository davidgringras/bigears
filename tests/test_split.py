"""Re-articulation splitting: fused repeats recovered, non-repeats untouched.

The thresholds in `_split_rearticulations` were set from measured onset-head
posteriors (see the constant block in transcribe.py). These tests pin the
behavioural consequences on real fixtures rather than re-deriving posteriors.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import soundfile as sf

from soloscribe.transcribe import transcribe

FIXTURE = "output/demo4/recording.wav"        # sampled-guitar bebop with 7 fused sites
SR = 22050


@pytest.fixture(scope="module")
def truth():
    import tests.licks as TL
    return TL.lick_events(TL.LICKS[0])


@pytest.mark.skipif(not os.path.exists(FIXTURE), reason="demo4 recording not built")
def test_fused_repeats_recovered(truth):
    events = transcribe(FIXTURE, mode="solo")
    starts = np.array(sorted(e.start for e in events))
    missing = [t for t in truth if np.min(np.abs(starts - t.start)) > 0.06]
    # 7 fused sites before the split pass; require at least 5 recovered and
    # print the residue so a regression is legible.
    print(f"missing after split pass: {len(missing)} -> "
          f"{[(m.pitch, round(m.start, 3)) for m in missing]}")
    assert len(missing) <= 2, [(m.pitch, round(m.start, 3)) for m in missing]
    # and nothing invented: every event matches a truth note within 60ms
    tstarts = np.array(sorted(t.start for t in truth))
    spurious = [e for e in events if np.min(np.abs(tstarts - e.start)) > 0.06]
    assert len(spurious) <= 1, [(e.pitch, round(e.start, 3)) for e in spurious]


def test_held_note_not_split(tmp_path):
    # A steady 2s tone with realistic decay must stay ONE event.
    t = np.arange(int(SR * 2.0)) / SR
    f = 440 * 2 ** ((57 - 69) / 12)
    y = sum(a * np.sin(2 * np.pi * f * (i + 1) * t)
            for i, a in enumerate([1.0, 0.5, 0.3, 0.15])) * np.exp(-t * 0.6)
    full = np.zeros(int(2.4 * SR))
    full[int(0.2 * SR):int(0.2 * SR) + len(y)] += y
    full /= np.abs(full).max()
    p = str(tmp_path / "held.wav")
    sf.write(p, full.astype(np.float32), SR)
    events = transcribe(p, mode="solo")
    at = [e for e in events if e.pitch == 57]
    assert len(at) == 1, [(e.pitch, round(e.start, 3), round(e.end, 3)) for e in events]


def test_deliberate_repeats_split(tmp_path):
    # Four hard re-picked same-pitch notes back to back: must come out as >= 3
    # events even though decay tails touch the next attack.
    t_note = np.arange(int(SR * 0.22)) / SR
    f = 440 * 2 ** ((62 - 69) / 12)
    note = sum(a * np.sin(2 * np.pi * f * (i + 1) * t_note)
               for i, a in enumerate([1.0, 0.6, 0.35, 0.2])) * np.exp(-t_note * 3.0)
    full = np.zeros(int(1.6 * SR))
    for k in range(4):
        i0 = int((0.2 + k * 0.22) * SR)
        full[i0:i0 + len(note)] += note
    full /= np.abs(full).max()
    p = str(tmp_path / "repeats.wav")
    sf.write(p, full.astype(np.float32), SR)
    events = transcribe(p, mode="solo")
    at = [e for e in events if e.pitch == 62]
    assert len(at) >= 3, [(round(e.start, 3), round(e.end, 3)) for e in at]
