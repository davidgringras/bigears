"""Variants fire only on their measured triggers, scored and honest."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from soloscribe.model import BeatGrid, NoteEvent
from soloscribe.quantize import quantize
from soloscribe.variants import generate_variants, CONF_DOUBT


def _grid(bpm=120.0, n=24):
    p = 60.0 / bpm
    return BeatGrid(beat_times=[i * p for i in range(n)], beats_per_bar=4, bpm_nominal=bpm)


def _line(conf_tail=1.0):
    evs = [NoteEvent(i * 0.5, i * 0.5 + 0.4, 60 + (i % 5), confidence=0.9)
           for i in range(12)]
    evs += [NoteEvent(1.0, 3.0, 41, confidence=conf_tail),
            NoteEvent(2.5, 4.5, 45, confidence=conf_tail)]
    return sorted(evs, key=lambda e: e.start)


def _score(evs, grid):
    return quantize(evs, grid, swing="off", key="C", chords=[], title="t")


def test_confident_only_fires_on_doubtful_notes():
    grid = _grid()
    evs = _line(conf_tail=0.2)  # two low-confidence long low notes (bleed-shaped)
    y = np.zeros(int(22050 * 6.5), dtype=float)
    vs = generate_variants(evs, grid, _score(evs, grid), y, 22050,
                           key="C", chords=[], title="t")
    slugs = [v.slug for v in vs]
    assert "confident-only" in slugs, slugs
    v = next(v for v in vs if v.slug == "confident-only")
    assert v.n_notes < len(evs)
    assert 0.0 <= v.f1_100 <= 1.0 and 0.0 <= v.coverage <= 1.0


def test_no_variant_when_everything_is_confident_and_straight():
    grid = _grid()
    evs = _line(conf_tail=0.9)
    y = np.zeros(int(22050 * 6.5), dtype=float)
    vs = generate_variants(evs, grid, _score(evs, grid), y, 22050,
                           key="C", chords=[], title="t")
    assert vs == [], [v.slug for v in vs]


def test_feel_flip_fires_only_in_the_grey_zone():
    grid = _grid()
    # offbeats at 0.58 of the beat: genuinely between straight and swing
    evs = []
    for b in range(8):
        evs.append(NoteEvent(b * 0.5, b * 0.5 + 0.2, 62, confidence=0.9))
        evs.append(NoteEvent((b + 0.58) * 0.5, (b + 0.58) * 0.5 + 0.15, 64, confidence=0.9))
    y = np.zeros(int(22050 * 5), dtype=float)
    vs = generate_variants(evs, grid, _score(evs, grid), y, 22050,
                           key="C", chords=[], title="t")
    assert any(v.slug.startswith("feel-") for v in vs), [v.slug for v in vs]
    # canonical swing (0.667) must NOT offer a flip — it is not a close call
    evs2 = []
    for b in range(8):
        evs2.append(NoteEvent(b * 0.5, b * 0.5 + 0.2, 62, confidence=0.9))
        evs2.append(NoteEvent((b + 2 / 3) * 0.5, (b + 2 / 3) * 0.5 + 0.15, 64, confidence=0.9))
    vs2 = generate_variants(evs2, grid, _score(evs2, grid), y, 22050,
                            key="C", chords=[], title="t")
    assert not any(v.slug.startswith("feel-") for v in vs2), [v.slug for v in vs2]
