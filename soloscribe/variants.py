"""Alternative readings, generated only where the pipeline made a
contestable call, scored so the listener chooses with numbers in hand.

Not N complete scores to diff by eye — that fails as an interface (variants
agree on most bars; finding the disagreements is the user's least favourite
job). Each variant here exists because a specific measured trigger fired:

  confident-only   ≥ MIN_DOUBTFUL notes fell below CONF_DOUBT — the class
                   the piano-roll colours pale (pad bleed, ghosts, marginal
                   double-stops). The alternate simply refuses them.
  feel-flip        the swing median landed inside the decision boundary's
                   grey zone, so the straight/swing call was genuinely close.
                   The alternate takes the other reading.

Scoring mirrors the audit's conventions (agreement F1 at ±100 ms against
what was heard; coverage of the clip's detected attacks at ±80 ms) without
the audit's rendering costs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import BeatGrid, NoteEvent, Score

CONF_DOUBT = 0.45
MIN_DOUBTFUL = 2
SWING_GREY = (0.53, 0.63)   # between the anchors: straight sits at 0.50, canonical swing at 0.667 — ambiguity is the gap between them, not the anchors themselves (0.66 IS swing, measured on the bebop fixture)


@dataclass
class Variant:
    slug: str
    description: str          # one plain sentence for the person choosing
    score: Score
    f1_100: float
    coverage: float
    n_notes: int


def _cheap_scores(score: Score, events: list[NoteEvent], y, sr: int) -> tuple[float, float]:
    import librosa
    import mir_eval.transcription as mt

    from .synth import score_note_seconds

    est = score_note_seconds(score)
    if not est or not events:
        return 0.0, 0.0
    ref_i = np.array([[e.start, e.end] for e in events])
    ref_p = np.array([440.0 * 2 ** ((e.pitch - 69) / 12) for e in events])
    est_i = np.array([[s, e] for _, s, e in est])
    est_p = np.array([440.0 * 2 ** ((n.pitch - 69) / 12) for n, _, _ in est])
    _, _, f1, _ = mt.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p, onset_tolerance=0.1, offset_ratio=None)

    detected = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    if len(detected) == 0:
        return float(f1), 1.0
    starts = np.array(sorted(s for _, s, _ in est))
    hits = sum(1 for t in detected if np.min(np.abs(starts - t)) <= 0.08)
    return float(f1), hits / len(detected)


def generate_variants(
    events: list[NoteEvent],
    grid: BeatGrid,
    main_score: Score,
    y,
    sr: int,
    *,
    key: str,
    chords: list[str | None],
    title: str,
) -> list[Variant]:
    from .fretting import assign_fretting
    from .quantize import detect_swing, quantize

    out: list[Variant] = []

    doubtful = [e for e in events if e.confidence < CONF_DOUBT]
    if len(doubtful) >= MIN_DOUBTFUL:
        kept = [e for e in events if e.confidence >= CONF_DOUBT]
        if kept:
            sc = quantize(kept, grid,
                          swing="on" if main_score.triplet_feel else "off",
                          key=key, chords=chords, title=title)
            assign_fretting(sc.qnotes, tuning=sc.tuning, capo=main_score.capo)
            sc.capo = main_score.capo
            f1, cov = _cheap_scores(sc, events, y, sr)
            out.append(Variant(
                slug="confident-only",
                description=(
                    f"the same reading minus the {len(doubtful)} notes I was "
                    "least sure of — cleaner if those were bleed or ghosts, "
                    "poorer if they were real"),
                score=sc, f1_100=f1, coverage=cov, n_notes=len(sc.qnotes)))

    swung, med = detect_swing(events, grid)
    if SWING_GREY[0] <= med <= SWING_GREY[1]:
        sc = quantize(events, grid,
                      swing="off" if main_score.triplet_feel else "on",
                      key=key, chords=chords, title=title)
        assign_fretting(sc.qnotes, tuning=sc.tuning, capo=main_score.capo)
        sc.capo = main_score.capo
        f1, cov = _cheap_scores(sc, events, y, sr)
        other = "straight" if main_score.triplet_feel else "swing"
        out.append(Variant(
            slug=f"feel-{other}",
            description=(
                f"the same notes read with a {other} feel — the swing "
                f"evidence sat near the boundary (median {med:.2f}), so "
                "your ears should make this call"),
            score=sc, f1_100=f1, coverage=cov, n_notes=len(sc.qnotes)))

    return out
