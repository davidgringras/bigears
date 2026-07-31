"""Audit self-consistency and sensitivity.

Two questions, and the second matters more than the first:

  1. Fed its own output, does the audit agree with itself? If resynthesizing a
     score and auditing that score against its own render does not come out
     near-perfect, the measurement apparatus is broken and every number it has
     ever printed is noise.

  2. Fed something it should object to, does it object? A gauge pinned at
     "excellent" is indistinguishable from a working one until you perturb the
     input. The 120 ms shift below is deliberately larger than both onset
     tolerances so the drop is unambiguous rather than marginal.
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soloscribe.audit import audit
from soloscribe.model import TICKS_PER_QUARTER, NoteEvent, QNote, Score
from soloscribe.synth import DEFAULT_SR, render_score, score_note_seconds

TEMPO = 120.0
QUARTER_SEC = 60.0 / TEMPO           # 0.5 s
BAR_TICKS = TICKS_PER_QUARTER * 4


def _build_score() -> Score:
    """Two bars, ten notes, an unambiguous tempo map, one tie chain.

    Notes are a quarter or longer and never closer than a quarter note apart,
    so a 120 ms perturbation lands squarely between grid positions and cannot
    accidentally match a neighbour.
    """
    pitches = [64, 67, 71, 74, 72, 69, 65, 62, 60, 67]
    qnotes: list[QNote] = []
    tick = 0
    for i, pitch in enumerate(pitches):
        dur = TICKS_PER_QUARTER
        qnotes.append(
            QNote(
                onset=tick,
                duration=dur,
                pitch=pitch,
                velocity=88 + (i % 4) * 8,
                confidence=0.65 + 0.03 * (i % 5),
            )
        )
        tick += dur
    # 10 quarters = 2.5 bars of 4/4; trim to a clean 2 bars by tying the last
    # note across the barline instead of leaving a ragged tail.
    qnotes = qnotes[:8]
    qnotes.append(
        QNote(onset=8 * TICKS_PER_QUARTER, duration=TICKS_PER_QUARTER,
              pitch=60, velocity=100, confidence=0.9)
    )
    qnotes.append(
        QNote(onset=9 * TICKS_PER_QUARTER, duration=TICKS_PER_QUARTER,
              pitch=60, velocity=100, confidence=0.9, tied_from_prev=True)
    )
    # Anchors every bar; strictly linear at 120 bpm, so tick_to_seconds is
    # exactly tick/960 * 0.5 and the test's own arithmetic can check it.
    tempo_map = [
        (bar * BAR_TICKS, bar * BAR_TICKS / TICKS_PER_QUARTER * QUARTER_SEC)
        for bar in range(4)
    ]
    return Score(
        qnotes=qnotes,
        tempo_bpm=TEMPO,
        tempo_map=tempo_map,
        title="audit self-test",
    )


def _events_from_score(score: Score) -> list[NoteEvent]:
    """Perfect raw transcription: exactly what the score says, in seconds."""
    return [
        NoteEvent(start=t0, end=t1, pitch=n.pitch, velocity=n.velocity,
                  confidence=n.confidence)
        for n, t0, t1 in score_note_seconds(score, use_tempo_map=True)
    ]


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    score = _build_score()
    y = render_score(score, sr=DEFAULT_SR, use_tempo_map=True)
    wav = tmp_path_factory.mktemp("audio") / "render.wav"
    sf.write(str(wav), y, DEFAULT_SR, subtype="PCM_16")
    return score, _events_from_score(score), str(wav)


@pytest.fixture(scope="module")
def clean(rendered, tmp_path_factory):
    score, events, wav = rendered
    out = tmp_path_factory.mktemp("clean")
    return audit(score, events, wav, None, str(out)), str(out)


# -- 1. tie merging is the shared premise of everything below ---------------

def test_ties_merge_into_one_sustained_note(rendered):
    score, events, _ = rendered
    merged = score_note_seconds(score, use_tempo_map=True)
    assert len(merged) == 9, "the tied pair must collapse to a single note"
    last, t0, t1 = merged[-1]
    assert last.pitch == 60
    assert last.duration == 2 * TICKS_PER_QUARTER
    assert t1 - t0 == pytest.approx(2 * QUARTER_SEC, abs=1e-6)
    assert len(events) == 9


# -- 2. self-consistency ---------------------------------------------------

def test_self_consistency_scores_near_perfect(clean):
    metrics, _ = clean
    q = metrics["quantization"]
    cov = metrics["coverage"]
    assert q["f1_50ms"] >= 0.95, q
    assert q["f1_100ms"] >= 0.95, q
    assert cov["onset_coverage"] >= 0.9, cov
    assert metrics["verdict"]["level"] == "high", metrics["verdict"]


def test_self_consistency_alignment_is_diagonal(clean):
    metrics, _ = clean
    c = metrics["chroma_dtw"]
    # Auditing a render against itself: the warping path has no reason to leave
    # the diagonal, and the cost has no reason to be nonzero.
    assert c["normalized_cost"] is not None
    assert c["normalized_cost"] < 1e-6, c
    assert c["diagonality_sec"] < 0.05, c


def test_per_measure_covers_every_bar(clean):
    metrics, _ = clean
    rows = metrics["per_measure"]
    assert [r["measure"] for r in rows] == list(range(1, len(rows) + 1))
    assert sum(r["n_notes"] for r in rows) == 9
    for r in rows:
        if r["n_notes"]:
            assert r["mean_confidence"] is not None
            # events are the score, so quantization moved nothing
            assert r["rms_quantization_error_ms"] == pytest.approx(0.0, abs=1.0)


# -- 3. perturbation: the gauge must move ----------------------------------

def test_shifted_events_degrade_scores_and_verdict(rendered, clean, tmp_path):
    score, events, wav = rendered
    clean_metrics, _ = clean

    shifted = [
        replace(ev, start=ev.start + 0.120, end=ev.end + 0.120) if i % 2 == 0
        else replace(ev)
        for i, ev in enumerate(events)
    ]
    assert sum(1 for a, b in zip(events, shifted) if a.start != b.start) == 5

    out = tmp_path / "perturbed"
    metrics = audit(score, shifted, wav, None, str(out))

    clean_f1 = clean_metrics["quantization"]["f1_50ms"]
    dirty_f1 = metrics["quantization"]["f1_50ms"]
    assert dirty_f1 < clean_f1 - 0.30, (clean_f1, dirty_f1)
    assert metrics["quantization"]["f1_100ms"] < clean_metrics["quantization"]["f1_100ms"] - 0.30

    levels = {"high": 2, "medium": 1, "low": 0}
    assert levels[metrics["verdict"]["level"]] < levels[clean_metrics["verdict"]["level"]]

    # The shifted half must show up as real timing error, not be absorbed.
    worst = max(
        (r["rms_quantization_error_ms"] or 0.0) for r in metrics["per_measure"]
    )
    assert worst > 50.0, metrics["per_measure"]


def test_dropped_notes_degrade_coverage(rendered, clean, tmp_path):
    """Coverage is measured against the audio, so it must fall when the score
    loses notes — independently of the raw-events comparison."""
    score, events, wav = rendered
    thin = Score(
        qnotes=[q for i, q in enumerate(score.qnotes) if i % 2 == 0],
        tempo_bpm=score.tempo_bpm,
        tempo_map=score.tempo_map,
        title=score.title,
    )
    metrics = audit(thin, events, wav, None, str(tmp_path / "thin"))
    assert metrics["coverage"]["onset_coverage"] < clean[0]["coverage"]["onset_coverage"]
    assert metrics["verdict"]["level"] != "high"


# -- 4. the report itself --------------------------------------------------

def test_report_is_written_and_self_contained(clean):
    metrics, out_dir = clean
    path = os.path.join(out_dir, "report.html")
    assert path == metrics["report_path"]
    assert os.path.exists(path)
    text = open(path, encoding="utf-8").read()

    assert metrics["verdict"]["level"] in text
    assert f"data-verdict='{metrics['verdict']['level']}'" in text
    assert "<audio" in text
    assert text.count("<audio") >= 2, "original and render must both be playable"
    assert "data:image/png;base64," in text
    # self-contained: nothing may be fetched at open time
    for token in ("http://", "https://", "<script"):
        assert token not in text, f"report reaches outside itself via {token}"
    for caveat_marker in ("Bends and vibrato", "Guitar Pro"):
        assert caveat_marker in text
    assert os.path.exists(metrics["render_path"])
    assert os.path.exists(metrics["piano_roll_path"])


def test_caveats_include_passed_in_and_standard(rendered, tmp_path):
    score, events, wav = rendered
    metrics = audit(
        score, events, wav, None, str(tmp_path / "cav"),
        caveats=["Tempo was supplied by hand, not detected."],
    )
    assert "Tempo was supplied by hand, not detected." in metrics["caveats"]
    assert any("Swing is notated straight" in c for c in metrics["caveats"])
    text = open(metrics["report_path"], encoding="utf-8").read()
    assert "Tempo was supplied by hand, not detected." in text


def test_metrics_dict_keys_are_stable(clean):
    """Integration reads these paths; a rename here is a breaking change."""
    metrics, _ = clean
    assert set(metrics["quantization"]) >= {
        "f1_50ms", "f1_100ms", "precision_50ms", "recall_50ms",
    }
    assert set(metrics["chroma_dtw"]) >= {"normalized_cost", "diagonality_sec"}
    assert set(metrics["coverage"]) >= {"onset_coverage", "onset_spuriousness"}
    assert set(metrics["verdict"]) == {"level", "reasons"}
    assert set(metrics["per_measure"][0]) == {
        "measure", "n_notes", "mean_confidence", "rms_quantization_error_ms",
    }
    assert metrics["comparison_source"] == "original"


def test_empty_score_does_not_crash_and_is_not_high(rendered, tmp_path):
    """An empty transcription is the one case where overselling is unforgivable."""
    _, events, wav = rendered
    empty = Score(qnotes=[], tempo_bpm=TEMPO, tempo_map=_build_score().tempo_map)
    metrics = audit(empty, events, wav, None, str(tmp_path / "empty"))
    assert metrics["verdict"]["level"] == "low"
    assert metrics["quantization"]["f1_100ms"] == 0.0
    assert os.path.exists(metrics["report_path"])


def test_spuriousness_polarity(rendered, tmp_path):
    """Spuriousness must rise when the score invents notes, not fall."""
    score, events, wav = rendered
    invented = list(score.qnotes) + [
        QNote(onset=t, duration=240, pitch=77, velocity=90, confidence=0.2)
        for t in range(120, 9 * TICKS_PER_QUARTER, 480)
    ]
    noisy = Score(qnotes=invented, tempo_bpm=TEMPO, tempo_map=score.tempo_map)
    metrics = audit(noisy, events, wav, None, str(tmp_path / "noisy"))
    base = audit(score, events, wav, None, str(tmp_path / "base"))
    assert (
        metrics["coverage"]["onset_spuriousness"]
        > base["coverage"]["onset_spuriousness"]
    )
