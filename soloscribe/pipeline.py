"""End-to-end pipeline orchestration.

This is the single entry point the CLI and web UI call. Stages:

  load → (separate) → transcribe → beat grid → quantize → fret → write GP5
       → synthesize score → audit vs original → HTML report

The beat grid is tracked on the FULL mix (the rhythm section carries the
pulse); note transcription runs on the separated guitar stem when separation
is active. Audit compares the finished score against both.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from .model import BeatGrid, NoteEvent, Score

ProgressFn = Callable[[str, float], None]  # (stage_name, fraction 0..1)


@dataclass
class PipelineResult:
    gp5_path: str
    report_path: str | None
    score: Score
    events: list[NoteEvent]          # raw transcription, pre-quantization
    grid: BeatGrid
    metrics: dict = field(default_factory=dict)  # audit metrics, see audit.py
    stem_path: str | None = None     # separated guitar stem, if separation ran
    warnings: list[str] = field(default_factory=list)


def run_pipeline(
    audio_path: str,
    out_dir: str,
    *,
    key: str = "C",
    bpm: float | None = None,        # None → auto beat-track
    beats_per_bar: int = 4,
    swing: str = "auto",             # "auto" | "on" | "off"
    separate: str = "auto",          # "auto" | "on" | "off"
    mode: str = "solo",              # "solo" (monophonic-biased) | "poly"
    chords: str | None = None,       # 'Fmaj7|D7|Gm7 C7|F6' — bars split on |
    title: str = "",
    downbeat: float | None = None,   # seconds where beat 1 of bar 1 falls
    start: float | None = None,      # trim: analyze only [start, end] seconds
    end: float | None = None,
    capo: int = 0,                   # frets; fingering and tab become capo-relative
    variants: bool = True,           # alternates where a contestable call fired
    progress: ProgressFn | None = None,
) -> PipelineResult:
    import librosa
    import soundfile as sf

    from .quantize import build_beat_grid, parse_chords, quantize
    from .fretting import assign_fretting
    from .gp5_writer import write_gp5

    ensure_out_dir(out_dir)
    warnings: list[str] = []

    _report(progress, "Listening", 0.02)
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    if start or end:
        s = int((start or 0.0) * sr)
        e = int(end * sr) if end else len(y)
        y = y[max(0, s):min(len(y), e)]
        if downbeat is not None and start:
            downbeat = max(0.0, downbeat - start)
    working = os.path.join(out_dir, "clip.wav")
    sf.write(working, y, sr)
    duration = len(y) / sr
    _report(progress, "Listening", 0.06)

    # Stem separation — transcribe the guitar, not the band.
    stem_path: str | None = None
    if separate in ("auto", "on"):
        _report(progress, "Isolating the guitar", 0.08)
        try:
            from .separate import separate as run_separation

            sep = run_separation(working, out_dir)
            stem_path = sep.stem_path
            if separate == "auto" and sep.stem_rel_rms >= 0.90:
                # Stem ≈ mix: the clip is already isolated guitar; separation
                # could only add artifacts.
                warnings.append(
                    "the clip already sounds like guitar on its own, so I "
                    "worked from your original recording"
                )
                stem_path = None
        except Exception as exc:  # separation is an enhancement, never a wall
            warnings.append(
                "I couldn't isolate the guitar (a technical hiccup on my "
                f"side: {type(exc).__name__}), so I listened to the full "
                "recording instead"
            )
        _report(progress, "Isolating the guitar", 0.35)

    _report(progress, "Working out the notes", 0.38)
    from .transcribe import transcribe

    # Separation can DESTROY the solo (a stem model that mis-classifies the
    # lead loses half the notes) — and a self-audit against that stem would
    # never know, because the missing notes are missing from its reference
    # too. Guard: transcribe both sources and keep the one whose notes
    # account for more of the ORIGINAL clip's audible attacks. Found by the
    # ground-truth validation campaign (stem recall 0.38 vs 0.95, while the
    # stem-referenced audit read "high fidelity").
    events: list[NoteEvent] = []
    stem_lost_notes = False
    if stem_path is not None:
        stem_events = transcribe(stem_path, mode=mode)
        _report(progress, "Working out the notes", 0.50)
        orig_events = transcribe(working, mode=mode)
        cov_stem = _onset_coverage_of(y, sr, stem_events)
        cov_orig = _onset_coverage_of(y, sr, orig_events)
        if separate == "on" or cov_stem >= cov_orig - 0.08:
            events = stem_events
            if cov_stem < cov_orig - 0.08:  # forced stem, but it lost notes
                warnings.append(
                    "isolating the guitar seems to have lost some notes "
                    f"(the isolated track accounts for {cov_stem:.0%} of the "
                    f"recording's attacks vs {cov_orig:.0%} without) — "
                    "consider re-running with isolation set to Skip"
                )
                stem_lost_notes = True
        else:
            events = orig_events
            warnings.append(
                "isolation lost some of the guitar, so I worked from your "
                "original recording instead "
                f"({cov_orig:.0%} vs {cov_stem:.0%} of attacks accounted for)"
            )
            stem_path = None
    else:
        events = transcribe(working, mode=mode)
    if not events:
        warnings.append(
            "I couldn't detect any notes — the clip may be very quiet, or "
            "try switching 'Isolate the guitar first?' to Skip"
        )
    _report(progress, "Working out the notes", 0.62)

    _report(progress, "Finding the beat", 0.64)
    grid = build_beat_grid(
        y, sr, bpm=bpm, downbeat=downbeat, beats_per_bar=beats_per_bar,
        cover_until=duration, onsets=[e.start for e in events],
        durations=[e.end - e.start for e in events],
    )
    _report(progress, "Finding the beat", 0.70)

    _report(progress, "Writing the notation", 0.72)
    chord_list = parse_chords(chords)
    score = quantize(
        events, grid, swing=swing, key=key, chords=chord_list, title=title,
    )
    score.capo = max(0, min(12, int(capo)))
    fret_warnings = assign_fretting(score.qnotes, tuning=score.tuning, capo=score.capo)
    warnings.extend(fret_warnings)
    gp5_path = os.path.join(out_dir, _safe_name(title) + ".gp5")
    warnings.extend(write_gp5(score, gp5_path))
    _report(progress, "Writing the notation", 0.80)

    if mode == "solo":
        # Unconditional, because it is undetectable: a suppressed octave
        # doubling leaves a clean single-note line with no measurable trace
        # (fold statistics were measured and do not separate the classes).
        # The user knows whether the solo uses octaves; we cannot.
        warnings.append(
            "transcribed in single-note mode — if this solo uses octave "
            "doubling (Wes Montgomery-style), the doubling will have been "
            "written as a single line; re-run with 'Chords too' to keep it"
        )
    # Also undetectable, measured on real two-guitar mixes (the transcription
    # was faithful to BOTH instruments; no metric can tell whose notes they
    # are without knowing which part the listener wanted):
    if stem_path is not None:
        warnings.append(
            "the band was isolated away first — but a second guitar or other "
            "instrument in the same register would have stayed with the solo, "
            "and its notes would be written down too"
        )
    else:
        warnings.append(
            "everything pitched in the clip gets written down — if another "
            "instrument shares it, its notes are in the tab too, and the "
            "checks below cannot tell whose notes are whose"
        )

    # Alternative readings, only where a measured trigger says the main
    # reading made a genuinely contestable call (see variants.py).
    variant_files: list[dict] = []
    if variants and events:
        try:
            from .variants import generate_variants

            for vr in generate_variants(events, grid, score, y, sr,
                                        key=key, chords=chord_list, title=title):
                vpath = os.path.join(out_dir, _safe_name(title) + f"-{vr.slug}.gp5")
                write_gp5(vr.score, vpath)
                variant_files.append({
                    "slug": vr.slug, "description": vr.description,
                    "file": vpath, "f1_100": round(vr.f1_100, 3),
                    "coverage": round(vr.coverage, 3), "n_notes": vr.n_notes,
                })
            if variant_files:
                warnings.append(
                    f"I also wrote {len(variant_files)} alternative "
                    "reading(s) of close calls — each is scored, so compare "
                    "before you choose"
                )
        except Exception:
            pass  # alternates are a bonus, never a wall

    _report(progress, "Checking my work against your recording", 0.82)
    report_path: str | None = None
    metrics: dict = {}
    try:
        from .audit import audit

        metrics = audit(score, events, working, stem_path, out_dir, caveats=warnings)
        # The audit references the stem when one is passed; a stem that lost
        # notes makes its agreement numbers flattering. Cap the verdict —
        # missing-from-the-reference is invisible to the audit by
        # construction, so the pipeline must carry this knowledge in.
        verdict = metrics.get("verdict") or {}
        if stem_lost_notes and verdict.get("level") == "high":
            verdict["level"] = "medium"
            verdict.setdefault("reasons", []).append(
                "guitar isolation lost some notes, so the comparison "
                "flatters the transcription — treat missing notes as likely"
            )
        report_path = os.path.join(out_dir, "report.html")
        if not os.path.exists(report_path):
            report_path = metrics.get("report_path")
    except Exception as exc:
        warnings.append(
            f"I couldn't finish checking my own work ({type(exc).__name__}) "
            "— the Guitar Pro file was still written"
        )
    _report(progress, "Checking my work against your recording", 1.0)

    metrics["variants"] = variant_files
    return PipelineResult(
        gp5_path=gp5_path,
        report_path=report_path,
        score=score,
        events=events,
        grid=grid,
        metrics=metrics,
        stem_path=stem_path,
        warnings=warnings,
    )


def _onset_coverage_of(y, sr: int, events: list[NoteEvent], tol: float = 0.08) -> float:
    """Fraction of the clip's detected attacks matched by an event onset."""
    import librosa
    import numpy as np

    if not events:
        return 0.0
    detected = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    if len(detected) == 0:
        return 1.0
    starts = np.array(sorted(e.start for e in events))
    hits = 0
    for t in detected:
        i = np.searchsorted(starts, t)
        near = min(
            (abs(starts[j] - t) for j in (i - 1, i) if 0 <= j < len(starts)),
            default=np.inf,
        )
        if near <= tol:
            hits += 1
    return hits / len(detected)


def _safe_name(title: str) -> str:
    keep = "".join(c if c.isalnum() or c in " -_" else "" for c in (title or "solo"))
    return (keep.strip().replace(" ", "-") or "solo")[:60]


def _report(progress: ProgressFn | None, stage: str, frac: float) -> None:
    if progress is not None:
        progress(stage, frac)
    else:
        print(f"[{stage}] {frac:.0%}")


def ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
