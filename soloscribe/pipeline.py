"""End-to-end pipeline orchestration.

This is the single entry point the CLI and web UI call. Stages:

  load → (separate) → transcribe → beat grid → quantize → fret → write GP5
       → synthesize score → audit vs original → HTML report

Interface is frozen; implementations land per-module.
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
    chords: str | None = None,       # 'Fmaj7|D7|Gm7 C7|F6' — bars split on |,
                                     # two chords in a bar split on space
    title: str = "",
    downbeat: float | None = None,   # seconds where beat 1 of bar 1 falls
    start: float | None = None,      # trim: analyze only [start, end] seconds
    end: float | None = None,
    progress: ProgressFn | None = None,
) -> PipelineResult:
    """Run the full audio → GP5 → audit pipeline. Implemented in integration."""
    raise NotImplementedError("wired up during integration")


def _report(progress: ProgressFn | None, stage: str, frac: float) -> None:
    if progress is not None:
        progress(stage, frac)
    else:
        print(f"[{stage}] {frac:.0%}")


def ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
