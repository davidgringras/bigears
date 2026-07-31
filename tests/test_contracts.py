"""Public-surface pins: cross-module contracts drift only deliberately.

During the build, two interfaces labelled frozen were extended by concurrent
sessions — compatibly, but noticed only because someone happened to re-read
the file. This test is the mechanism that replaces that luck: any change to
a pinned dataclass field list or public signature fails here and must be
made deliberately, with this file updated in the same commit (which is the
reviewable record that every consumer was considered).

Additions land at the END of a signature with a default, or at the end of a
dataclass — that keeps existing call sites working, and the pin update
documents the extension.
"""
import inspect
from dataclasses import fields

from soloscribe import model
from soloscribe.audit import audit
from soloscribe.fretting import assign_fretting
from soloscribe.gp5_writer import write_gp5
from soloscribe.pipeline import PipelineResult, run_pipeline
from soloscribe.quantize import build_beat_grid, quantize
from soloscribe.synth import render_score
from soloscribe.transcribe import transcribe

PINNED_FIELDS = {
    model.NoteEvent: ["start", "end", "pitch", "velocity", "confidence", "bend", "vibrato"],
    model.BeatGrid: ["beat_times", "beats_per_bar", "first_downbeat", "bpm_nominal", "swing"],
    model.QNote: ["onset", "duration", "pitch", "velocity", "string", "fret",
                  "tied_from_prev", "vibrato", "bend", "confidence"],
    model.Score: ["qnotes", "beats_per_bar", "denominator", "tempo_bpm", "triplet_feel",
                  "key", "tempo_map", "chords", "tuning", "capo", "title", "subtitle"],
    PipelineResult: ["gp5_path", "report_path", "score", "events", "grid",
                     "metrics", "stem_path", "warnings"],
}

PINNED_SIGNATURES = {
    transcribe: ["audio_path", "mode", "min_pitch", "max_pitch", "stats"],
    quantize: ["events", "grid", "swing", "key", "chords", "title", "min_cell"],
    build_beat_grid: ["y", "sr", "bpm", "downbeat", "beats_per_bar", "cover_until"],
    assign_fretting: ["qnotes", "tuning", "capo", "max_fret"],
    write_gp5: ["score", "path"],
    audit: ["score", "events", "original_path", "stem_path", "out_dir", "caveats"],
    render_score: ["score", "sr", "use_tempo_map"],
    run_pipeline: ["audio_path", "out_dir", "key", "bpm", "beats_per_bar", "swing",
                   "separate", "mode", "chords", "title", "downbeat", "start", "end",
                   "progress"],
}


def test_dataclass_fields_pinned():
    for cls, expected in PINNED_FIELDS.items():
        actual = [f.name for f in fields(cls)]
        assert actual == expected, (
            f"{cls.__name__} fields changed: {actual} != pinned {expected}. "
            "If deliberate, update tests/test_contracts.py in the same commit."
        )


def test_public_signatures_pinned():
    for fn, expected in PINNED_SIGNATURES.items():
        actual = list(inspect.signature(fn).parameters)
        assert actual == expected, (
            f"{fn.__module__}.{fn.__name__} signature changed: {actual} != "
            f"pinned {expected}. If deliberate, update tests/test_contracts.py "
            "in the same commit."
        )


def test_ticks_per_quarter_pinned():
    # Every tick computation in the tree assumes this; 960 divides cleanly by
    # 2, 3, 4, 6, 8 which is what makes the duration table expressible.
    assert model.TICKS_PER_QUARTER == 960
