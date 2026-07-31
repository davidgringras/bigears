"""Command-line entry point.

    .venv/bin/python -m soloscribe.cli INPUT.mp3 --key F --bpm 152 --out output/solo
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="soloscribe",
        description="Transcribe a guitar solo from audio into a Guitar Pro (.gp5) file, "
        "then audit the result against the original recording.",
    )
    p.add_argument("audio", help="input audio file (mp3/wav/m4a/aiff/flac)")
    p.add_argument("--out", default="output/run", help="output directory")
    p.add_argument("--key", default="C", help="key, e.g. F, Bb, Em")
    p.add_argument("--bpm", type=float, default=None, help="tempo; omit to auto-detect")
    p.add_argument("--beats-per-bar", type=int, default=4, choices=(3, 4, 6))
    p.add_argument("--swing", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--separate", choices=("auto", "on", "off"), default="auto",
                   help="isolate the guitar with Demucs first")
    p.add_argument("--mode", choices=("solo", "poly"), default="solo")
    p.add_argument("--chords", default=None, help="chord per bar: 'Gm7|C7|Fmaj7|...'")
    p.add_argument("--title", default="")
    p.add_argument("--downbeat", type=float, default=None,
                   help="seconds where bar 1 beat 1 falls")
    p.add_argument("--start", type=float, default=None, help="trim start (seconds)")
    p.add_argument("--end", type=float, default=None, help="trim end (seconds)")
    args = p.parse_args(argv)

    from .pipeline import run_pipeline

    result = run_pipeline(
        args.audio,
        args.out,
        key=args.key,
        bpm=args.bpm,
        beats_per_bar=args.beats_per_bar,
        swing=args.swing,
        separate=args.separate,
        mode=args.mode,
        chords=args.chords,
        title=args.title,
        downbeat=args.downbeat,
        start=args.start,
        end=args.end,
    )
    print(f"\nGP5:    {result.gp5_path}")
    if result.report_path:
        print(f"Report: {result.report_path}")
    verdict = result.metrics.get("verdict", {})
    if verdict:
        print(f"Audit:  {verdict.get('level', '?')} fidelity — "
              + "; ".join(verdict.get("reasons", [])))
    for w in result.warnings:
        print(f"note: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
