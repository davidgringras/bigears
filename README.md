# soloscribe

Audio in, Guitar Pro out — with an honest audit of itself.

Give it a recording of a guitar solo (MP3, WAV, M4A…) and it will isolate the
guitar from the band, work out the notes, find the beat, write real notation —
swing-aware, key-aware, with playable fingerings — into a `.gp5` file that
opens in Guitar Pro or TuxGuitar, and then **check its own work**: it
resynthesizes the transcription, lines it up against your recording, and hands
you a report saying exactly how much to trust each bar.

That last step is the point. Automatic transcription is imperfect on real
recordings; a tool that pretends otherwise wastes your time. This one measures
itself and says so.

## Quick start

```bash
bin/install.sh                      # one-time setup
open "bin/Start SoloScribe.command" # opens the app in your browser
```

Or from the command line:

```bash
.venv/bin/python -m soloscribe.cli solo.mp3 \
    --key F --bpm 152 --swing auto --chords "Gm7|C7|Fmaj7|Fmaj7" \
    --title "My Solo" --out output/my-solo
```

## What it does, honestly

- **Best case** (clear solo, guitar well ahead of the band): most notes right,
  rhythm notated sensibly, a usable chart after light cleanup in Guitar Pro or
  TuxGuitar. The audit report tells you which bars to check first.
- **Worst case** (guitar buried, heavy effects, rubato): the audit will say
  "low fidelity" — believe it. No tool transcribes what it cannot hear.
- Swing is notated the way real transcribers do it: straight eighths with a
  "swing feel" marking, not a fog of triplets. Genuine triplet figures still
  get triplet notation.
- Bends, vibrato and double-stops survive into the GP5 where detected;
  slides/hammer-ons and exact articulation are beyond it — that's the human
  cleanup pass.

Measured accuracy on synthesized ground-truth licks (where every note is
known): note-level F1 0.96–0.99 on clean solo renders, 0.90–0.97 with a
backing pad, 6/6 correct swing/straight calls — full table, methodology and
honest limits in `docs/VALIDATION.md`. Real recordings will land below the
synthetic numbers; each run's own report is the number that matters for that
clip.

## Pipeline

| Stage | Module | Method |
|---|---|---|
| Stem separation | `separate.py` | Demucs htdemucs_6s (guitar stem), MPS-accelerated |
| Note transcription | `transcribe.py` | Spotify basic-pitch (CoreML) + pyin contour cross-check |
| Beat tracking | `quantize.py` | librosa, user-BPM prior, drift-following grid |
| Quantization | `quantize.py` | per-beat subdivision inference, swing detection |
| Fingering | `fretting.py` | Viterbi DP over hand positions |
| GP5 writing | `gp5_writer.py` | PyGuitarPro, round-trip + MuseScore-validated |
| Self-audit | `synth.py`, `audit.py` | Karplus-Strong resynthesis, mir_eval F1, chroma DTW |

## Requirements

macOS (Apple Silicon), Homebrew Python 3.11, ffmpeg. Everything else installs
into `.venv` via `bin/install.sh`. First run downloads the Demucs model
(~300 MB, one-time).

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

`tests/licks.py` defines ground-truth licks (known note-for-note) that the
E2E suite renders to audio and pushes through the full pipeline — accuracy
claims here are measured, not vibes.
