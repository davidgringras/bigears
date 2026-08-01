# Measured accuracy

Every number here is measured against synthesized ground-truth licks — test
phrases defined note-for-note in `tests/licks.py`, rendered to audio with an
independent Karplus-Strong synth (verified in tune to ≤0.18 cents), and pushed
through the complete pipeline exactly as a user would run it. Nothing below is
an estimate.

Run it yourself: `.venv/bin/python tests/run_validation.py` (results land in
`output/validation/results.json`). The pytest suite (`tests/test_e2e.py`)
gates a subset of these numbers on every test run.

## Full pipeline, default settings (auto separation, auto swing)

Note-level F1 against ground truth, onset tolerance 100 ms, pitch-matched
(mir_eval `precision_recall_f1_overlap`, `offset_ratio=None`):

| lick | material | backing | P | R | F1 | swing call |
|---|---|---|---|---|---|---|
| bebop_f | 8 bars, swung 8ths + triplets, 150 bpm | none | 1.000 | 0.983 | 0.991 | correct |
| bebop_f | | chord pad | 0.966 | 0.983 | 0.974 | correct |
| bebop_f | | pad + noise | 0.950 | 0.983 | 0.966 | correct |
| blues_a | 4 bars, straight 8ths/16ths + qtr-triplet, 92 bpm | none | 1.000 | 0.964 | 0.982 | correct |
| blues_a | | chord pad | 0.929 | 0.929 | 0.929 | correct |
| blues_a | | pad + noise | 0.900 | 0.964 | 0.931 | correct |
| funk_e | 4 bars, syncopated straight 16ths, 104 bpm | none | 0.950 | 0.950 | 0.950 | correct |
| funk_e | | chord pad | 0.860 | 0.925 | 0.892 | correct |
| funk_e | | pad + noise | 0.860 | 0.925 | 0.892 | correct |

A re-articulation splitter now recovers repeated fast notes that reach the
transcriber as one fused activation (its residual recall class): it splits
only where basic-pitch's own onset head fires ≥0.80 at the note's pitch bin
AND audio energy rises through the claimed re-attack (measured: true re-picks
1.07–1.22 energy ratio, false-split sites 0.90 — ratio 1.00 is the physical
boundary; full measurement table in transcribe.py). On the sampled-guitar
demo this cut missed notes from 3 to 2 of 58 with zero invented notes. On
the Karplus-Strong rows above the effect is mixed within ±0.02 — that
renderer's ring-down blurs the energy boundary — and per policy the
thresholds will only be retuned against real-guitar evidence, not another
synthetic curve.

Swing detection called all nine cases correctly, including refusing to stamp
"swing" on the straight funk line whose 0.75-position sixteenths would fool a
naive offbeat-median test. Added pink noise costs only a few F1 points over
the pad alone.

Quantization in isolation (ground-truth events, perfect grid): 126/126 onsets
exact across all three licks, including triplet figures inside swing context.

## Real audio (GuitarSet)

The first contact with real recordings — GuitarSet v1.1.0, real close-mic'd
acoustic guitar with note-level ground truth — is written up in full in
[REAL-AUDIO-BENCH.md](REAL-AUDIO-BENCH.md). Headlines, cold start (no key or
tempo hints): **median F1@100ms 0.850 on solo material** (rock 0.887, bossa
0.884, jazz 0.860, funk 0.792, strummy singer-songwriter 0.660 — the last is
not single-note material); chordal comping 0.47 (dense strums exceed the
transcriber's simultaneity capacity: precision 0.85, recall 0.20 on the worst
case); constructed two-guitar mixes fail by PRECISION — the pipeline
faithfully transcribes both guitars and cannot know which one was wanted, and
instrument-class separation cannot split guitar from guitar. The audit now
carries a standing caveat saying exactly that. Real-world advice these
numbers support: clips where the solo carries the register do well; a second
guitar in the same range is the honest frontier.

## What the self-audit is, and is not

Each run's report scores agreement between the written score and what the
transcriber heard, plus coverage of the recording's audible attacks. It
CANNOT see notes the transcriber never heard. The validation campaign caught
exactly this failure: stem separation destroyed half a solo, and the audit —
referenced against the same impoverished stem — read "high fidelity" while
true recall was 0.38. Two guards now exist:

1. In auto mode the pipeline transcribes both the stem and the original and
   keeps whichever accounts for more of the original clip's attacks
   (bebop-with-pad F1: 0.581 → 0.974 from this guard alone).
2. If the user forces separation and the stem measurably lost notes, the
   verdict is capped at "medium" with an explanatory reason.

## Honest limits of this evidence

- **All test material is synthesized.** The Karplus-Strong timbre is
  guitar-like, not guitar. No real recording was ground-truth-scored in this
  validation (there is no way to know the true notes of a real recording
  without hand-transcribing it first). Expect real-world numbers below these,
  especially on dense mixes; the per-run report is the number that matters.
- Demucs' 6-stem model did not recognize the synthetic timbre as "guitar"
  (rel-RMS fallback to the 4-stem model fired). Its behavior on real guitar
  is untested here, in either direction.
- **Octave double-stops (Wes-style octave melodies) collapse to single notes
  in solo mode** — measured on a dedicated octave-melody fixture: doubling
  recall 0.000, melody recall 1.000 (the line survives; the texture doesn't).
  This is a deliberate trade: the same amplitude filter that removes harmonic
  ghosts (funk precision 0.826 → 0.974) cannot distinguish them from played
  octaves, which reach the transcriber at ghost-like relative amplitude
  (0.52× measured at equal played gain), and duration distributions overlap
  with the ghost median higher. **The workaround ships in the product:
  "Chords too" (poly) mode recovers 0.682 of octave doublings** (measured,
  same fixture) at some cost in single-line cleanliness. A test pins the
  trade so it cannot move silently.
- Bends and vibrato are detected and written to the GP5, but no ground-truth
  lick exercises them end-to-end; the audit's resynthesis ignores bend
  contours (scores are pessimistic on heavily bent phrases, and say so).
- Time signatures beyond 4/4 and 3/4 are best-effort (6/8 is handled as six
  beats per bar).
- MPS acceleration is verified for the production separation model: a direct
  probe ran htdemucs_6s (the HTDemucs transformer) on MPS with finite 6-source
  output, and the validation runs exercised its "guitar" stem by name (the
  rel-RMS fallback lines in the logs are that stem being read and measured).
  A CPU fallback remains for any MPS failure (slower, not wrong).
- Separation is seeded for reproducibility in validation runs; interactive
  runs may vary ±1 note between repeats (demucs draws a random shift).
