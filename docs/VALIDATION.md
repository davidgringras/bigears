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
| blues_a | 4 bars, straight 8ths/16ths + qtr-triplet, 92 bpm | none | 1.000 | 0.964 | 0.982 | correct |
| blues_a | | chord pad | 0.926 | 0.893 | 0.909 | correct |
| funk_e | 4 bars, syncopated straight 16ths, 104 bpm | none | 0.974 | 0.950 | 0.962 | correct |
| funk_e | | chord pad | 0.881 | 0.925 | 0.902 | correct |

Swing detection called all six cases correctly, including refusing to stamp
"swing" on the straight funk line whose 0.75-position sixteenths would fool a
naive offbeat-median test.

Quantization in isolation (ground-truth events, perfect grid): 126/126 onsets
exact across all three licks, including triplet figures inside swing context.

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
- No ground-truth lick contains octave double-stops; the transcriber's
  octave-ghost filter is untested against them (flagged by its author).
- Bends and vibrato are detected and written to the GP5, but no ground-truth
  lick exercises them end-to-end; the audit's resynthesis ignores bend
  contours (scores are pessimistic on heavily bent phrases, and say so).
- Time signatures beyond 4/4 and 3/4 are best-effort (6/8 is handled as six
  beats per bar).
- MPS acceleration is verified for the test separation model only; the
  production models fall back to CPU on any MPS failure (slower, not wrong).
- Separation is seeded for reproducibility in validation runs; interactive
  runs may vary ±1 note between repeats (demucs draws a random shift).
