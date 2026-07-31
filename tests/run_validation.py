"""Full-pipeline validation campaign against ground-truth licks.

Runs every lick x {none, pad} through run_pipeline (separation ON for pad
variants) and scores the resulting Score against the lick's ground truth.
Produces the table for docs/VALIDATION.md. Deterministic: seeds demucs's
shift RNG.
"""
import os, random, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import mir_eval.transcription as mt

from tests.licks import LICKS, lick_events
from soloscribe.pipeline import run_pipeline
from soloscribe.synth import merge_tied, score_note_seconds

def score_vs_truth(score, lick):
    truth = lick_events(lick)
    ref_i = np.array([[e.start, e.end] for e in truth])
    ref_p = np.array([440.0 * 2 ** ((e.pitch - 69) / 12) for e in truth])
    est = score_note_seconds(score)  # [(MergedNote, start_s, end_s)] via tempo map
    if not len(est):
        return 0.0, 0.0, 0.0
    est_i = np.array([[s, e] for _, s, e in est])
    est_p = np.array([440.0 * 2 ** ((n.pitch - 69) / 12) for n, _, _ in est])
    p, r, f, _ = mt.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p, onset_tolerance=0.1, offset_ratio=None)
    return p, r, f

rows = []
for lick in LICKS:
    for backing in ("none", "pad"):
        random.seed(20260731); np.random.seed(20260731)
        fx = f"tests/fixtures/{lick.name}__{backing}.wav"
        out = f"output/validation/{lick.name}_{backing}"
        res = run_pipeline(
            fx, out, key=lick.key, bpm=lick.bpm, downbeat=0.0,
            beats_per_bar=lick.beats_per_bar, swing="auto",
            separate="auto" if backing != "none" else "off",
            mode="solo", title=f"{lick.name} {backing}",
            progress=lambda *_: None,
        )
        p, r, f = score_vs_truth(res.score, lick)
        v = res.metrics.get("verdict", {})
        rows.append(dict(lick=lick.name, backing=backing, swing_truth=lick.swing,
                         swing_detected=res.score.triplet_feel,
                         precision=round(p,3), recall=round(r,3), f1=round(f,3),
                         self_f1=round(res.metrics.get("quantization",{}).get("f1_100ms",0),3),
                         verdict=v.get("level","?")))
        print(f"{lick.name:8s} {backing:5s}  P={p:.3f} R={r:.3f} F1={f:.3f}  "
              f"self-audit={rows[-1]['self_f1']:.3f} verdict={rows[-1]['verdict']} "
              f"swing {lick.swing}->{res.score.triplet_feel}")
json.dump(rows, open("output/validation/results.json", "w"), indent=1)
print("\nwrote output/validation/results.json")
