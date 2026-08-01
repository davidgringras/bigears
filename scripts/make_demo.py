"""Rebuild the landing-page demo end-to-end, gated by the acceptance harness.

  .venv/bin/python scripts/make_demo.py

Pipeline: render ground-truth bebop on sampled guitar (A) → transcribe that
audio → render what was heard, same voice, calibrated (B) → acceptance
harness → write site assets only if ALL PASS. Every calibration is measured
per run (fluidsynth lag, onset-convention gap), never assumed — assuming a
previously-measured lag was this demo's second shipped defect.
"""
import json
import os
import pickle
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

SF2 = "/Applications/MuseScore 4.app/Contents/Resources/sound/MS Basic.sf3"
OUT = "output/demo4"
SR = 22050


def fluid(midi_path, wav_path):
    subprocess.run(["fluidsynth", "-ni", "-F", wav_path, "-r", str(SR), "-g", "0.9",
                    SF2, midi_path], capture_output=True, check=True)


def mono(path):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(1)
    return y.astype(float), sr


def measured_lag(y, sr, midi_starts):
    on = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)
    return float(np.median([on[np.argmin(np.abs(on - t))] - t for t in midi_starts]))


def events_to_midi(evs, path, articulate=True):
    starts = [e.start for e in evs]
    pm = pretty_midi.PrettyMIDI(initial_tempo=150)
    inst = pretty_midi.Instrument(program=26)
    for i, e in enumerate(evs):
        dur = e.end - e.start
        if articulate:
            gap = (starts[i + 1] - e.start) if i + 1 < len(evs) else dur
            dur = min(dur, max(0.07, 0.85 * gap))
        inst.notes.append(pretty_midi.Note(
            velocity=int(np.clip(e.velocity, 20, 127)), pitch=e.pitch,
            start=float(e.start), end=float(e.start + max(dur, 0.06))))
    pm.instruments.append(inst)
    pm.write(path)


def main():
    random.seed(20260731)
    np.random.seed(20260731)
    os.makedirs(OUT, exist_ok=True)
    import tests.licks as TL
    from soloscribe.pipeline import run_pipeline
    import tests.demo_acceptance as ACC

    truth = sorted(TL.lick_events(TL.LICKS[0]), key=lambda e: e.start)

    # A: the recording = ground truth performance, sampled guitar
    events_to_midi(truth, f"{OUT}/truth.mid", articulate=False)
    fluid(f"{OUT}/truth.mid", f"{OUT}/recording_raw.wav")
    a, sr = mono(f"{OUT}/recording_raw.wav")
    lag_a = measured_lag(a, sr, [e.start for e in truth])
    a = a[max(0, int(lag_a * sr)):]
    a *= 0.9 / np.abs(a).max()
    L = int((max(e.end for e in truth) + 0.9) * sr)
    a = np.pad(a, (0, max(0, L - len(a))))[:L]
    sf.write(f"{OUT}/recording.wav", a.astype(np.float32), sr)
    print(f"A rendered (lag {lag_a*1000:+.0f}ms)")

    # Transcribe THAT audio; full pipeline for the chart/report artifacts
    res = run_pipeline(
        f"{OUT}/recording.wav", f"{OUT}/run",
        key="F", bpm=150, downbeat=0.0, swing="auto", separate="off", mode="solo",
        chords="Gm7|C7|Fmaj7|Fmaj7|Gm7|C7|Fmaj7|Fmaj7",
        title="Bebop demo", progress=lambda *_: None)
    events = sorted(res.events, key=lambda e: e.start)
    pickle.dump(events, open(f"{OUT}/events.pkl", "wb"))
    print(f"transcribed {len(events)} events vs {len(truth)} truth")

    # B: what it heard, same voice, calibrated
    events_to_midi(events, f"{OUT}/heard.mid", articulate=True)
    fluid(f"{OUT}/heard.mid", f"{OUT}/heard_raw.wav")
    b, srb = mono(f"{OUT}/heard_raw.wav")
    lag_b = measured_lag(b, srb, [e.start for e in events])
    oa = librosa.onset.onset_detect(y=a, sr=sr, units="time", backtrack=False)
    ev_starts = np.array([e.start for e in events])
    deltas = [t - ev_starts[np.argmin(np.abs(ev_starts - t))] for t in oa
              if np.min(np.abs(ev_starts - t)) <= 0.06]
    delta = float(np.median(deltas))
    trim = lag_b - delta
    b = b[int(trim * srb):] if trim > 0 else np.concatenate(
        [np.zeros(int(-trim * srb)), b])
    b = np.pad(b, (0, max(0, L - len(b))))[:L]
    rms = lambda x: np.sqrt((x[np.abs(x) > 0.001] ** 2).mean())
    b *= rms(a) / rms(b)
    if np.abs(b).max() > 0.97:
        b *= 0.97 / np.abs(b).max()
    sf.write(f"{OUT}/heard.wav", b.astype(np.float32), srb)
    print(f"B rendered (lag {lag_b*1000:+.0f}ms, convention gap {delta*1000:+.0f}ms)")

    ok = ACC.run(f"{OUT}/recording.wav", f"{OUT}/heard.wav",
                 [e.start for e in truth], events)
    if not ok:
        print("acceptance FAILED — site assets NOT updated")
        return 1

    # Assets
    json.dump({"notes": [{"t": round(e.start, 4), "d": round(max(0.05, e.end - e.start), 4),
                          "p": e.pitch, "c": round(e.confidence, 3)} for e in events],
               "beats": [round(s, 4) for _, s in res.score.tempo_map if 0 <= s <= 14.5],
               "beats_per_bar": 4, "chords": res.score.chords, "bpm": 150.0},
              open("site/assets/demo-notes.json", "w"))
    for src_, dst in [(f"{OUT}/recording.wav", "site/assets/demo-original.mp3"),
                      (f"{OUT}/heard.wav", "site/assets/demo-machine.mp3")]:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src_,
                        "-b:a", "128k", dst], check=True)
    subprocess.run(["cp", f"{OUT}/run/Bebop-demo.gp5", "site/assets/demo.gp5"], check=True)
    subprocess.run(["cp", f"{OUT}/run/report.html", "site/report.html"], check=True)
    ms = "/Applications/MuseScore 4.app/Contents/MacOS/mscore"
    subprocess.run([ms, "-f", "-o", "site/assets/score-1.svg",
                    f"{OUT}/run/Bebop-demo.gp5"], capture_output=True)
    subprocess.run([ms, "-f", "-o", "site/score.pdf",
                    f"{OUT}/run/Bebop-demo.gp5"], capture_output=True)
    svg = open("site/assets/score-1.svg").read().replace(
        'width="215.9mm" height="279.4mm" viewBox="0 0 10200 13200"',
        'width="215.9mm" height="110.1mm" viewBox="0 0 10200 5200"')
    open("site/assets/score-1.svg", "w").write(svg)
    print("site assets updated (ALL PASS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
