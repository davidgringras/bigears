# Real-audio accuracy: GuitarSet

`docs/VALIDATION.md` measures the pipeline against synthesized licks and says
plainly that no real recording had been ground-truth-scored. This document is
that missing measurement. Every number below comes from real acoustic guitar
recorded through a microphone, scored against human-verified per-string
annotations, with no tempo or key hint given to the pipeline.

**Headline: on monophonic single-line solos the median note F1 is 0.850 at a
±100 ms onset tolerance (0.821 at ±50 ms), against 0.96–0.99 on the synthetic
licks. Polyphonic comping falls to 0.471. Constructed two-guitar mixes fall to
0.354 at 0 dB and 0.295 with the solo 6 dB down.** The self-audit never called
a real-audio run "high" confidence, which is the good news; the reason it gave
was wrong every time, which is not.

---

## 1. Dataset and licence

**GuitarSet** (Xi, Bittner, Pauwels, Ye, Bello, ISMIR 2018), Zenodo record
[10.5281/zenodo.3371780](https://zenodo.org/records/3371780), version 1.1.0.
The record page states the licence as **Creative Commons Attribution 4.0
International (CC-BY-4.0)**; verified on the record page before downloading.

Downloaded to `output/guitarset/` (gitignored):

| file | bytes | listed size |
|---|---|---|
| `annotation.zip` | 39,132,574 | 39.1 MB |
| `audio_mono-mic.zip` | 656,927,981 | 656.9 MB |

`audio_mono-mic` is the mono reference-microphone mix — a real acoustic guitar
in a room, not a pickup feed. 360 excerpts, 180 solo and 180 comping, 6 players
× 30 lead sheets × 2 versions.

### Naming convention

The README on [marl/GuitarSet](https://github.com/marl/GuitarSet) does not
document the filename fields, and neither does
[guitarset.weebly.com](https://guitarset.weebly.com/) — both were fetched and
checked. The convention was therefore **derived from the data and verified
against the annotations themselves**: for `05_Jazz3-137-Eb_solo`, the file's own
`tempo` annotation reads `137.0` and its `key_mode` annotation reads
`Eb:major`, matching the filename's third and fourth fields. That check was run
across all 14 selected excerpts: **14/14 filename tempi match the in-file tempo
annotation exactly**.

    <player 00-05>_<Style><progression 1-3>-<tempo BPM>-<key>_<solo|comp>

Enumerating all 360 annotation filenames gives exactly five style tokens, 24
files each per progression: **BN, Funk, Jazz, Rock, SS**. The dataset website
names the five styles as Rock, Singer-Songwriter, Bossa Nova, Jazz and Funk, so
`BN` = Bossa Nova and `SS` = Singer-Songwriter.

> **Correction to the brief.** The task specified two *blues* excerpts and
> glossed `BN` as blues. GuitarSet contains no blues style; `BN` is Bossa Nova.
> The selection below substitutes 2 Bossa Nova + 2 Singer-Songwriter for the
> requested blues pair, giving 2 excerpts in each of the five styles that
> actually exist.

### Ground truth

Extracted by parsing the JAMS JSON directly (the format is plain JSON; parsing
it in-process avoids installing a library into the venv the measurement depends
on). Verified by inspecting `05_Jazz3-137-Eb_solo.jams`:

- Six annotations carry namespace `note_midi`, one per string, distinguished by
  `annotation_metadata.data_source` = `"0"`…`"5"`. Their pitch ranges rise
  monotonically with that index (source 1 spans 51.0–56.0, source 5 spans
  72.0–74.0), so `"0"` is the low E string and `"5"` the high E.
- Each observation is `{time, duration, value, confidence}`. `value` is a
  **float** MIDI pitch (e.g. `53.0227`) — the pitch actually sounded, not a
  quantized fret.

**Truth for an excerpt is the union of all six strings' observations.** No
filtering, no rounding. Across all 180 solo files that is 16,861 notes.

Float pitches are converted to Hz and matched under mir_eval's default 50-cent
tolerance. Deviation from the nearest integer semitone has median 6.6 cents,
p95 23 cents, **max 49.99 cents** — so every ground-truth note lies within the
tolerance of some integer, and the integer-MIDI output of the transcriber is
never disadvantaged by a boundary effect.

---

## 2. Method

Selection was **pre-registered before any audio was scored**: seeded
(`random.Random(20260801)`), 2 solo excerpts sampled per style from the sorted
filename list, 4 comping excerpts sampled from 4 styles with seed+1. The
selection is frozen in `output/guitarset/selection.json`. Nothing was re-rolled
after seeing a score.

Transcription layer called directly, at stock settings:

```python
from soloscribe.transcribe import transcribe
events = transcribe(wav_path, mode="solo")   # "poly" additionally for comping
```

Scoring: `mir_eval.transcription.precision_recall_f1_overlap` with
`onset_tolerance` 0.05 and 0.10, `pitch_tolerance=50.0`, **`offset_ratio=None`**
(onset + pitch only; note durations are not scored). No bpm, key, downbeat or
chord hint is supplied anywhere in this document — this is the cold test.

### Pinned source revision

**The soloscribe source changed underneath this benchmark while it was
running.** A first full pass over the ten solos completed at ~01:14; at 01:19
`soloscribe/transcribe.py` was rewritten by a concurrent session landing commit
`effc5a7` ("Re-articulation splitter"), with `synth.py` (01:22) and `audit.py`
(01:26) following. Every measurement taken before that point describes a
different program. §7 reports the measured size of the shift, because it is
worth knowing.

All numbers in §3–§6 were re-derived in a single pass against these file
hashes, which are what actually pin the measurement:

    transcribe.py b0304a002e41af70a495c980859e4ee64f15b75504409716d6edc8cac996428d
    model.py      2e7d44d7c8435bd5efa18e9c9f1eff15b5c4fbddbda0697fc4ab768654da3188
    pipeline.py   d71002c15cd2e841e2c559c08f73ac7679d9faf94cb7426245ca710d621e81bd
    audit.py      3f64946fb3d99b009b0297780b898622dadbcf0ab56707a710a4f6af63602078
    synth.py      69e9cccf51a93855ad07bbe901f604cd4701b050b5ba1aa9e176a4df7a78f96c

That was `HEAD` = `effc5a7` at the time of the run. `HEAD` has since advanced
to `0eda1f7`, which touched only `synth.py` (to the hash above, already
re-verified) and added a `space/soloscribe/` deployment copy — **all five
hashes above are unchanged at `0eda1f7`**, so these numbers describe the
current transcription path, not a superseded one. Verify by hash, not by
commit: in a repository this actively edited, `HEAD` moves for reasons
unrelated to what is being measured.

Those four hashes were captured before the re-run and re-verified unchanged
after it.

`soloscribe/synth.py` — which `audit.py` imports for its resynthesis — was
edited again by the concurrent session *after* the §6 pipeline runs completed.
Those runs were therefore repeated against the newer `synth.py`
(`69e9cccf…`) and every reported field came back **identical**: same verdict,
same `f1_100ms`, same onset coverage, same true F1, on all three runs. The §6
findings are not an artefact of a stale resynthesis module.

`transcribe` was separately confirmed reproducible at this revision:
two complete passes over the ten solos were **bit-identical in every note count
and every F1**, three repeats within one process agreed, six repeats in
separate processes agreed, and results were unchanged under 12× concurrent CPU
load. The instability in §7 is a source change, not a nondeterministic model.

---

## 3. Solo excerpts (n = 10, `mode="solo"`)

| file | style | GT | est | P@50 | R@50 | **F1@50** | P@100 | R@100 | **F1@100** | s |
|---|---|---|---|---|---|---|---|---|---|---|
| 02_BN3-154-E_solo | BN | 79 | 95 | 0.747 | 0.899 | 0.816 | 0.758 | 0.911 | 0.828 | 5.7 |
| 05_BN3-154-E_solo | BN | 56 | 63 | 0.873 | 0.982 | 0.924 | 0.889 | 1.000 | **0.941** | 3.9 |
| 01_Funk2-119-G_solo | Funk | 137 | 127 | 0.732 | 0.679 | 0.705 | 0.756 | 0.701 | 0.727 | 5.0 |
| 05_Funk3-112-C#_solo | Funk | 70 | 70 | 0.857 | 0.857 | 0.857 | 0.857 | 0.857 | 0.857 | 5.4 |
| 01_Jazz1-130-D_solo | Jazz | 68 | 70 | 0.743 | 0.765 | 0.754 | 0.800 | 0.824 | 0.812 | 3.6 |
| 02_Jazz2-110-Bb_solo | Jazz | 54 | 54 | 0.907 | 0.907 | 0.907 | 0.907 | 0.907 | 0.907 | 5.5 |
| 02_Rock1-130-A_solo | Rock | 65 | 61 | 0.853 | 0.800 | 0.825 | 0.885 | 0.831 | 0.857 | 3.5 |
| 02_Rock3-148-C_solo | Rock | 65 | 66 | 0.894 | 0.908 | 0.901 | 0.909 | 0.923 | 0.916 | 4.0 |
| 00_SS1-68-E_solo | SS | 161 | 128 | 0.492 | 0.391 | 0.436 | 0.539 | 0.429 | **0.477** | 6.7 |
| 03_SS3-98-C_solo | SS | 111 | 117 | 0.778 | 0.820 | 0.798 | 0.821 | 0.865 | 0.842 | 7.5 |

**Medians:** F1@50 **0.821**, F1@100 **0.850**; P@100 0.839, R@100 0.861.
Range at ±100 ms: 0.477 – 0.941. Totals: 866 ground-truth notes, 851 estimated.

Per style (median F1@100, n = 2 each):

| style | F1@100 | F1@50 |
|---|---|---|
| Rock | 0.887 | 0.863 |
| BN (Bossa Nova) | 0.884 | 0.870 |
| Jazz | 0.860 | 0.831 |
| Funk | 0.792 | 0.781 |
| SS (Singer-Songwriter) | **0.660** | 0.617 |

With two excerpts per style these are indicative, not decisive. The Singer-
Songwriter median is dragged down entirely by one clip (0.477 vs 0.842), and
that clip is the only texturally polyphonic item in the set — see §6.

---

## 4. Comping excerpts (n = 4, polyphonic stress case)

| file | mode | GT | est | P@100 | R@100 | **F1@100** | F1@50 | s |
|---|---|---|---|---|---|---|---|---|
| 02_Jazz2-110-Bb_comp | solo | 262 | 127 | 0.677 | 0.309 | 0.417 | 0.386 | 16.4 |
| 02_Jazz2-110-Bb_comp | poly | 262 | 131 | 0.664 | 0.305 | 0.417 | 0.382 | 0.8 |
| 05_Funk3-98-A_comp | solo | 184 | 172 | 0.535 | 0.522 | 0.528 | 0.523 | 14.8 |
| 05_Funk3-98-A_comp | poly | 184 | 172 | 0.535 | 0.522 | 0.528 | 0.523 | 0.8 |
| 00_Rock1-90-C#_comp | solo | 665 | 154 | 0.851 | 0.197 | 0.320 | 0.300 | 12.0 |
| 00_Rock1-90-C#_comp | poly | 665 | 154 | 0.851 | 0.197 | 0.320 | 0.300 | 0.7 |
| 02_BN1-129-Eb_comp | solo | 154 | 90 | 0.744 | 0.405 | 0.525 | 0.492 | 7.8 |
| 02_BN1-129-Eb_comp | poly | 154 | 92 | 0.717 | 0.396 | 0.520 | 0.488 | 0.5 |

**Medians — `solo`: F1@100 0.471** (P 0.674, R 0.362). **`poly`: F1@100 0.469**
(P 0.661, R 0.364).

Two things are worth naming. First, comping fails through **recall**, not
precision: median recall 0.36 against precision 0.67. The transcriber is not
inventing chords, it is hearing two or three notes of a six-note voicing.
Second, **`poly` mode is statistically indistinguishable from `solo` mode here
(ΔF1@100 median 0.002) while running 16× faster** (0.7 s vs 11.3 s per 30 s of
audio). On dense chordal material the pyin refinement pass costs an order of
magnitude in time and buys nothing measurable — on two of the four excerpts it
returned a byte-identical note set. That is consistent with the module's own
documented guard, which refuses octave relabelling on any span that has a
concurrent note: on comping, almost every span does.

---

## 5. Hard mixes (constructed — see caveat)

**These mixes do not exist in GuitarSet; I built them.** Each pairs a solo
excerpt with a comping excerpt of the *same lead sheet* (identical style,
progression, tempo and key) played by a *different player*, summed sample-wise
aligned at t = 0, with the comping at unity gain and the solo attenuated by
0 dB or −6 dB. Where the sum exceeded 0.99 full-scale both parts were scaled
together to prevent clipping. Scored against the **solo's** ground truth only —
the comping guitar is pure interference. Files in `output/guitarset/mixes/`.

Choosing a same-lead-sheet partner makes the result a plausible duo rather than
noise, which is the harder and fairer test: the interfering guitar is in the
same key, at the same tempo, playing the same changes.

| solo | interferer | dB | GT | est | P@100 | R@100 | **F1@100** | F1@50 |
|---|---|---|---|---|---|---|---|---|
| 01_Funk2-119-G | 00_Funk2-119-G_comp | 0 | 137 | 312 | 0.231 | 0.525 | 0.321 | 0.316 |
| 01_Funk2-119-G | 00_Funk2-119-G_comp | −6 | 137 | 282 | 0.195 | 0.402 | **0.263** | 0.239 |
| 02_BN3-154-E | 00_BN3-154-E_comp | 0 | 79 | 208 | 0.288 | 0.861 | 0.432 | 0.425 |
| 02_BN3-154-E | 00_BN3-154-E_comp | −6 | 79 | 204 | 0.264 | 0.785 | 0.396 | 0.389 |
| 02_Jazz2-110-Bb | 00_Jazz2-110-Bb_comp | 0 | 54 | 183 | 0.230 | 0.778 | 0.354 | 0.346 |
| 02_Jazz2-110-Bb | 00_Jazz2-110-Bb_comp | −6 | 54 | 229 | 0.175 | 0.741 | 0.283 | 0.276 |
| 05_Funk3-112-C# | 00_Funk3-112-C#_comp | 0 | 70 | 246 | 0.228 | 0.800 | 0.354 | 0.335 |
| 05_Funk3-112-C# | 00_Funk3-112-C#_comp | −6 | 70 | 275 | 0.193 | 0.757 | 0.307 | 0.273 |

**Medians — 0 dB: F1@100 0.354** (P 0.230, R 0.781). **−6 dB: F1@100 0.295**
(P 0.194, R 0.725).

The failure mode is unambiguous and it is *not* masking. Recall of the target
solo stays high (median 0.78 at 0 dB, 0.73 at −6 dB) while precision collapses
to ~0.23. The system still hears the solo; it also faithfully transcribes the
other guitar and has no concept of which one you wanted. Direct confirmation
from the error breakdown on `02_Jazz2` at 0 dB: of 141 false positives, **92
have no ground-truth solo note sounding anywhere within ±100 ms at any
octave/harmonic relation** — they are the second guitar's notes, correctly
heard and wrongly attributed.

### Source selection on a mix (`separate="auto"`)

`run_pipeline` on the `01_Funk2 + 00_Funk2_comp` 0 dB mix:

- **Which source won: the original clip. Separation was declined outright.**
  The pipeline's own explanation: *"the clip already sounds like guitar on its
  own, so I worked from your original recording"* — the `sep.stem_rel_rms ≥
  0.90` branch in `pipeline.py` fired, because demucs' guitar stem of a
  two-guitar mix is very nearly the whole mix.
- Resulting F1@100 **0.326** on raw events (P 0.235, R 0.533), 0.335 for the
  written score. Runtime 35.2 s.

The heuristic behaved correctly on its own terms, and the deeper point is
structural rather than a bug: demucs separates by **instrument class**, so no
setting of this switch can split a lead guitar from a rhythm guitar. The
pipeline has no defence against same-instrument interference, and the "auto"
path cannot acquire one.

---

## 6. Full pipeline, end to end, on real audio

`run_pipeline(separate="off", swing="auto")`, no bpm, no key, no downbeat.

| | 01_Jazz1-130-D_solo | 02_Jazz2-110-Bb_solo |
|---|---|---|
| runtime | 17.2 s | 18.5 s |
| tempo tracked / true | 129.2 / **130** | **54.98 / 110** |
| swing called | straight | straight |
| true F1@100, raw events | 0.812 | 0.907 |
| true F1@100, **written score** | 0.797 | 0.852 |
| true F1@50, written score | 0.725 | 0.704 |
| GP5 written | 3,643 B | 2,849 B |
| **MuseScore 4 render** | **OK** (rc 0, 48,701 B PDF) | **OK** (rc 0, 44,282 B PDF) |
| audit verdict | low | low |

Both files render in MuseScore 4 without error
(`/Applications/MuseScore 4.app/Contents/MacOS/mscore -f -o out.pdf in.gp5`).
Quantization costs 0.015–0.055 F1 relative to the raw events at ±100 ms, and
considerably more at ±50 ms on the second clip (0.907 → 0.704) — that clip is
the half-tempo one, and a wrong grid misplaces onsets within the tighter
window.

**Beat tracking got one of two jazz clips exactly half wrong: 54.98 BPM against
a true 110.** The other landed at 129.2 against 130 (0.6% error). Octave-error-
in-tempo is a classic beat-tracker failure; on a swung jazz clip with no bpm
hint it is a coin flip, and this is the coin landing badly. Neither clip was
called swung.

### The self-audit's blind spot, measured on real audio

`VALIDATION.md` says the audit "CANNOT see notes the transcriber never heard".
Real audio exposes the mirror image, which is worse, because the audit's
headline number is *maximally flattering exactly where the system is weakest*:

| run | audit `f1_100ms` | audit coverage | audit verdict | **true F1@100** | **true precision** |
|---|---|---|---|---|---|
| 01_Jazz1 solo | **1.000** | 0.521 | low | 0.812 | 0.800 |
| 02_Jazz2 solo | 0.926 | 0.370 | low | 0.907 | 0.907 |
| Funk2 0 dB mix (`separate="auto"`) | **1.000** | 0.772 | medium | **0.326** | **0.235** |

On the worst-performing run in this entire benchmark — 77% of its notes
invented — the audit's own note-agreement metric reads **1.000**, and it was
held below "high" only by an unrelated blocker (the resynthesis needed 0.98 s
of stretching). That metric measures how many transcribed notes survived
quantization. It is a self-consistency check and it cannot fail when the
transcriber is confidently wrong.

The coverage proxy is the guard that actually fires, and it is the one that is
miscalibrated on real acoustic guitar. It reported 37% and 52% of attacks
accounted for on the two clean jazz solos, whose true recall was 0.907 and
0.824. `librosa.onset.onset_detect` finds 94 attacks in a clip with 68 notes
and 130 in a clip with 54 — real guitar produces string noise, body thumps and
finger squeaks that a synthetic Karplus-Strong render does not.

The practical upshot is mixed. **The verdict level was never wrong in a
dangerous direction: no real-audio run was called "high".** But the reason
attached to every one of them was "Notes are missing" when on the mixes the
actual defect is the opposite — 3–4× too many notes. A user reading that report
is told to look for gaps in a transcription whose real problem is a second
guitar written into their solo.

---

## 7. Reproducibility: the source moved mid-benchmark

A complete first pass over the ten solos ran against the pre-`effc5a7` tree.
Re-running the identical script after that commit landed changed **every single
file**:

| file | est notes (pre → post) | F1@100 (pre → post) | Δ |
|---|---|---|---|
| 02_BN3-154-E_solo | 132 → 95 | 0.692 → 0.828 | +0.136 |
| 05_BN3-154-E_solo | 84 → 63 | 0.800 → 0.941 | +0.141 |
| 01_Funk2-119-G_solo | 169 → 127 | 0.765 → 0.727 | −0.037 |
| 05_Funk3-112-C#_solo | 114 → 70 | 0.728 → 0.857 | +0.129 |
| 01_Jazz1-130-D_solo | 101 → 70 | 0.781 → 0.812 | +0.030 |
| 02_Jazz2-110-Bb_solo | 84 → 54 | 0.754 → 0.907 | +0.154 |
| 02_Rock1-130-A_solo | 88 → 61 | 0.797 → 0.857 | +0.060 |
| 02_Rock3-148-C_solo | 99 → 66 | 0.780 → 0.916 | +0.136 |
| 00_SS1-68-E_solo | 224 → 128 | 0.525 → 0.477 | −0.047 |
| 03_SS3-98-C_solo | 169 → 117 | 0.707 → 0.842 | +0.135 |

Median F1@100 moved **0.759 → 0.850**; note counts fell to a median 0.69× of
their former value; two files got worse. Both regimes are internally
reproducible — the difference is the code, not the machine (CoreML is loaded
`CPU_ONLY` at `basic_pitch/inference.py:99`, and results were invariant under
concurrent CPU load and across processes).

Two things follow. The numbers reported here are the *more favourable* of the
two regimes observed, and they are pinned to a revision hash for exactly that
reason. And any future comparison against this document must state its own
revision, or it is comparing two unknowns.

---

## 8. The three worst cases, diagnosed

Diagnosis is from the measured error structure and the ground-truth texture
statistics (`output/guitarset/diagnosis.json`), not from inspection of the
score. Every false positive is classified by what it is relative to the truth
sounding at that instant: an octave of a real note, a harmonic (+7/+16/+19/+24),
a same-pitch re-trigger within 120 ms, a pitch-correct note whose onset missed
the window, or genuinely unrelated.

**1. `00_SS1-68-E_solo` — F1@100 0.477** (P 0.539, R 0.429; 161 GT, 128 est,
92 missed). *It is not a single-note solo.* It carries **1.64 ground-truth
notes per attack** and **91% of its notes sound while another note is still
ringing** — a fingerstyle passage with sustained overlapping voices. Every
other solo in the set averages 1.00–1.02 notes per attack. `mode="solo"` is a
monophonic-biased path by construction, so it discards the voices it is
designed to discard, and pyin — which locks onto a *common subharmonic* on
dyads, as the module docstring warns — drives 20 of the 59 false positives as
octave relabels (13 up, 7 down). Across the ten solos, F1@100 correlates
−0.90 with notes-per-attack, but that coefficient is almost entirely this one
point's leverage; the rank correlation is a much weaker −0.43. The honest
statement is not "density predicts accuracy" but "the one polyphonic clip is
the one that fell over".

**2. `00_Rock1-90-C#_comp` — F1@100 0.320** (P **0.851**, R **0.197**; 665 GT,
154 est, 534 missed). Precision is the best in the whole comping set; recall is
the worst anywhere. The excerpt is 665 notes across **192 strum events, mean
3.46 notes per strum, 28% of them five or six notes wide, median inter-onset
gap 7.2 ms**. Full six-string chords struck as a block. The transcriber emits
roughly one note per two strummed strings and is right about nearly all of
them. This is a capacity limit of the note decoder on dense simultaneities, not
a confusion — nothing here is being misheard, four-fifths of it is simply not
being reported.

**3. `02_Jazz2-110-Bb_comp` — F1@100 0.417** (P 0.677, R 0.309; 262 GT, 127
est, 181 missed). The same class as above but with confusion added: mean 2.13
notes per attack, 33% of attacks three or more notes wide, and 46 false
positives of which **19 are octave errors (11 up, 8 down) and 7 more are
harmonics** — 26 of 46, over half. Rootless jazz voicings stack thirds and
sevenths into
a register where the second and fourth harmonics of the lower notes land on
real chord tones, and the amplitude-ratio ghost filter cannot separate a
harmonic from a voiced note at the same pitch.

Worst overall, for completeness: the `01_Funk2 −6 dB` constructed mix at F1@100
**0.263** (P 0.195). That is the interference case of §5, not a property of the
recording.

---

## 9. Runtime

Apple Silicon, CPU-only CoreML backend, measured wall-clock inside
`transcribe()`.

| path | per 30 s of audio | range observed |
|---|---|---|
| `mode="solo"`, single-line solo | **4.7 s** | 3.5–7.5 s (22–42 s clips) |
| `mode="solo"`, dense comping | 11.3 s | 7.8–16.4 s |
| `mode="solo"`, two-guitar mix | 13.4 s | 9.9–17.7 s |
| `mode="poly"`, dense comping | **0.7 s** | 0.5–0.8 s |
| full `run_pipeline` incl. audit + GP5 | ~17–18 s per clip | 22–35 s clips |

Cost scales with note count, not clip length: the pyin refinement and the
re-articulation splitter both work per note. `poly` mode's 15× advantage on
comping is the same number from §4 seen from the other side. The pipeline
figure includes beat tracking, quantization, fretting, GP5 writing and the full
audit resynthesis, which together account for roughly two-thirds of it.

Nothing crashed. 36 transcription runs, 3 full pipeline runs, 3 MuseScore
renders, zero exceptions and zero non-zero exit codes.

---

## 10. What this means

Real audio costs roughly **0.11–0.14 F1** against the synthetic licks on the
material the pipeline is actually built for — median 0.850 at ±100 ms and 0.821
at ±50 ms, against 0.96–0.99 synthetic. That is a real and substantial drop and
it should be stated as such: the synthetic numbers are not predictive of
real-world performance, and `VALIDATION.md`'s warning that they would be
optimistic is now quantified rather than anticipated. A user handing this a
clean single-line guitar solo should expect roughly **one note in seven to be
wrong** — either invented or missed — not one in fifty.

The shape of the degradation is more useful than its size:

- **Monophonic single-line playing is where the system works.** Nine of the ten
  solos average 1.00–1.02 ground-truth notes per attack, and eight of those
  nine score 0.727–0.941. Style barely matters within that group; Rock, Bossa
  Nova and Jazz medians sit within 0.03 of each other.
- **Texture is what breaks it, and it breaks it hard.** The single polyphonic
  "solo" scores 0.477. Comping — which is polyphonic by definition — has a
  median of 0.471 and fails through recall, hearing two or three notes of a
  six-note voicing. If a user's material is chordal, this pipeline will
  under-report it by roughly two-thirds, and the product should say so before
  they upload rather than after.
- **Funk is the weakest of the five styles** on single-line material (median
  0.792 vs 0.86–0.89 for Rock, Bossa Nova and Jazz), though with two excerpts
  per style that ordering is indicative only.
- **Interference is the unsolved problem.** Two guitars at equal level takes
  F1 to 0.354 and precision to 0.23, and the separation front-end cannot help
  because it separates by instrument class. Attenuating the solo by 6 dB makes
  it worse (0.295) but not catastrophically so — the system was never really
  tracking the loud one, it was transcribing everything.
- **The self-audit's verdict is trustworthy in direction and untrustworthy in
  explanation.** It never claimed high confidence on real audio. But its
  headline agreement metric read 1.000 on the run with 23.5% precision, and its
  stated reason was "notes are missing" on every run, including the ones whose
  defect was 3–4× too many notes.

### Caveats, stated plainly

- **No bpm, key, downbeat or chord hint was given anywhere.** Supplying a tempo
  would very likely have prevented the half-tempo failure in §6, and the
  product does expose that input. These numbers are the cold-start floor, not
  the ceiling.
- **GuitarSet is still clean solo-microphone audio in a quiet room.** One
  guitar, close-mic'd, no drums, no bass, no vocals, no amplifier, no room
  reverb to speak of, no MP3 artefacts. It is real audio, not difficult audio.
  Numbers on a band mix or a phone recording of a rehearsal will be lower and
  are not measured here.
- **The mixes in §5 are constructed by me, not recorded.** Summing two dry
  close-mic'd tracks at t = 0 is not the same as two players in a room: there is
  no bleed, no shared reverb, no interaction, and both parts sit in exactly the
  same frequency band with no spatial separation. The result is plausibly
  *harder* than a real duo recording in some respects and easier in others.
  Treat §5 as a probe of same-instrument interference, not as a band-mix
  benchmark.
- **Two excerpts per style, ten solos total.** Per-style medians rest on n = 2.
  The aggregate median over ten is the number worth quoting; the style ordering
  is a hypothesis.
- **Offsets are not scored** (`offset_ratio=None`), so note durations — which
  the notation layer very much cares about — are untested here.
- **Ground truth is GuitarSet's, not mine.** Its per-string annotations are
  themselves derived from a hexaphonic pickup with human verification; where
  they are wrong, this benchmark inherits the error.

### Reproducing

    output/guitarset/selection.json          frozen excerpt selection (seed 20260801)
    output/guitarset/bench_lib.py            truth extraction + scoring
    output/guitarset/run_bench.py            solo | comp | mix stages
    output/guitarset/run_pipeline_bench.py   e2e | sep stages
    output/guitarset/diagnose.py             error-structure classification
    output/guitarset/results_*.json          all raw numbers
    output/guitarset/results_solo_RUN1.json  pre-effc5a7 pass, kept for §7
    output/guitarset/*_STALE.json            pre-effc5a7 comp/mix/pipeline runs,
                                             superseded; retained as evidence only
    output/guitarset/*_PRESYNTH.json         §6 runs before the later synth.py edit,
                                             verified field-identical to current

Re-running requires the two Zenodo zips in `output/guitarset/`; nothing under
`soloscribe/` was modified to produce any number in this document.
