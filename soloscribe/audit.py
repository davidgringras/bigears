"""Self-audit: resynthesize the transcription, align it against the recording,
and tell the user the truth about it.

Nothing here is allowed to flatter the pipeline. Every number is computed from
the score as actually written (ties merged, ticks mapped back through the
tempo_map into the recording's clock), compared against either the separated
guitar stem or the original mix, and reported next to a plain-English sentence
saying what it means for someone holding a guitar rather than a spectrogram.

Four independent views, because each one is blind to a different failure:

  quantization  Did writing the transcriber's notes onto a musical grid move
                them somewhere they don't belong? Compares raw NoteEvents to
                the quantized Score. Blind to the transcriber being wrong in
                the first place — if it hallucinated a note, both sides agree.
  chroma_dtw    Does the resynthesis sound like the recording, pitch-class by
                pitch-class, and does it stay in step with it? Catches wrong
                notes and octave errors that the grid comparison cannot see.
  coverage      Does every audible pick attack have a note, and does every
                written note have an audible attack? Catches missing and
                invented notes directly from the audio.
  per_measure   Where, specifically, in the piece it went wrong.

Metrics dict shape (integration depends on these exact names):

  {
    "quantization": {"f1_50ms", "precision_50ms", "recall_50ms",
                     "f1_100ms", "precision_100ms", "recall_100ms",
                     "avg_overlap_ratio_50ms", "n_ref_events", "n_est_notes"},
    "chroma_dtw":   {"normalized_cost", "diagonality_sec",
                     "median_deviation_sec", "n_frames", "path_length"},
    "coverage":     {"onset_coverage", "onset_spuriousness", "onset_precision",
                     "n_detected", "n_score_onsets", "n_matched",
                     "tolerance_sec"},
    "per_measure":  [{"measure", "n_notes", "mean_confidence",
                      "rms_quantization_error_ms"}, ...],
    "verdict":      {"level": "high"|"medium"|"low", "reasons": [str, ...]},
    "caveats":      [str, ...],
    "comparison_source": "stem"|"original",
    "render_path": str, "piano_roll_path": str, "report_path": str,
    "durations": {"comparison_sec", "render_sec"},
  }
"""
from __future__ import annotations

import base64
import html as _html
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # report generation must not require a display

import librosa
import mir_eval
import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from scipy.spatial.distance import cdist

from .model import NoteEvent, Score
from .synth import DEFAULT_SR, merge_tied, render_score, score_note_seconds

SR = DEFAULT_SR
HOP = 512

# 80 ms: about the width of a relaxed pick attack, and comfortably inside the
# ~50 ms window where two attacks stop being heard as separate events.
ONSET_TOL_SEC = 0.080

# How far a raw event may sit from a score note and still be considered "the
# same note, moved by quantization". Beyond half a second it isn't the same
# note, it's a different one.
EVENT_PAIR_WINDOW_SEC = 0.5

# Verdict thresholds. Deliberately asymmetric: it is much cheaper to tell a
# guitarist to check their tab than to tell them it is fine when it isn't.
HIGH_F1 = 0.85
HIGH_COVERAGE = 0.80
LOW_F1 = 0.60
LOW_COVERAGE = 0.55
# A transcription can score well on notes and still slide out of time with the
# take; either of these blocks "high" without on its own condemning the run.
DIAGONALITY_BLOCK_SEC = 0.75
CHROMA_COST_BLOCK = 0.45
# If the render and the comparison audio are not roughly the same span, the
# chroma and onset numbers are answering a question nobody asked.
DURATION_MISMATCH_RATIO = 0.20

MAX_EMBED_SEC = 240.0  # cap on audio embedded in the HTML, keeps the file sane

STANDARD_CAVEATS = [
    "Articulation is approximate. Slides, hammer-ons, pull-offs and legato "
    "phrasing are written as separately picked notes; the tab will say the "
    "right pitches but not always the right left hand.",
    "Bends and vibrato are written into the .gp5 but are NOT reproduced in the "
    "resynthesis used for this audit. A heavily bent phrase therefore scores "
    "worse here than the transcription actually deserves.",
    "Swing is notated straight with a feel marker, which is the convention. "
    "Swung eighths will look like even eighths on the page and should be "
    "played with the marked feel, not as written.",
    "This is a first pass, not a finished transcription. Open the .gp5 in "
    "Guitar Pro or TuxGuitar and fix it by ear — this report is a map of where "
    "to look, not a verdict on what to write.",
]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    fallback = "/opt/homebrew/bin/ffmpeg"
    return fallback if os.path.exists(fallback) else None


def _greedy_match(
    a_times: np.ndarray,
    b_times: np.ndarray,
    tol: float,
    a_pitch: np.ndarray | None = None,
    b_pitch: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """One-to-one nearest-first matching of two time series within `tol`.

    A plain "is there anything within tolerance" test lets a single note claim
    credit for three attacks, which quietly inflates coverage on exactly the
    dense passages where transcription is hardest. Insisting on a one-to-one
    matching costs nothing here (both sequences are small) and cannot flatter.
    """
    cands: list[tuple[float, int, int]] = []
    for i, ta in enumerate(a_times):
        for j, tb in enumerate(b_times):
            d = abs(float(ta) - float(tb))
            if d > tol:
                continue
            if a_pitch is not None and b_pitch is not None:
                if int(a_pitch[i]) != int(b_pitch[j]):
                    continue
            cands.append((d, i, j))
    cands.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    out: list[tuple[int, int]] = []
    for _, i, j in cands:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((i, j))
    return out


def _intervals(spans: list[tuple[float, float]]) -> np.ndarray:
    """(n, 2) float array, shaped correctly even when empty.

    mir_eval.util.validate_intervals (util.py:738) checks `ndim != 2 or
    shape[1] != 2` before anything else, so an empty list must still arrive as
    (0, 2) or it raises instead of returning the documented zeros.
    """
    if not spans:
        return np.zeros((0, 2), dtype=float)
    arr = np.asarray(spans, dtype=float)
    arr[:, 0] = np.maximum(arr[:, 0], 0.0)
    # validate_intervals (util.py:759) rejects non-positive durations outright.
    arr[:, 1] = np.maximum(arr[:, 1], arr[:, 0] + 1e-3)
    return arr


def _safe(x) -> float | None:
    if x is None:
        return None
    f = float(x)
    return None if (math.isnan(f) or math.isinf(f)) else f


# --------------------------------------------------------------------------
# metric blocks
# --------------------------------------------------------------------------

def _quantization_metrics(
    events: list[NoteEvent], note_secs: list[tuple]
) -> dict:
    """Raw transcription vs. the score as written, via mir_eval.

    mir_eval.transcription.precision_recall_f1_overlap (transcription.py:477)
    takes pitches in HERTZ, not MIDI, and returns a 4-tuple
    (precision, recall, f_measure, avg_overlap_ratio). offset_ratio=None
    (documented at transcription.py:535-541) makes it ignore note offsets, so
    this measures onset placement and pitch only — the right call, since note
    durations in a solo are a notation choice as much as an acoustic fact.
    """
    ref = _intervals([(ev.start, ev.end) for ev in events])
    ref_hz = np.array(
        [mir_eval.util.midi_to_hz(ev.pitch) for ev in events], dtype=float
    )
    est = _intervals([(t0, t1) for _, t0, t1 in note_secs])
    est_hz = np.array(
        [mir_eval.util.midi_to_hz(n.pitch) for n, _, _ in note_secs], dtype=float
    )

    out: dict = {"n_ref_events": len(events), "n_est_notes": len(note_secs)}
    if len(events) == 0 or len(note_secs) == 0:
        out.update(
            f1_50ms=0.0, precision_50ms=0.0, recall_50ms=0.0,
            f1_100ms=0.0, precision_100ms=0.0, recall_100ms=0.0,
            avg_overlap_ratio_50ms=0.0,
        )
        return out

    for tol, tag in ((0.05, "50ms"), (0.10, "100ms")):
        p, r, f, aor = mir_eval.transcription.precision_recall_f1_overlap(
            ref, ref_hz, est, est_hz, onset_tolerance=tol, offset_ratio=None
        )
        out[f"precision_{tag}"] = float(p)
        out[f"recall_{tag}"] = float(r)
        out[f"f1_{tag}"] = float(f)
        if tag == "50ms":
            out["avg_overlap_ratio_50ms"] = float(aor)
    return out


def _chroma_dtw_metrics(y_cmp: np.ndarray, y_ren: np.ndarray, sr: int) -> dict:
    """Align the resynthesis to the recording on pitch-class content.

    Both signals are zero-padded to a common length first, so the two chroma
    matrices have identical frame counts and the "diagonal" is literally the
    n == m line. That is only a meaningful reference because render_score lays
    the score onto the recording's own clock via the tempo_map — if the render
    used nominal tempo instead, drift in the take would show up here as a
    transcription error, which it isn't.
    """
    n = max(y_cmp.size, y_ren.size)
    empty = {
        "normalized_cost": None, "diagonality_sec": None,
        "median_deviation_sec": None, "n_frames": 0, "path_length": 0,
    }
    if n < HOP * 4:
        return empty
    a = np.pad(y_cmp, (0, n - y_cmp.size))
    b = np.pad(y_ren, (0, n - y_ren.size))
    if not np.any(a) or not np.any(b):
        return empty

    ca = librosa.feature.chroma_cqt(y=a, sr=sr, hop_length=HOP)
    cb = librosa.feature.chroma_cqt(y=b, sr=sr, hop_length=HOP)

    # Cosine distance is undefined for an all-zero frame, and librosa's dtw
    # refuses a cost matrix containing NaN (sequence.py:426). A silent frame is
    # a real thing in a solo, so nudge every vector off the origin rather than
    # dropping frames: a silent frame becomes uniform, which reads as ~0 cost
    # against other silence and ~0.7 against a pitched frame. That is the
    # behaviour we want.
    ca = ca + 1e-8
    cb = cb + 1e-8
    C = cdist(ca.T, cb.T, metric="cosine")
    C = np.nan_to_num(C, nan=1.0, posinf=1.0, neginf=0.0)

    D, wp = librosa.sequence.dtw(C=C, backtrack=True)
    dev_frames = np.abs(wp[:, 0].astype(float) - wp[:, 1].astype(float))
    per_frame = HOP / float(sr)
    return {
        # Path-length normalization; without it the cost just measures duration.
        "normalized_cost": _safe(D[-1, -1] / max(len(wp), 1)),
        "diagonality_sec": _safe(dev_frames.max() * per_frame),
        # The max is one frame's worth of bad luck away from being misleading;
        # the median says whether the drift is structural.
        "median_deviation_sec": _safe(np.median(dev_frames) * per_frame),
        "n_frames": int(C.shape[0]),
        "path_length": int(len(wp)),
    }


def _coverage_metrics(
    y_cmp: np.ndarray, sr: int, score_onsets: np.ndarray
) -> dict:
    """Compare audible pick attacks against written note starts.

    onset_coverage      fraction of DETECTED attacks that a score note covers
                        (low => notes are missing from the tab)
    onset_spuriousness  fraction of SCORE notes with NO detected attack near
                        them (high => the tab invents notes)

    Note the polarity: spuriousness is 1 - precision, not precision. A metric
    named for a defect must go up when the defect gets worse, or a skimmer
    reads the report backwards.
    """
    detected = librosa.onset.onset_detect(
        y=y_cmp, sr=sr, hop_length=HOP, backtrack=True, units="time"
    )
    detected = np.asarray(detected, dtype=float)
    n_det = int(detected.size)
    n_sco = int(np.asarray(score_onsets).size)
    if n_det == 0 or n_sco == 0:
        return {
            "onset_coverage": 0.0 if n_det else None,
            "onset_spuriousness": 1.0 if n_sco else None,
            "onset_precision": 0.0 if n_sco else None,
            "n_detected": n_det, "n_score_onsets": n_sco, "n_matched": 0,
            "tolerance_sec": ONSET_TOL_SEC,
        }
    pairs = _greedy_match(detected, np.asarray(score_onsets, float), ONSET_TOL_SEC)
    matched = len(pairs)
    precision = matched / n_sco
    return {
        "onset_coverage": matched / n_det,
        "onset_spuriousness": 1.0 - precision,
        "onset_precision": precision,
        "n_detected": n_det,
        "n_score_onsets": n_sco,
        "n_matched": matched,
        "tolerance_sec": ONSET_TOL_SEC,
    }


def _per_measure_metrics(
    score: Score, events: list[NoteEvent], note_secs: list[tuple]
) -> list[dict]:
    """Bar-by-bar note count, transcriber confidence, and quantization error.

    Each raw event is paired with the score note that most plausibly is the
    same note (same pitch, nearest in time, one-to-one), and the distance it
    had to move to land on the grid is charged to the bar its score note sits
    in. Events with no plausible partner contribute nothing here — they are a
    recall failure, which the quantization and coverage blocks already report.
    """
    tpb = score.ticks_per_bar
    n_bars = max(1, score.n_measures())

    ev_t = np.array([ev.start for ev in events], dtype=float)
    ev_p = np.array([ev.pitch for ev in events], dtype=int)
    sc_t = np.array([t0 for _, t0, _ in note_secs], dtype=float)
    sc_p = np.array([n.pitch for n, _, _ in note_secs], dtype=int)

    errors: dict[int, list[float]] = {}
    if ev_t.size and sc_t.size:
        for i, j in _greedy_match(
            ev_t, sc_t, EVENT_PAIR_WINDOW_SEC, a_pitch=ev_p, b_pitch=sc_p
        ):
            bar = int(note_secs[j][0].onset // tpb)
            errors.setdefault(bar, []).append(abs(ev_t[i] - sc_t[j]) * 1000.0)

    counts: dict[int, list[float]] = {}
    for n, _, _ in note_secs:
        counts.setdefault(int(n.onset // tpb), []).append(float(n.confidence))

    rows: list[dict] = []
    for bar in range(n_bars):
        confs = counts.get(bar, [])
        errs = errors.get(bar, [])
        rows.append(
            {
                "measure": bar + 1,  # musicians count from 1
                "n_notes": len(confs),
                "mean_confidence": (
                    float(np.mean(confs)) if confs else None
                ),
                # None, not 0.0 — "no events landed here to measure" and
                # "landed perfectly" are opposite facts.
                "rms_quantization_error_ms": (
                    float(np.sqrt(np.mean(np.square(errs)))) if errs else None
                ),
            }
        )
    return rows


def _verdict(quant: dict, chroma: dict, cov: dict, extra_blocks: list[str]) -> dict:
    """Three levels, thresholds fixed up front, no discretion at report time."""
    f1 = quant.get("f1_100ms") or 0.0
    coverage = cov.get("onset_coverage")
    coverage = 0.0 if coverage is None else float(coverage)
    reasons: list[str] = []

    if f1 < LOW_F1:
        level = "low"
        reasons.append(
            f"Only {f1:.0%} of the transcribed notes survived quantization at a "
            f"±100 ms tolerance (below the {LOW_F1:.0%} floor). The written "
            "rhythm does not represent what was played."
        )
    elif coverage < LOW_COVERAGE:
        level = "low"
        reasons.append(
            f"Only {coverage:.0%} of the pick attacks audible in the recording "
            f"have a note in the tab (below the {LOW_COVERAGE:.0%} floor). "
            "Notes are missing."
        )
    elif f1 >= HIGH_F1 and coverage >= HIGH_COVERAGE:
        level = "high"
        reasons.append(
            f"{f1:.0%} note agreement at ±100 ms and {coverage:.0%} of audible "
            "attacks accounted for."
        )
    else:
        level = "medium"
        if f1 < HIGH_F1:
            reasons.append(
                f"Note agreement at ±100 ms is {f1:.0%}, short of the "
                f"{HIGH_F1:.0%} needed for high confidence."
            )
        if coverage < HIGH_COVERAGE:
            reasons.append(
                f"{coverage:.0%} of audible attacks are accounted for, short of "
                f"the {HIGH_COVERAGE:.0%} needed for high confidence."
            )

    # Independent blockers: a run can pass the note tests and still be wrong.
    blockers: list[str] = list(extra_blocks)
    diag = chroma.get("diagonality_sec")
    cost = chroma.get("normalized_cost")
    if diag is not None and diag > DIAGONALITY_BLOCK_SEC:
        blockers.append(
            f"The resynthesis had to be stretched by up to {diag:.2f} s to line "
            "up with the recording, so the tab slips out of time somewhere."
        )
    if cost is not None and cost > CHROMA_COST_BLOCK:
        blockers.append(
            f"The resynthesis does not match the recording harmonically "
            f"(alignment cost {cost:.3f}), which usually means wrong notes "
            "rather than wrong rhythm."
        )
    if blockers:
        reasons.extend(blockers)
        if level == "high":
            level = "medium"
    return {"level": level, "reasons": reasons}


# --------------------------------------------------------------------------
# report assets
# --------------------------------------------------------------------------

def _encode_audio(src_path: str, work: str, tag: str) -> tuple[str, str] | None:
    """Return (mime, base64) for embedding, mp3 via ffmpeg where available."""
    ff = _ffmpeg()
    if ff:
        dst = os.path.join(work, f"{tag}.mp3")
        proc = subprocess.run(
            [ff, "-y", "-loglevel", "error", "-i", src_path,
             "-t", str(MAX_EMBED_SEC), "-vn", "-ac", "1", "-b:a", "96k", dst],
            capture_output=True,
        )
        if proc.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst):
            with open(dst, "rb") as fh:
                return "audio/mpeg", base64.b64encode(fh.read()).decode("ascii")
    # No ffmpeg: mono 22 kHz WAV, still bounded so the HTML stays openable.
    try:
        y, sr = librosa.load(src_path, sr=SR, mono=True, duration=MAX_EMBED_SEC)
    except Exception:
        return None
    dst = os.path.join(work, f"{tag}.wav")
    sf.write(dst, y, sr, subtype="PCM_16")
    with open(dst, "rb") as fh:
        return "audio/wav", base64.b64encode(fh.read()).decode("ascii")


_CONFIDENCE_CMAP = LinearSegmentedColormap.from_list(
    "soloscribe_confidence",
    # warm/uncertain -> cool/confident; readable in greyscale print too
    ["#b4552f", "#c08a2e", "#5d7f4e", "#2f6f8f"],
)


def _piano_roll(
    score: Score,
    events: list[NoteEvent],
    note_secs: list[tuple],
    out_png: str,
    duration: float,
) -> None:
    """Raw events as grey outlines under the quantized score, on a beat grid."""
    fig, ax = plt.subplots(figsize=(11.5, 4.4), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    pitches = [ev.pitch for ev in events] + [n.pitch for n, _, _ in note_secs]
    lo = (min(pitches) - 2) if pitches else 55
    hi = (max(pitches) + 2) if pitches else 79

    # Beat grid, read off the tempo_map so it bends with the recording.
    tpb_beat = score.ticks_per_beat
    tpb_bar = score.ticks_per_bar
    for bar in range(score.n_measures() + 1):
        for beat in range(score.beats_per_bar):
            tick = bar * tpb_bar + beat * tpb_beat
            t = score.tick_to_seconds(tick)
            if t > duration + 0.5:
                break
            if beat == 0:
                ax.axvline(t, color="#9aa0a6", lw=0.9, alpha=0.55, zorder=0)
                ax.text(
                    t, hi + 0.35, str(bar + 1), fontsize=7, color="#6b7075",
                    ha="left", va="bottom", clip_on=False,
                )
            else:
                ax.axvline(t, color="#c9ced3", lw=0.6, alpha=0.5, zorder=0)

    for ev in events:
        ax.add_patch(
            Rectangle(
                (ev.start, ev.pitch - 0.42), max(ev.duration, 0.01), 0.84,
                facecolor="none", edgecolor="#8b9096", lw=0.9, zorder=2,
            )
        )
    norm = Normalize(vmin=0.0, vmax=1.0)
    for n, t0, t1 in note_secs:
        ax.add_patch(
            Rectangle(
                (t0, n.pitch - 0.30), max(t1 - t0, 0.01), 0.60,
                facecolor=_CONFIDENCE_CMAP(norm(float(n.confidence))),
                edgecolor="none", alpha=0.9, zorder=3,
            )
        )

    ax.set_xlim(0, max(duration, 1.0))
    ax.set_ylim(lo, hi + 1)
    ax.set_xlabel("seconds (original recording)", fontsize=9)
    ax.set_ylabel("MIDI pitch", fontsize=9)
    ax.tick_params(labelsize=8, colors="#3c4043")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9ced3")
    ax.grid(axis="y", color="#eceff1", lw=0.6, zorder=0)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#8b9096", lw=0.9),
        Rectangle((0, 0), 1, 1, facecolor="#2f6f8f", edgecolor="none"),
    ]
    ax.legend(
        handles, ["heard (raw transcription)", "written (quantized score)"],
        loc="upper right", fontsize=8, frameon=False, ncol=2,
        bbox_to_anchor=(1.0, 1.16),
    )
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=_CONFIDENCE_CMAP),
        ax=ax, pad=0.012, fraction=0.03,
    )
    cbar.set_label("note confidence", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_png, facecolor="#ffffff")
    plt.close(fig)


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f4f2ee; color: #1c1c1a;
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
  "Helvetica Neue", Arial, sans-serif; -webkit-font-smoothing: antialiased; }
.wrap { max-width: 920px; margin: 0 auto; padding: 40px 24px 72px; }
header { border-bottom: 1px solid #dcd8d0; padding-bottom: 18px; margin-bottom: 26px; }
h1 { font-size: 25px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }
.sub { color: #6a665e; font-size: 13px; margin: 0; }
h2 { font-size: 15px; font-weight: 600; margin: 38px 0 12px;
  text-transform: uppercase; letter-spacing: 0.07em; color: #55514a; }
p { margin: 0 0 12px; }
.lede { color: #4a463f; font-size: 14px; max-width: 66ch; }
.verdict { border-left: 5px solid; border-radius: 3px; padding: 16px 20px;
  margin: 0 0 8px; background: #fff; }
.verdict h3 { margin: 0 0 6px; font-size: 17px; font-weight: 600;
  letter-spacing: 0.02em; }
.verdict ul { margin: 8px 0 0; padding-left: 19px; font-size: 14px;
  color: #3d3a34; }
.verdict li { margin-bottom: 5px; }
.v-high { border-color: #3f7d5b; } .v-high h3 { color: #2c5c42; }
.v-medium { border-color: #b58326; } .v-medium h3 { color: #7d5a13; }
.v-low { border-color: #a8503f; } .v-low h3 { color: #7d3527; }
.players { display: flex; flex-wrap: wrap; gap: 16px; }
.player { flex: 1 1 260px; background: #fff; border: 1px solid #e2ded6;
  border-radius: 3px; padding: 13px 15px; }
.player b { display: block; font-size: 13px; font-weight: 600; margin-bottom: 3px; }
.player span { display: block; font-size: 12px; color: #6a665e; margin-bottom: 9px; }
audio { width: 100%; height: 34px; }
figure { margin: 0; background: #fff; border: 1px solid #e2ded6;
  border-radius: 3px; padding: 12px; overflow-x: auto; }
figure img { display: block; width: 100%; min-width: 640px; height: auto; }
figcaption { font-size: 12px; color: #6a665e; margin-top: 9px; }
table { width: 100%; border-collapse: collapse; background: #fff;
  border: 1px solid #e2ded6; border-radius: 3px; font-size: 14px; }
th, td { text-align: left; padding: 9px 14px; border-bottom: 1px solid #eeebe5;
  vertical-align: top; }
th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
  color: #6a665e; font-weight: 600; background: #faf9f6; }
tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; white-space: nowrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 13px; font-variant-numeric: tabular-nums; }
td.why { color: #5b574f; font-size: 13px; line-height: 1.5; }
.scroll { overflow-x: auto; }
.flag { color: #a8503f; font-weight: 600; }
.caveats { background: #fff; border: 1px solid #e2ded6; border-radius: 3px;
  padding: 6px 22px 14px; }
.caveats li { margin: 9px 0; font-size: 13.5px; color: #45413a; max-width: 74ch; }
footer { margin-top: 44px; padding-top: 16px; border-top: 1px solid #dcd8d0;
  font-size: 12px; color: #797469; max-width: 74ch; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px;
  background: #eeebe5; padding: 1px 4px; border-radius: 2px; }
"""

_VERDICT_HEADLINE = {
    "high": (
        "HIGH CONFIDENCE",
        "The transcription tracks the recording closely. Check it by ear "
        "anyway — the caveats below are not boilerplate.",
    ),
    "medium": (
        "MEDIUM CONFIDENCE",
        "The shape of the solo is right; the details need a human. Work "
        "through the weakest bars listed below.",
    ),
    "low": (
        "LOW CONFIDENCE",
        "Treat this as a sketch, not a transcription. Something upstream went "
        "wrong — wrong tempo, wrong downbeat, or audio the transcriber could "
        "not hear through.",
    ),
}


def _fmt(x, digits: int = 3, pct: bool = False, suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    if pct:
        return f"{float(x) * 100:.1f}%"
    return f"{float(x):.{digits}f}{suffix}"


def _metric_rows(m: dict) -> list[tuple[str, str, str, bool]]:
    """(label, value, plain-English meaning, flag_as_bad)."""
    q, c, cv = m["quantization"], m["chroma_dtw"], m["coverage"]
    rows = [
        (
            "Note agreement, ±100 ms",
            _fmt(q["f1_100ms"], pct=True),
            "Of everything the transcriber heard, how much still lines up "
            "after being written onto a musical grid. This is the headline "
            "number: how much of the playing survived being written down.",
            (q["f1_100ms"] or 0) < HIGH_F1,
        ),
        (
            "Note agreement, ±50 ms",
            _fmt(q["f1_50ms"], pct=True),
            "The same test with a stricter stopwatch. A big drop from the "
            "±100 ms figure means the notes are right but the rhythm is "
            "written a little loose.",
            False,
        ),
        (
            "Precision, ±50 ms",
            _fmt(q["precision_50ms"], pct=True),
            "Of the notes written in the tab, how many correspond to "
            "something actually played. Low means the tab has notes the "
            "recording doesn't.",
            (q["precision_50ms"] or 0) < 0.6,
        ),
        (
            "Recall, ±50 ms",
            _fmt(q["recall_50ms"], pct=True),
            "Of the notes actually played, how many made it into the tab. "
            "Low means notes are missing.",
            (q["recall_50ms"] or 0) < 0.6,
        ),
        (
            "Harmonic alignment cost",
            _fmt(c["normalized_cost"]),
            "How different the resynthesis sounds from the recording, "
            "pitch-class by pitch-class. 0 is identical. Above ~0.45 usually "
            "means wrong notes rather than wrong rhythm.",
            (c["normalized_cost"] or 0) > CHROMA_COST_BLOCK,
        ),
        (
            "Worst timing drift",
            _fmt(c["diagonality_sec"], digits=2, suffix=" s"),
            "The furthest the resynthesis had to be shifted to line up with "
            "the recording. Small means the tab follows the take; a large "
            "number means it slips out of time somewhere.",
            (c["diagonality_sec"] or 0) > DIAGONALITY_BLOCK_SEC,
        ),
        (
            "Typical timing drift",
            _fmt(c["median_deviation_sec"], digits=2, suffix=" s"),
            "The same measurement taken across the whole solo rather than at "
            "its worst point. If this is small and the figure above is large, "
            "the problem is one spot, not the whole transcription.",
            False,
        ),
        (
            "Onset coverage",
            _fmt(cv["onset_coverage"], pct=True),
            f"Share of the pick attacks audible in the recording that have a "
            f"note within {int(ONSET_TOL_SEC * 1000)} ms in the tab. Low means "
            "the tab is missing notes you can hear.",
            (cv["onset_coverage"] or 0) < HIGH_COVERAGE,
        ),
        (
            "Onset spuriousness",
            _fmt(cv["onset_spuriousness"], pct=True),
            "Share of notes in the tab with no audible attack anywhere near "
            "them. High means the tab is inventing notes. (Some of this is "
            "normal: tied and legato notes have no fresh attack to detect.)",
            (cv["onset_spuriousness"] or 0) > 0.4,
        ),
    ]
    return rows


def _build_html(
    metrics: dict,
    title: str,
    source_name: str,
    audio: list[tuple[str, str, str, str]],
    roll_b64: str,
    caveats: list[str],
) -> str:
    level = metrics["verdict"]["level"]
    headline, blurb = _VERDICT_HEADLINE[level]
    e = _html.escape
    out: list[str] = []
    a = out.append

    a("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    a("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    a(f"<title>{e(title)} — soloscribe audit</title>")
    a(f"<style>{_CSS}</style></head><body><div class='wrap'>")

    a("<header>")
    a(f"<h1>{e(title)}</h1>")
    a(
        "<p class='sub'>soloscribe transcription audit &middot; "
        f"{e(source_name)} &middot; compared against the "
        f"{e(metrics['comparison_source'])} &middot; "
        f"{datetime.now().strftime('%d %b %Y, %H:%M')}</p>"
    )
    a("</header>")

    a(f"<div class='verdict v-{level}' data-verdict='{level}'>")
    a(f"<h3>{headline}</h3>")
    a(f"<p class='lede'>{e(blurb)}</p><ul>")
    for reason in metrics["verdict"]["reasons"]:
        a(f"<li>{e(reason)}</li>")
    a("</ul></div>")

    a("<h2>Listen</h2>")
    a(
        "<p class='lede'>Play these against each other. Your ears will find "
        "the errors faster than the table below will, and the table exists "
        "mainly to tell you which bar to rewind to.</p>"
    )
    a("<div class='players'>")
    for label, note, mime, b64 in audio:
        a("<div class='player'>")
        a(f"<b>{e(label)}</b><span>{e(note)}</span>")
        a(f"<audio controls preload='none' src='data:{mime};base64,{b64}'></audio>")
        a("</div>")
    a("</div>")

    a("<h2>What was heard, what was written</h2>")
    a("<figure>")
    a(f"<img alt='piano roll of raw transcription against quantized score' "
      f"src='data:image/png;base64,{roll_b64}'>")
    a(
        "<figcaption>Grey outlines are what the transcriber heard, in real "
        "time. Solid blocks are what was written into the tab, shaded by how "
        "confident the transcriber was about each note. Vertical lines are the "
        "beat grid as laid back onto the recording — where a solid block sits "
        "well away from its outline, quantization moved that note."
        "</figcaption>"
    )
    a("</figure>")

    a("<h2>Metrics</h2>")
    a("<div class='scroll'><table><thead><tr>")
    a("<th>Measurement</th><th class='num'>Value</th><th>What it means</th>")
    a("</tr></thead><tbody>")
    for label, value, why, bad in _metric_rows(metrics):
        cls = " class='flag'" if bad else ""
        a(f"<tr><td>{e(label)}</td><td class='num'{cls}>{e(value)}</td>"
          f"<td class='why'>{e(why)}</td></tr>")
    a("</tbody></table></div>")

    rows = metrics["per_measure"]
    a("<h2>Bar by bar</h2>")
    a(
        "<p class='lede'>Timing error is how far the played notes had to move "
        "to land on the grid, averaged over the bar. A bar with high error and "
        "low confidence is where to start editing.</p>"
    )
    a("<div class='scroll'><table><thead><tr>")
    a("<th class='num'>Bar</th><th class='num'>Notes</th>"
      "<th class='num'>Mean confidence</th>"
      "<th class='num'>Timing error (RMS)</th><th>&nbsp;</th>")
    a("</tr></thead><tbody>")
    for r in rows:
        err = r["rms_quantization_error_ms"]
        conf = r["mean_confidence"]
        flags = []
        if err is not None and err > 60:
            flags.append("loose against the grid")
        if conf is not None and conf < 0.5:
            flags.append("transcriber unsure")
        if r["n_notes"] == 0:
            flags.append("no notes written")
        bad = " class='num flag'" if flags else " class='num'"
        a(
            f"<tr><td class='num'>{r['measure']}</td>"
            f"<td class='num'>{r['n_notes']}</td>"
            f"<td class='num'>{_fmt(conf, 2)}</td>"
            f"<td{bad}>{_fmt(err, 1, suffix=' ms')}</td>"
            f"<td class='why'>{e(', '.join(flags))}</td></tr>"
        )
    a("</tbody></table></div>")

    a("<h2>Caveats</h2>")
    a("<div class='caveats'><ul>")
    for c in caveats:
        a(f"<li>{e(c)}</li>")
    a("</ul></div>")

    a(
        "<footer>Generated by soloscribe. The audit resynthesizes the "
        "transcription as a plucked string, lays it back onto the original "
        "recording's clock through the score's tempo map, and compares the two "
        "— it is measuring its own output against the source, not asserting "
        "that its output is correct. Numbers here are evidence, not a grade. "
        "Where the report and your ears disagree, your ears are right."
        "</footer>"
    )
    a("</div></body></html>")
    return "".join(out)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def audit(
    score: Score,
    events: list[NoteEvent],
    original_path: str,
    stem_path: str | None,
    out_dir: str,
    caveats: list[str] | None = None,
) -> dict:
    """Audit a transcription against its source and write an HTML report.

    original_path must be the same span of audio the score covers — if the
    pipeline trimmed to [start, end] before transcribing, pass the trimmed
    clip. A score covering 20 seconds compared against a five-minute file
    produces numbers that are technically correct and completely useless; the
    report flags a large mismatch rather than quietly absorbing it.

    Returns the metrics dict documented at the top of this module and writes
    {out_dir}/report.html, {out_dir}/render.wav and {out_dir}/piano_roll.png.
    """
    os.makedirs(out_dir, exist_ok=True)
    comparison_path = stem_path if stem_path else original_path
    comparison_source = "stem" if stem_path else "original"

    y_cmp, _ = librosa.load(comparison_path, sr=SR, mono=True)
    y_ren = render_score(score, sr=SR, use_tempo_map=True)

    note_secs = score_note_seconds(score, use_tempo_map=True)
    score_onsets = np.array([t0 for _, t0, _ in note_secs], dtype=float)

    quant = _quantization_metrics(events, note_secs)
    chroma = _chroma_dtw_metrics(y_cmp, y_ren, SR)
    cov = _coverage_metrics(y_cmp, SR, score_onsets)
    per_measure = _per_measure_metrics(score, events, note_secs)

    dur_cmp = y_cmp.size / SR
    dur_ren = y_ren.size / SR
    all_caveats: list[str] = list(caveats or [])
    blockers: list[str] = []

    longer = max(dur_cmp, dur_ren)
    if longer > 0 and abs(dur_cmp - dur_ren) / longer > DURATION_MISMATCH_RATIO:
        msg = (
            f"The audio audited against is {dur_cmp:.1f} s long but the "
            f"transcription only covers {dur_ren:.1f} s. The harmonic and "
            "onset figures compare a part against a whole and are not "
            "trustworthy here."
        )
        all_caveats.append(msg)
        blockers.append(msg)
    if comparison_source == "original":
        all_caveats.append(
            "No isolated guitar track was available, so the audit compares "
            "against the full mix. Bass, drums and keys all count against the "
            "harmonic and onset figures, which makes those two numbers "
            "pessimistic — usually by a lot."
        )
    if not events:
        all_caveats.append(
            "No raw transcription events were supplied, so the note-agreement "
            "and bar-by-bar timing figures are empty rather than zero."
        )
    all_caveats.extend(STANDARD_CAVEATS)
    if _ffmpeg() is None:
        all_caveats.append(
            "ffmpeg was not found, so the audio below is embedded as WAV and "
            f"truncated at {int(MAX_EMBED_SEC)} s."
        )

    verdict = _verdict(quant, chroma, cov, blockers)

    render_path = os.path.join(out_dir, "render.wav")
    sf.write(render_path, y_ren, SR, subtype="PCM_16")
    roll_path = os.path.join(out_dir, "piano_roll.png")
    _piano_roll(score, events, note_secs, roll_path, max(dur_cmp, dur_ren))
    report_path = os.path.join(out_dir, "report.html")

    metrics = {
        "quantization": quant,
        "chroma_dtw": chroma,
        "coverage": cov,
        "per_measure": per_measure,
        "verdict": verdict,
        "caveats": all_caveats,
        "comparison_source": comparison_source,
        "render_path": render_path,
        "piano_roll_path": roll_path,
        "report_path": report_path,
        "durations": {"comparison_sec": dur_cmp, "render_sec": dur_ren},
    }

    with tempfile.TemporaryDirectory() as work:
        audio: list[tuple[str, str, str, str]] = []
        blocks = [
            ("Original", "what you gave us", original_path, "original"),
            ("Isolated guitar", "after source separation", stem_path, "stem"),
            (
                "Transcription, resynthesized",
                "the tab played back as a plucked string, on the recording's "
                "own clock",
                render_path,
                "render",
            ),
        ]
        for label, note, path, tag in blocks:
            if not path or not os.path.exists(path):
                continue
            enc = _encode_audio(path, work, tag)
            if enc:
                audio.append((label, note, enc[0], enc[1]))
        with open(roll_path, "rb") as fh:
            roll_b64 = base64.b64encode(fh.read()).decode("ascii")

    title = score.title or os.path.splitext(os.path.basename(original_path))[0]
    html_text = _build_html(
        metrics, title, os.path.basename(original_path), audio, roll_b64,
        all_caveats,
    )
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(html_text)

    return metrics
