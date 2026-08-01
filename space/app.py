"""Big Ears on Hugging Face Spaces — the same pipeline, reachable from a phone.

The desktop app (``soloscribe/webapp/``) assumes a Mac with the repo checked
out. This is the same ``run_pipeline`` behind a Gradio page, so a guitarist can
open a URL on whatever device is to hand, drop in a recording, and get back the
Guitar Pro file and the audit report.

Three things about the free CPU tier shape this file:

  * TWO vCPUs. Demucs separation on 2 cores runs for minutes on a 90-second
    clip, so the isolation checkbox DEFAULTS TO OFF and says in its own label
    what it costs. The desktop app defaults it to "auto" because a laptop can
    afford it; here that default would read as a hang.
  * A SHARED, SLEEPING machine. Free Spaces sleep when idle, so the first
    request after a quiet spell pays a cold start on top of the transcription.
    Clips are capped at MAX_CLIP_SECONDS for the same reason — one long upload
    would hold the only worker while everyone else waits.
  * NOBODY IS WATCHING THE LOGS. Every failure has to come back as a sentence
    the guitarist can act on, so the handler catches everything and renders an
    apology rather than letting Gradio surface a traceback.

The transcription backend is NOT the same as on the developer's Mac — see
``pin_transcription_backend`` below, which is where that gets decided and said
out loud.
"""
from __future__ import annotations

import html
import os
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    # The Space runs from its own root with the package vendored alongside this
    # file (see make_space.py), where `python app.py` already puts this
    # directory first on sys.path — so this line is only insurance for the case
    # where something IMPORTS app.py from elsewhere.
    #
    # APPEND, do not insert. Prepending would let the vendored snapshot shadow
    # the real soloscribe/ package for the rest of the process, which in the
    # repo's own test run means every test after this one silently exercising a
    # copy that may be a commit or two behind. make_space.py --check is what
    # guards the copy; sys.path is the wrong place to do it.
    sys.path.append(str(HERE))

import gradio as gr

# Bound as a module global on purpose: the tests monkeypatch
# `app.run_pipeline`, exactly as tests/test_webapp.py does for the desktop
# server. The handler must therefore call the global, never a captured local.
from soloscribe.pipeline import run_pipeline

# --------------------------------------------------------------------------
# limits
# --------------------------------------------------------------------------

MAX_CLIP_SECONDS = 180.0
# A ceiling on the upload itself, so a mis-clicked album rip is rejected at the
# door rather than after it has been buffered. Three minutes of 24-bit stereo
# WAV at 96 kHz is about 100 MB; anything larger is not the clip they meant.
MAX_UPLOAD = "150mb"

# Everything the pipeline writes lands under here, and launch(allowed_paths=...)
# is told about it so Gradio will serve the finished files back.
OUTPUT_ROOT = Path(
    os.environ.get("SOLOSCRIBE_OUT", Path(tempfile.gettempdir()) / "soloscribe-space")
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

KEYS: list[tuple[str, str]] = [
    ("C major", "C"), ("D♭ major", "Db"), ("D major", "D"),
    ("E♭ major", "Eb"), ("E major", "E"), ("F major", "F"),
    ("F♯ major", "F#"), ("G major", "G"), ("A♭ major", "Ab"),
    ("A major", "A"), ("B♭ major", "Bb"), ("B major", "B"),
    ("A minor", "Am"), ("B♭ minor", "Bbm"), ("B minor", "Bm"),
    ("C minor", "Cm"), ("C♯ minor", "C#m"), ("D minor", "Dm"),
    ("E♭ minor", "Ebm"), ("E minor", "Em"), ("F minor", "Fm"),
    ("F♯ minor", "F#m"), ("G minor", "Gm"), ("G♯ minor", "G#m"),
]
KEY_CHOICES = [label for label, _ in KEYS]
KEY_VALUES = dict(KEYS)

FEEL_CHOICES = ["Let me work it out", "Swing", "Straight"]
FEEL_VALUES = {"Let me work it out": "auto", "Swing": "on", "Straight": "off"}

MODE_CHOICES = ["A single-note solo", "Chords as well (experimental — single notes are far more accurate)"]
MODE_VALUES = {"A single-note solo": "solo", "Chords as well (experimental — single notes are far more accurate)": "poly"}

BEATS_CHOICES = ["4/4 — four in a bar", "3/4 — waltz time", "6/8 — counted as six"]
BEATS_VALUES = {BEATS_CHOICES[0]: 4, BEATS_CHOICES[1]: 3, BEATS_CHOICES[2]: 6}

# Plain readings for failures whose own message would tell a guitarist nothing.
# Lifted from soloscribe/webapp/server.py so the two front ends apologise in the
# same words; keyed on class name so this layer need not import the audio stack.
PLAIN_ERRORS: dict[str, str] = {
    "NoBackendError": (
        "I could not read that recording at all. The file may be damaged, or it "
        "may not really be audio. Try playing it in another app to check, then "
        "save it again as a WAV or an MP3."
    ),
    "DecodeError": (
        "I could not read that recording at all. The file may be damaged, or it "
        "may not really be audio. Try playing it in another app to check, then "
        "save it again as a WAV or an MP3."
    ),
    "FileNotFoundError": "I lost track of your recording. Please choose it again.",
    "MemoryError": (
        "That recording is too big for me to hold all at once. Try a shorter "
        "stretch of it."
    ),
}

VERDICT_COPY: dict[str, tuple[str, str, str]] = {
    "high": (
        "high",
        "This one came out well.",
        "Most of what you played is on the page. Read the report anyway — it "
        "names the bars I was least sure of, and that is where your ears are "
        "worth more than mine.",
    ),
    "medium": (
        "medium",
        "A fair transcription, with places to check.",
        "Enough of it is right to be worth working from, but I was guessing in "
        "places. The report marks which bars; start there rather than at the "
        "beginning.",
    ),
    "low": (
        "low",
        "I struggled with this one.",
        "I would not learn this tab without the recording in your ears, bar by "
        "bar. The report says where it went wrong, and usually why.",
    ),
}

ACCENT = {"high": "#2f7d46", "medium": "#b0700d", "low": "#b3261e"}


# --------------------------------------------------------------------------
# transcription backend
# --------------------------------------------------------------------------

# basic-pitch ships the same model in four formats and picks one at import
# time, in a fixed if/elif order — TensorFlow, then CoreML, then TFLite, then
# ONNX (basic_pitch/__init__.py:81-88) — from whichever runtimes happen to be
# importable. Which of those pip installed is decided by markers on the
# package's own dependencies rather than by anything in requirements.txt:
# coremltools on macOS, tflite-runtime on Linux under Python 3.10, TensorFlow
# on Linux under Python 3.11 or newer (basic_pitch-0.4.0.dist-info/METADATA).
# So the developer's Mac runs CoreML and the Space runs something else, and the
# `[onnx]` extra alone does NOT make it ONNX — the extra installs onnxruntime,
# but ONNX is last in that chain and only wins when the other three are absent.
#
# That is a fragile way to decide what actually does the work, so this pins it:
# pick a backend whose runtime is importable AND whose model file is on disk,
# and hand that path to basic-pitch explicitly. `Model.__init__` still tries
# each present runtime in order (inference.py:77-150), but it only ever gets
# the one file, so the choice below is the choice that happens.
BACKEND_PREFERENCE = ("onnx", "tflite", "tf", "coreml")
_FLAG_FOR = {
    "onnx": "ONNX_PRESENT",
    "tflite": "TFLITE_PRESENT",
    "tf": "TF_PRESENT",
    "coreml": "CT_PRESENT",
}


def pin_transcription_backend() -> str:
    """Fix basic-pitch's model format. Returns a line for the Space logs.

    Never raises: a Space that will not start is worse than one that starts on
    whatever basic-pitch would have chosen unaided.
    """
    try:
        import basic_pitch
        import basic_pitch.inference as inference
        from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    except Exception as exc:
        # basic_pitch/__init__.py:95 raises NameError rather than anything
        # descriptive when no runtime at all is importable, because
        # _default_model_type is only assigned inside the if/elif chain.
        return (
            f"transcription backend: UNAVAILABLE ({type(exc).__name__}: {exc}). "
            "No transcription will be possible — check that basic-pitch and "
            "one of onnxruntime / tflite-runtime / tensorflow installed."
        )

    present = {name: bool(getattr(basic_pitch, flag, False)) for name, flag in _FLAG_FOR.items()}
    auto = getattr(getattr(basic_pitch, "ICASSP_2022_MODEL_PATH", None), "name", "unknown")

    for name in BACKEND_PREFERENCE:
        if not present[name]:
            continue
        try:
            path = build_icassp_2022_model_path(FilenameSuffix[name])
        except Exception:
            continue
        if not Path(path).exists():
            continue
        original = inference.predict

        def predict(audio_path, model_or_model_path=path, *args, **kwargs):
            return original(audio_path, model_or_model_path, *args, **kwargs)

        # transcribe.py imports `predict` from this module inside the function
        # body at call time (transcribe.py:562), so rebinding the attribute
        # here reaches it. Rebinding basic_pitch.ICASSP_2022_MODEL_PATH would
        # not: it is baked into predict's default argument at import.
        inference.predict = predict
        installed = ", ".join(n for n, ok in present.items() if ok) or "none"
        return (
            f"transcription backend: {name} ({Path(path).name}); "
            f"runtimes present: {installed}; basic-pitch would have chosen {auto}"
        )

    return (
        "transcription backend: left to basic-pitch "
        f"(would choose {auto}); no preferred runtime had a model file on disk"
    )


# --------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------


def probe_duration(path: str) -> float:
    """Seconds of audio, read from the header where possible.

    soundfile reads the header alone, which is instant and does not decode the
    file; librosa is the fallback for anything libsndfile will not open. A
    failure here is not fatal — the caller treats an unknown duration as
    acceptable and lets the pipeline be the one to complain about the file.
    """
    try:
        import soundfile as sf

        info = sf.info(path)
        if info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        import librosa

        return float(librosa.get_duration(path=path))
    except Exception:
        return 0.0


def _minutes(seconds: float) -> str:
    """'3 minutes 12 seconds' — how a person says a duration out loud."""
    total = int(round(seconds))
    m, s = divmod(total, 60)
    if m and s:
        return f"{m} minute{'s' if m != 1 else ''} {s} second{'s' if s != 1 else ''}"
    if m:
        return f"{m} minute{'s' if m != 1 else ''}"
    return f"{s} second{'s' if s != 1 else ''}"


def _number(value, label: str, *, minimum: float | None = None,
            zero_means_blank: bool = False) -> float | None:
    """A Gradio number box, which hands back None, '', a float or a string.

    ``zero_means_blank``: deployed Gradio number widgets were observed to hand
    back 0 for fields the user never touched (the first real user hit this —
    a phantom tempo of 0 refused the run). For fields where zero carries no
    meaning (a tempo, an end time), 0 is read as "left blank".
    """
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        raise ValueError(
            f"I could not read {label} as a number. Leave it blank if you are not sure."
        )
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(
            f"I could not read {label} as a number. Leave it blank if you are not sure."
        )
    if zero_means_blank and number == 0:
        return None
    if minimum is not None and number < minimum:
        raise ValueError(f"{label.capitalize()} cannot be less than {minimum:g}.")
    return number


def normalise_chords(raw: str | None) -> str | None:
    """Textarea (one bar per line, or bars split on |) into the pipeline's format."""
    text = (raw or "").strip()
    if not text:
        return None
    bars: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        for bar in line.split("|"):
            bar = " ".join(bar.split())
            if bar:
                bars.append(bar)
    return "|".join(bars) or None


# --------------------------------------------------------------------------
# the page's replies
# --------------------------------------------------------------------------


def _note(body: str, *, heading: str = "", accent: str = "#8a8a8a") -> str:
    head = f"<h3 style='margin:0 0 8px;font-size:17px;'>{html.escape(heading)}</h3>" if heading else ""
    paras = "".join(
        f"<p style='margin:0 0 10px;'>{html.escape(chunk.strip())}</p>"
        for chunk in body.split("\n\n")
        if chunk.strip()
    )
    return (
        f"<div style='border-left:4px solid {accent};padding:14px 18px;"
        f"background:rgba(127,127,127,0.09);border-radius:4px;line-height:1.55;'>"
        f"{head}{paras}</div>"
    )


def _bullets(heading: str, items: list[str]) -> str:
    if not items:
        return ""
    rows = "".join(f"<li style='margin-bottom:5px;'>{html.escape(str(i))}</li>" for i in items)
    return (
        f"<h4 style='margin:18px 0 6px;font-size:15px;'>{html.escape(heading)}</h4>"
        f"<ul style='margin:0;padding-left:20px;line-height:1.55;'>{rows}</ul>"
    )


def _variants_html(metrics: dict) -> str:
    rows = metrics.get("variants") or []
    if not rows:
        return ""
    items = "".join(
        "<li><strong>{slug}</strong> — {desc} <em>({n} notes, agreement "
        "{f1:.0%}, attacks covered {cov:.0%})</em></li>".format(
            slug=html.escape(str(v.get("slug", ""))),
            desc=html.escape(str(v.get("description", ""))),
            n=v.get("n_notes", "?"), f1=float(v.get("f1_100") or 0),
            cov=float(v.get("coverage") or 0))
        for v in rows)
    return (
        "<h4 style='margin:14px 0 6px'>Alternative readings</h4>"
        "<p style='margin:0 0 6px'>Where a call was genuinely close, the other "
        "reading is attached below, with the same measurements — choose with "
        "your ears and the numbers together.</p><ul>" + items + "</ul>")


def summary_html(result) -> str:
    """The verdict and the caveats, in the words the report itself uses."""
    metrics = dict(getattr(result, "metrics", None) or {})
    verdict = dict(metrics.get("verdict") or {})
    level = str(verdict.get("level") or "").lower()
    key, heading, blurb = VERDICT_COPY.get(
        level,
        (
            "unknown",
            "Finished, though I could not grade myself.",
            "The Guitar Pro file is written. My own check of it did not "
            "complete, so I have nothing to tell you about how much of it is "
            "right — treat it as unverified and use your ears.",
        ),
    )
    accent = ACCENT.get(key, "#8a8a8a")
    warnings = [str(w) for w in (getattr(result, "warnings", None) or [])]
    reasons = [str(r) for r in (verdict.get("reasons") or [])]

    parts = [
        f"<div style='border-left:5px solid {accent};padding:16px 20px;"
        f"background:rgba(127,127,127,0.09);border-radius:4px;line-height:1.55;'>",
        f"<h3 style='margin:0 0 8px;font-size:18px;'>{html.escape(heading)}</h3>",
        f"<p style='margin:0;'>{html.escape(blurb)}</p>",
        _bullets("Why I say that", reasons),
        _bullets("Worth knowing before you trust it", warnings),
        _variants_html(metrics),
        "<p style='margin:18px 0 0;font-size:14px;opacity:0.85;'>"
        "Both files are above. The report is a single web page — download it "
        "and open it in any browser; it has the piano roll, the bar-by-bar "
        "numbers, and your recording next to my reconstruction so you can hear "
        "the difference."
        "</p>",
        "</div>",
    ]
    return "".join(parts)


# --------------------------------------------------------------------------
# the handler
# --------------------------------------------------------------------------


def transcribe_solo(
    audio_path: str | None,
    key_label: str,
    tempo,
    feel_label: str,
    mode_label: str,
    chords: str | None,
    title: str | None,
    isolate: bool,
    beats_label: str = BEATS_CHOICES[0],
    downbeat=None,
    capo=None,
    start=None,
    end=None,
    progress=gr.Progress(),
):
    """One transcription, start to finish.

    Returns ``(gp5_path | None, report_path | None, html)``. It never raises:
    every exit renders a sentence into the third slot, because a traceback in
    front of a guitarist is the same as no answer at all.
    """
    def tick(frac: float, desc: str) -> None:
        if progress is None:
            return
        try:
            progress(frac, desc=desc)
        except Exception:
            # Progress reporting is decoration. Losing it must not lose the run.
            pass

    if not audio_path:
        return None, None, None, _note(
            "Choose a recording first — an MP3, WAV, M4A, AIFF, FLAC, or an MP4/MOV video of the "
            "solo you want written out.",
            heading="Nothing to listen to yet",
        )

    try:
        bpm = _number(tempo, "the tempo", minimum=20.0, zero_means_blank=True)
        downbeat_s = _number(downbeat, "where bar one begins", minimum=0.0, zero_means_blank=True)
        start_s = _number(start, "the start time", zero_means_blank=True, minimum=0.0)
        end_s = _number(end, "the end time", minimum=0.0, zero_means_blank=True)
        capo_fret = _number(capo, "the capo fret", minimum=0.0)
    except ValueError as exc:
        return None, None, None, _note(str(exc), heading="I could not read one of the boxes")

    if start_s is not None and end_s is not None and end_s <= start_s:
        return None, None, None, _note(
            "The end time needs to come after the start time.",
            heading="I could not read one of the boxes",
        )

    total = probe_duration(audio_path)
    # Cap what will actually be ANALYSED, not what was uploaded: someone who
    # trims a five-minute track down to a two-minute solo with the boxes below
    # has done exactly the right thing and should not be turned away for it.
    heard = total
    if total > 0:
        first = min(start_s or 0.0, total)
        last = min(end_s, total) if end_s is not None else total
        heard = max(0.0, last - first)
    if heard > MAX_CLIP_SECONDS:
        return None, None, None, _note(
            f"That is {_minutes(heard)} of audio, and I take three minutes at a "
            "time here — this runs on a small shared machine and a longer clip "
            "would keep everyone else queueing. Trim it to the solo itself "
            "before you upload, or open “If you know a little more” and give me "
            "a three-minute stretch of it with the From and To boxes.",
            heading="That clip is a little long for me",
        )

    out_dir = tempfile.mkdtemp(prefix="soloscribe-", dir=str(OUTPUT_ROOT))
    tick(0.0, "Getting started")
    try:
        result = run_pipeline(
            audio_path,
            out_dir,
            key=KEY_VALUES.get(key_label, "C"),
            bpm=bpm,
            beats_per_bar=BEATS_VALUES.get(beats_label, 4),
            swing=FEEL_VALUES.get(feel_label, "auto"),
            # Checked maps to "auto", not "on": both run Demucs and cost the
            # same minutes, but "auto" keeps the pipeline's own guards — it
            # falls back to the full mix when isolation turns out to have lost
            # notes (pipeline.py:119-136), which "on" forces it to ignore.
            separate="auto" if isolate else "off",
            mode=MODE_VALUES.get(mode_label, "solo"),
            chords=normalise_chords(chords),
            title=(title or "").strip(),
            downbeat=downbeat_s,
            start=start_s,
            end=end_s,
            capo=int(capo_fret or 0),
            progress=lambda stage, frac: tick(float(frac), str(stage) or "Working"),
        )
    except Exception as exc:  # noqa: BLE001 — a person is reading the result
        kind = type(exc).__name__
        message = PLAIN_ERRORS.get(kind) or str(exc).strip() or (
            "Something went wrong and it did not tell me what."
        )
        print(f"[soloscribe] {kind} while transcribing:\n{traceback.format_exc()}", flush=True)
        return None, None, None, _note(
            f"{message}\n\nYour recording is untouched. If it happens again with "
            f"the same file, try a shorter excerpt, or save it as a WAV and start "
            f"over. The error was called {kind}, which is the bit David will want.",
            heading="I am sorry, that did not work",
            accent=ACCENT["low"],
        )

    gp5 = getattr(result, "gp5_path", None)
    report = getattr(result, "report_path", None)
    gp5 = gp5 if gp5 and os.path.exists(gp5) else None
    report = report if report and os.path.exists(report) else None
    if gp5 is None:
        return None, None, None, _note(
            "The transcription finished but no Guitar Pro file came out of it. "
            "Please tell David — this one is not your fault.",
            heading="I am sorry, that did not work",
            accent=ACCENT["low"],
        )

    tick(1.0, "Finished")
    alts = [v["file"] for v in (result.metrics or {}).get("variants", [])
            if v.get("file") and os.path.exists(v["file"])] or None
    return gp5, report, alts, summary_html(result)


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

INTRO = """
# Big Ears

Give me a recording of a solo and I will write it out for you: a Guitar Pro
file you can slow down, loop and read, and an honest report on how much of it
I actually got right.

No transcription software gets everything right — anyone who says otherwise is
selling something. The difference here is that this one measures how much it
got right and shows you, so your cleanup time goes exactly where it is needed.
"""

CLOSING = """
Up to three minutes of audio at a time. A clip trimmed to just the solo works
better than a whole track. Nothing you upload is kept: the files are written to
a scratch folder on a machine that forgets them when it restarts.
"""


def build_ui() -> gr.Blocks:
    # theme belongs on launch() from Gradio 6.0 onward; passing it here warns.
    with gr.Blocks(title="Big Ears") as demo:
        gr.Markdown(INTRO)

        with gr.Row():
            with gr.Column(scale=3):
                audio = gr.File(
                    label="Your recording — MP3, WAV, M4A, AIFF, FLAC, or a "
                          "video (MP4/MOV); I use the sound track",
                    file_types=[".mp3", ".wav", ".m4a", ".aiff", ".aif",
                                ".flac", ".mp4", ".mov", ".aac", ".ogg"],
                    type="filepath",
                )
                gr.Markdown(
                    "### A few things that help me get it right\n"
                    "Every one of these is optional. If you are not sure, leave it be."
                )
                with gr.Row():
                    key = gr.Dropdown(
                        KEY_CHOICES, value=KEY_CHOICES[0], label="Key",
                        info="Sets how the sharps and flats get spelled.",
                    )
                    tempo = gr.Textbox(
                        label="Tempo, in beats per minute", value="",
                        placeholder="Leave blank",
                        info="Leave blank and I will work it out from your playing.",
                    )
                feel = gr.Radio(
                    FEEL_CHOICES, value=FEEL_CHOICES[0], label="Feel",
                    info="Swung eighths are the lilt in blues and jazz. Straight is rock and pop.",
                )
                mode = gr.Radio(
                    MODE_CHOICES, value=MODE_CHOICES[0], label="What am I listening for?",
                    info=(
                        "One note at a time is much easier to get right. Choose "
                        "Chords as well for Wes Montgomery-style octaves, which "
                        "single-note mode strips out as overtone ringing."
                    ),
                )
                chords = gr.Textbox(
                    label="Chord chart", lines=4,
                    placeholder="Fmaj7\nD7\nGm7 | C7\nF6",
                    info=(
                        "One bar per line, or separate the bars with a vertical "
                        "bar. Knowing the chords helps me choose sensible fingerings."
                    ),
                )
                title = gr.Textbox(
                    label="Title", placeholder="Untitled solo", max_lines=1,
                    info="This names the piece inside Guitar Pro, and the file you download.",
                )
                isolate = gr.Checkbox(
                    value=False,
                    label="Isolate the guitar first — adds several minutes",
                    info=(
                        "Stripping the band away helps me hear the guitar, but on "
                        "this small machine it is slow. Leave it off if the guitar "
                        "is already on its own, or if you would rather not wait."
                    ),
                )

                with gr.Accordion("If you know a little more", open=False):
                    beats = gr.Radio(
                        BEATS_CHOICES, value=BEATS_CHOICES[0], label="Time signature",
                        info="Nearly everything is 4/4.",
                    )
                    with gr.Row():
                        downbeat = gr.Textbox(
                            label="Where bar one begins, in seconds", value="",
                            placeholder="Leave blank",
                            info="The moment you would count “one”. Useful after a count-in.",
                        )
                        capo = gr.Textbox(
                            label="Capo, in frets", value="", placeholder="Leave blank",
                            info="If you played with a capo, the tab comes out relative to it.",
                        )
                    with gr.Row():
                        start = gr.Textbox(
                            label="Use only from, in seconds", value="",
                            placeholder="Leave blank",
                            info="Leave both blank to transcribe the whole thing.",
                        )
                        end = gr.Textbox(label="Use only up to, in seconds",
                                         value="", placeholder="Leave blank")

                go = gr.Button("Transcribe this solo", variant="primary")

            with gr.Column(scale=2):
                gp5_out = gr.File(label="The Guitar Pro file (.gp5)")
                report_out = gr.File(label="The report on itself (report.html)")
                alts_out = gr.Files(
                    label="Alternative readings — close calls, each scored",
                    visible=True,
                )
                summary_out = gr.HTML()
                gr.Markdown(CLOSING)

        go.click(
            fn=transcribe_solo,
            inputs=[audio, key, tempo, feel, mode, chords, title, isolate,
                    beats, downbeat, capo, start, end],
            outputs=[gp5_out, report_out, alts_out, summary_out],
            api_name="transcribe",
        )
    return demo


demo = build_ui()


def main() -> None:
    print(f"[soloscribe] {pin_transcription_backend()}", flush=True)
    # One heavy job at a time. Separation and transcription are both CPU-bound
    # and there are two cores: three concurrent jobs is a worse experience than
    # three sequential ones.
    demo.queue(default_concurrency_limit=1, max_size=8)
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ["GRADIO_SERVER_PORT"]) if os.environ.get("GRADIO_SERVER_PORT") else None,
        theme=gr.themes.Soft(),
        max_file_size=MAX_UPLOAD,
        allowed_paths=[str(OUTPUT_ROOT)],
        show_error=True,
    )


if __name__ == "__main__":
    main()
