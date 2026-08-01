"""SoloScribe on Streamlit Community Cloud.

The free web tier of the transcription app: upload a solo, get back a Guitar
Pro file and the honest audit report. Guitar isolation is OFF here — the
Mac app carries it — and the interface says so rather than hiding it.

`pin_transcription_backend` and `probe_duration` are copied from
space/app.py in the main repository (same author, same session, documented
there); a gradio import at that module's top level is the only reason this
file does not import them.
"""
from __future__ import annotations

import os
import tempfile
import traceback
from pathlib import Path

import streamlit as st

MAX_SECONDS = 180.0
KEYS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb",
        "Am", "Em", "Bm", "F#m", "C#m", "G#m", "Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm"]

BACKEND_PREFERENCE = ["ONNX", "COREML", "TFLITE", "TF"]
_FLAG_FOR = {"ONNX": "ONNX_PRESENT", "COREML": "CT_PRESENT",
             "TFLITE": "TFLITE_PRESENT", "TF": "TF_PRESENT"}


def pin_transcription_backend() -> str:
    """Fix basic-pitch's model format. Never raises. (Copied: space/app.py.)"""
    try:
        import basic_pitch
        import basic_pitch.inference as inference
        from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    except Exception as exc:
        return (f"transcription backend: UNAVAILABLE ({type(exc).__name__}: {exc})")

    present = {name: bool(getattr(basic_pitch, flag, False))
               for name, flag in _FLAG_FOR.items()}
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

        inference.predict = predict
        return f"transcription backend: {name} ({Path(path).name})"
    return f"transcription backend: left to basic-pitch (would choose {auto})"


def probe_duration(path: str) -> float:
    """Seconds of audio from the header where possible. (Copied: space/app.py.)"""
    try:
        import soundfile as sf
        return float(sf.info(path).duration)
    except Exception:
        pass
    try:
        import librosa
        return float(librosa.get_duration(path=path))
    except Exception:
        return 0.0


def nice_duration(seconds: float) -> str:
    m, s = int(seconds // 60), int(round(seconds % 60))
    if m and s:
        return f"{m} minute{'s' if m != 1 else ''} {s} seconds"
    if m:
        return f"{m} minute{'s' if m != 1 else ''}"
    return f"{s} seconds"


def process(audio_path: str, out_dir: str, *, key: str, bpm: float | None,
            beats_per_bar: int, swing: str, mode: str, chords: str | None,
            title: str, downbeat: float | None, start: float | None,
            end: float | None, capo: int, progress=None):
    """Plain-function core so tests can drive it without Streamlit."""
    from soloscribe.pipeline import run_pipeline

    return run_pipeline(
        audio_path, out_dir, key=key, bpm=bpm, beats_per_bar=beats_per_bar,
        swing=swing, separate="off", mode=mode, chords=chords, title=title,
        downbeat=downbeat, start=start, end=end, capo=capo, progress=progress,
    )


def _opt(x) -> float | None:
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


st.set_page_config(page_title="SoloScribe", page_icon="🎸", layout="centered")

if "backend_note" not in st.session_state:
    st.session_state.backend_note = pin_transcription_backend()
    print(st.session_state.backend_note, flush=True)

st.title("SoloScribe")
st.markdown(
    "Give me a recording of a guitar solo and I will write it out for you: a "
    "**Guitar Pro file** you can slow down, loop and read, and an honest "
    "**report** on how much of it I actually got right — you can listen to "
    "your recording and my reading of it side by side.\n\n"
    "*This is the web version: it listens to your clip exactly as it is. If "
    "the guitar is buried in a loud band, the Mac app's isolation step does "
    "better — or send the clip to David.*"
)

with st.form("job"):
    clip = st.file_uploader(
        "Your recording — MP3, WAV, M4A, AIFF or FLAC, up to 3 minutes",
        type=["mp3", "wav", "m4a", "aiff", "aif", "flac"],
    )
    c1, c2 = st.columns(2)
    with c1:
        key = st.selectbox("Key", KEYS, index=8)  # F, the jazz default
        feel = st.radio("Feel", ["Let me detect it", "Swing", "Straight"],
                        horizontal=True)
        listening = st.radio("What am I listening for?",
                             ["A single-note solo", "Chords too"], horizontal=True)
    with c2:
        bpm_in = st.text_input("Tempo, beats per minute",
                               placeholder="Leave blank and I'll work it out")
        timesig = st.selectbox("Time signature", ["4/4", "3/4", "6/8"])
        title = st.text_input("Title", placeholder="What shall I call it?")
    chords = st.text_area(
        "Chord chart (optional)", height=80,
        placeholder="One bar per line, or separated by | — e.g.  Gm7 | C7 | Fmaj7",
    )
    with st.expander("If you know a little more"):
        d1, d2 = st.columns(2)
        with d1:
            downbeat_in = st.text_input("Where bar one begins (seconds in)",
                                        placeholder="Leave blank")
            capo_in = st.text_input("Capo fret", placeholder="Leave blank for none")
        with d2:
            start_in = st.text_input("Use from (seconds)", placeholder="Start")
            end_in = st.text_input("…to (seconds)", placeholder="End")
    go = st.form_submit_button("Transcribe", type="primary",
                               use_container_width=True)

if go:
    if clip is None:
        st.warning("Choose a recording first — MP3, WAV, M4A, AIFF or FLAC.")
        st.stop()
    workdir = tempfile.mkdtemp(prefix="soloscribe_")
    src = os.path.join(workdir, clip.name)
    with open(src, "wb") as f:
        f.write(clip.getbuffer())

    start_v, end_v = _opt(start_in), _opt(end_in)
    total = probe_duration(src)
    effective = (min(end_v, total) - (start_v or 0.0)) if end_v else (total - (start_v or 0.0))
    if total and effective > MAX_SECONDS:
        st.error(
            f"That clip runs to {nice_duration(total)} — more than I can chew "
            "through here. Trim it to the solo, or use the From/To boxes under "
            "'If you know a little more' to point me at the right part."
        )
        st.stop()

    capo_v = int(_opt(capo_in) or 0)
    stages = st.status("Working on it…", expanded=True)

    def on_progress(stage: str, frac: float):
        stages.update(label=f"{stage}…")
        stages.write(f"{stage} — {frac:.0%}")

    try:
        result = process(
            src, os.path.join(workdir, "out"),
            key=key, bpm=_opt(bpm_in),
            beats_per_bar={"4/4": 4, "3/4": 3, "6/8": 6}[timesig],
            swing={"Let me detect it": "auto", "Swing": "on", "Straight": "off"}[feel],
            mode={"A single-note solo": "solo", "Chords too": "poly"}[listening],
            chords=chords or None, title=title or "", downbeat=_opt(downbeat_in),
            start=start_v, end=end_v, capo=capo_v, progress=on_progress,
        )
        stages.update(label="Done.", state="complete", expanded=False)
    except Exception as exc:
        stages.update(label="Something went wrong.", state="error")
        st.error(
            "I couldn't finish that one, and it's my fault, not yours. If you "
            f"tell David, the name he needs is `{type(exc).__name__}`."
        )
        print(traceback.format_exc(), flush=True)
        st.stop()

    verdict = (result.metrics or {}).get("verdict", {})
    level = verdict.get("level", "unknown")
    badge = {"high": "🟢 High fidelity", "medium": "🟡 Medium fidelity",
             "low": "🔴 Low fidelity"}.get(level, level)
    st.subheader(badge)
    for reason in verdict.get("reasons", []):
        st.markdown(f"- {reason}")
    if result.warnings:
        with st.expander("Worth knowing before you trust it", expanded=True):
            for w in result.warnings:
                st.markdown(f"- {w}")

    g1, g2 = st.columns(2)
    with g1:
        with open(result.gp5_path, "rb") as f:
            st.download_button("Download the Guitar Pro file (.gp5)", f.read(),
                               file_name=os.path.basename(result.gp5_path),
                               use_container_width=True)
    with g2:
        if result.report_path and os.path.exists(result.report_path):
            with open(result.report_path, "rb") as f:
                st.download_button("Download the report", f.read(),
                                   file_name="report.html", mime="text/html",
                                   use_container_width=True)
    if result.report_path and os.path.exists(result.report_path):
        with st.expander("Read the report here"):
            import streamlit.components.v1 as components
            components.html(open(result.report_path).read(), height=900,
                            scrolling=True)
