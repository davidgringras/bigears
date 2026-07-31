"""Local web UI for soloscribe.

One page in front of the frozen ``run_pipeline``. The browser POSTs a recording
plus a handful of plain-language options to ``/api/jobs``; a daemon thread runs
the pipeline and reports progress into an in-process job store; the page polls
``/api/jobs/{id}`` until the Guitar Pro file and the audit report are ready.

Two deliberate constraints:

  * One heavy job at a time (a semaphore, not a thread pool). Separation and
    transcription are both CPU-hungry; three concurrent jobs on a laptop is a
    worse experience than three sequential ones, and the ``queued`` status
    exists precisely so the page can say so.
  * No authentication. Bind to 127.0.0.1 and keep it that way — see
    ``bin/Start SoloScribe.command``.

The job store is process-local, so this must run under a single uvicorn worker.
"""
from __future__ import annotations

import math
import os
import re
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..pipeline import run_pipeline

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
REPO_ROOT = HERE.parents[1]

# Module-level so tests can point them at a tmp_path. Both are created lazily,
# on first use, rather than at import or startup — creating them eagerly would
# defeat monkeypatching and litter the repo on a bare import.
UPLOAD_DIR = REPO_ROOT / "uploads"
OUTPUT_DIR = REPO_ROOT / "output"

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
# Multipart wrapping (boundaries, headers, the other form fields) adds a little
# on top of the file itself; the header check is a coarse early reject and the
# exact one happens against UploadFile.size.
_MULTIPART_SLACK = 1024 * 1024

AUDIO_SUFFIXES = (".mp3", ".wav", ".m4a", ".aiff", ".aif", ".flac")
SWING_CHOICES = ("auto", "on", "off")
SEPARATE_CHOICES = ("auto", "on", "off")
MODE_CHOICES = ("solo", "poly")
# Tonic, optional accidental, optional 'm' — matches model.Score.key ("F", "Bb",
# "Em"). A regex rather than a list of 24 so an enharmonic spelling from a
# future CLI ("Gb", "D#m") is not rejected for being unfashionable.
KEY_RE = re.compile(r"^[A-G](#|b)?m?$")

TOO_BIG_MESSAGE = (
    "That recording is larger than 200 MB. Try a shorter excerpt, or save it "
    "as an MP3 first."
)
NOT_WIRED_MESSAGE = (
    "The transcription engine is not connected in this build yet, so I could "
    "not turn your recording into notation. Nothing is wrong with your file."
)

_UNREADABLE = (
    "I could not read that recording at all. The file may be damaged, or it may "
    "not really be audio. Try playing it in another app to check, then save it "
    "again as a WAV or an MP3."
)

# Plain readings for failures whose own message would tell the person reading it
# nothing. Keyed on class name so this layer need not import the audio stack.
# NoBackendError and DecodeError come from audioread, whose docstring says "The
# file could not be decoded by any backend" — the wording below claims no more
# than that. Only add an entry whose meaning has been read from its source.
PLAIN_ERRORS: dict[str, str] = {
    "NoBackendError": _UNREADABLE,
    "DecodeError": _UNREADABLE,
    "FileNotFoundError": "I lost track of your recording. Please choose it again.",
    "MemoryError": (
        "That recording is too big for me to hold all at once. Try transcribing "
        "a shorter stretch of it."
    ),
}

# One checklist step per stage run_pipeline actually reports, in the order it
# reports them. Read off the _report(progress, ...) call sites in pipeline.py:
# Listening → Isolating the guitar → Working out the notes → Finding the beat
# → Writing the notation → Checking my work against your recording. Note that
# transcription really does run before beat tracking (the grid is tracked on
# the full mix while the notes come off the separated stem), and that fretting
# is reported inside the writing stage rather than on its own.
#
# The stage strings are not part of the frozen contract, so each step also
# carries substrings — matched against the whole reported name, taking the LAST
# step that matches — to survive a rename. test_webapp.py reads the literals
# straight out of pipeline.py and fails if any of them stops mapping.
STAGES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("listen", "Listening to your recording",
     ("listen", "load", "read", "decode", "resample", "audio")),
    ("isolate", "Isolating the guitar from the band",
     ("isolat", "separat", "stem", "demucs")),
    ("notes", "Working out the notes",
     ("note", "transcri", "pitch", "onset", "basic")),
    ("beat", "Finding the beat",
     ("beat", "tempo", "grid", "downbeat", "metre", "meter", "quantis", "quantiz")),
    ("write", "Writing the Guitar Pro file",
     ("writ", "notation", "gp5", "guitar pro", "guitarpro", "export", "fret", "finger")),
    ("check", "Checking my work against your recording",
     ("check", "audit", "synth", "resynth", "report", "compar")),
)
STAGE_KEYS = tuple(key for key, _, _ in STAGES)

app = FastAPI(title="SoloScribe", docs_url=None, redoc_url=None, openapi_url=None)


# --------------------------------------------------------------------------
# job store
# --------------------------------------------------------------------------


@dataclass
class Job:
    """One transcription, start to finish. Mutated only under ``_LOCK``."""

    id: str
    title: str = ""
    filename: str = ""
    separate: str = "auto"            # drives whether the isolation step shows
    status: str = "queued"            # queued | running | done | error
    stage: str = ""                   # raw stage name from the pipeline
    frac: float = 0.0                 # raw fraction from the pipeline
    overall: float = 0.0              # monotone 0..1 for the progress bar
    step_index: int = -1              # index into STAGES, -1 before the first
    error: str | None = None          # one friendly sentence for the page
    error_kind: str = ""              # exception class name, for reporting it on
    error_detail: str | None = None   # traceback, for whoever is integrating
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    result_paths: dict[str, str | None] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def steps(self) -> list[dict]:
        """The checklist the page renders, one entry per friendly stage."""
        out: list[dict] = []
        for i, (key, label, _) in enumerate(STAGES):
            if key == "isolate" and self.separate == "off":
                continue  # asked to skip it, so do not promise it
            if self.status == "done":
                state = "done"
            elif self.status == "error":
                state = "failed" if i == max(self.step_index, 0) else (
                    "done" if i < self.step_index else "pending"
                )
            elif i < self.step_index:
                state = "done"
            elif i == self.step_index and self.status == "running":
                state = "active"
            else:
                state = "pending"
            out.append({"key": key, "label": label, "state": state})
        return out

    def stage_label(self) -> str:
        if self.status == "queued":
            return "Waiting for the previous recording to finish"
        if self.status == "done":
            return "Finished"
        if 0 <= self.step_index < len(STAGES):
            return STAGES[self.step_index][1]
        if self.stage:
            return self.stage[:1].upper() + self.stage[1:]
        return "Getting started"

    def as_dict(self) -> dict:
        finished = self.finished_at or time.time()
        return {
            "id": self.id,
            "title": self.title,
            "filename": self.filename,
            "status": self.status,
            "stage": self.stage,
            "stage_label": self.stage_label(),
            "frac": self.frac,
            "overall": round(1.0 if self.status == "done" else self.overall, 4),
            "steps": self.steps(),
            "history": list(self.history),
            "error": self.error,
            "error_kind": self.error_kind,
            "error_detail": self.error_detail,
            "warnings": list(self.warnings),
            "metrics": self.metrics,
            "result_paths": dict(self.result_paths),
            "downloads": {
                "gp5": f"/api/jobs/{self.id}/gp5" if self.result_paths.get("gp5") else None,
                "report": f"/api/jobs/{self.id}/report" if self.result_paths.get("report") else None,
            },
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed": round(finished - (self.started_at or self.created_at), 2),
        }


_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()
_SLOT = threading.Semaphore(1)  # one heavy pipeline run at a time


def _match_stage(stage: str) -> int:
    """Map a raw pipeline stage name onto a checklist index, -1 if unrecognised."""
    needle = stage.lower()
    best = -1
    for i, (_, _, matchers) in enumerate(STAGES):
        if any(m in needle for m in matchers):
            best = i
    return best


def _clamp01(value: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return min(1.0, max(0.0, f))


def _record_progress(job_id: str, stage: str, frac: float) -> None:
    """Progress callback handed to run_pipeline. Runs on the worker thread."""
    now = time.time()
    stage = str(stage)
    frac = _clamp01(frac)
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job.stage = stage
        job.frac = frac
        idx = _match_stage(stage)
        if idx >= 0:
            # Never walk backwards: an unrecognised stage, or a pipeline that
            # revisits an earlier one, should not un-tick the checklist.
            job.step_index = max(job.step_index, idx)
        # run_pipeline reports frac as a WHOLE-RUN fraction: its call sites walk
        # 0.02 → 1.0 across the pipeline rather than restarting each stage. So
        # use it as given. A reporter that only marks stage boundaries would
        # send 0.0 forever, hence the fall back to the checklist position; the
        # clamp keeps the bar monotone under either.
        candidate = frac if frac > 0.0 else max(job.step_index, 0) / len(STAGES)
        job.overall = max(job.overall, min(1.0, candidate))
        job.history.append({
            "stage": stage,
            "frac": frac,
            "t": now,
            "elapsed": round(now - (job.started_at or job.created_at), 3),
            "step": STAGE_KEYS[idx] if idx >= 0 else None,
        })


def _run_job(job_id: str, audio_path: Path, out_dir: Path, options: dict) -> None:
    """Worker body. Every exit path must leave the job in done or error."""
    with _SLOT:
        with _LOCK:
            job = _JOBS.get(job_id)
            if job is None:
                return
            job.status = "running"
            job.started_at = time.time()
        try:
            os.makedirs(out_dir, exist_ok=True)
            # Resolved from the module globals at call time, which is what lets
            # the tests monkeypatch soloscribe.webapp.server.run_pipeline.
            result = run_pipeline(
                str(audio_path),
                str(out_dir),
                progress=lambda stage, frac: _record_progress(job_id, stage, frac),
                **options,
            )
        except NotImplementedError:
            _fail(job_id, NOT_WIRED_MESSAGE, traceback.format_exc(), "NotImplementedError")
            return
        except Exception as exc:  # noqa: BLE001 — the page shows this to a human
            message, kind = _explain(exc)
            _fail(job_id, message, traceback.format_exc(), kind)
            return
        _finish(job_id, result)


def _explain(exc: BaseException) -> tuple[str, str]:
    """A sentence for the person reading it, plus the name David would want."""
    kind = exc.__class__.__name__
    if kind in PLAIN_ERRORS:
        return PLAIN_ERRORS[kind], kind
    message = str(exc).strip()
    if message:
        return message, kind
    # Bare exceptions do exist — audioread's NoBackendError carries no message
    # at all — and the class name alone is not something to hand to a guitarist.
    return "Something went wrong and it did not tell me what.", kind


def _fail(job_id: str, message: str, detail: str | None = None, kind: str = "") -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job.status = "error"
        job.error = message
        job.error_detail = detail
        job.error_kind = kind
        job.finished_at = time.time()


def _finish(job_id: str, result) -> None:
    gp5 = getattr(result, "gp5_path", None)
    report = getattr(result, "report_path", None)
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job.result_paths = {
            "gp5": str(gp5) if gp5 and os.path.exists(gp5) else None,
            "report": str(report) if report and os.path.exists(report) else None,
        }
        job.warnings = list(getattr(result, "warnings", None) or [])
        job.metrics = dict(getattr(result, "metrics", None) or {})
        stem = getattr(result, "stem_path", None)
        job.result_paths["stem"] = str(stem) if stem and os.path.exists(stem) else None
        if not job.result_paths["gp5"]:
            job.status = "error"
            job.error = (
                "The transcription finished but no Guitar Pro file was written. "
                "Please tell David — this one is not your fault."
            )
        else:
            job.status = "done"
            job.step_index = len(STAGES) - 1
            job.overall = 1.0
        job.finished_at = time.time()


# --------------------------------------------------------------------------
# request parsing
# --------------------------------------------------------------------------


def _bad(message: str, status: int = 400) -> HTTPException:
    return HTTPException(status_code=status, detail=message)


def _opt_float(raw: str, label: str, *, minimum: float | None = None) -> float | None:
    text = (raw or "").strip().replace(",", ".")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        raise _bad(f"I could not read {label} as a number. Leave it blank if you are not sure.")
    if not math.isfinite(value):
        raise _bad(f"I could not read {label} as a number. Leave it blank if you are not sure.")
    if minimum is not None and value < minimum:
        raise _bad(f"{label.capitalize()} cannot be less than {minimum:g}.")
    return value


def _choice(raw: str, allowed: tuple[str, ...], label: str, default: str) -> str:
    value = (raw or "").strip().lower() or default
    if value not in allowed:
        raise _bad(f"I did not recognise the {label} setting '{value}'.")
    return value


def _normalise_chords(raw: str) -> str | None:
    """Textarea (one bar per line, or bars separated by |) → the pipeline's format."""
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


def _safe_stem(title: str) -> str:
    """A download filename a human would recognise, from a title they typed."""
    stem = re.sub(r"[^\w \-'()&.,]+", " ", title or "")
    stem = " ".join(stem.split()).strip(" .")
    return stem[:80] or "solo"


@app.middleware("http")
async def _limit_request_size(request: Request, call_next):
    """Reject an oversized upload from its Content-Length, before it is buffered."""
    if request.method == "POST":
        raw = request.headers.get("content-length") or ""
        if raw.isdigit() and int(raw) > MAX_UPLOAD_BYTES + _MULTIPART_SLACK:
            return JSONResponse(status_code=413, content={"detail": TOO_BIG_MESSAGE})
    return await call_next(request)


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    """Used by the launcher to wait for the server before opening a browser."""
    return {"ok": True, "app": "soloscribe"}


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    key: str = Form("C"),
    bpm: str = Form(""),
    beats_per_bar: str = Form("4"),
    swing: str = Form("auto"),
    separate: str = Form("auto"),
    mode: str = Form("solo"),
    chords: str = Form(""),
    title: str = Form(""),
    downbeat: str = Form(""),
    start: str = Form(""),
    end: str = Form(""),
) -> dict:
    name = (file.filename or "").strip()
    suffix = Path(name).suffix.lower()
    if not name or not suffix:
        raise _bad("Please choose an audio file — MP3, WAV, M4A, AIFF or FLAC.")
    if suffix not in AUDIO_SUFFIXES:
        raise _bad(
            f"I cannot read {suffix} files. Please choose an MP3, WAV, M4A, "
            "AIFF or FLAC recording."
        )
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise _bad(TOO_BIG_MESSAGE, status=413)

    key_value = (key or "").strip() or "C"
    if not KEY_RE.match(key_value):
        raise _bad(f"I did not recognise the key '{key_value}'.")

    try:
        beats = int((beats_per_bar or "4").strip() or 4)
    except ValueError:
        raise _bad("I could not read the time signature.")
    if not 1 <= beats <= 16:
        raise _bad("The time signature needs between 1 and 16 beats in a bar.")

    options = {
        "key": key_value,
        "bpm": _opt_float(bpm, "the tempo", minimum=1.0),
        "beats_per_bar": beats,
        "swing": _choice(swing, SWING_CHOICES, "feel", "auto"),
        "separate": _choice(separate, SEPARATE_CHOICES, "guitar isolation", "auto"),
        "mode": _choice(mode, MODE_CHOICES, "transcription", "solo"),
        "chords": _normalise_chords(chords),
        "title": " ".join((title or "").split())[:120],
        "downbeat": _opt_float(downbeat, "the first downbeat", minimum=0.0),
        "start": _opt_float(start, "the start time", minimum=0.0),
        "end": _opt_float(end, "the end time", minimum=0.0),
    }
    if options["start"] is not None and options["end"] is not None:
        if options["end"] <= options["start"]:
            raise _bad("The finish time needs to come after the start time.")

    job_id = uuid.uuid4().hex[:12]
    upload_dir = Path(UPLOAD_DIR) / job_id
    out_dir = Path(OUTPUT_DIR) / job_id
    os.makedirs(upload_dir, exist_ok=True)

    audio_path = upload_dir / (Path(name).name or f"recording{suffix}")
    written = 0
    with open(audio_path, "wb") as fh:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                fh.close()
                audio_path.unlink(missing_ok=True)
                raise _bad(TOO_BIG_MESSAGE, status=413)
            fh.write(chunk)
    await file.close()
    if written == 0:
        audio_path.unlink(missing_ok=True)
        raise _bad("That file is empty. Please choose the recording again.")

    job = Job(
        id=job_id,
        title=options["title"],
        filename=Path(name).name,
        separate=options["separate"],
    )
    with _LOCK:
        _JOBS[job_id] = job
    threading.Thread(
        target=_run_job,
        args=(job_id, audio_path, out_dir, options),
        name=f"soloscribe-{job_id}",
        daemon=True,
    ).start()
    return {"job_id": job_id, "status": job.status}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail="I have lost track of that transcription. Please start it again.",
            )
        return job.as_dict()


def _artifact(job_id: str, kind: str) -> tuple[Job, str]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="I have lost track of that transcription.")
        status, path = job.status, job.result_paths.get(kind)
    if status == "error":
        raise HTTPException(status_code=404, detail="That transcription did not finish.")
    if status != "done":
        raise HTTPException(status_code=404, detail="That transcription is still running.")
    if not path or not os.path.exists(path):
        missing = "Guitar Pro file" if kind == "gp5" else "accuracy report"
        raise HTTPException(status_code=404, detail=f"There is no {missing} for this recording.")
    return job, path


@app.get("/api/jobs/{job_id}/gp5")
def download_gp5(job_id: str) -> FileResponse:
    job, path = _artifact(job_id, "gp5")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"{_safe_stem(job.title)}.gp5",
    )


@app.get("/api/jobs/{job_id}/report")
def download_report(job_id: str) -> FileResponse:
    job, path = _artifact(job_id, "report")
    # inline, not attachment: "Open the accuracy report" should open a tab.
    return FileResponse(
        path,
        media_type="text/html",
        filename=f"{_safe_stem(job.title)} - accuracy report.html",
        content_disposition_type="inline",
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8746, log_level="warning")
