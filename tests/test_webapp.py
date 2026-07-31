"""Tests for the local web UI.

The real ``run_pipeline`` is not implemented yet, so every test that needs a
finished job monkeypatches ``soloscribe.webapp.server.run_pipeline``. One test
deliberately does not, to pin the behaviour the page depends on today: an
unimplemented pipeline must surface as a job error, never as a hung job or a
500.
"""
from __future__ import annotations

import array
import io
import math
import os
import sys
import time
import wave
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `.venv/bin/pytest tests/test_webapp.py`
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from soloscribe.model import BeatGrid, Score  # noqa: E402
from soloscribe.pipeline import PipelineResult  # noqa: E402
from soloscribe.webapp import server  # noqa: E402

GP5_BYTES = b"FICHIER GUITAR PRO v5.10\x00fake-but-nonempty"
REPORT_HTML = "<!doctype html><title>Audit</title><p>4 of 5 notes matched.</p>"


def _wav_bytes(seconds: float = 0.25, rate: int = 8000, freq: float = 220.0) -> bytes:
    """A small real WAV, so the upload path is exercised with plausible bytes."""
    frames = array.array(
        "h",
        (int(12000 * math.sin(2 * math.pi * freq * i / rate)) for i in range(int(rate * seconds))),
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames.tobytes())
    return buf.getvalue()


def _result(out_dir: str, *, warnings: list[str] | None = None) -> PipelineResult:
    os.makedirs(out_dir, exist_ok=True)
    gp5 = os.path.join(out_dir, "solo.gp5")
    report = os.path.join(out_dir, "report.html")
    Path(gp5).write_bytes(GP5_BYTES)
    Path(report).write_text(REPORT_HTML, encoding="utf-8")
    return PipelineResult(
        gp5_path=gp5,
        report_path=report,
        score=Score(title="Take 3"),
        events=[],
        grid=BeatGrid(beat_times=[0.0, 0.5, 1.0, 1.5]),
        metrics={"notes_matched": 4, "notes_total": 5},
        warnings=list(warnings or []),
    )


def _happy_pipeline(audio_path, out_dir, *, progress=None, **kwargs):
    assert os.path.exists(audio_path), "the server must save the upload before running"
    for stage in ("load", "separate", "transcribe", "beat grid", "quantize",
                  "fret", "write GP5", "audit", "HTML report"):
        if progress is not None:
            progress(stage, 1.0)
    _happy_pipeline.seen = kwargs
    return _result(out_dir, warnings=["Bars 9 to 12 were quiet; those notes are a guess."])


def _boom_pipeline(audio_path, out_dir, *, progress=None, **kwargs):
    if progress is not None:
        progress("load", 0.5)
    raise RuntimeError("boom")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path / "output")
    with server._LOCK:
        server._JOBS.clear()
    yield TestClient(server.app)
    with server._LOCK:
        server._JOBS.clear()


def _post(client, *, filename="take.wav", data=None, content=None):
    form = {
        "key": "Am",
        "bpm": "",
        "beats_per_bar": "4",
        "swing": "auto",
        "separate": "auto",
        "mode": "solo",
        "chords": "Am7 | D7\nGmaj7",
        "title": "Dad's Blues Solo",
        "downbeat": "",
        "start": "",
        "end": "",
    }
    form.update(data or {})
    return client.post(
        "/api/jobs",
        files={"file": (filename, content if content is not None else _wav_bytes(), "audio/wav")},
        data=form,
    )


def _settle(client, job_id, timeout=15.0):
    """Poll like the page does, but faster, until the job stops moving."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"done", "error"}:
            return payload
        time.sleep(0.02)
    pytest.fail(f"job {job_id} never settled (last status: {payload['status']})")


def test_happy_path_produces_both_downloads(client, monkeypatch):
    monkeypatch.setattr(server, "run_pipeline", _happy_pipeline)

    created = _post(client)
    assert created.status_code == 200, created.text
    job_id = created.json()["job_id"]

    job = _settle(client, job_id)
    assert job["status"] == "done", job.get("error")
    assert job["overall"] == 1.0
    assert [s["state"] for s in job["steps"]] == ["done"] * 6
    assert job["warnings"] == ["Bars 9 to 12 were quiet; those notes are a guess."]
    assert job["metrics"] == {"notes_matched": 4, "notes_total": 5}
    assert job["downloads"] == {
        "gp5": f"/api/jobs/{job_id}/gp5",
        "report": f"/api/jobs/{job_id}/report",
    }
    # Every stage the fake reported was recognised by the checklist mapping.
    assert [entry["step"] for entry in job["history"]].count(None) == 0

    gp5 = client.get(f"/api/jobs/{job_id}/gp5")
    assert gp5.status_code == 200
    assert gp5.content == GP5_BYTES
    assert "Dad" in gp5.headers["content-disposition"]
    assert ".gp5" in gp5.headers["content-disposition"]

    report = client.get(f"/api/jobs/{job_id}/report")
    assert report.status_code == 200
    assert report.content.decode() == REPORT_HTML
    assert report.headers["content-disposition"].startswith("inline")


def test_form_fields_reach_run_pipeline_as_contract_kwargs(client, monkeypatch):
    monkeypatch.setattr(server, "run_pipeline", _happy_pipeline)
    _happy_pipeline.seen = None

    created = _post(client, data={
        "key": "Bb", "bpm": "138.5", "beats_per_bar": "3", "swing": "on",
        "separate": "off", "mode": "poly", "downbeat": "1.25",
        "start": "4", "end": "40",
    })
    _settle(client, created.json()["job_id"])

    assert _happy_pipeline.seen == {
        "key": "Bb",
        "bpm": 138.5,
        "beats_per_bar": 3,
        "swing": "on",
        "separate": "off",
        "mode": "poly",
        "chords": "Am7|D7|Gmaj7",  # textarea lines and pipes both become bars
        "title": "Dad's Blues Solo",
        "downbeat": 1.25,
        "start": 4.0,
        "end": 40.0,
    }


def test_pipeline_exception_becomes_a_job_error(client, monkeypatch):
    monkeypatch.setattr(server, "run_pipeline", _boom_pipeline)

    job_id = _post(client).json()["job_id"]
    job = _settle(client, job_id)

    assert job["status"] == "error"
    assert job["error"] == "boom"
    assert "RuntimeError" in job["error_detail"]
    assert [s["state"] for s in job["steps"]][0] == "failed"
    assert job["downloads"] == {"gp5": None, "report": None}
    assert client.get(f"/api/jobs/{job_id}/gp5").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/report").status_code == 404


def test_unimplemented_pipeline_is_reported_gracefully(client):
    """No monkeypatch: today's real run_pipeline raises NotImplementedError."""
    job = _settle(client, _post(client).json()["job_id"])
    assert job["status"] == "error"
    assert "not connected" in job["error"]
    assert "NotImplementedError" in job["error_detail"]


@pytest.mark.parametrize("filename", ["riff.txt", "riff.mp4", "riff"])
def test_non_audio_uploads_are_refused(client, filename):
    response = _post(client, filename=filename, content=b"not audio at all")
    assert 400 <= response.status_code < 500
    assert response.json()["detail"]
    with server._LOCK:
        assert not server._JOBS, "a rejected upload must not create a job"


@pytest.mark.parametrize("field,value", [
    ("key", "H"), ("bpm", "quickish"), ("swing", "sideways"),
    ("mode", "orchestral"), ("beats_per_bar", "99"), ("end", "2"),
])
def test_nonsense_options_are_refused_with_a_readable_message(client, field, value):
    data = {field: value}
    if field == "end":
        data["start"] = "30"  # finish before start
    response = _post(client, data=data)
    assert response.status_code == 400
    assert len(response.json()["detail"]) > 10


# The stages pipeline.py's docstring commits to, in the order it commits to.
DOCUMENTED_STAGES = (
    "load", "separate", "transcribe", "beat grid", "quantize", "fret",
    "write GP5", "synthesize score", "audit vs original", "HTML report",
)


def test_checklist_never_walks_backwards_through_the_real_stage_order(client):
    """Transcription runs before beat tracking, so the mapping must not un-tick."""
    job = server.Job(id="ordering")
    with server._LOCK:
        server._JOBS["ordering"] = job
    job.status = "running"

    seen = []
    for stage in DOCUMENTED_STAGES:
        server._record_progress("ordering", stage, 0.5)
        seen.append((stage, job.step_index, job.overall))

    unmapped = [entry["stage"] for entry in job.history if entry["step"] is None]
    assert not unmapped, f"no checklist step matches {unmapped}"
    indices = [i for _, i, _ in seen]
    assert indices == sorted(indices), f"checklist walked backwards: {seen}"
    overalls = [o for _, _, o in seen]
    assert overalls == sorted(overalls), f"progress bar went backwards: {seen}"
    assert indices[0] == 0 and indices[-1] == len(server.STAGES) - 1
    # Every checklist step is reached by at least one documented stage.
    assert set(indices) == set(range(len(server.STAGES)))


def test_out_of_order_stages_do_not_un_tick_the_checklist(client):
    """The stage names are not frozen; if the real order differs, degrade quietly."""
    job = server.Job(id="jumbled")
    with server._LOCK:
        server._JOBS["jumbled"] = job
    job.status = "running"

    server._record_progress("jumbled", "write GP5", 1.0)
    ahead = (job.step_index, job.overall)
    server._record_progress("jumbled", "beat grid", 0.1)

    assert job.step_index == ahead[0], "a late-arriving early stage rewound the checklist"
    assert job.overall == ahead[1], "a late-arriving early stage rewound the progress bar"
    assert job.stage == "beat grid", "the raw stage should still be reported honestly"


def test_an_unrecognised_stage_name_is_shown_rather_than_swallowed(client):
    job = server.Job(id="odd")
    with server._LOCK:
        server._JOBS["odd"] = job
    job.status = "running"
    server._record_progress("odd", "polishing the brass", 0.4)

    assert job.step_index == -1
    assert job.stage_label() == "Polishing the brass"
    assert job.history[-1]["step"] is None


def test_unknown_job_is_a_polite_404(client):
    response = client.get("/api/jobs/nope")
    assert response.status_code == 404
    assert "start it again" in response.json()["detail"]


def test_index_and_static_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200
    body = page.text
    assert "SoloScribe" in body
    for asset in ("/static/style.css", "/static/app.js"):
        assert asset in body
        assert client.get(asset).status_code == 200
