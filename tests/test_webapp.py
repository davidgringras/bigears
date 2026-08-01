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
import re
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

PIPELINE_SOURCE = (REPO_ROOT / "soloscribe" / "pipeline.py").read_text(encoding="utf-8")

#: (stage name, fraction) for every progress report run_pipeline actually makes,
#: read out of its source in file order. The web UI's checklist and progress bar
#: are both derived from these strings, and neither is part of the frozen
#: contract — so read them rather than assume them. A stage rename in the
#: pipeline shows up here as a failure in this file, not as a stuck progress bar
#: in front of the person using it.
REPORTED_STAGES = [
    (name, float(frac))
    for name, frac in re.findall(
        r'_report\(\s*progress\s*,\s*"([^"]+)"\s*,\s*([0-9.]+)\s*\)', PIPELINE_SOURCE
    )
]


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
    for stage, frac in REPORTED_STAGES:
        if progress is not None:
            progress(stage, frac)
    _happy_pipeline.seen = kwargs
    return _result(out_dir, warnings=["Bars 9 to 12 were quiet; those notes are a guess."])


def _boom_pipeline(audio_path, out_dir, *, progress=None, **kwargs):
    if progress is not None:
        progress("Listening", 0.02)
    raise RuntimeError("boom")


def _unwired_pipeline(audio_path, out_dir, *, progress=None, **kwargs):
    raise NotImplementedError("wired up during integration")


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
        "start": "4", "end": "40", "capo": "3",
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
        "capo": 3,
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


def test_unimplemented_pipeline_is_reported_gracefully(client, monkeypatch):
    """A NotImplementedError is a missing part of the app, not the user's fault.

    run_pipeline was a stub when this server was written and is implemented now,
    so the stub is faked rather than called. The branch stays because a stage
    module can go back to raising it.
    """
    monkeypatch.setattr(server, "run_pipeline", _unwired_pipeline)
    job = _settle(client, _post(client).json()["job_id"])
    assert job["status"] == "error"
    assert "not connected" in job["error"]
    assert "wired up during integration" not in job["error"], "spare him the jargon"
    assert "NotImplementedError" in job["error_detail"]


class NoBackendError(Exception):
    """Stands in for audioread's, which is raised carrying no message at all."""


def test_an_undecodable_recording_is_explained_not_named(client, monkeypatch):
    def undecodable(audio_path, out_dir, *, progress=None, **kwargs):
        raise NoBackendError()

    monkeypatch.setattr(server, "run_pipeline", undecodable)
    job = _settle(client, _post(client).json()["job_id"])

    assert job["status"] == "error"
    assert job["error"] == server.PLAIN_ERRORS["NoBackendError"]
    assert "NoBackendError" not in job["error"], "that word means nothing to him"
    assert job["error_kind"] == "NoBackendError", "but David still needs the name"


def test_a_silent_exception_does_not_leave_the_apology_empty(client, monkeypatch):
    def mute(audio_path, out_dir, *, progress=None, **kwargs):
        raise ValueError()

    monkeypatch.setattr(server, "run_pipeline", mute)
    job = _settle(client, _post(client).json()["job_id"])

    assert job["status"] == "error"
    assert job["error"] not in ("", "ValueError"), "a bare class name is not an apology"
    assert len(job["error"]) > 20
    assert job["error_kind"] == "ValueError"


def test_an_oversized_recording_is_refused_before_any_work_starts(client, monkeypatch):
    """Cap tested by lowering it rather than by uploading 200 MB."""
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 512)
    response = _post(client, content=_wav_bytes(seconds=1.0))

    assert response.status_code == 413
    assert "200 MB" in response.json()["detail"]
    with server._LOCK:
        assert not server._JOBS


# .mp4 moved to the SUPPORTED list (iPhone videos; ffmpeg extracts the
# audio) — garbage bytes inside a supported container now fail at decode
# with the friendly message instead of at the extension gate.
@pytest.mark.parametrize("filename", ["riff.txt", "riff.docx", "riff"])
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


def test_the_pipeline_still_reports_stages_this_ui_can_read():
    """Drift guard: the checklist is derived from strings pipeline.py owns."""
    assert REPORTED_STAGES, "found no _report(progress, ...) calls to read"
    unmapped = [name for name, _ in REPORTED_STAGES if server._match_stage(name) < 0]
    assert not unmapped, f"no checklist step matches these pipeline stages: {unmapped}"

    indices = [server._match_stage(name) for name, _ in REPORTED_STAGES]
    assert indices == sorted(indices), (
        "the pipeline reports stages in an order the checklist would walk "
        f"backwards through: {list(zip([n for n, _ in REPORTED_STAGES], indices))}"
    )
    assert set(indices) == set(range(len(server.STAGES))), (
        "every checklist step should correspond to a stage the pipeline reports; "
        f"reached {sorted(set(indices))} of {len(server.STAGES)}"
    )


def test_progress_fractions_are_whole_run_not_per_stage():
    """The progress bar reads frac directly, which only holds if it is global."""
    fracs = [frac for _, frac in REPORTED_STAGES]
    assert fracs == sorted(fracs), f"fractions are not monotone: {fracs}"
    assert fracs[-1] == 1.0, f"the last report should be 1.0, got {fracs[-1]}"
    assert fracs[0] < 0.5, "a per-stage fraction would restart near 0 each stage"
    # A per-stage reading would have several stages ending at 1.0.
    assert fracs.count(1.0) == 1


def test_checklist_and_bar_track_a_full_run_of_the_real_stages(client):
    job = server.Job(id="ordering")
    with server._LOCK:
        server._JOBS["ordering"] = job
    job.status = "running"

    seen = []
    for stage, frac in REPORTED_STAGES:
        server._record_progress("ordering", stage, frac)
        seen.append((stage, job.step_index, job.overall))

    indices = [i for _, i, _ in seen]
    overalls = [o for _, _, o in seen]
    assert indices == sorted(indices), f"checklist walked backwards: {seen}"
    assert overalls == sorted(overalls), f"progress bar went backwards: {seen}"
    assert indices[0] == 0 and indices[-1] == len(server.STAGES) - 1
    assert not [entry for entry in job.history if entry["step"] is None]
    # The bar shows what the pipeline reports, not a re-derivation of it. Re-
    # deriving from the checklist position would have the bar reading 44 per
    # cent while the pipeline was telling us 62.
    assert overalls == [frac for _, frac in REPORTED_STAGES]


def test_skipping_isolation_drops_it_from_the_checklist(client, monkeypatch):
    monkeypatch.setattr(server, "run_pipeline", _happy_pipeline)

    kept = _settle(client, _post(client, data={"separate": "auto"}).json()["job_id"])
    skipped = _settle(client, _post(client, data={"separate": "off"}).json()["job_id"])

    keys = lambda job: [step["key"] for step in job["steps"]]
    assert "isolate" in keys(kept)
    assert "isolate" not in keys(skipped), "do not promise a step that was skipped"
    assert len(keys(skipped)) == len(keys(kept)) - 1


def test_out_of_order_stages_do_not_un_tick_the_checklist(client):
    """The stage names are not frozen; if the real order differs, degrade quietly."""
    job = server.Job(id="jumbled")
    with server._LOCK:
        server._JOBS["jumbled"] = job
    job.status = "running"

    server._record_progress("jumbled", "Writing the notation", 0.8)
    ahead = (job.step_index, job.overall)
    server._record_progress("jumbled", "Finding the beat", 0.1)

    assert job.step_index == ahead[0], "a late-arriving early stage rewound the checklist"
    assert job.overall == ahead[1], "a late-arriving early stage rewound the progress bar"
    assert job.stage == "Finding the beat", "the raw stage should still be reported honestly"


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
    assert "Big Ears" in body
    for asset in ("/static/style.css", "/static/app.js"):
        assert asset in body
        assert client.get(asset).status_code == 200
