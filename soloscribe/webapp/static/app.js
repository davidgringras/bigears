/* SoloScribe — the whole front end. No framework, no build step, no network
   beyond this machine. Pick a file, post it, poll the job, offer the results. */
(function () {
  "use strict";

  var POLL_MS = 1500;
  var MAX_BYTES = 200 * 1024 * 1024;
  var ALLOWED = [".mp3", ".wav", ".m4a", ".aiff", ".aif", ".flac"];

  var $ = function (id) { return document.getElementById(id); };

  var els = {
    veil: $("drop-veil"),
    form: $("form"),
    dropzone: $("dropzone"),
    file: $("file"),
    empty: $("dropzone-empty"),
    filled: $("dropzone-filled"),
    chosenName: $("chosen-name"),
    chosenMeta: $("chosen-meta"),
    browse: $("browse"),
    rechoose: $("rechoose"),
    fileError: $("file-error"),
    title: $("title"),
    submit: $("submit"),
    cards: {
      setup: $("card-setup"),
      working: $("card-working"),
      done: $("card-done"),
      error: $("card-error")
    },
    workingTitle: $("working-title"),
    meter: $("meter"),
    meterFill: $("meter-fill"),
    stageLabel: $("stage-label"),
    elapsed: $("elapsed"),
    checklist: $("checklist"),
    doneSub: $("done-sub"),
    downloadGp5: $("download-gp5"),
    openReport: $("open-report"),
    warnings: $("warnings"),
    warningsList: $("warnings-list"),
    again: $("again"),
    errorMessage: $("error-message"),
    errorKind: $("error-kind"),
    retry: $("retry")
  };

  var chosen = null;
  var jobId = null;
  var pollTimer = null;
  var startedAt = null;
  var dragDepth = 0;
  var renderedSteps = "";
  /* Whether the title in the box was put there by us rather than typed. Without
     this the title of an abandoned attempt sticks to the next recording, and it
     is the title that names the downloaded file. */
  var titleIsOurs = false;

  /* ------------------------------------------------------------ helpers */

  function show(which) {
    Object.keys(els.cards).forEach(function (name) {
      els.cards[name].hidden = name !== which;
    });
    if (which !== "setup") {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }

  function extensionOf(name) {
    var dot = name.lastIndexOf(".");
    return dot < 0 ? "" : name.slice(dot).toLowerCase();
  }

  function formatSize(bytes) {
    if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    if (bytes >= 1024) return Math.round(bytes / 1024) + " KB";
    return bytes + " bytes";
  }

  function formatClock(seconds) {
    var total = Math.max(0, Math.round(seconds));
    var mins = Math.floor(total / 60);
    var secs = total % 60;
    if (mins === 0) return secs + " seconds";
    if (mins === 1) return "1 minute " + secs + " seconds";
    return mins + " minutes " + secs + " seconds";
  }

  function setFileError(message) {
    els.fileError.textContent = message || "";
    els.fileError.hidden = !message;
  }

  /* -------------------------------------------------------- file choice */

  function chooseFile(file) {
    if (!file) return;
    var ext = extensionOf(file.name);
    if (ALLOWED.indexOf(ext) < 0) {
      chosen = null;
      renderChoice();
      setFileError(
        "I cannot read " + (ext || "files without an extension") +
        ". Please choose an MP3, WAV, M4A, AIFF or FLAC recording."
      );
      return;
    }
    if (file.size > MAX_BYTES) {
      chosen = null;
      renderChoice();
      setFileError(
        "That recording is " + formatSize(file.size) +
        ", which is over the 200 MB limit. Try a shorter excerpt."
      );
      return;
    }
    setFileError("");
    chosen = file;
    renderChoice();
    describeDuration(file);
    if (!els.title.value.trim() || titleIsOurs) {
      els.title.value = file.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim();
      titleIsOurs = true;
    }
  }

  els.title.addEventListener("input", function () { titleIsOurs = false; });

  function renderChoice() {
    var has = Boolean(chosen);
    els.empty.hidden = has;
    els.filled.hidden = !has;
    if (has) {
      els.chosenName.textContent = chosen.name;
      els.chosenMeta.textContent = formatSize(chosen.size);
    }
  }

  function describeDuration(file) {
    /* The browser can tell us how long it is; if it cannot, say nothing
       rather than guess. */
    var url = URL.createObjectURL(file);
    var probe = new Audio();
    var done = function () { URL.revokeObjectURL(url); };
    probe.addEventListener("loadedmetadata", function () {
      if (chosen === file && isFinite(probe.duration) && probe.duration > 0) {
        els.chosenMeta.textContent =
          formatClock(probe.duration) + " long, " + formatSize(file.size);
      }
      done();
    });
    probe.addEventListener("error", done);
    probe.src = url;
  }

  els.browse.addEventListener("click", function () { els.file.click(); });
  els.rechoose.addEventListener("click", function () { els.file.click(); });
  els.file.addEventListener("change", function () {
    chooseFile(els.file.files && els.file.files[0]);
  });
  els.file.addEventListener("focus", function () { els.dropzone.classList.add("is-focused"); });
  els.file.addEventListener("blur", function () { els.dropzone.classList.remove("is-focused"); });
  els.dropzone.addEventListener("click", function (event) {
    if (event.target === els.dropzone) els.file.click();
  });

  /* Dropping anywhere on the page works, not only on the dashed rectangle. */
  ["dragenter", "dragover", "dragleave", "drop"].forEach(function (name) {
    window.addEventListener(name, function (event) { event.preventDefault(); });
  });
  window.addEventListener("dragenter", function () {
    dragDepth += 1;
    if (!els.cards.setup.hidden) {
      els.veil.classList.add("is-showing");
      els.dropzone.classList.add("is-hot");
    }
  });
  window.addEventListener("dragleave", function () {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) clearDragState();
  });
  window.addEventListener("drop", function (event) {
    dragDepth = 0;
    clearDragState();
    if (els.cards.setup.hidden) return;
    var files = event.dataTransfer && event.dataTransfer.files;
    if (files && files.length) chooseFile(files[0]);
  });

  function clearDragState() {
    els.veil.classList.remove("is-showing");
    els.dropzone.classList.remove("is-hot");
  }

  /* ------------------------------------------------------------- submit */

  els.form.addEventListener("submit", function (event) {
    event.preventDefault();
    if (!chosen) {
      setFileError("Please choose a recording first.");
      els.dropzone.scrollIntoView({ block: "center", behavior: "smooth" });
      return;
    }
    startJob();
  });

  function startJob() {
    var data = new FormData();
    data.append("file", chosen, chosen.name);
    ["key", "bpm", "beats_per_bar", "chords", "title", "downbeat", "start", "end"]
      .forEach(function (name) {
        var el = els.form.elements[name];
        data.append(name, el ? el.value : "");
      });
    ["swing", "separate", "mode"].forEach(function (name) {
      var picked = els.form.querySelector('input[name="' + name + '"]:checked');
      data.append(name, picked ? picked.value : "auto");
    });

    els.submit.disabled = true;
    renderedSteps = "";
    els.workingTitle.textContent =
      "Working on " + ((els.title.value || "").trim() || chosen.name);
    els.stageLabel.textContent = "Sending your recording";
    els.elapsed.textContent = "";
    els.meterFill.style.width = "0%";
    els.checklist.innerHTML = "";
    show("working");

    /* XHR rather than fetch: a 200 MB file deserves an honest upload bar. */
    var request = new XMLHttpRequest();
    request.open("POST", "/api/jobs", true);
    request.upload.addEventListener("progress", function (event) {
      if (!event.lengthComputable) return;
      var pct = Math.round((event.loaded / event.total) * 100);
      els.stageLabel.textContent =
        pct >= 100 ? "Recording received" : "Sending your recording, " + pct + " per cent";
    });
    request.addEventListener("load", function () {
      els.submit.disabled = false;
      var payload = parseJson(request.responseText);
      if (request.status !== 200 || !payload || !payload.job_id) {
        fail((payload && payload.detail) ||
             "The transcriber turned the recording away and did not say why.");
        return;
      }
      jobId = payload.job_id;
      startedAt = Date.now();
      els.stageLabel.textContent = "Getting started";
      poll();
      pollTimer = window.setInterval(poll, POLL_MS);
    });
    request.addEventListener("error", function () {
      els.submit.disabled = false;
      fail("I lost contact with SoloScribe. Is the window that started it still open?");
    });
    request.send(data);
  }

  function parseJson(text) {
    try { return JSON.parse(text); } catch (err) { return null; }
  }

  /* --------------------------------------------------------------- poll */

  function poll() {
    if (!jobId) return;
    fetch("/api/jobs/" + encodeURIComponent(jobId), { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) throw new Error("status " + response.status);
        return response.json();
      })
      .then(render)
      .catch(function () {
        stopPolling();
        fail("I lost contact with SoloScribe. Is the window that started it still open?");
      });
  }

  function stopPolling() {
    if (pollTimer) window.clearInterval(pollTimer);
    pollTimer = null;
  }

  function render(job) {
    renderChecklist(job.steps || []);
    var pct = Math.round((job.overall || 0) * 100);
    els.meterFill.style.width = pct + "%";
    els.meter.setAttribute("aria-valuenow", String(pct));
    els.stageLabel.textContent = job.stage_label || "Working";
    if (startedAt) {
      els.elapsed.textContent = formatClock((Date.now() - startedAt) / 1000) + " so far";
    }

    if (job.status === "done") {
      stopPolling();
      finish(job);
    } else if (job.status === "error") {
      stopPolling();
      fail(job.error || "Something went wrong and I do not have a good explanation.",
           job.error_kind);
    }
  }

  function renderChecklist(steps) {
    var signature = steps.map(function (s) { return s.key + ":" + s.state; }).join(",");
    if (signature === renderedSteps) return;
    renderedSteps = signature;
    els.checklist.innerHTML = "";
    steps.forEach(function (step) {
      var li = document.createElement("li");
      li.setAttribute("data-state", step.state);
      var mark = document.createElement("span");
      mark.className = "step-mark";
      mark.setAttribute("aria-hidden", "true");
      var text = document.createElement("span");
      text.textContent = step.label;
      li.appendChild(mark);
      li.appendChild(text);
      els.checklist.appendChild(li);
    });
  }

  /* ------------------------------------------------------------ outcome */

  function finish(job) {
    var elapsed = job.elapsed ? formatClock(job.elapsed) : null;
    els.doneSub.textContent = elapsed
      ? "That took " + elapsed + "."
      : "";

    els.downloadGp5.href = (job.downloads && job.downloads.gp5) || "#";
    var report = job.downloads && job.downloads.report;
    els.openReport.hidden = !report;
    if (report) els.openReport.href = report;

    var warnings = job.warnings || [];
    els.warnings.hidden = warnings.length === 0;
    els.warningsList.innerHTML = "";
    warnings.forEach(function (warning) {
      var li = document.createElement("li");
      li.textContent = warning;
      els.warningsList.appendChild(li);
    });

    show("done");
  }

  function fail(message, kind) {
    stopPolling();
    els.submit.disabled = false;
    els.errorMessage.textContent = message;
    /* Gives him something useful to say on the phone instead of a bare token
       where the apology should be. */
    els.errorKind.hidden = !kind;
    if (kind) {
      els.errorKind.textContent = "";
      els.errorKind.appendChild(document.createTextNode("If you tell David about this, the name he needs is "));
      var code = document.createElement("code");
      code.textContent = kind;
      els.errorKind.appendChild(code);
      els.errorKind.appendChild(document.createTextNode("."));
    }
    show("error");
  }

  function reset() {
    stopPolling();
    jobId = null;
    startedAt = null;
    renderedSteps = "";
    chosen = null;
    els.file.value = "";
    if (titleIsOurs) {
      els.title.value = "";
      titleIsOurs = false;
    }
    renderChoice();
    setFileError("");
    els.submit.disabled = false;
    show("setup");
    els.dropzone.scrollIntoView({ block: "center", behavior: "smooth" });
  }

  els.again.addEventListener("click", reset);
  els.retry.addEventListener("click", reset);

})();
