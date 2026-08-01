---
title: Big Ears
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 6.22.0
python_version: "3.10"
app_file: app.py
license: other
short_description: Audio in, Guitar Pro out, with an honest report on itself.
---

# Big Ears

Give it a recording of a guitar solo and it writes the solo out: a Guitar Pro
`.gp5` file with notation, tab and playable fingerings, plus a report in which
it resynthesizes its own transcription, lines that up against your recording,
and says bar by bar how much of it to trust.

Up to three minutes of audio at a time. Guitar isolation is off by default —
this runs on two shared CPU cores, where Demucs takes minutes rather than
seconds.

---

# Deploying this — notes for David

Everything below the front matter is repo documentation; the Space itself is
`app.py`. The directory you are reading is the complete Space: push it as-is.

## Before you start: this may cost money

Hugging Face's own documentation now says, on
[Spaces Overview](https://huggingface.co/docs/hub/spaces-overview):

> Static Spaces are free for everyone. Gradio and Docker Spaces run on compute
> and require a paid plan to create: PRO for personal accounts, Team or
> Enterprise for organizations. Free personal accounts in good standing can
> still host up to 2 Gradio Spaces running on ZeroGPU.

So the "free CPU basic" assumption this was built against no longer holds for
*creating* a Gradio Space on a personal account. Three ways forward, in the
order I would try them:

1. **PRO account** ($9/month at the time of writing). CPU Basic hardware then
   costs nothing per hour; you are paying for the right to run a Gradio Space
   at all.
2. **ZeroGPU**, the documented free exception (up to two Gradio Spaces). I have
   not tested this app on ZeroGPU and would not assume it works unchanged —
   ZeroGPU Spaces expect GPU-decorated functions, and this pipeline is
   CPU-bound end to end. Treat it as an experiment, not a fallback.
3. **Somewhere else entirely.** The app is a plain Gradio program with no
   Hugging Face dependencies; `python app.py` serves it anywhere that has
   Python 3.10 and the packages in `requirements.txt`.

I could not verify current pricing or plan mechanics beyond the documentation
quoted above.

## The three steps

**1. Get an account.** [huggingface.co/join](https://huggingface.co/join).
Then, on this machine:

```bash
pip install -U "huggingface_hub"   # provides the `hf` command
hf auth login                      # paste a token with WRITE permission
```

Tokens are made at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). A
read token will let you log in and then fail at the push, which is a confusing
way to find out, so make it a write token.

**2. Create the Space.** [huggingface.co/new-space](https://huggingface.co/new-space).
Set:

- **Space name**: `soloscribe` (the URL becomes `huggingface.co/spaces/<you>/soloscribe`)
- **License**: your choice — see the note below, the repo currently declares none
- **SDK**: **Gradio** (do not pick the Gradio template; this repo replaces it)
- **Hardware**: **CPU basic**
- **Visibility**: public or private, both work. Private means only you can open
  the app, which for a link you are sending to one guitarist is arguably the
  right answer.

**3. Push this directory.** From the repo root:

```bash
.venv/bin/python space/make_space.py          # vendor the package in
cd space
hf upload <your-username>/soloscribe . . --repo-type=space --exclude="**/__pycache__/*"
```

`hf upload` takes `[repo_id] [local_path] [path_in_repo]`, so `. .` means "this
directory, at the root of the Space". The build starts by itself. Watch it
under the **Logs** tab on the Space page.

The `--exclude` is belt and braces: `make_space.py` already prunes bytecode
caches on every run, but running the app locally from inside `space/` puts them
straight back, and I did not confirm whether `hf upload` reads `.gitignore`.

Git works too, if you prefer to see what you are sending:

```bash
git clone https://huggingface.co/spaces/<your-username>/soloscribe
cp -R space/. soloscribe/
cd soloscribe && git add -A && git commit -m "Big Ears" && git push
```

Over HTTPS, git asks for your username and then a **write token** as the
password — not your account password.

## Redeploying after changing the pipeline

The Space carries a *copy* of `soloscribe/`, taken at packaging time. A change
to `soloscribe/transcribe.py` does not reach the Space until you rerun the
packaging script:

```bash
.venv/bin/python space/make_space.py --check   # has the copy drifted?
.venv/bin/python space/make_space.py           # refresh it
cd space && hf upload <your-username>/soloscribe . . --repo-type=space
```

`--check` exits non-zero when the vendored copy no longer matches the package,
which is the cheap way to notice before pushing rather than after.

## The first build takes a while

Ten minutes is an optimistic estimate and I have not measured it on the
platform. What is being downloaded and installed, with sizes I did verify:

| What | Size | Why |
| --- | --- | --- |
| `torch==2.13.0+cpu` | 183 MB | Demucs needs it. Verified against `download.pytorch.org/whl/cpu`. |
| `fluid-soundfont-gm` | ~120 MB apt | The sampled-guitar soundfont. See the note below — currently inert. |
| `basic-pitch` + `onnxruntime` | ~17 MB + models | The transcription model ships inside the wheel; nothing downloads at runtime. |
| `gradio`, `librosa`, `scipy`, `matplotlib` | tens of MB | The usual. |

Two runtime downloads happen later, not during the build:

- **Demucs model weights (~300 MB)** land on first use of the isolation
  checkbox, into the Space's ephemeral disk. Hugging Face documents 50 GB of
  *non-persistent* disk, so this re-downloads after any rebuild or restart. The
  first isolated transcription is therefore much slower than the ones after it.
- Nothing else. The transcription model is in the wheel.

Free Spaces sleep when idle, so the first request after a quiet spell also pays
a cold start.

## Two things I would change if the build fails

**If `basic-pitch` drags TensorFlow in**, the `python_version: "3.10"` line in
the front matter has stopped being honoured. basic-pitch 0.4.0's own metadata
makes TensorFlow a hard dependency on Linux under Python 3.11 and newer, and
`tflite-runtime` a hard dependency under 3.10 — the `[onnx]` extra does not
change that, it only adds onnxruntime alongside. TensorFlow 2.15.0 is a 475 MB
wheel that also pins `numpy<2`, which is why the runtime is pinned to 3.10.

**If the soundfont install is slow**, you can delete `fluid-soundfont-gm` and
`fluidsynth` from `packages.txt`, but you would be giving something up. Both
are load-bearing as of this packaging: `soloscribe/synth.py` gained
`render_events_fluid`, `audit.py:919` calls it, and it voices the report's
resynthesis with a sampled guitar instead of a plucked string — much easier to
compare against your own playing by ear. It finds the binary with
`shutil.which("fluidsynth")` and the soundfont at
`/usr/share/sounds/sf2/FluidR3_GM.sf2`, which is exactly the path Debian's
`fluid-soundfont-gm` installs (verified against Debian's package file list).

Remove them and nothing breaks: `render_events_fluid` returns `None` when
either is missing, logs the reason, and the report falls back to the
Karplus-Strong synthesis and says so. You lose the sampled voice, not the
report.

That work landed from a parallel session *while this Space was being packaged*,
so `synth.py` is the one file here most likely to be a snapshot of something
still moving. `make_space.py --check` is the way to find out.

## About the license field

The front matter says `license: other` because this repository declares no
license — there is no `LICENSE` file and no license statement in the README. I
did not want to invent one. Set it to whatever you actually intend before the
Space goes public; the field is a plain string in the front matter above.

## What runs where

The developer machine and the Space do not use the same transcription backend,
and it is worth knowing which is which when a result differs:

| | This Mac | The Space |
| --- | --- | --- |
| Python | 3.11 | 3.10 |
| basic-pitch backend | CoreML | ONNX Runtime, pinned by `app.py` |
| Chosen by | basic-pitch's macOS dependency markers | `pin_transcription_backend()` in `app.py` |

Same model, different runtimes. `app.py` pins the choice rather than leaving it
to whatever pip happened to install, and prints the result to the Space logs at
startup: look for a line beginning `[soloscribe] transcription backend:`.

## What I verified, and what I did not

**Verified on this machine.** The app's request path end to end against
`tests/fixtures/blues_a__none.wav` — a real `.gp5` that `guitarpro.parse`
reads, and a real `report.html`. The ONNX backend transcribing without CoreML,
in an isolated environment. The dependency metadata quoted above, read from the
installed packages and from PyPI. `fluid-soundfont-gm`'s file list, read from
Debian's package index. `tests/test_space_app.py` covers the handler's
branches.

**Not verified — expected, from documentation.** Everything about the Hugging
Face platform: that the build succeeds, that `python_version: "3.10"` is
honoured, that the apt packages install, that the Demucs weights download
through the Space's network, that CPU Basic is fast enough to be pleasant. None
of that can be tested without pushing. The build log is the first real evidence
and is worth reading rather than skimming.
