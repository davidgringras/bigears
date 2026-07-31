"""Source separation: pull the guitar out of a full-band mix with Demucs.

Strategy: run the 6-source model (`htdemucs_6s`), which has a dedicated guitar
stem, and measure how much of the mix survived into it. Jazz recordings where
the guitar sits inside a horn section — or where the 6-source model simply
mis-assigns — produce a near-silent guitar stem; below `MIN_REL_RMS` we fall
back to the 4-source `htdemucs` "other" bucket, which is where guitar lands in
a drums/bass/vocals/other decomposition.

All demucs API usage below is verified against the installed source at
`.venv/lib/python3.11/site-packages/demucs/`:

  * `Separator.__init__(model, repo, device, shifts, overlap, split, segment,
    jobs, progress, callback, callback_arg)`               — api.py:53-67
  * `Separator.separate_audio_file(file) -> (wav, dict)`   — api.py:290-304
  * the dict is `dict(zip(self._model.sources, out[0]))`,
    i.e. stem name -> tensor of shape (channels, samples) — api.py:288
  * the returned `wav` is the mix *already resampled* to the model rate and
    channel count, so mix and stem RMS are directly comparable
                                                           — api.py:267-268, 288
  * `save_audio(wav, path, samplerate, ...)`               — api.py:32, audio.py:297-304
  * `segment=None` means "use the model's own segment" and keeps `split=True`
    chunking at the trained window (memory-sane for long inputs)
                                                           — apply.py:261-266
  * `shifts` draws a random offset from the global `random` module, so runs are
    not bit-reproducible unless the caller seeds it        — apply.py:245

Stem names live in the downloaded checkpoint, not in the package, so
`GUITAR_STEM` is looked up in `separator.model.sources` rather than assumed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

PRIMARY_MODEL = "htdemucs_6s"
PRIMARY_STEM = "guitar"
FALLBACK_MODEL = "htdemucs"
FALLBACK_STEM = "other"

# Stem RMS as a fraction of mix RMS. Below this the primary stem carries
# essentially no signal and the fallback model is worth its runtime.
MIN_REL_RMS = 0.05


@dataclass
class SeparationResult:
    """Which model produced the stem on disk, and how loud it came out."""

    stem_path: str
    model_name: str
    stem_name: str
    stem_rel_rms: float          # rms(stem) / rms(mix), both post-resample


def _pick_device(device: str | None) -> str:
    """Explicit device wins; otherwise MPS when the Metal backend is available."""
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _rms(x: torch.Tensor) -> float:
    """RMS over every sample and channel; 0.0 for an empty tensor."""
    if x.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(x.to(torch.float32) ** 2)))


def _run_model(
    audio_path: str, model_name: str, stem_name: str, device: str
) -> tuple[torch.Tensor, float, int]:
    """Separate once and return (stem, rel_rms, samplerate).

    Falls back to CPU if the requested device raises — MPS coverage for the
    transformer blocks in htdemucs is not something this module can assume.
    Raises KeyError if `stem_name` is absent from the checkpoint's source list.
    """
    from demucs.api import Separator

    def _separate(dev: str) -> tuple[torch.Tensor, float, int]:
        sep = Separator(
            model=model_name,
            device=dev,
            shifts=1,
            overlap=0.25,
            split=True,        # chunk the input; the 2-min-clip memory guard
            segment=None,      # apply.py:261-262 → model.segment
            jobs=0,
            progress=False,
        )
        sources = list(sep.model.sources)
        if stem_name not in sources:
            raise KeyError(
                f"model {model_name!r} has no stem {stem_name!r}; it emits {sources}"
            )
        mix, stems = sep.separate_audio_file(Path(audio_path))
        stem = stems[stem_name].detach().to("cpu")
        mix_rms = _rms(mix)
        rel = _rms(stem) / mix_rms if mix_rms > 0 else 0.0
        sr = sep.samplerate
        # Drop the model and the other five stems before the caller decides
        # whether to load a second model.
        del sep, stems, mix
        return stem, rel, sr

    try:
        return _separate(device)
    except Exception as exc:                      # noqa: BLE001 — device probe
        if device == "cpu" or isinstance(exc, KeyError):
            raise
        print(f"[separate] {device} failed ({exc!r}); retrying on cpu")
        return _separate("cpu")


def separate(audio_path: str, out_dir: str, device: str | None = None) -> SeparationResult:
    """Write a guitar stem for `audio_path` into `out_dir` and describe it.

    First call downloads model weights (~hundreds of MB) into the torch hub
    cache. `stem_rel_rms` is measured on the raw model output, before
    `save_audio`'s peak rescaling (audio.py:279-294), so it describes the
    separation rather than the file.
    """
    dev = _pick_device(device)
    os.makedirs(out_dir, exist_ok=True)

    stem, rel, sr = _run_model(audio_path, PRIMARY_MODEL, PRIMARY_STEM, dev)
    model_name, stem_name = PRIMARY_MODEL, PRIMARY_STEM

    if rel < MIN_REL_RMS:
        print(
            f"[separate] {PRIMARY_MODEL}/{PRIMARY_STEM} rel RMS {rel:.4f} "
            f"< {MIN_REL_RMS}; falling back to {FALLBACK_MODEL}/{FALLBACK_STEM}"
        )
        del stem
        stem, rel, sr = _run_model(audio_path, FALLBACK_MODEL, FALLBACK_STEM, dev)
        model_name, stem_name = FALLBACK_MODEL, FALLBACK_STEM

    from demucs.api import save_audio

    base = os.path.splitext(os.path.basename(audio_path))[0]
    stem_path = os.path.join(out_dir, f"{base}_{stem_name}.wav")
    save_audio(stem, stem_path, samplerate=sr)

    return SeparationResult(
        stem_path=stem_path,
        model_name=model_name,
        stem_name=stem_name,
        stem_rel_rms=rel,
    )


if __name__ == "__main__":  # pragma: no cover — exercised by hand / integration
    import argparse

    ap = argparse.ArgumentParser(description="Separate a guitar stem with Demucs.")
    ap.add_argument("audio")
    ap.add_argument("-o", "--out-dir", default="output/stems")
    ap.add_argument("-d", "--device", default=None, help="mps | cpu | cuda")
    a = ap.parse_args()

    res = separate(a.audio, a.out_dir, device=a.device)
    print(
        f"model={res.model_name} stem={res.stem_name} "
        f"rel_rms={res.stem_rel_rms:.4f}\n{res.stem_path}"
    )
