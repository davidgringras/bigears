"""Assemble the deployable Hugging Face Space directory.

The Space is a *self-contained* git repository: Hugging Face clones it, runs
``pip install -r requirements.txt``, and starts ``app.py``. It has no access to
this repo, so the ``soloscribe`` package has to travel with it. This script
vendors it — a plain directory copy of ``soloscribe/`` into ``space/``, minus
``webapp/`` (the desktop FastAPI UI, which the Space replaces) and the usual
build litter.

    .venv/bin/python space/make_space.py            # copy the package in
    .venv/bin/python space/make_space.py --check    # is the copy current?

REDEPLOYING AFTER A CHANGE TO soloscribe/: rerun this script, then commit and
push the Space repo. The vendored copy is a snapshot, not a symlink, so a
change to ``soloscribe/transcribe.py`` reaches the Space only once this has run
again. ``--check`` exits non-zero when the snapshot has drifted, which is the
cheap way to find out before pushing rather than after.

Hand-written files in ``space/`` (app.py, requirements.txt, packages.txt,
README.md, this script) are never touched — the script only ever writes inside
``space/soloscribe/``.
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

SPACE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SPACE_DIR.parent
SOURCE_PKG = REPO_ROOT / "soloscribe"
VENDORED_PKG = SPACE_DIR / "soloscribe"

# webapp/ is the desktop UI (FastAPI + static files); app.py is its replacement
# and pulling it along would drag fastapi and uvicorn into the Space's
# requirements for no reason. Nothing outside webapp/ imports it — verified by
# grepping the package for "webapp".
EXCLUDE_DIRS = {"webapp", "__pycache__", ".pytest_cache"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}

# Everything the Space needs that this script does NOT generate. Listed so a
# missing file is caught here rather than by a failed build on the platform.
REQUIRED_FILES = ("app.py", "requirements.txt", "packages.txt", "README.md")


def _wanted(root: Path) -> list[Path]:
    """Every file under `root` that belongs in the Space, as relative paths."""
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if path.is_file() and path.suffix not in EXCLUDE_SUFFIXES:
            out.append(rel)
    return out


def _copy(verbose: bool = True) -> list[Path]:
    if VENDORED_PKG.exists():
        shutil.rmtree(VENDORED_PKG)
    files = _wanted(SOURCE_PKG)
    for rel in files:
        dest = VENDORED_PKG / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_PKG / rel, dest)
    if verbose:
        total = sum((VENDORED_PKG / r).stat().st_size for r in files)
        print(f"vendored {len(files)} files ({total / 1024:.0f} KiB) into {VENDORED_PKG}")
        for rel in files:
            print(f"  soloscribe/{rel}")
    return files


def _prune_litter() -> list[Path]:
    """Drop __pycache__ and friends from anywhere under space/.

    Running the app or the tests locally leaves bytecode caches inside the very
    directory that gets pushed, and `hf upload` sends what it is given. This
    runs on every pass so the directory is clean at the moment it is uploaded,
    rather than relying on whoever pushes to remember an --exclude flag.
    """
    dropped: list[Path] = []
    for path in sorted(SPACE_DIR.rglob("*"), reverse=True):
        if path.is_dir() and path.name in {"__pycache__", ".pytest_cache"}:
            shutil.rmtree(path)
            dropped.append(path.relative_to(SPACE_DIR))
    return dropped


def _check() -> int:
    """Exit 0 when the vendored copy matches the package it was made from."""
    if not VENDORED_PKG.exists():
        print(f"MISSING: {VENDORED_PKG} — run this script without --check", file=sys.stderr)
        return 1
    want = set(_wanted(SOURCE_PKG))
    have = set(_wanted(VENDORED_PKG))
    problems: list[str] = []
    for rel in sorted(want - have):
        problems.append(f"missing from the Space:  soloscribe/{rel}")
    for rel in sorted(have - want):
        problems.append(f"stale, not in the source: soloscribe/{rel}")
    for rel in sorted(want & have):
        if not filecmp.cmp(SOURCE_PKG / rel, VENDORED_PKG / rel, shallow=False):
            problems.append(f"out of date:              soloscribe/{rel}")
    if problems:
        print("The vendored package has drifted from soloscribe/:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        print("\nRerun: .venv/bin/python space/make_space.py", file=sys.stderr)
        return 1
    print(f"vendored copy is current ({len(want)} files)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the vendored copy is current; do not write anything",
    )
    ap.add_argument("--quiet", action="store_true", help="print only the summary line")
    args = ap.parse_args(argv)

    if not SOURCE_PKG.is_dir():
        print(f"cannot find the package at {SOURCE_PKG}", file=sys.stderr)
        return 2

    missing = [f for f in REQUIRED_FILES if not (SPACE_DIR / f).is_file()]
    if missing:
        print(
            "these hand-written Space files are missing: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 2

    if args.check:
        return _check()

    _copy(verbose=not args.quiet)
    for rel in _prune_litter():
        if not args.quiet:
            print(f"  removed {rel}")
    print(f"\nSpace directory ready: {SPACE_DIR}")
    print("Push this directory to the Space repo — see space/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
