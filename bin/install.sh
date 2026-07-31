#!/bin/bash
# Sets SoloScribe up on this Mac. Safe to run again at any time.
#
#   bash bin/install.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BREW_PYTHON="/opt/homebrew/opt/python@3.11/bin/python3.11"
VENV="$REPO_ROOT/.venv"

echo "Setting up SoloScribe"
echo "Folder: $REPO_ROOT"
echo

# 1. Homebrew ---------------------------------------------------------------

if ! command -v brew > /dev/null 2>&1; then
  echo "Homebrew is missing. SoloScribe needs it to install Python 3.11."
  echo
  echo "Copy the line below, paste it into Terminal, press return, and follow"
  echo "the instructions it gives you:"
  echo
  echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
  echo
  echo "That is the installer published at https://brew.sh, where you can check"
  echo "it for yourself before running it."
  echo
  echo "When Homebrew has finished, run this script again."
  exit 1
fi
echo "Homebrew: found"

# 2. Python 3.11 ------------------------------------------------------------

if [ ! -x "$BREW_PYTHON" ]; then
  echo
  echo "Python 3.11 is missing. Run this line in Terminal:"
  echo
  echo "  brew install python@3.11"
  echo
  echo "Then run this script again."
  exit 1
fi
echo "Python 3.11: $BREW_PYTHON"

# 3. The environment --------------------------------------------------------

if [ -x "$VENV/bin/python" ]; then
  echo "Environment: already at .venv, reusing it"
else
  echo "Environment: creating .venv"
  if ! "$BREW_PYTHON" -m venv "$VENV"; then
    echo
    echo "Could not create the environment at $VENV."
    exit 1
  fi
fi

# 4. The parts --------------------------------------------------------------

echo
echo "Installing the parts SoloScribe needs."
echo "The first run downloads a lot and can take ten minutes or more."
echo

"$VENV/bin/python" -m pip install --quiet --disable-pip-version-check --upgrade pip

if ! "$VENV/bin/python" -m pip install --disable-pip-version-check -r "$REPO_ROOT/requirements.txt"; then
  echo
  echo "Something went wrong while installing. The lines above say what."
  echo "If it mentions running out of space, free some up and run this again."
  exit 1
fi

# 5. Finish -----------------------------------------------------------------

chmod +x "$REPO_ROOT/bin/Start SoloScribe.command" 2> /dev/null

echo
echo "Done."
echo "Now double-click Start SoloScribe.command in the bin folder."
