#!/bin/zsh
# Double-click this in Finder to run SoloScribe.
#
# Starts the local web server and opens it in your browser. Keep the Terminal
# window open while you are using it; closing the window stops SoloScribe.

set -u

REPO="${0:A:h:h}"
PORT=8746
URL="http://127.0.0.1:${PORT}"
PY="$REPO/.venv/bin/python"

printf '\033]0;SoloScribe\007'   # name the Terminal window

cd "$REPO" || exit 1

stop_here() {
  print ""
  print "Press return to close this window."
  read -r _
  exit 1
}

if [[ ! -x "$PY" ]]; then
  print "SoloScribe has not been set up on this Mac yet."
  print ""
  print "Open Terminal and run this line:"
  print ""
  print "    bash \"$REPO/bin/install.sh\""
  stop_here
fi

# Already running from an earlier double-click? Just bring it up.
if curl -fsS --max-time 2 "$URL/api/health" > /dev/null 2>&1; then
  print "SoloScribe is already running. Opening it now."
  open "$URL"
  exit 0
fi

# Check the app loads before handing the window over to the server, so a broken
# install says so in plain words instead of flashing a traceback.
if ! "$PY" -c "import soloscribe.webapp.server" > /tmp/soloscribe-start.log 2>&1; then
  print "SoloScribe could not start. Some of its parts are missing or broken."
  print ""
  print "Open Terminal and run this line to repair it:"
  print ""
  print "    bash \"$REPO/bin/install.sh\""
  print ""
  print "The technical detail, if it helps David:"
  print ""
  tail -n 12 /tmp/soloscribe-start.log
  stop_here
fi

print "Starting SoloScribe."
print "Your browser will open in a moment."
print ""
print "Leave this window open while you use it. Close it to stop SoloScribe."
print ""

# Wait for the server to answer, then open the browser. Runs alongside the
# server below, which replaces this shell.
{
  repeat 60; do
    if curl -fsS --max-time 1 "$URL/api/health" > /dev/null 2>&1; then
      open "$URL"
      exit 0
    fi
    sleep 0.5
  done
} &

exec "$PY" -m uvicorn soloscribe.webapp.server:app \
  --host 127.0.0.1 --port "$PORT" --log-level warning
