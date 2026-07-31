# SoloScribe — a guide for the guitarist

You give it a recording of a solo. It gives you back two things:

1. **A Guitar Pro file (.gp5)** — the solo written out in notation and tab,
   with fingerings chosen the way a player would choose them. Opens in
   Guitar Pro, or in TuxGuitar, which is free.
2. **A report on itself** — it re-plays what it wrote, lines that up against
   your recording, and tells you honestly which bars to trust and which to
   check by ear. You can listen to the original and its reconstruction side
   by side on the report page.

No transcription software gets everything right — anyone who says otherwise
is selling something. The difference here is that this one measures how much
it got right and shows you, so your cleanup time goes exactly where it's
needed.

## Using it

Double-click **Start SoloScribe** (in the `bin` folder). Your browser opens
the app. Then:

1. **Drop your recording in.** MP3, WAV, M4A — whatever you have. A clip
   trimmed to just the solo works better than a whole track (there are
   start/end boxes under "If you know a little more" if you'd rather not
   trim it yourself).
2. **Tell it what you know.** Every box is optional, but each one helps:
   - **Key** — you usually know this from the chart.
   - **Tempo** — even roughly. If the recording was made to a click or a
     backing track, also give the **downbeat** (the moment bar 1 starts, in
     seconds) under "If you know a little more" — with both, the timing
     comes out exact.
   - **Feel** — Swing or Straight if you know; it detects this well on its
     own, but telling it removes the guess.
   - **Chord chart** — one bar per line (or separated by `|`). These print
     above the bars in the Guitar Pro file, so the solo reads in context.
3. **Press Transcribe** and watch it work through the steps. A ninety-second
   clip takes a few minutes the first time (it downloads its
   guitar-isolating model once) and is faster after that.
4. **Download the Guitar Pro file, then read the report.** The report's
   verdict — high, medium or low fidelity — is honest. The piano-roll
   picture colors each note by how confident it was: the pale ones are where
   to point your ears first.

## What to expect

- Where the guitar is clearly audible, most notes land, and the rhythm is
  written the way a transcriber would write it — swung eighths come out as
  straight eighths marked "Swing 8ths", the way real jazz charts do it.
- Bends and vibrato are marked where it hears them. Slides, hammer-ons and
  the finer articulation are yours to add — that's a five-minute pass in
  Guitar Pro with the recording in your ears, not an evening of note-hunting.
- If the band is loud and the guitar buried, it will do worse, and the
  report will say so rather than pretend.
- It assumes standard tuning. A capo can be set on the command line but not
  yet in the app.
- If the solo has Wes Montgomery-style octave passages, set "What am I
  listening for?" to **Chords too** — in single-note mode the octave doubling
  gets mistaken for overtone ringing and stripped out; in Chords-too mode
  most of it survives.

## If something goes wrong

The app apologises and shows a short error name. Send David a screenshot —
he built the thing, he can fix it.
