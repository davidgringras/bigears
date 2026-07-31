"""Score → .gp5 via PyGuitarPro.

Design constraints this module enforces (Guitar Pro renders garbage otherwise):
  - Every measure's voice-0 beat durations sum EXACTLY to the measure length.
  - A sub-beat duration never crosses a beat boundary; whole-beat chunks only
    start on a beat. Notes that span boundaries become tie chains.
  - Within one beat, content is single-grid (straight or triplet) — the
    quantizer guarantees this; the decomposer relies on it.

Empirically verified against PyGuitarPro 0.11 (write→parse round-trip):
string 1 = high E; BendPoint.value is in semitones (gp3.py writeBend scales
by bendSemitone/semitoneLength); TripletFeel.eighth survives GP5 write.
"""
from __future__ import annotations

import guitarpro
from guitarpro import models as gp

from .model import QNote, Score, TICKS_PER_QUARTER

# tick span → (Duration.value, isDotted, tuplet or None), longest first.
_DURATIONS: list[tuple[int, tuple[int, bool, tuple[int, int] | None]]] = [
    (3840, (1, False, None)),
    (2880, (2, True, None)),
    (1920, (2, False, None)),
    (1440, (4, True, None)),
    (1280, (2, False, (3, 2))),
    (960, (4, False, None)),
    (720, (8, True, None)),
    (640, (4, False, (3, 2))),
    (480, (8, False, None)),
    (360, (16, True, None)),
    (320, (8, False, (3, 2))),
    (240, (16, False, None)),
    (160, (16, False, (3, 2))),
    (120, (32, False, None)),
    (80, (32, False, (3, 2))),
    (60, (64, False, None)),
]
_DUR_BY_TICKS = dict(_DURATIONS)

# Alignment a chunk's within-beat offset must satisfy (single-grid-per-beat
# makes these sufficient for exact fill).
_ALIGN = {3840: 960, 2880: 960, 1920: 960, 1440: 960, 1280: 320, 960: 960,
          720: 960, 640: 320, 480: 480, 360: 240, 320: 320, 240: 240,
          160: 160, 120: 120, 80: 80, 60: 60}


def _decompose(start: int, length: int, ticks_per_beat: int, ticks_per_bar: int) -> list[tuple[int, int]]:
    """Split [start, start+length) into representable (pos, ticks) chunks.

    `start` is measured from the start of the measure.
    """
    out: list[tuple[int, int]] = []
    pos, remaining = start, length
    while remaining > 0:
        into_beat = pos % ticks_per_beat
        beat_idx = pos // ticks_per_beat
        take = 0
        if into_beat == 0:
            # Whole-beat-aligned big values first (quarter-based meters only;
            # other denominators fall through to the generic path).
            if ticks_per_beat == TICKS_PER_QUARTER:
                if beat_idx == 0 and remaining >= ticks_per_bar and ticks_per_bar in _DUR_BY_TICKS:
                    take = ticks_per_bar          # whole-bar note/rest
                elif beat_idx % 4 == 0 and remaining >= 3840 and ticks_per_bar >= 3840:
                    take = 3840
                elif beat_idx % 2 == 0 and remaining >= 1920:
                    take = 1920
                elif remaining == 1440:
                    take = 1440                   # dotted quarter on the beat
                elif remaining >= 960:
                    take = 960
        if take == 0:
            to_beat_end = ticks_per_beat - into_beat
            avail = min(remaining, to_beat_end)
            for ticks, _ in _DURATIONS:
                if ticks <= avail and ticks <= 960 and into_beat % _ALIGN[ticks] == 0:
                    take = ticks
                    break
        if take == 0:
            # Off-grid residue should not occur; absorb at 32nd resolution
            # rather than corrupt the measure sum.
            take = min(remaining, 120 - pos % 120 or 120)
            if take not in _DUR_BY_TICKS:
                take = 60
        out.append((pos, take))
        pos += take
        remaining -= take
    return out


def _make_duration(ticks: int) -> gp.Duration:
    value, dotted, tuplet = _DUR_BY_TICKS[ticks]
    return gp.Duration(
        value=value,
        isDotted=dotted,
        tuplet=gp.Tuplet(*tuplet) if tuplet else gp.Tuplet(1, 1),
    )


_LETTER_OK = set("ABCDEFG")


def key_signature_for(key: str) -> tuple[gp.KeySignature, str | None]:
    """'F' → FMajor, 'Bb' → BMajorFlat, 'F#m' → FMinorSharp. Returns (sig, warning)."""
    k = key.strip()
    minor = k.endswith("m") and not k.endswith("dim")
    if minor:
        k = k[:-1]
    if not k or k[0].upper() not in _LETTER_OK:
        return gp.KeySignature.CMajor, f"unrecognized key {key!r}; wrote C major"
    name = k[0].upper() + ("Minor" if minor else "Major")
    if len(k) > 1:
        name += {"#": "Sharp", "b": "Flat"}.get(k[1], "")
    sig = getattr(gp.KeySignature, name, None)
    if sig is None:
        return gp.KeySignature.CMajor, f"unrecognized key {key!r}; wrote C major"
    return sig, None


def _apply_bend(note: gp.Note, bend: list[tuple[float, float]]) -> None:
    """Map (rel_time 0..1, semitones) contour to a GP5 bend graph.

    GP5 bend values are QUARTER-TONES (4 = full tone): verified against a
    Guitar-Pro-authored file whose full-tone bend stores wire value 100 =
    model value 4. PyGuitarPro's `semitoneLength = 1` constant is a misnomer.
    """
    if not bend:
        return
    peak = max(v for _, v in bend)
    if peak < 0.4:
        return
    effect = gp.BendEffect()
    released = bend[-1][1] < peak - 0.4
    effect.type = gp.BendType.bendRelease if released else gp.BendType.bend
    effect.value = round(peak * 2)  # semitones → quarter-tone units
    points = [(0.0, 0.0)] + [(t, v) for t, v in bend if 0.0 < t <= 1.0]
    seen: set[int] = set()
    for t, v in points:
        posn = min(12, max(0, round(t * 12)))
        if posn in seen:
            continue
        seen.add(posn)
        effect.points.append(
            gp.BendPoint(position=posn, value=max(0, round(v * 2)))
        )
    effect.points.sort(key=lambda p: p.position)
    note.effect.bend = effect


def write_gp5(score: Score, path: str) -> list[str]:
    """Write the Score to a .gp5 file. Returns human-readable warnings."""
    warnings: list[str] = []
    song = gp.Song()
    song.title = score.title or "Transcription"
    song.subtitle = score.subtitle
    song.artist = "Transcribed by soloscribe"
    song.tempo = max(30, min(320, round(score.tempo_bpm)))
    track = song.tracks[0]
    track.name = "Guitar"
    track.channel.instrument = 26  # GM Electric Guitar (jazz)
    track.strings = [
        gp.GuitarString(i + 1, v) for i, v in enumerate(score.tuning)
    ]
    track.fretCount = 24
    if score.capo:
        track.offset = score.capo

    key_sig, key_warn = key_signature_for(score.key)
    if key_warn:
        warnings.append(key_warn)
    feel = gp.TripletFeel.eighth if score.triplet_feel else gp.TripletFeel.none

    n_measures = score.n_measures()
    while len(song.measureHeaders) < n_measures:
        song.newMeasure()
    for i, header in enumerate(song.measureHeaders):
        header.keySignature = key_sig
        header.tripletFeel = feel
        header.timeSignature.numerator = score.beats_per_bar
        header.timeSignature.denominator = gp.Duration(value=score.denominator)
        if i < len(score.chords) and score.chords[i]:
            header.marker = gp.Marker(title=str(score.chords[i]))

    tpb = score.ticks_per_beat
    bar_len = score.ticks_per_bar

    # Group into verticals (chords) by onset; a vertical's duration is the
    # minimum of its members (longer members are truncated — flagged).
    by_onset: dict[int, list[QNote]] = {}
    for q in sorted(score.qnotes, key=lambda q: (q.onset, q.pitch)):
        if q.duration <= 0:
            continue
        by_onset.setdefault(q.onset, []).append(q)
    verticals: list[tuple[int, int, list[QNote]]] = []
    truncated = 0
    for onset in sorted(by_onset):
        group = by_onset[onset]
        dur = min(q.duration for q in group)
        if any(q.duration != dur for q in group):
            truncated += 1
        verticals.append((onset, dur, group))
    # Clip overlaps between successive verticals (monophonic-biased output).
    clipped = 0
    for i in range(len(verticals) - 1):
        onset, dur, group = verticals[i]
        nxt = verticals[i + 1][0]
        if onset + dur > nxt:
            verticals[i] = (onset, nxt - onset, group)
            clipped += 1
    verticals = [v for v in verticals if v[1] > 0]
    if truncated:
        warnings.append(
            f"in {truncated} place(s), notes struck together rang for different "
            "lengths — I shortened them to match so the notation stays readable"
        )
    if clipped:
        warnings.append(
            f"{clipped} note(s) rang into the next note — I trimmed them so "
            "the line reads cleanly"
        )

    # Emit measure by measure: notes as tie chains, gaps as rests.
    def emit(measure_idx: int, chunk_pos: int, chunk_len: int, group: list[QNote] | None, tie: bool) -> bool:
        measure = track.measures[measure_idx]
        voice = measure.voices[0]
        beat = gp.Beat(voice)
        beat.duration = _make_duration(chunk_len)
        if group is None:
            beat.status = gp.BeatStatus.rest
        else:
            beat.status = gp.BeatStatus.normal
            for q in group:
                note = gp.Note(beat)
                note.type = gp.NoteType.tie if tie else gp.NoteType.normal
                note.velocity = max(15, min(127, q.velocity))
                if q.string is not None and q.fret is not None:
                    note.string = q.string
                    note.value = q.fret
                else:
                    note.string = 1
                    note.value = max(0, q.pitch - score.tuning[0] - score.capo)
                if q.vibrato:
                    note.effect.vibrato = True
                if not tie:
                    _apply_bend(note, q.bend)
                beat.notes.append(note)
        voice.beats.append(beat)
        return True

    cursor = 0
    swing_label_pending = score.triplet_feel  # visible marking, not just the feel flag
    for onset, dur, group in verticals:
        if onset > cursor:  # rest gap
            gap_pos = cursor
            while gap_pos < onset:
                mi = gap_pos // bar_len
                seg_end = min(onset, (mi + 1) * bar_len)
                for cpos, clen in _decompose(gap_pos - mi * bar_len, seg_end - gap_pos, tpb, bar_len):
                    emit(mi, cpos, clen, None, False)
                gap_pos = seg_end
        note_pos = onset
        note_end = onset + dur
        first = True
        while note_pos < note_end:
            mi = note_pos // bar_len
            seg_end = min(note_end, (mi + 1) * bar_len)
            for cpos, clen in _decompose(note_pos - mi * bar_len, seg_end - note_pos, tpb, bar_len):
                tie = (not first) or (first and group[0].tied_from_prev)
                emit(mi, cpos, clen, group, tie)
                if swing_label_pending:
                    track.measures[mi].voices[0].beats[-1].text = "Swing 8ths"
                    swing_label_pending = False
                first = False
            note_pos = seg_end
        cursor = max(cursor, note_end)

    # Fill the final partial measure with rests.
    total = n_measures * bar_len
    if cursor < total:
        gap_pos = cursor
        while gap_pos < total:
            mi = gap_pos // bar_len
            seg_end = min(total, (mi + 1) * bar_len)
            for cpos, clen in _decompose(gap_pos - mi * bar_len, seg_end - gap_pos, tpb, bar_len):
                emit(mi, cpos, clen, None, False)
            gap_pos = seg_end

    # Invariant: every measure sums exactly.
    for mi, measure in enumerate(track.measures):
        ticks = sum(_beat_ticks(b) for b in measure.voices[0].beats)
        if ticks != bar_len:
            warnings.append(f"measure {mi + 1} sums to {ticks}/{bar_len} ticks — notation may misrender")

    guitarpro.write(song, path)
    return warnings


def _beat_ticks(beat: gp.Beat) -> int:
    d = beat.duration
    ticks = TICKS_PER_QUARTER * 4 // d.value
    if d.isDotted:
        ticks = ticks * 3 // 2
    if d.tuplet.enters != d.tuplet.times:
        ticks = ticks * d.tuplet.times // d.tuplet.enters
    return ticks
