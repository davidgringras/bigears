"""Round-trip verification of the GP5 writer against PyGuitarPro's parser."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import guitarpro
from guitarpro import models as gp

from soloscribe.gp5_writer import write_gp5, _beat_ticks, _decompose
from soloscribe.model import QNote, Score, TICKS_PER_QUARTER


def _score(qnotes, **kw):
    defaults = dict(beats_per_bar=4, tempo_bpm=150.0, key="F", triplet_feel=True,
                    title="Test", chords=["Gm7", "C7"])
    defaults.update(kw)
    return Score(qnotes=qnotes, **defaults)


def _roundtrip(score, tmp_path):
    path = str(tmp_path / "out.gp5")
    warnings = write_gp5(score, path)
    return guitarpro.parse(path), warnings


def test_basic_phrase_roundtrip(tmp_path):
    q = [
        QNote(0, 960, 65, string=1, fret=1),            # quarter F4
        QNote(960, 480, 67, string=1, fret=3),          # 8th G4
        QNote(1440, 480, 69, string=1, fret=5),         # 8th A4
        QNote(1920, 320, 70, string=1, fret=6),         # triplet 8ths
        QNote(2240, 320, 72, string=1, fret=8),
        QNote(2560, 320, 74, string=1, fret=10),
        QNote(2880, 480, 72, string=1, fret=8),         # 8th + implicit 8th rest
    ]
    song, warnings = _roundtrip(_score(q), tmp_path)
    assert not [w for w in warnings if "misrender" in w], warnings
    measures = song.tracks[0].measures
    assert len(measures) == 2  # chords list forces 2 bars
    beats = measures[0].voices[0].beats
    # 7 notes + trailing 8th rest in bar 1
    assert len(beats) == 8
    assert beats[-1].status == gp.BeatStatus.rest
    trip = [b for b in beats if b.duration.tuplet.enters == 3]
    assert len(trip) == 3
    assert song.measureHeaders[0].tripletFeel == gp.TripletFeel.eighth
    assert song.measureHeaders[0].keySignature == gp.KeySignature.FMajor
    assert song.measureHeaders[0].marker.title == "Gm7"
    assert song.measureHeaders[1].marker.title == "C7"


def test_measure_sums_exact(tmp_path):
    # Awkward spans: dotted quarter off-beat, note across barline, long gap.
    q = [
        QNote(480, 1440, 60, string=2, fret=1),          # off-beat dotted-quarter span
        QNote(2880, 1920, 64, string=2, fret=5),         # crosses barline → tie
        QNote(6720, 240, 67, string=1, fret=3),          # 16th late in bar 2
    ]
    song, warnings = _roundtrip(_score(q, chords=[]), tmp_path)
    assert not [w for w in warnings if "misrender" in w], warnings
    for meas in song.tracks[0].measures:
        assert sum(_beat_ticks(b) for b in meas.voices[0].beats) == 3840


def test_tie_across_barline(tmp_path):
    q = [QNote(2880, 1920, 64, string=2, fret=5)]
    song, _ = _roundtrip(_score(q, chords=[]), tmp_path)
    m1, m2 = song.tracks[0].measures[:2]
    tail = m1.voices[0].beats[-1]
    head = m2.voices[0].beats[0]
    assert tail.notes and tail.notes[0].type == gp.NoteType.normal
    assert head.notes and head.notes[0].type == gp.NoteType.tie
    assert head.notes[0].string == tail.notes[0].string
    assert head.notes[0].value == tail.notes[0].value


def test_double_stop_and_effects(tmp_path):
    q = [
        QNote(0, 960, 72, string=2, fret=1, vibrato=True),
        QNote(0, 960, 76, string=1, fret=0),
        QNote(960, 960, 74, string=2, fret=3,
              bend=[(0.0, 0.0), (0.5, 2.0), (1.0, 2.0)]),
    ]
    song, _ = _roundtrip(_score(q, chords=[]), tmp_path)
    beats = song.tracks[0].measures[0].voices[0].beats
    assert len(beats[0].notes) == 2
    assert any(n.effect.vibrato for n in beats[0].notes)
    bend = beats[1].notes[0].effect.bend
    assert bend is not None and bend.points, "bend did not survive round-trip"
    # GP5 bend units are quarter-tones: a full-tone (2-semitone) bend = 4.
    assert max(p.value for p in bend.points) == 4
    assert bend.value == 4


def test_decompose_invariants():
    tpb = TICKS_PER_QUARTER
    for start in range(0, 3840, 240):
        for length in (240, 480, 720, 960, 1440, 1920, 3840 - start):
            if length <= 0 or start + length > 3840:
                continue
            chunks = _decompose(start, length, tpb, 3840)
            assert sum(c for _, c in chunks) == length, (start, length, chunks)
    # triplet-grid spans
    for start in (0, 320, 640, 960, 1280):
        chunks = _decompose(start, 320, tpb, 3840)
        assert sum(c for _, c in chunks) == 320


def test_capo_written_as_track_offset(tmp_path):
    q = [QNote(0, 960, 62, string=2, fret=1)]  # D4 = string 2 open(59)+capo2+fret1
    song, warnings = _roundtrip(_score(q, capo=2, chords=[]), tmp_path)
    assert song.tracks[0].offset == 2
    note = song.tracks[0].measures[0].voices[0].beats[0].notes[0]
    assert (note.string, note.value) == (2, 1)  # fret stays capo-relative
    assert not [w for w in warnings if "misrender" in w]


def test_waltz_time(tmp_path):
    q = [QNote(i * 960, 960, 60 + i, string=2, fret=1 + i) for i in range(6)]
    song, warnings = _roundtrip(_score(q, beats_per_bar=3, chords=[]), tmp_path)
    assert not [w for w in warnings if "misrender" in w], warnings
    assert song.measureHeaders[0].timeSignature.numerator == 3
    for meas in song.tracks[0].measures:
        assert sum(_beat_ticks(b) for b in meas.voices[0].beats) == 2880
