"""Tests for the fretting engine.

The invariant every test leans on is `_assert_playable`: a fretting is only
correct if it reproduces its own pitch, so `tuning[string - 1] + capo + fret`
must equal `pitch` for every note. Out-of-range pitches are rewritten by the
engine (and warned about), which is exactly why the identity still holds there.
"""
from __future__ import annotations

from soloscribe.fretting import assign_fretting
from soloscribe.model import STANDARD_TUNING, TICKS_PER_QUARTER, QNote

QUARTER = TICKS_PER_QUARTER
EIGHTH = TICKS_PER_QUARTER // 2
SIXTEENTH = TICKS_PER_QUARTER // 4


def _line(pitches: list[int], step: int, start: int = 0) -> list[QNote]:
    return [
        QNote(onset=start + i * step, duration=step, pitch=p)
        for i, p in enumerate(pitches)
    ]


def _assert_playable(
    qnotes: list[QNote],
    tuning: tuple[int, ...] = STANDARD_TUNING,
    capo: int = 0,
    max_fret: int = 22,
) -> None:
    for q in qnotes:
        assert q.string is not None, f"unfretted note at tick {q.onset}"
        assert q.fret is not None, f"unfretted note at tick {q.onset}"
        assert 1 <= q.string <= len(tuning)
        assert 0 <= q.fret <= max_fret
        assert tuning[q.string - 1] + capo + q.fret == q.pitch, (
            f"string {q.string} fret {q.fret} does not sound pitch {q.pitch}"
        )


def _fretted(qnotes: list[QNote]) -> list[int]:
    """Frets of the stopped notes; open strings sit outside position math."""
    return [q.fret for q in qnotes if q.fret]


# --------------------------------------------------------------------------
# the six required cases
# --------------------------------------------------------------------------
def test_c_major_scale_is_playable_and_positionally_coherent():
    notes = _line([48, 50, 52, 53, 55, 57, 59, 60], QUARTER)  # C3 up to C4
    warnings = assign_fretting(notes)

    assert warnings == []
    _assert_playable(notes)
    frets = _fretted(notes)
    for a, b in zip(frets, frets[1:]):
        assert abs(a - b) <= 5, f"hand jumps {abs(a - b)} frets inside a scale"


def test_dense_chromatic_run_stays_inside_one_hand_span():
    notes = _line(list(range(64, 77)), SIXTEENTH)  # E4 to E5, thirteen sixteenths
    warnings = assign_fretting(notes)

    assert warnings == []
    _assert_playable(notes)
    frets = _fretted(notes)
    window = 8
    assert len(frets) >= window, "a chromatic run should be stopped, not played open"
    spreads = [
        max(frets[i : i + window]) - min(frets[i : i + window])
        for i in range(len(frets) - window + 1)
    ]
    # A single-string slide would pass a narrow window and fail this one, which
    # is the point: eight consecutive sixteenths belong under one hand.
    assert max(spreads) <= 5, f"rolling position spread {max(spreads)} frets"


def test_double_stop_takes_distinct_strings_within_a_reachable_span():
    notes = [
        QNote(onset=0, duration=QUARTER, pitch=60),
        QNote(onset=0, duration=QUARTER, pitch=64),
    ]
    warnings = assign_fretting(notes)

    assert warnings == []
    _assert_playable(notes)
    assert notes[0].string != notes[1].string
    frets = _fretted(notes)
    if frets:
        assert max(frets) - min(frets) <= 4


def test_tied_chain_keeps_one_string_and_fret_throughout():
    chain = [
        QNote(onset=i * QUARTER, duration=QUARTER, pitch=60, tied_from_prev=i > 0)
        for i in range(4)
    ]
    # A line moving up the neck between the chain's attacks: the held note must
    # not follow the hand.
    moving = [
        QNote(onset=EIGHTH, duration=EIGHTH, pitch=79),
        QNote(onset=QUARTER + EIGHTH, duration=EIGHTH, pitch=81),
        QNote(onset=2 * QUARTER + EIGHTH, duration=EIGHTH, pitch=83),
    ]
    notes = chain + moving
    warnings = assign_fretting(notes)

    assert warnings == []
    _assert_playable(notes)
    assert len({(q.string, q.fret) for q in chain}) == 1


def test_pitch_below_the_low_e_is_transposed_up_an_octave_with_a_warning():
    notes = [
        QNote(onset=0, duration=QUARTER, pitch=30),
        QNote(onset=QUARTER, duration=QUARTER, pitch=45),
    ]
    warnings = assign_fretting(notes)

    assert len(warnings) == 1
    assert "30" in warnings[0] and "42" in warnings[0]
    assert "transposed" in warnings[0]
    assert notes[0].pitch == 42
    _assert_playable(notes)


def test_top_fret_boundary_is_playable_and_silent():
    notes = [QNote(onset=0, duration=QUARTER, pitch=86)]  # 64 + 22, high E string
    warnings = assign_fretting(notes)

    assert warnings == []
    assert (notes[0].string, notes[0].fret) == (1, 22)
    _assert_playable(notes)


def test_pitch_identity_holds_across_a_mixed_line():
    """The property check, over open strings, chords, ties and the extremes."""
    notes = _line([40, 55, 64, 76, 47, 62, 69, 52], EIGHTH)
    notes += [
        QNote(onset=8 * EIGHTH, duration=QUARTER, pitch=60),
        QNote(onset=8 * EIGHTH, duration=QUARTER, pitch=64),
        QNote(onset=8 * EIGHTH, duration=QUARTER, pitch=67),
        QNote(onset=8 * EIGHTH + QUARTER, duration=QUARTER, pitch=60, tied_from_prev=True),
        QNote(onset=9 * EIGHTH + QUARTER, duration=QUARTER, pitch=86),
        QNote(onset=10 * EIGHTH + QUARTER, duration=QUARTER, pitch=41),
    ]
    warnings = assign_fretting(notes)

    assert warnings == []
    _assert_playable(notes)


# --------------------------------------------------------------------------
# supporting behaviour
# --------------------------------------------------------------------------
def test_empty_input_is_a_no_op():
    assert assign_fretting([]) == []


def test_capo_moves_the_whole_fretboard():
    notes = _line([66, 68, 70, 71], QUARTER)  # open position with a capo at 2
    warnings = assign_fretting(notes, capo=2)

    assert warnings == []
    _assert_playable(notes, capo=2)


def test_pitch_above_the_range_is_clamped_to_the_top_fret():
    notes = [QNote(onset=0, duration=QUARTER, pitch=95)]
    warnings = assign_fretting(notes)

    assert len(warnings) == 1
    assert "clamped" in warnings[0]
    assert notes[0].pitch == 86
    assert (notes[0].string, notes[0].fret) == (1, 22)
    _assert_playable(notes)


def test_fast_alternation_crosses_strings_instead_of_sliding():
    notes = _line([65, 72] * 6, SIXTEENTH)
    warnings = assign_fretting(notes)

    assert warnings == []
    _assert_playable(notes)
    for a, b in zip(notes, notes[1:]):
        assert not (a.string == b.string and abs(a.fret - b.fret) >= 5), (
            f"5+ fret slide on string {a.string} at sixteenth-note speed"
        )


def test_orphan_tie_is_fretted_and_warned_about():
    notes = [
        QNote(onset=0, duration=QUARTER, pitch=60, tied_from_prev=True),
        QNote(onset=QUARTER, duration=QUARTER, pitch=62),
    ]
    warnings = assign_fretting(notes)

    assert len(warnings) == 1
    assert "no contiguous predecessor" in warnings[0]
    _assert_playable(notes)


def test_result_is_deterministic():
    first = _line([48, 50, 52, 53, 55, 57, 59, 60], EIGHTH)
    second = _line([48, 50, 52, 53, 55, 57, 59, 60], EIGHTH)
    assign_fretting(first)
    assign_fretting(second)

    assert [(q.string, q.fret) for q in first] == [(q.string, q.fret) for q in second]


def test_seven_string_tuning_is_supported():
    tuning = STANDARD_TUNING + (35,)  # low B, string 7
    notes = _line([35, 40, 47, 55], QUARTER)
    warnings = assign_fretting(notes, tuning=tuning)

    assert warnings == []
    _assert_playable(notes, tuning=tuning)
    assert notes[0].string == 7 and notes[0].fret == 0
