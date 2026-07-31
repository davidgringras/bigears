"""Fretting — deciding where on the neck each quantized note is played.

Pitch does not determine tab. Most pitches sit under three to five playable
(string, fret) pairs on a standard-tuned guitar, and choosing among them is the
whole job: the same eight notes are a comfortable second-position scale or an
unplayable zig-zag depending on decisions no single note can make locally. So
the choice is made globally, by a Viterbi pass over note GROUPS (notes sharing
an onset — a double stop or a chord is one group), minimizing internal cost
(how awkward a fingering is) plus transition cost (how far the hand travels).

Two things about the state are worth knowing before tuning any weight here:

* The hidden state is a fingering PLUS a hand position (the fret the index
  finger occupies), not a fingering alone. Scoring movement as the fret
  distance between consecutive NOTES is the obvious formulation and it is
  wrong: a chromatic E4->E5 in sixteenths costs 12 units as a single-string
  slide up the high E and 17 units played in one position with two string
  crossings, so a note-distance cost prefers the slide no guitarist would
  play. Measured against a persistent hand anchor the same comparison is 8-10
  units versus about 1, which is the answer we want and the reason the anchor
  is carried in the state rather than inferred from the frets.
* Open strings are transparent to position — reachable from anywhere, so they
  neither pin the hand nor pay for movement. A group of nothing but open
  strings inherits whatever position preceded it.

Everything else is preference expressed as cost: stretch, string crossings,
high positions, open strings in low positions, and a jazz prior that keeps
dense lines out of open position (a bebop eighth-note line played on open
strings loses the legato; the same line at the fifth fret does not).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from .model import GRID_SIXTEENTH, STANDARD_TUNING, TICKS_PER_QUARTER, QNote

# --- hand geometry ---------------------------------------------------------
COMFORT_SPAN = 3        # frets covered one-finger-per-fret from the index finger
HAND_SPAN = 4           # frets reachable with a stretch (pinky at position + 4)
MAX_GROUP_SPAN = 4      # fret span allowed inside one simultaneous group
LOW_POSITION = 3        # at or below this the open strings are idiomatic
HIGH_POSITION = 15      # above this the neck gets cramped and the tab unreadable

# --- costs (all in the same arbitrary unit; one unit ~ one fret of travel) --
W_MOVE = 1.0            # per fret the hand position shifts between groups
W_STRING_CROSS = 0.35   # per string crossed between groups
W_REACH = 0.25          # per fret stretched beyond COMFORT_SPAN
W_SPAN = 0.6            # per fret of stretch inside a group
W_HIGH_POS = 0.45       # per fret above HIGH_POSITION
W_OPEN_LOW = 0.4        # bonus (subtracted) per open string in a low position
W_DENSE_LOW = 0.5       # per fret below LOW_POSITION when the line is dense
W_VOICE_CROSS = 0.5     # per inverted pair inside a group
W_FAST_SLIDE = 1.2      # per fret beyond SLIDE_FRETS on a fast same-string jump

# --- thresholds ------------------------------------------------------------
DENSE_IOI = TICKS_PER_QUARTER // 2      # median gap at or under an eighth = dense
FAST_IOI = TICKS_PER_QUARTER // 2       # gap at or under an eighth = fast
SLIDE_FRETS = 5                         # same-string jump that stops being a reach
MOVE_DISCOUNT_FLOOR = 0.4               # a long rest still costs something to cross
TIE_SLOP = GRID_SIXTEENTH               # how far a tie may miss its predecessor
MAX_CANDIDATES = 24                     # K in the O(n * K^2) budget
MAX_PLACEMENTS = 4096                   # guard against pathological chord groups


@dataclass(frozen=True)
class _Fingering:
    """One way to play a group, with the fretting hand anchored somewhere."""

    placements: tuple[tuple[int, int], ...]  # (string, fret), parallel to the group
    position: int                            # index-finger fret; opens ignore it
    internal: float
    mean_string: float
    lone: tuple[int, int] | None             # (string, fret) if the group is one note


class _WarningLog:
    """Collects warnings, folding repeats of one kind into a single line."""

    def __init__(self) -> None:
        self._text: dict[tuple, str] = {}
        self._count: dict[tuple, int] = {}
        self._first: dict[tuple, int] = {}

    def log(self, key: tuple, text: str, onset: int) -> None:
        if key in self._count:
            self._count[key] += 1
            self._first[key] = min(self._first[key], onset)
        else:
            self._text[key] = text
            self._count[key] = 1
            self._first[key] = onset

    def render(self) -> list[str]:
        out = []
        for key, text in self._text.items():
            n = self._count[key]
            where = f"first at tick {self._first[key]}"
            out.append(f"{text} ({where}{f', {n} notes affected' if n > 1 else ''})")
        return out


def assign_fretting(
    qnotes: list[QNote],
    tuning: tuple[int, ...] = STANDARD_TUNING,
    capo: int = 0,
    max_fret: int = 22,
) -> list[str]:
    """Fill in string/fret on every QNote; return human-readable warnings.

    Notes are mutated in place. A pitch outside the instrument's range has its
    `pitch` rewritten as well — up an octave if it is below the lowest open
    string, clamped to the top fret if it is above the last one — so that
    string/fret and pitch never disagree downstream (the audit resynthesizes
    the score, and a tab that sounds an octave off its own pitch field would
    quietly poison every metric). Both rewrites are warned about explicitly.
    """
    log = _WarningLog()
    if not qnotes:
        return []
    if not tuning:
        raise ValueError("tuning must name at least one string")
    if max_fret < 0:
        raise ValueError("max_fret must be non-negative")

    order = sorted(range(len(qnotes)), key=lambda i: (qnotes[i].onset, -qnotes[i].pitch, i))
    _normalize_range(qnotes, order, tuning, capo, max_fret, log)

    locked = _tie_chains(qnotes, order, log)
    participants = [i for i in order if i not in locked]
    groups = _group_by_onset(qnotes, participants)
    dense = _is_dense(qnotes)

    chosen = _solve(qnotes, groups, tuning, capo, max_fret, dense, log)
    for gi, (_, idxs) in enumerate(groups):
        for note_idx, (string, fret) in zip(idxs, chosen[gi]):
            qnotes[note_idx].string = string
            qnotes[note_idx].fret = fret

    # A tie is a hard constraint, not a preference: the continuation is the same
    # finger still holding the same string, so it copies rather than chooses.
    for i in order:
        head = locked.get(i)
        if head is not None:
            qnotes[i].string = qnotes[head].string
            qnotes[i].fret = qnotes[head].fret

    _repair_collisions(qnotes, tuning, capo, max_fret, log)
    return log.render()


# --------------------------------------------------------------------------
# range, ties, grouping
# --------------------------------------------------------------------------
def _normalize_range(
    qnotes: list[QNote],
    order: list[int],
    tuning: tuple[int, ...],
    capo: int,
    max_fret: int,
    log: _WarningLog,
) -> None:
    """Fold out-of-range pitches back onto the fretboard, warning as we go."""
    lowest = min(tuning) + capo
    highest = max(tuning) + capo + max_fret
    top_string = tuning.index(max(tuning)) + 1
    for i in order:
        q = qnotes[i]
        if q.pitch < lowest:
            octaves = -((q.pitch - lowest) // 12)
            new = q.pitch + 12 * octaves
            log.log(
                ("low", q.pitch, new),
                f"pitch {q.pitch} is below the lowest playable pitch {lowest}; "
                f"transposed up {octaves} octave(s) to {new}",
                q.onset,
            )
            q.pitch = new
        if q.pitch > highest:
            log.log(
                ("high", q.pitch, highest),
                f"pitch {q.pitch} is above the highest playable pitch {highest}; "
                f"clamped to string {top_string} fret {max_fret} (pitch {highest})",
                q.onset,
            )
            q.pitch = highest


def _tie_chains(qnotes: list[QNote], order: list[int], log: _WarningLog) -> dict[int, int]:
    """Map each tied continuation to the chain head it must copy.

    Continuations are held out of the search entirely: the hand is already
    where it needs to be, so they neither choose a fingering nor pay for one.
    """
    locked: dict[int, int] = {}
    head_of: dict[int, int] = {}    # pitch -> index of the chain head
    chain_end: dict[int, int] = {}  # chain head index -> offset of its last member
    for i in order:
        q = qnotes[i]
        head = head_of.get(q.pitch)
        contiguous = head is not None and chain_end[head] >= q.onset - TIE_SLOP
        if q.tied_from_prev and contiguous:
            locked[i] = head
            chain_end[head] = max(chain_end[head], q.offset)
            continue
        if q.tied_from_prev:
            log.log(
                ("orphan-tie", q.pitch),
                f"tied note at pitch {q.pitch} has no contiguous predecessor; "
                "fretted as a fresh note",
                q.onset,
            )
        head_of[q.pitch] = i
        chain_end[i] = q.offset
    return locked


def _group_by_onset(qnotes: list[QNote], participants: list[int]) -> list[tuple[int, list[int]]]:
    """Bucket time-ordered participants into simultaneous groups."""
    groups: list[tuple[int, list[int]]] = []
    for i in participants:
        onset = qnotes[i].onset
        if groups and groups[-1][0] == onset:
            groups[-1][1].append(i)
        else:
            groups.append((onset, [i]))
    return groups


def _is_dense(qnotes: list[QNote]) -> bool:
    """True when the median gap between attacks is an eighth note or shorter."""
    onsets = sorted({q.onset for q in qnotes})
    if len(onsets) < 2:
        return False
    return statistics.median(b - a for a, b in zip(onsets, onsets[1:])) <= DENSE_IOI


# --------------------------------------------------------------------------
# candidate fingerings
# --------------------------------------------------------------------------
def _note_options(
    pitch: int, tuning: tuple[int, ...], capo: int, max_fret: int
) -> list[tuple[int, int]]:
    """Every (string, fret) that sounds `pitch`, string 1 first."""
    out = []
    for idx, open_pitch in enumerate(tuning):
        fret = pitch - (open_pitch + capo)
        if 0 <= fret <= max_fret:
            out.append((idx + 1, fret))
    return out


def _placements(
    options: list[list[tuple[int, int]]], max_span: int, limit: int
) -> list[tuple[tuple[int, int], ...]]:
    """Fingerings for a group: distinct strings, fretted span within max_span."""
    results: list[tuple[tuple[int, int], ...]] = []
    chosen: list[tuple[int, int]] = []
    used: set[int] = set()

    def walk(i: int, lo: int | None, hi: int | None) -> None:
        if len(results) >= limit:
            return
        if i == len(options):
            results.append(tuple(chosen))
            return
        for string, fret in options[i]:
            if string in used:
                continue
            nlo, nhi = lo, hi
            if fret > 0:
                nlo = fret if lo is None else min(lo, fret)
                nhi = fret if hi is None else max(hi, fret)
                if nhi - nlo > max_span:
                    continue
            chosen.append((string, fret))
            used.add(string)
            walk(i + 1, nlo, nhi)
            chosen.pop()
            used.discard(string)

    walk(0, None, None)
    return results


def _positions_for(placement: tuple[tuple[int, int], ...], inherited: list[int]) -> list[int]:
    """Index-finger frets consistent with a placement (opens constrain nothing)."""
    fretted = [f for _, f in placement if f > 0]
    if not fretted:
        return inherited
    lo, hi = min(fretted), max(fretted)
    start = max(0, hi - HAND_SPAN)
    if start > lo:
        # Wider than the hand — anchor at the lowest note and let W_REACH price it.
        return [lo]
    return list(range(start, lo + 1))


def _build_layer(
    placements: list[tuple[tuple[int, int], ...]],
    pitches: tuple[int, ...],
    inherited: list[int],
    dense: bool,
) -> list[_Fingering]:
    """All (fingering, hand position) states for one group, pruned to MAX_CANDIDATES."""
    states: list[_Fingering] = []
    for placement in placements:
        mean_string = sum(s for s, _ in placement) / len(placement)
        lone = placement[0] if len(placement) == 1 else None
        for position in _positions_for(placement, inherited):
            states.append(
                _Fingering(
                    placements=placement,
                    position=position,
                    internal=_internal_cost(placement, pitches, position, dense),
                    mean_string=mean_string,
                    lone=lone,
                )
            )
    states.sort(key=lambda s: (s.internal, s.placements, s.position))
    return states[:MAX_CANDIDATES]


def _internal_cost(
    placement: tuple[tuple[int, int], ...],
    pitches: tuple[int, ...],
    position: int,
    dense: bool,
) -> float:
    """How awkward this fingering is, independent of what came before it."""
    cost = 0.0
    fretted = [f for _, f in placement if f > 0]
    if fretted:
        cost += W_SPAN * (max(fretted) - min(fretted))
        for fret in fretted:
            cost += W_REACH * max(0, fret - position - COMFORT_SPAN)
    cost += W_HIGH_POS * max(0, position - HIGH_POSITION)

    n_open = len(placement) - len(fretted)
    if n_open and not dense and position <= LOW_POSITION:
        cost -= W_OPEN_LOW * n_open
    if dense:
        # Jazz prior: a dense line lives at the third fret and above, where every
        # note can be damped, bent and slurred. Open position is for ballads.
        cost += W_DENSE_LOW * max(0, LOW_POSITION - position)

    # Voicing: pitches arrive high to low, so strings should ascend 1, 2, 3...
    for a in range(len(placement) - 1):
        if pitches[a] > pitches[a + 1] and placement[a][0] > placement[a + 1][0]:
            cost += W_VOICE_CROSS
    return cost


def _transition_cost(prev: _Fingering, cur: _Fingering, ioi: int) -> float:
    """What it costs the hand to get from one group to the next."""
    # A long gap is a free ride: the hand has time to move, so discount the shift.
    scale = 1.0 if ioi <= TICKS_PER_QUARTER else max(MOVE_DISCOUNT_FLOOR, TICKS_PER_QUARTER / ioi)
    cost = W_MOVE * abs(cur.position - prev.position) * scale
    cost += W_STRING_CROSS * abs(cur.mean_string - prev.mean_string)
    if ioi <= FAST_IOI and prev.lone is not None and cur.lone is not None:
        (ps, pf), (cs, cf) = prev.lone, cur.lone
        if ps == cs and pf > 0 and cf > 0:
            jump = abs(cf - pf)
            if jump >= SLIDE_FRETS:
                # At speed a long same-string slide is a stumble; cross instead.
                cost += W_FAST_SLIDE * (jump - SLIDE_FRETS + 1)
    return cost


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------
def _solve(
    qnotes: list[QNote],
    groups: list[tuple[int, list[int]]],
    tuning: tuple[int, ...],
    capo: int,
    max_fret: int,
    dense: bool,
    log: _WarningLog,
) -> list[tuple[tuple[int, int], ...]]:
    """Viterbi over groups; returns the chosen placement per group."""
    if not groups:
        return []

    per_group: list[tuple[list[tuple[tuple[int, int], ...]], tuple[int, ...]]] = []
    for onset, idxs in groups:
        pitches = tuple(qnotes[i].pitch for i in idxs)
        options = [_note_options(p, tuning, capo, max_fret) for p in pitches]
        placements = _placements(options, MAX_GROUP_SPAN, MAX_PLACEMENTS)
        if not placements:
            placements = _placements(options, max_fret, MAX_PLACEMENTS)
            if placements:
                log.log(
                    ("wide-group", pitches),
                    f"group {list(pitches)} needs a fret span wider than {MAX_GROUP_SPAN}",
                    onset,
                )
        if not placements:
            # More notes than strings, or two voices that only one string can play.
            placements = [tuple(opts[0] for opts in options)]
            log.log(
                ("crowded-group", pitches),
                f"group {list(pitches)} cannot be spread over distinct strings; "
                "doubled onto shared strings",
                onset,
            )
        per_group.append((placements, pitches))

    layers: list[list[_Fingering]] = []
    backs: list[list[int]] = []
    prev_layer: list[_Fingering] = []
    prev_cost: list[float] = []
    prev_onset = groups[0][0]

    for gi, (onset, _) in enumerate(groups):
        placements, pitches = per_group[gi]
        inherited = sorted({s.position for s in prev_layer}) or [0]
        layer = _build_layer(placements, pitches, inherited, dense)
        costs: list[float] = []
        back: list[int] = []
        ioi = max(1, onset - prev_onset)
        for state in layer:
            if not prev_layer:
                costs.append(state.internal)
                back.append(-1)
                continue
            best, best_k = float("inf"), 0
            for k, prior in enumerate(prev_layer):
                c = prev_cost[k] + _transition_cost(prior, state, ioi)
                if c < best:
                    best, best_k = c, k
            costs.append(best + state.internal)
            back.append(best_k)
        layers.append(layer)
        backs.append(back)
        prev_layer, prev_cost, prev_onset = layer, costs, onset

    k = min(range(len(prev_cost)), key=lambda j: (prev_cost[j], j))
    chosen: list[tuple[tuple[int, int], ...]] = []
    for gi in range(len(groups) - 1, -1, -1):
        chosen.append(layers[gi][k].placements)
        k = backs[gi][k]
    chosen.reverse()
    return chosen


def _repair_collisions(
    qnotes: list[QNote],
    tuning: tuple[int, ...],
    capo: int,
    max_fret: int,
    log: _WarningLog,
) -> None:
    """Two notes cannot share a string at the same instant; ties win the argument."""
    by_onset: dict[int, list[int]] = {}
    for i, q in enumerate(qnotes):
        by_onset.setdefault(q.onset, []).append(i)
    for onset, idxs in by_onset.items():
        if len(idxs) < 2:
            continue
        # Held notes are pinned, so they claim their string first.
        used: dict[int, int] = {}
        for i in sorted(idxs, key=lambda j: (not qnotes[j].tied_from_prev, j)):
            q = qnotes[i]
            if q.string not in used:
                used[q.string] = i
                continue
            free = [
                opt
                for opt in _note_options(q.pitch, tuning, capo, max_fret)
                if opt[0] not in used
            ]
            if not free:
                log.log(
                    ("stacked", q.pitch),
                    f"pitch {q.pitch} shares a string with a held note and has "
                    "nowhere else to go",
                    onset,
                )
                continue
            string, fret = min(free, key=lambda o: (abs(o[1] - (q.fret or 0)), o[0]))
            log.log(
                ("moved", q.pitch),
                f"pitch {q.pitch} moved to string {string} fret {fret}; a tied note "
                "was holding its string",
                onset,
            )
            q.string, q.fret = string, fret
            used[string] = i
