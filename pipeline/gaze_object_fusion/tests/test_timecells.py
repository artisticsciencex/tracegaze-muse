"""Unit tests for the timestamp-defined frame-cell assignment rule."""

import pytest

from gaze_object_fusion.timecells import (ASSIGNED, NO_MATCHING_FRAME,
                                          FrameCells, TimeCellError)

from futil import T0, WORLD_TS

START = WORLD_TS[0]                       # trim semantics: b[0] = world[0]
END = WORLD_TS[-1] + 34_000_000           # time after last frame


def make_cells(**kw):
    return FrameCells(WORLD_TS, START, END, **kw)


def test_boundaries_are_exact_integer_midpoints():
    c = make_cells()
    assert c.boundaries[0] == START
    assert c.boundaries[-1] == END
    for i in range(1, len(WORLD_TS)):
        assert c.boundaries[i] == (WORLD_TS[i - 1] + WORLD_TS[i]) // 2


def test_half_open_membership_at_exact_boundaries():
    c = make_cells()
    for i in range(1, c.n_frames):
        b = c.boundaries[i]
        assert c.assign(b).frame_index == i, "boundary belongs to the right"
        assert c.assign(b - 1).frame_index == i - 1
    # interval edges
    assert c.assign(START).frame_index == 0
    assert c.assign(START - 1).status == NO_MATCHING_FRAME
    assert c.assign(END - 1).frame_index == c.n_frames - 1
    assert c.assign(END).status == NO_MATCHING_FRAME
    assert c.assign(END + 10**9).status == NO_MATCHING_FRAME


def test_signed_offsets_preserved_exactly():
    c = make_cells()
    # sample 7 ms BEFORE frame 5's timestamp, inside frame 5's cell
    # (cell of frame 5 starts at midpoint(166,180)=173 ms)
    ts = WORLD_TS[5] - 7_000_000
    a = c.assign(ts)
    assert a.frame_index == 5
    assert a.offset_ns == -7_000_000
    # sample 24 ms AFTER frame 1 (cell [25 ms, 75 ms) around 50 ms) —
    # a fixed 16/20 ms threshold would have dropped this; the rule must not
    a = c.assign(WORLD_TS[1] + 24_000_000)
    assert a.frame_index == 1
    assert a.offset_ns == 24_000_000
    assert a.status == ASSIGNED


def test_no_global_threshold_wide_cell_far_sample():
    """The 40 ms gap between frames 5 (180 ms) and 6 (220 ms): a sample
    19 ms after frame 5 is still frame 5's (midpoint = 200 ms)."""
    c = make_cells()
    a = c.assign(WORLD_TS[5] + 19_000_000)
    assert a.frame_index == 5 and a.offset_ns == 19_000_000
    a = c.assign(WORLD_TS[5] + 21_000_000)     # past the midpoint
    assert a.frame_index == 6
    assert a.offset_ns == 21_000_000 - 40_000_000


def test_midpoint_floor_division_odd_sum():
    w = [T0 + 10, T0 + 15]                     # midpoint floor = T0+12
    c = FrameCells(w, T0 + 10, T0 + 100)
    assert c.boundaries[1] == T0 + 12
    assert c.assign(T0 + 11).frame_index == 0
    assert c.assign(T0 + 12).frame_index == 1


def test_exact_at_nanosecond_scale_beyond_float53():
    """T0 > 2**53: ±1 ns decisions must be exact (would corrupt via float)."""
    c = make_cells()
    b3 = c.boundaries[3]
    assert b3 > 2 ** 53, "sanity: beyond exact float53 integer range"
    assert c.assign(b3 - 1).frame_index == 2
    assert c.assign(b3).frame_index == 3


def test_anomalous_gap_reported_but_samples_still_assigned():
    w = [T0, T0 + 50_000_000, T0 + 300_000_000]      # 250 ms gap
    c = FrameCells(w, T0, T0 + 340_000_000, sanity_gap_ms=200.0)
    gaps = c.anomalous_gaps()
    assert len(gaps) == 1
    assert gaps[0]["between_frames"] == [1, 2]
    assert gaps[0]["gap_ms"] == pytest.approx(250.0)
    # a sample deep inside the anomalous gap is STILL assigned by midpoints
    a = c.assign(T0 + 120_000_000)     # midpoint = 175 ms -> frame 1
    assert a.frame_index == 1 and a.status == ASSIGNED
    # default threshold does not flag ordinary VFR
    assert make_cells().anomalous_gaps() == []


def test_single_frame_recording():
    c = FrameCells([T0], T0, T0 + 34_000_000)
    assert c.boundaries == [T0, T0 + 34_000_000]
    assert c.assign(T0 + 33_999_999).frame_index == 0
    assert c.assign(T0 + 34_000_000).status == NO_MATCHING_FRAME


def test_rejects_bad_inputs():
    with pytest.raises(TimeCellError):
        FrameCells([], T0, T0 + 1)
    with pytest.raises(TimeCellError):
        FrameCells([float(T0)], T0, T0 + 10**9)      # float timestamp
    with pytest.raises(TimeCellError):
        FrameCells([T0, T0], T0, T0 + 10**9)         # not increasing
    with pytest.raises(TimeCellError):
        FrameCells([T0], T0 + 1, T0 + 10**9)         # start after frame 0
    with pytest.raises(TimeCellError):
        FrameCells([T0], T0, T0)                     # end not after last
    c = make_cells()
    with pytest.raises(TimeCellError):
        c.assign(float(T0))                          # float gaze ts


def test_cell_bounds_cover_interval_without_overlap():
    c = make_cells()
    lo0, _ = c.cell_bounds(0)
    assert lo0 == START
    prev_hi = None
    for i in range(c.n_frames):
        lo, hi = c.cell_bounds(i)
        assert lo < hi
        if prev_hi is not None:
            assert lo == prev_hi, "cells tile the interval exactly"
        assert lo <= c.world[i] < hi, "each frame's timestamp is in its cell"
        prev_hi = hi
    assert prev_hi == END
