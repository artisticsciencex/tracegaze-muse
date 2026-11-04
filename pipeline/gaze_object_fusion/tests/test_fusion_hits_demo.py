"""Integration tests over fuse(): hit ordering, validity, demo selection.

Built on REAL trimmed + tracked artifacts (synthetic 10-frame recording).
"""

import pytest

from gaze_object_fusion.fusion import FusionInputs, fuse

from futil import WORLD_TS, make_tracked_artifact, trim_selection

# object geometry: BIG contains SMALL_A == SMALL_B (identical boxes)
BIG = {"name": "big", "box": (20, 20, 100, 100), "score": 0.95}       # 6400
SMALL_A = {"name": "small-a", "box": (40, 40, 60, 60), "score": 0.9}  # 400
SMALL_B = {"name": "small-b", "box": (40, 40, 60, 60), "score": 0.9}  # 400
SMALL_HI = {"name": "small-hi", "box": (40, 40, 60, 60), "score": 0.99}

INSIDE_ALL = {"x": 50.5, "y": 50.2}       # inside SMALL boxes and BIG
INSIDE_BIG = {"x": 90.5, "y": 30.4}       # inside BIG only
OUTSIDE_ALL = {"x": 5.0, "y": 5.0}        # inside no mask


def run(tmp_path, gaze_rows, objects, **fuse_kw):
    tracked, trim, oids = make_tracked_artifact(tmp_path, gaze_rows, objects)
    inputs = FusionInputs(tracked)
    fused, cells, validation, extras = fuse(inputs, **fuse_kw)
    assert all(v["ok"] for v in validation), validation
    return fused, cells, extras, oids


def test_hit_order_area_then_score_then_id(tmp_path):
    gaze = [
        {"ts": WORLD_TS[3] + 1_000_000, **INSIDE_ALL},
        {"ts": WORLD_TS[4] + 1_000_000, **INSIDE_BIG},
        {"ts": WORLD_TS[6] + 1_000_000, **OUTSIDE_ALL},
    ]
    fused, _, _, oids = run(tmp_path, gaze,
                            [BIG, SMALL_HI, SMALL_A, SMALL_B])
    big_id, hi_id, a_id, b_id = oids

    r = fused.iloc[0]
    # area splits SMALLs from BIG; score puts small-hi first; the equal
    # (area, score) pair small-a/small-b falls back to stable id order
    assert r["hit_object_ids"] == [hi_id, a_id, b_id, big_id]
    assert r["primary_object_id"] == hi_id, "first entry IS the primary"
    assert r["n_hits"] == 4

    r = fused.iloc[1]
    assert r["hit_object_ids"] == [big_id]
    assert r["primary_object_id"] == big_id
    assert r["primary_mask_area"] == 6400

    r = fused.iloc[2]
    assert r["hit_object_ids"] == [] and r["n_hits"] == 0
    import pandas as pd
    assert pd.isna(r["primary_object_id"])
    assert r["assignment"] == "assigned", "no hit is still a valid sample"


def test_validity_flags_and_frame_assignment_independent(tmp_path):
    gaze = [
        {"ts": WORLD_TS[2] + 2_000_000, **INSIDE_ALL, "worn": 0.0},
        {"ts": WORLD_TS[2] + 3_000_000, "x": -5.0, "y": 60.0},
        {"ts": WORLD_TS[2] + 4_000_000, "x": float("nan"), "y": 60.0},
        {"ts": WORLD_TS[2] + 5_000_000, **INSIDE_ALL},
    ]
    fused, _, _, _ = run(tmp_path, gaze, [BIG])
    worn0, oob, nonf, ok = (fused.iloc[i] for i in range(4))

    for r in (worn0, oob, nonf, ok):        # assignment is timestamp-only
        assert r["frame_index"] == 2 and r["assignment"] == "assigned"

    assert not worn0["gaze_valid"] and worn0["invalid_reason"] == "not_worn"
    assert worn0["n_hits"] == 0
    assert oob["gaze_valid"] and not oob["coords_in_frame"]
    assert oob["n_hits"] == 0
    assert not nonf["gaze_valid"] and nonf["invalid_reason"] \
        == "nonfinite_coords"
    assert ok["gaze_valid"] and ok["coords_in_frame"] and ok["n_hits"] == 1
    # only the fully valid row is eligible for (and selected as) the cursor
    assert list(fused["demo_selected"]) == [False, False, False, True]


def test_demo_smallest_abs_offset_and_tiebreak(tmp_path):
    gaze = [
        {"ts": WORLD_TS[4] - 8_000_000, **INSIDE_ALL},
        {"ts": WORLD_TS[4] + 2_000_000, **INSIDE_ALL},
        {"ts": WORLD_TS[4] + 5_000_000, **INSIDE_ALL},
        # frame 7: symmetric ±3 ms -> tie on |offset| -> EARLIER ts wins
        {"ts": WORLD_TS[7] - 3_000_000, **INSIDE_ALL},
        {"ts": WORLD_TS[7] + 3_000_000, **INSIDE_ALL},
    ]
    fused, _, extras, _ = run(tmp_path, gaze, [BIG])
    assert list(fused["demo_selected"]) == [False, True, False,
                                            True, False]
    assert extras["demo_rows"][4] == 1
    assert extras["demo_rows"][7] == 3


def test_demo_never_borrows_from_adjacent_frame(tmp_path):
    """Frame 5 has no sample; frame 6's earliest sample sits 1 ns past the
    5/6 boundary. Frame 5 must show NO cursor."""
    sel_mid_5_6 = (WORLD_TS[5] + WORLD_TS[6]) // 2
    gaze = [
        {"ts": WORLD_TS[4] + 1_000_000, **INSIDE_ALL},      # frame 4
        {"ts": sel_mid_5_6, **INSIDE_ALL},                  # frame 6 (bound)
    ]
    fused, cells, extras, _ = run(tmp_path, gaze, [BIG])
    assert fused.iloc[1]["frame_index"] == 6
    assert 5 not in extras["demo_rows"], \
        "no borrowing: frame 5 has no assigned row, shows no cursor"
    assert extras["demo_rows"][6] == 1


def test_no_matching_frame_outside_trim_interval(tmp_path):
    """Out-of-interval samples (injected post-trim: the trim stage itself
    drops them) get no_matching_frame; interval edges are half-open."""
    import pandas as pd

    probe = make_tracked_artifact(
        tmp_path / "probe", [{"ts": WORLD_TS[0], **INSIDE_ALL}], [BIG])
    sel = trim_selection(probe[1])
    start, end = sel["absolute_start_ns"], sel["absolute_end_boundary_ns"]

    tracked, _, _ = make_tracked_artifact(
        tmp_path / "main",
        [{"ts": start, **INSIDE_ALL},
         {"ts": end - 1, **INSIDE_ALL}], [BIG],
        post_trim_gaze=[{"ts": start - 1, **INSIDE_ALL},
                        {"ts": end, **INSIDE_ALL}])
    fused, _, validation, _ = fuse(FusionInputs(tracked))
    assert all(v["ok"] for v in validation), validation
    by_ts = {int(r["gaze_ts_ns"]): r for _, r in fused.iterrows()}
    assert by_ts[start]["assignment"] == "assigned"
    assert by_ts[start]["frame_index"] == 0
    assert by_ts[end - 1]["assignment"] == "assigned"
    assert by_ts[end - 1]["frame_index"] == len(WORLD_TS) - 1
    assert by_ts[start - 1]["assignment"] == "no_matching_frame"
    assert by_ts[end]["assignment"] == "no_matching_frame"
    assert pd.isna(by_ts[start - 1]["frame_index"])
    assert pd.isna(by_ts[start - 1]["frame_offset_ns"])


def test_offsets_recorded_against_assigned_frame(tmp_path):
    gaze = [{"ts": WORLD_TS[6] - 12_345_678, **INSIDE_ALL}]
    fused, _, _, _ = run(tmp_path, gaze, [BIG])
    r = fused.iloc[0]
    # midpoint(180,220)=200ms; 220-12.345678=207.65ms -> frame 6's cell
    assert r["frame_index"] == 6
    assert r["frame_offset_ns"] == -12_345_678
    assert r["frame_offset_ms"] == pytest.approx(-12.345678)


def test_object_absent_frames_produce_no_hits(tmp_path):
    obj = dict(BIG, frames=list(range(0, 5)))       # tracked frames 0..4 only
    gaze = [
        {"ts": WORLD_TS[3] + 1_000_000, **INSIDE_ALL},
        {"ts": WORLD_TS[8] + 1_000_000, **INSIDE_ALL},
    ]
    fused, _, _, _ = run(tmp_path, gaze, [obj])
    assert fused.iloc[0]["n_hits"] == 1
    assert fused.iloc[1]["n_hits"] == 0, \
        "no mask rows at frame 8 -> gaze hits nothing (missing = not visible)"
