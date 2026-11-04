"""Presentation layer (display-only) and analytical dwell contracts.

The two layers must stay strictly separate: interpolation may make the
demo smooth, but it may never touch the authoritative 23-column table or
add a millisecond of dwell time.
"""

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from gaze_object_fusion import dwell as dwell_mod
from gaze_object_fusion import presentation as pres
from gaze_object_fusion.artifact import publish_fusion_artifact
from gaze_object_fusion.fusion import PUBLIC_COLUMNS, FusionInputs, fuse

from futil import WORLD_TS, make_tracked_artifact

BIG = {"name": "big", "box": (20, 20, 100, 100), "score": 0.95}
SMALL = {"name": "small", "box": (40, 40, 60, 60), "score": 0.5}
IN_BOTH = {"x": 50.5, "y": 50.5}
IN_BIG = {"x": 90.5, "y": 30.5}
OUT = {"x": 5.0, "y": 5.0}


def half_rate_gaze():
    """10 Hz-style gaze over 20 Hz-style video: a sample on every SECOND
    frame — exactly the real Aria pattern that produced the flicker."""
    return [{"ts": WORLD_TS[i] + 1_000, **(IN_BOTH if i % 4 == 0 else IN_BIG)}
            for i in range(0, len(WORLD_TS), 2)]


def build(tmp_path, gaze, objects=(BIG,), **kw):
    tracked, trim, oids = make_tracked_artifact(tmp_path, gaze, list(objects))
    inputs = FusionInputs(tracked)
    fused, cells, validation, extras = fuse(inputs)
    return tracked, inputs, fused, cells, oids


# ------------------------------------------------- presentation timeline

def test_half_rate_gaze_becomes_continuous_presentation(tmp_path):
    tracked, inputs, fused, cells, _ = build(tmp_path, half_rate_gaze())
    rows, policy = pres.build_timeline(fused, cells.world)
    assert len(rows) == cells.n_frames, "one row per video frame"
    assert [r["frame_index"] for r in rows] == list(range(cells.n_frames))
    counts = pres.method_counts(rows)
    # every frame between the first and last sample must show a cursor
    inner = rows[1:-1]
    assert all(r["presentation_cursor_visible"] for r in inner), \
        "the alternating blank-frame flicker must be gone"
    assert counts["exact"] >= len(half_rate_gaze()) - 1
    assert counts["linear_interpolation"] > 0
    # exact frames keep the RAW coordinates untouched
    for r in rows:
        if r["presentation_method"] == pres.METHOD_EXACT:
            raw = fused.iloc[r["left_gaze_sample_index"]]
            assert r["display_gaze_x_px"] == float(raw["gaze_x_px"])
            assert r["display_gaze_y_px"] == float(raw["gaze_y_px"])


def test_interpolation_uses_the_stated_linear_formula(tmp_path):
    gaze = [{"ts": WORLD_TS[0] + 1000, "x": 10.0, "y": 20.0},
            {"ts": WORLD_TS[4] + 1000, "x": 50.0, "y": 60.0}]
    tracked, inputs, fused, cells, _ = build(tmp_path, gaze)
    rows, policy = pres.build_timeline(fused, cells.world)
    mid = [r for r in rows
           if r["presentation_method"] == pres.METHOD_INTERP]
    assert mid, "frames between the two samples must interpolate"
    for r in mid:
        t0 = int(fused["gaze_ts_ns"].iat[r["left_gaze_sample_index"]])
        t1 = int(fused["gaze_ts_ns"].iat[r["right_gaze_sample_index"]])
        x0 = float(fused["gaze_x_px"].iat[r["left_gaze_sample_index"]])
        x1 = float(fused["gaze_x_px"].iat[r["right_gaze_sample_index"]])
        alpha = (r["frame_timestamp_ns"] - t0) / (t1 - t0)
        assert r["interpolation_alpha"] == pytest.approx(alpha)
        assert r["display_gaze_x_px"] == pytest.approx(x0 + alpha * (x1 - x0))
        assert 0.0 < alpha < 1.0


def test_adaptive_gap_policy_matches_the_documented_rule():
    ts = [i * 100_000_000 for i in range(10)]          # 100 ms cadence
    p = pres.gap_policy(ts)
    assert p["median_interval_ms"] == 100.0
    assert p["max_interpolation_gap_ms"] == 250.0      # clamped at 250
    fast = pres.gap_policy([i * 5_000_000 for i in range(10)])
    assert fast["max_interpolation_gap_ms"] == 50.0    # clamped at 50
    mid = pres.gap_policy([i * 30_000_000 for i in range(10)])
    assert mid["max_interpolation_gap_ms"] == 90.0     # 3 x 30 ms


def test_long_gaps_are_not_bridged_and_boundaries_are_limited(tmp_path):
    gaze = [{"ts": WORLD_TS[0] + 1000, **IN_BOTH},
            {"ts": WORLD_TS[9] - 1000, **IN_BOTH}]     # far apart
    tracked, inputs, fused, cells, _ = build(tmp_path, gaze)
    rows, policy = pres.build_timeline(fused, cells.world)
    methods = {r["frame_index"]: r["presentation_method"] for r in rows}
    assert pres.METHOD_NONE in methods.values(), \
        "a gap beyond the adaptive limit must not be bridged"
    assert not any(m == pres.METHOD_INTERP for m in methods.values())


def test_invalid_and_outside_samples_break_interpolation(tmp_path):
    gaze = [{"ts": WORLD_TS[0] + 1000, **IN_BOTH},
            {"ts": WORLD_TS[2] + 1000, **IN_BOTH, "worn": 0.0},
            {"ts": WORLD_TS[4] + 1000, "x": -20.0, "y": 50.0},
            {"ts": WORLD_TS[6] + 1000, **IN_BOTH}]
    tracked, inputs, fused, cells, _ = build(tmp_path, gaze)
    rows, _ = pres.build_timeline(fused, cells.world)
    used = {r["left_gaze_sample_index"] for r in rows
            if r["left_gaze_sample_index"] is not None}
    used |= {r["right_gaze_sample_index"] for r in rows
             if r["right_gaze_sample_index"] is not None}
    assert 1 not in used and 2 not in used, \
        "invalid / outside-frame rows may never support a display position"


def test_presentation_never_changes_the_authoritative_table(tmp_path):
    gaze = half_rate_gaze()
    tracked, inputs, fused, cells, _ = build(tmp_path, gaze)
    before = fused.copy(deep=True)
    rows, _ = pres.build_timeline(fused, cells.world)
    pd.testing.assert_frame_equal(fused, before)
    assert all("presentation" not in c for c in PUBLIC_COLUMNS)


# ---------------------------------------------------------- dwell layer

def test_dwell_uses_raw_support_intervals_without_double_counting(tmp_path):
    gaze = half_rate_gaze()
    tracked, inputs, fused, cells, oids = build(tmp_path, gaze,
                                                objects=(BIG, SMALL))
    policy = pres.gap_policy([int(t) for t in fused["gaze_ts_ns"]])
    iv = dwell_mod.support_intervals(fused, inputs.interval_start_ns,
                                     inputs.interval_end_ns,
                                     policy["max_interpolation_gap_ns"])
    # intervals never overlap -> no interval is counted twice
    ordered = sorted(iv, key=lambda x: x["start_ns"])
    for a, b in zip(ordered, ordered[1:]):
        assert a["end_ns"] <= b["start_ns"] + 1
    for x in iv:
        assert inputs.interval_start_ns <= x["start_ns"] < x["end_ns"] \
            <= inputs.interval_end_ns
    meta = {o: {"name": o, "color": "#ffffff"} for o in oids}
    acc = dwell_mod.accumulate(iv, cells.world, inputs.interval_end_ns, meta)
    total = sum(acc["totals_ns"].values())
    assert total == acc["valid_supported_ns"]
    assert total <= inputs.interval_end_ns - inputs.interval_start_ns


def test_interpolated_positions_add_no_dwell(tmp_path):
    """Dwell must be identical whether or not the demo interpolates."""
    gaze = half_rate_gaze()
    tracked, inputs, fused, cells, oids = build(tmp_path, gaze)
    policy = pres.gap_policy([int(t) for t in fused["gaze_ts_ns"]])
    meta = {o: {"name": o, "color": "#fff"} for o in oids}
    iv = dwell_mod.support_intervals(fused, inputs.interval_start_ns,
                                     inputs.interval_end_ns,
                                     policy["max_interpolation_gap_ns"])
    acc = dwell_mod.accumulate(iv, cells.world, inputs.interval_end_ns, meta)
    rows, _ = pres.build_timeline(fused, cells.world)
    n_interp = sum(1 for r in rows
                   if r["presentation_method"] == pres.METHOD_INTERP)
    assert n_interp > 0
    # recomputing after building the presentation layer changes nothing
    iv2 = dwell_mod.support_intervals(fused, inputs.interval_start_ns,
                                      inputs.interval_end_ns,
                                      policy["max_interpolation_gap_ns"])
    acc2 = dwell_mod.accumulate(iv2, cells.world, inputs.interval_end_ns,
                                meta)
    assert acc2["totals_ns"] == acc["totals_ns"]


def test_cumulative_is_monotonic_and_ends_at_the_totals(tmp_path):
    gaze = half_rate_gaze()
    tracked, inputs, fused, cells, oids = build(tmp_path, gaze)
    policy = pres.gap_policy([int(t) for t in fused["gaze_ts_ns"]])
    meta = {o: {"name": o, "color": "#fff"} for o in oids}
    iv = dwell_mod.support_intervals(fused, inputs.interval_start_ns,
                                     inputs.interval_end_ns,
                                     policy["max_interpolation_gap_ns"])
    acc = dwell_mod.accumulate(iv, cells.world, inputs.interval_end_ns, meta)
    cum = acc["cumulative_ns_by_frame"]
    assert len(cum) == cells.n_frames
    keys = set().union(*[set(c) for c in cum]) if cum else set()
    for k in keys:
        seq = [c.get(k, 0) for c in cum]
        assert all(b >= a for a, b in zip(seq, seq[1:])), "never decreases"
        assert seq[-1] == acc["totals_ns"].get(k, 0), \
            "final frame equals the final total"


def test_top_eight_aggregation_is_deterministic():
    totals = {f"obj_{i:04d}": (20 - i) * 1_000_000 for i in range(1, 12)}
    totals[dwell_mod.NO_HIT_ID] = 5_000_000
    acc = {"totals_ns": totals, "counts": {},
           "first_hit_ns": {}, "last_hit_ns": {},
           "order": sorted(totals, key=lambda k: (-totals[k],
                                                  k == dwell_mod.NO_HIT_ID,
                                                  k)),
           "top": [], "other": [], "cumulative_ns_by_frame": [],
           "valid_supported_ns": sum(totals.values()), "bar_scale_ns": 0}
    ranked = [k for k in acc["order"] if k != dwell_mod.NO_HIT_ID]
    acc["top"], acc["other"] = ranked[:8], ranked[8:]
    meta = {k: {"name": k, "color": "#fff"} for k in totals}
    bars = dwell_mod.panel_rows(acc, meta)
    assert len(bars) == 9 and bars[-1]["name"] == dwell_mod.OTHER_GROUP
    assert [b["key"] for b in bars[:8]] == ranked[:8]
    assert bars[-1]["final_ns"] == sum(totals[o] for o in acc["other"])
    rows = dwell_mod.summary_rows(acc, meta)
    assert [r["display_rank"] for r in rows] == list(range(1, len(rows) + 1))
    assert sum(1 for r in rows if r["included_in_top_eight"]) == 8
    assert any(r["aggregation_group"] == "no_hit" for r in rows)


# ------------------------------------------------- published artifact

def test_published_artifact_carries_both_layers(tmp_path):
    gaze = half_rate_gaze() + [{"ts": WORLD_TS[5] + 2_000, **OUT}]
    gaze.sort(key=lambda r: r["ts"])
    tracked, trim, oids = make_tracked_artifact(tmp_path, gaze, [BIG, SMALL])
    out = tmp_path / "fusion"
    m = publish_fusion_artifact(tracked, out, make_demo=True)
    assert all(v["ok"] for v in m["validation"])

    # authoritative table untouched by rendering mode
    pub = pd.read_parquet(out / "gaze_samples.parquet")
    assert list(pub.columns) == PUBLIC_COLUMNS
    out2 = tmp_path / "fusion_nodemo"
    m2 = publish_fusion_artifact(tracked, out2, make_demo=False)
    pub2 = pd.read_parquet(out2 / "gaze_samples.parquet")
    pd.testing.assert_frame_equal(pub, pub2)
    assert (out / "gaze_samples.csv").read_text() == \
        (out2 / "gaze_samples.csv").read_text()

    # presentation sidecar: one ordered row per frame + disclosure
    text = (out / "demo_presentation.csv").read_text()
    assert text.splitlines()[0].startswith("# ")
    assert "not an analytical gaze table" in text
    prows = pres.read_presentation_csv(out / "demo_presentation.csv")
    assert list(prows[0].keys()) == pres.PRESENTATION_COLUMNS
    assert [int(r["frame_index"]) for r in prows] == \
        list(range(m["video"]["n_frames"]))

    # dwell summary agrees with the manifest
    with open(out / "object_dwell_summary.csv") as f:
        drows = list(csv.DictReader(f))
    assert list(drows[0].keys()) == dwell_mod.SUMMARY_COLUMNS
    total = sum(float(r["primary_gaze_time_s"]) for r in drows)
    assert total == pytest.approx(m["dwell"]["valid_supported_ns"] / 1e9,
                                  abs=1e-6)
    assert m["presentation_layer"]["authoritative"] is False
    assert m["presentation_layer"]["gap_policy"]["median_interval_ms"] > 0
    assert m["dwell"]["ordering"].startswith("fixed")
    assert m["files"]["demo_presentation"] == "demo_presentation.csv"
    assert m["files"]["dwell_summary"] == "object_dwell_summary.csv"
    assert m["demo"]["frames"] == m["video"]["n_frames"]
    assert m["demo"]["max_pts_dev_ms"] <= 2.0
    assert m["demo"]["audio"] is False       # synthetic source has no audio
