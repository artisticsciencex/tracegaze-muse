"""Hardening tests from the adversarial review: float-pathway refusal,
missing-hash gate, full-path gap reporting, worn-absent input, pixel-edge
coordinates, demo gates, loader integrity checks."""

import json
from pathlib import Path

import pandas as pd
import pytest

from common.artifacts import ArtifactError
from gaze_object_fusion.artifact import (load_fusion_artifact,
                                         publish_fusion_artifact)
from gaze_object_fusion.fusion import (FusionInputs, fuse, load_gaze,
                                       load_world_ts)

from futil import W, H, WORLD_TS, make_tracked_artifact

BIG = {"name": "big", "box": (20, 20, 100, 100), "score": 0.95}
IN = {"x": 50.5, "y": 50.5}


# ------------------------------------------------ float pathway refusal

def _write_csv(path, header, rows):
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def test_float_formatted_gaze_timestamps_refused(tmp_path):
    """A float-formatted 'timestamp [ns]' would silently truncate through
    pandas float64 (±hundreds of ns on >2^53 values) — must be refused."""
    p = tmp_path / "gaze.csv"
    hdr = "section id,recording id,timestamp [ns],gaze x [px],gaze y [px]"
    _write_csv(p, hdr, [f"s,r,{WORLD_TS[0]},1.0,1.0",
                        f"s,r,{WORLD_TS[1]}.0,1.0,1.0"])   # trailing .0
    with pytest.raises(ArtifactError, match="non-integer text"):
        load_gaze(p)
    _write_csv(p, hdr, ["s,r,1.7332008399e+18,1.0,1.0"])   # scientific
    with pytest.raises(ArtifactError, match="non-integer text"):
        load_gaze(p)


def test_float_formatted_world_timestamps_refused(tmp_path):
    p = tmp_path / "world_timestamps.csv"
    hdr = "section id,recording id,timestamp [ns]"
    _write_csv(p, hdr, [f"s,r,{WORLD_TS[0]}", f"s,r,{WORLD_TS[1]}.0"])
    with pytest.raises(ArtifactError, match="non-integer text"):
        load_world_ts(p)


def test_integer_text_parses_exactly(tmp_path):
    p = tmp_path / "world_timestamps.csv"
    odd = WORLD_TS[0] + 1                       # ±1 ns must survive
    _write_csv(p, "section id,recording id,timestamp [ns]",
               [f"s,r,{odd}", f"s,r,{WORLD_TS[1]}"])
    assert load_world_ts(p) == [odd, WORLD_TS[1]]


# ------------------------------------------------ missing-hash gate

def test_missing_parent_hash_refused(tmp_path):
    tracked, trim, _ = make_tracked_artifact(
        tmp_path, [{"ts": WORLD_TS[0] + 1_000_000, **IN}], [BIG])
    mf = Path(tracked) / "tracking_manifest.json"
    m = json.loads(mf.read_text())
    m["parent"]["gaze_sha256"] = None
    mf.write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="records no parent hash"):
        publish_fusion_artifact(tracked, tmp_path / "fusion",
                                make_demo=False)


# ------------------------------------- anomalous gaps through full fuse()

def test_anomalous_gaps_reported_via_full_fuse_and_manifest(tmp_path):
    """sanity threshold 45 ms flags exactly the two 50 ms VFR gaps; samples
    INSIDE those gaps stay assigned (reporting never drops samples)."""
    deep_in_gap = WORLD_TS[0] + 24_000_000     # 24 ms into the 50 ms gap
    gaze = [{"ts": deep_in_gap, **IN},
            {"ts": WORLD_TS[5] + 1_000_000, **IN}]
    tracked, _, _ = make_tracked_artifact(tmp_path, gaze, [BIG])

    fused, cells, validation, extras = fuse(FusionInputs(tracked),
                                            sanity_gap_ms=45.0)
    assert [g["between_frames"] for g in extras["gaps"]] == [[0, 1], [1, 2]]
    assert all(v["ok"] for v in validation)
    assert fused.iloc[0]["assignment"] == "assigned"
    assert fused.iloc[0]["frame_index"] == 0        # 24 ms < 25 ms midpoint
    assert fused.iloc[0]["frame_offset_ns"] == 24_000_000

    manifest = publish_fusion_artifact(tracked, tmp_path / "fusion",
                                       make_demo=False, sanity_gap_ms=45.0)
    gaps = manifest["assignment_rule"]["anomalous_gaps"]
    assert len(gaps) == 2 and gaps[0]["gap_ms"] == pytest.approx(50.0)
    assert manifest["counts"]["samples_assigned"] == 2


# ------------------------------------------------ worn column absent

def test_worn_column_absent_entirely(tmp_path):
    gaze = [{"ts": WORLD_TS[3] + 1_000_000, **IN}]
    tracked, _, _ = make_tracked_artifact(tmp_path, gaze, [BIG],
                                          with_worn=False)
    fused, _, validation, extras = fuse(FusionInputs(tracked))
    assert all(v["ok"] for v in validation)
    assert "worn" not in fused.columns
    r = fused.iloc[0]
    assert r["gaze_valid"] and r["n_hits"] == 1 and r["demo_selected"]


# ------------------------------------------------ pixel-edge coordinates

def test_coords_at_exact_frame_edges(tmp_path):
    gaze = [
        {"ts": WORLD_TS[2] + 1_000_000, "x": 0.0, "y": 0.0},          # in
        {"ts": WORLD_TS[2] + 2_000_000, "x": float(W), "y": 10.0},    # out
        {"ts": WORLD_TS[2] + 3_000_000, "x": 10.0, "y": float(H)},    # out
        {"ts": WORLD_TS[2] + 4_000_000, "x": W - 0.5, "y": H - 0.5},  # in
    ]
    fused, _, _, _ = fuse(FusionInputs(
        make_tracked_artifact(tmp_path, gaze, [BIG])[0]))
    assert list(fused["coords_in_frame"]) == [True, False, False, True]
    assert (fused["assignment"] == "assigned").all(), \
        "frame assignment is timestamp-only, unaffected by coordinates"


# ------------------------------------------------ demo gates

def test_demo_frame_count_gate(tmp_path):
    from gaze_object_fusion.demo_video import render_demo
    tracked, trim, _ = make_tracked_artifact(
        tmp_path, [{"ts": WORLD_TS[0] + 1_000_000, **IN}], [BIG])
    video = Path(trim) / "scene.mp4"
    if not video.exists():
        video = next(Path(trim).glob("*.mp4"))
    with pytest.raises(ArtifactError, match="frames"):
        render_demo(video, tmp_path / "demo.mp4",
                    n_frames=len(WORLD_TS) + 1, objects_by_frame={},
                    cursor_by_frame={}, has_audio=False)
    assert not (tmp_path / "demo.mp4").exists(), "no partial output"


def test_demo_cursor_pixels_at_selected_position(tmp_path):
    """Decode the published demo and verify the FIXED cursor was drawn at
    the selected row's coordinates (red center dot, green ring)."""
    import av
    from gaze_object_fusion.demo_video import cursor_radius
    cx, cy = 130.5, 30.5                       # OUTSIDE the mask box
    gaze = [{"ts": WORLD_TS[1] + 1_000_000, "x": cx, "y": cy}]
    tracked, _, _ = make_tracked_artifact(tmp_path, gaze, [BIG])
    publish_fusion_artifact(tracked, tmp_path / "fusion", make_demo=True)
    with av.open(str(tmp_path / "fusion" / "gaze_demo.mp4")) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    f1 = frames[1]                             # background gray 23
    R = cursor_radius(W, H)
    center = f1[int(cy), int(cx)]
    assert center[0] > 140 and center[1] < 110, \
        f"red center dot missing: {center}"
    ring = f1[int(cy), int(cx) + R]
    assert ring[1] > 140 and ring[0] < 120, f"green ring missing: {ring}"
    assert f1[110, 20].mean() < 60, "background must stay untouched"
    f0 = frames[0]                             # frame 0 has no gaze row
    assert f0[int(cy), int(cx)].mean() < 60, \
        "no cursor may appear on a frame without a valid assigned row"


def test_fixed_cursor_identical_on_object_and_background(tmp_path):
    """The cursor is ONE fixed design: same colors whether the gaze is on
    a tracked object or on background; halo is dark; no cursor on frames
    whose only samples are invalid."""
    import av
    import numpy as np
    from gaze_object_fusion.demo_video import cursor_radius
    on_obj = (50.5, 50.5)                      # inside BIG's mask
    on_bg = (130.5, 30.5)                      # outside every mask
    gaze = [
        {"ts": WORLD_TS[1] + 1_000_000, "x": on_bg[0], "y": on_bg[1]},
        {"ts": WORLD_TS[2] + 1_000_000, "x": on_obj[0], "y": on_obj[1]},
        {"ts": WORLD_TS[3] + 1_000_000, "x": on_obj[0], "y": on_obj[1],
         "worn": 0.0},                          # invalid -> no cursor
    ]
    tracked, _, _ = make_tracked_artifact(tmp_path, gaze, [BIG])
    publish_fusion_artifact(tracked, tmp_path / "fusion", make_demo=True)
    with av.open(str(tmp_path / "fusion" / "gaze_demo.mp4")) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    R = cursor_radius(W, H)

    def probe(img, cx, cy):
        return {"center": img[int(cy), int(cx)].astype(int),
                "ring": img[int(cy), int(cx) + R].astype(int),
                "halo": img[int(cy) - R - 4, int(cx)].astype(int)}

    bg = probe(frames[1], *on_bg)
    ob = probe(frames[2], *on_obj)
    for p, whr in ((bg, "background"), (ob, "object")):
        assert p["center"][0] > 140 and p["center"][1] < 110, \
            f"{whr}: red dot missing {p['center']}"
        assert p["ring"][1] > 140 and p["ring"][0] < 120, \
            f"{whr}: green ring missing {p['ring']}"
        assert p["halo"].sum() < 150, f"{whr}: black halo missing {p['halo']}"
    assert int(np.abs(bg["ring"] - ob["ring"]).max()) < 60, \
        "cursor must be identical on object and background"
    # frame 3's only sample is invalid: fixed design pixels must be absent
    f3 = frames[3]
    c3 = f3[int(on_obj[1]), int(on_obj[0])].astype(int)
    assert not (c3[0] > 140 and c3[1] < 110), \
        "no cursor may be drawn for invalid gaze"


# ------------------------------------------------ loader integrity

def _publish(tmp_path):
    gaze = [{"ts": WORLD_TS[i] + 1_000_000, **IN}
            for i in range(len(WORLD_TS))]
    tracked, _, _ = make_tracked_artifact(tmp_path, gaze, [BIG])
    out = tmp_path / "fusion"
    publish_fusion_artifact(tracked, out, make_demo=False)
    return out


def test_loader_rejects_tampered_assigned_count(tmp_path):
    out = _publish(tmp_path)
    mf = out / "fusion_manifest.json"
    m = json.loads(mf.read_text())
    m["counts"]["samples_assigned"] -= 1
    m["counts"]["samples_no_matching_frame"] += 1
    mf.write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="samples_assigned"):
        load_fusion_artifact(out)


def test_loader_rejects_tampered_table_via_hash(tmp_path):
    out = _publish(tmp_path)
    t = pd.read_parquet(out / "gaze_objects.parquet")
    t.loc[0, "primary_object_id"] = "obj-FAKE"
    t.to_parquet(out / "gaze_objects.parquet", index=False)
    with pytest.raises(ArtifactError, match="sha256"):
        load_fusion_artifact(out)


def test_loader_requires_recorded_file_hashes(tmp_path):
    out = _publish(tmp_path)
    mf = out / "fusion_manifest.json"
    m = json.loads(mf.read_text())
    del m["files_sha256"]
    mf.write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="files_sha256"):
        load_fusion_artifact(out)
