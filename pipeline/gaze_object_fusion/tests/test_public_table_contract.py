"""Contract tests for the authoritative public gaze_samples table
(CSV + Parquet) and the corrected demo-highlight semantics."""

import csv
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from gaze_object_fusion.artifact import (load_fusion_artifact,
                                         publish_fusion_artifact)
from gaze_object_fusion.fusion import PUBLIC_COLUMNS

from futil import W, H, WORLD_TS, make_tracked_artifact, trim_selection

BIG = {"name": "big", "box": (20, 20, 100, 100), "score": 0.95}
SMALL = {"name": "small", "box": (40, 40, 60, 60), "score": 0.5}
IN_BOTH = {"x": 50.5, "y": 50.25}
IN_BIG = {"x": 90.5, "y": 30.5}
NOWHERE = {"x": 5.0, "y": 5.0}


def build(tmp_path, *, demo=False):
    """A fixture covering EVERY gaze_status + null/valued ID fields."""
    probe = make_tracked_artifact(tmp_path / "probe",
                                  [{"ts": WORLD_TS[0], **IN_BIG}], [BIG])
    sel = trim_selection(probe[1])
    start, end = sel["absolute_start_ns"], sel["absolute_end_boundary_ns"]

    gaze = [
        {"ts": WORLD_TS[0] + 1_000_000, **IN_BOTH, "fixation": 7},
        {"ts": WORLD_TS[1] + 1_000_000, **IN_BIG, "fixation": 7},
        {"ts": WORLD_TS[2] + 1_000_000, **NOWHERE, "fixation": 8},
        {"ts": WORLD_TS[3] + 1_000_000, **IN_BOTH, "worn": 0.0},
        {"ts": WORLD_TS[4] + 1_000_000, "x": -3.0, "y": 40.0},
        {"ts": WORLD_TS[5] - 12_345_678, **IN_BOTH, "blink": 3},
    ]
    tracked, trim, oids = make_tracked_artifact(
        tmp_path / "main", gaze, [BIG, SMALL],
        post_trim_gaze=[{"ts": end, **IN_BOTH}])
    out = tmp_path / "fusion"
    manifest = publish_fusion_artifact(tracked, out, make_demo=demo)
    return tracked, out, manifest, oids, (start, end)


def _read_csv_text(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------- schema

def test_both_files_exist_with_exact_columns_in_order(tmp_path):
    _, out, manifest, _, _ = build(tmp_path)
    pq = Path(out) / "gaze_samples.parquet"
    cs = Path(out) / "gaze_samples.csv"
    assert pq.is_file() and cs.is_file()
    assert list(pd.read_parquet(pq).columns) == PUBLIC_COLUMNS
    with open(cs) as f:
        assert f.readline().strip().split(",") == PUBLIC_COLUMNS
    assert manifest["table_schema"]["columns"] == PUBLIC_COLUMNS


def test_csv_parquet_logical_equality(tmp_path):
    """Field-by-field: CSV text decodes to exactly the parquet values."""
    _, out, _, _, _ = build(tmp_path)
    pq = pd.read_parquet(Path(out) / "gaze_samples.parquet")
    rows = _read_csv_text(Path(out) / "gaze_samples.csv")
    assert len(rows) == len(pq)
    INT = {"gaze_sample_index", "timestamp_ns", "frame_index",
           "frame_timestamp_ns", "gaze_pixel_x", "gaze_pixel_y",
           "hit_object_count", "primary_mask_area", "fixation_id",
           "blink_id"}
    FLT = {"relative_time_s", "frame_offset_ms", "gaze_x_px", "gaze_y_px",
           "worn", "primary_track_score"}
    BOOL = {"gaze_valid", "is_gaze_at_object"}
    LST = {"hit_object_ids", "hit_object_names"}
    for i, r in enumerate(rows):
        for col in PUBLIC_COLUMNS:
            pv = pq[col].iloc[i]
            cv = r[col]
            if col in LST:
                assert json.loads(cv) == list(pv), (i, col)
            elif cv == "":
                assert pv is None or (not isinstance(pv, (list,))
                                      and pd.isna(pv)), (i, col, pv)
            elif col in INT:
                assert int(cv) == int(pv), (i, col)
            elif col in FLT:
                assert float(cv) == float(pv), (i, col)
            elif col in BOOL:
                assert (cv == "True") == bool(pv), (i, col)
            else:                       # strings (status, ids, names)
                assert cv == str(pv), (i, col)


def test_ordering_index_and_exact_int_timestamps(tmp_path):
    tracked, out, _, _, (start, end) = build(tmp_path)
    pq = pd.read_parquet(Path(out) / "gaze_samples.parquet")
    # stable zero-based index in original gaze-row order
    assert list(pq["gaze_sample_index"]) == list(range(len(pq)))
    src = _read_csv_text(Path(tracked).parent.parent / "main" / "trim"
                         / "gaze.csv")
    assert [int(r["timestamp [ns]"]) for r in src] \
        == [int(t) for t in pq["timestamp_ns"]], \
        "row order and exact int timestamps preserved from the source CSV"
    # CSV carries the very same integer TEXT (no float formatting)
    rows = _read_csv_text(Path(out) / "gaze_samples.csv")
    for sr, rr in zip(src, rows):
        assert rr["timestamp_ns"] == sr["timestamp [ns]"]
    # frame_timestamp_ns is the authoritative world timestamp
    for _, r in pq[pq["frame_index"].notna()].iterrows():
        assert int(r["frame_timestamp_ns"]) == WORLD_TS[int(r["frame_index"])]
    # relative_time_s measured from the trimmed-video start
    r0 = pq.iloc[0]
    assert float(r0["relative_time_s"]) == pytest.approx(
        (int(r0["timestamp_ns"]) - start) / 1e9)


def test_status_boolean_hit_consistency_all_statuses_present(tmp_path):
    _, out, _, oids, _ = build(tmp_path)
    big_id, small_id = oids
    pq = pd.read_parquet(Path(out) / "gaze_samples.parquet")
    assert list(pq["gaze_status"]) == [
        "on_object", "on_object", "background", "invalid_gaze",
        "outside_frame", "on_object", "no_matching_frame"]
    for _, r in pq.iterrows():
        assert bool(r["is_gaze_at_object"]) \
            == (r["gaze_status"] == "on_object")
        assert int(r["hit_object_count"]) == len(list(r["hit_object_ids"]))
        if r["hit_object_count"] > 0:
            assert r["primary_object_id"] == list(r["hit_object_ids"])[0]
        else:
            assert pd.isna(r["primary_object_id"])
        if r["gaze_status"] == "invalid_gaze":
            assert not bool(r["gaze_valid"])
    # deterministic overlapping rule retained: smaller area first
    assert list(pq.iloc[0]["hit_object_ids"]) == [small_id, big_id]
    # exact integer pixels used for the mask lookup
    assert int(pq.iloc[0]["gaze_pixel_x"]) == math.floor(IN_BOTH["x"])
    assert int(pq.iloc[0]["gaze_pixel_y"]) == math.floor(IN_BOTH["y"])
    # nullable ids stay null (never a misleading numeric)
    assert pd.isna(pq.iloc[0]["blink_id"])
    assert int(pq.iloc[0]["fixation_id"]) == 7
    assert int(pq.iloc[5]["blink_id"]) == 3
    assert pq["fixation_id"].dtype.name == "Int64"
    assert pq["blink_id"].dtype.name == "Int64"


def test_loader_validates_public_contract(tmp_path):
    _, out, _, _, _ = build(tmp_path)
    art = load_fusion_artifact(out)
    assert Path(art.gaze_samples_csv_path).name == "gaze_samples.csv"
    # column-order tampering is refused
    pq_path = Path(out) / "gaze_samples.parquet"
    df = pd.read_parquet(pq_path)
    df[df.columns[::-1]].to_parquet(pq_path, index=False)
    with pytest.raises(Exception):
        load_fusion_artifact(out)


def test_manifest_provenance_complete(tmp_path):
    """fusion_manifest.json must include/resolve the full provenance set
    (recorded ONCE in the manifest — deliberately not per CSV row)."""
    _, out, manifest, oids, _ = build(tmp_path)
    m = manifest
    assert m["artifact_id"]
    pa = m["parent_artifact"]
    assert pa["artifact_id"] and pa["tracking_manifest_sha256"] \
        and pa["objects_table_sha256"], "tracked-artifact id + hashes"
    pr = m["parent_recording"]
    assert pr["video"] and pr["hashes_sha256"]["video"], "video identity"
    assert pr["gaze"] and pr["hashes_sha256"]["gaze"], "gaze identity"
    assert pr["world_timestamps"] \
        and pr["hashes_sha256"]["world_timestamps"], "world identity"
    cat = {o["object_id"]: o for o in m["objects"]}
    assert set(oids) <= set(cat)
    for o in cat.values():
        assert o["name"] and o["color"].startswith("#") \
            and o["review_status"]
    rule = m["assignment_rule"]
    assert rule["method"] == "timestamp_defined_frame_cells"
    assert "floor" in rule["pixel_convention"], "gaze-pixel rounding rule"
    assert "mask area" in rule["hit_order"], "overlap/primary rule"
    bg = rule["background_definition"]
    assert "NONE of the accepted tracked-object masks" in bg
    assert "does NOT prove" in bg and "untracked or unlabeled" in bg
    # the public table must NOT duplicate provenance per row
    import pandas as pd
    cols = set(pd.read_parquet(Path(out) / "gaze_samples.parquet").columns)
    assert not ({"video_sha256", "artifact_id", "recording_id"} & cols)


def test_parent_tracked_artifact_byte_identical_after_fusion(tmp_path):
    probe = make_tracked_artifact(tmp_path / "p",
                                  [{"ts": WORLD_TS[0] + 1, **IN_BIG}], [BIG])
    tracked = probe[0]

    def digest():
        h = hashlib.sha256()
        for f in sorted(Path(tracked).rglob("*")):
            if f.is_file():
                h.update(f.name.encode())
                h.update(f.read_bytes())
        return h.hexdigest()

    before = digest()
    publish_fusion_artifact(tracked, tmp_path / "fusion", make_demo=True)
    assert digest() == before, "fusion must never touch the parent artifact"


# ------------------------------------------------------- demo highlight

def _mix(bg, rgb, alpha):
    return [(1 - alpha) * bg + alpha * c for c in rgb]


def _dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def test_demo_highlight_agrees_with_primary_object_id(tmp_path):
    """Pixel-level: the brightly filled object on each frame IS that
    frame's primary_object_id; others stay subtle."""
    import av

    # disjoint boxes so interiors are attributable; gaze sits near each
    # box's TOP-LEFT corner so the fixed cursor (ring+halo ~13 px) cannot
    # contaminate the bottom-right interior region sampled below
    A = {"name": "obj-a", "box": (20, 20, 60, 60), "score": 0.9}
    B = {"name": "obj-b", "box": (90, 20, 130, 60), "score": 0.9}
    gaze = [
        {"ts": WORLD_TS[2] + 1_000_000, "x": 27.5, "y": 27.5},   # on A
        {"ts": WORLD_TS[3] + 1_000_000, "x": 97.5, "y": 27.5},   # on B
        {"ts": WORLD_TS[4] + 1_000_000, "x": 5.0, "y": 100.0},   # background
    ]
    tracked, _, oids = make_tracked_artifact(tmp_path, gaze, [A, B])
    out = tmp_path / "fusion"
    publish_fusion_artifact(tracked, out, make_demo=True)

    pq = pd.read_parquet(Path(out) / "gaze_samples.parquet")
    ann = json.loads((Path(tracked) / "object_annotations.json").read_text())
    colors = {o["object_id"]:
              tuple(int(o["color"].lstrip("#")[i:i + 2], 16)
                    for i in (0, 2, 4))
              for o in ann["objects"]}
    with av.open(str(Path(out) / "gaze_demo.mp4")) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]

    def interior_mean(img, box):
        # bottom-right interior quadrant: far from the top-left gaze
        # cursor, still well inside the mask
        x0, y0, x1, y1 = box
        return img[y1 - 14:y1 - 4, x1 - 14:x1 - 4].reshape(-1, 3) \
            .mean(axis=0)

    demo = pq[pq["frame_index"].notna()]
    for _, row in demo.iterrows():
        fi = int(row["frame_index"])
        if fi not in (2, 3, 4):
            continue
        bg = (fi * 23) % 255
        img = frames[fi]
        primary = (row["primary_object_id"]
                   if not pd.isna(row["primary_object_id"]) else None)
        for oid, box in zip(oids, (A["box"], B["box"])):
            got = interior_mean(img, box)
            strong = _mix(bg, colors[oid], 0.50)
            subtle = _mix(bg, colors[oid], 0.12)
            if oid == primary:
                assert _dist(got, strong) < _dist(got, subtle), \
                    f"frame {fi}: primary {oid} not brightly highlighted"
            else:
                assert _dist(got, subtle) < _dist(got, strong), \
                    f"frame {fi}: non-primary {oid} not subtle"
    # background frame: no bright highlight at all (both subtle)
    assert pd.isna(demo[demo["frame_index"] == 4]
                   .iloc[0]["primary_object_id"])
