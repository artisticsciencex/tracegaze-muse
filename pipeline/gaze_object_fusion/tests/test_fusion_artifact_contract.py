"""gaze_object_fusion artifact contract: publish, validate, load, refuse."""

import json
from pathlib import Path

import pandas as pd
import pytest

from common.artifacts import ArtifactError
from gaze_object_fusion.artifact import (load_fusion_artifact,
                                         publish_fusion_artifact)
from gaze_object_fusion.cli import default_out_dir, main as cli_main

from futil import WORLD_TS, make_tracked_artifact

BIG = {"name": "big", "box": (20, 20, 100, 100), "score": 0.95}
GAZE = sorted(
    [{"ts": WORLD_TS[i] + 1_000_000, "x": 50.5, "y": 50.5}
     for i in range(len(WORLD_TS))]
    + [{"ts": WORLD_TS[4] + 3_000_000, "x": 5.0, "y": 5.0}],
    key=lambda r: r["ts"])


def publish(tmp_path, **kw):
    tracked, trim, oids = make_tracked_artifact(tmp_path, GAZE, [BIG])
    out = tmp_path / "fusion"
    manifest = publish_fusion_artifact(tracked, out, **kw)
    return tracked, out, manifest


def test_publish_load_roundtrip_with_demo(tmp_path):
    tracked, out, manifest = publish(tmp_path, make_demo=True)
    assert manifest["artifact_type"] == "gaze_object_fusion"
    assert all(v["ok"] for v in manifest["validation"])
    assert manifest["demo"]["present"] is True
    assert manifest["demo"]["frames"] == len(WORLD_TS)
    assert manifest["demo"]["max_pts_dev_ms"] <= 2.0
    assert manifest["counts"]["samples_total"] == len(GAZE)
    assert manifest["counts"]["samples_assigned"] == len(GAZE)
    assert manifest["counts"]["frames_with_cursor"] == len(WORLD_TS)
    assert (manifest["assignment_rule"]["method"]
            == "timestamp_defined_frame_cells")

    art = load_fusion_artifact(out)
    assert art.artifact_id == manifest["artifact_id"]
    fused = pd.read_parquet(art.gaze_samples_parquet_path)
    assert len(fused) == len(GAZE)
    assert Path(art.gaze_samples_csv_path).exists()
    assert art.demo_path is not None and Path(art.demo_path).exists()
    # parent linkage recorded with hashes
    pa = manifest["parent_artifact"]
    assert pa["artifact_id"] and pa["tracking_manifest_sha256"] \
        and pa["objects_table_sha256"] and pa["grandparent_artifact_id"]


def test_refuses_existing_destination(tmp_path):
    tracked, out, _ = publish(tmp_path, make_demo=False)
    with pytest.raises(Exception):
        publish_fusion_artifact(tracked, out, make_demo=False)


def test_detects_parent_changed_since_tracking(tmp_path):
    tracked, trim, _ = make_tracked_artifact(tmp_path, GAZE, [BIG])
    with open(Path(trim) / "gaze.csv", "a") as f:
        f.write(f"sec-1,rec-1,{WORLD_TS[0] + 500},1,1,1.0\n")
    with pytest.raises(ArtifactError, match="changed since"):
        publish_fusion_artifact(tracked, tmp_path / "fusion",
                                make_demo=False)


def test_refuses_non_trimmed_parent(tmp_path):
    """Tracked artifacts whose parent is a bare CSV export carry no
    validated trim interval — fusion v1 must refuse, clearly."""
    from object_tracking import intake, trackstore
    from object_tracking.annotations import AnnotationStore
    from object_tracking.artifact import publish_tracked_artifact
    from futil import encode_vfr_video, write_csvs, rle_for_box, N

    src = tmp_path / "srcexport"
    src.mkdir()
    encode_vfr_video(src / "scene.mp4")
    write_csvs(src, GAZE)
    bundle = intake.open_bundle(src)
    assert bundle.source_type != "trimmed_artifact"
    store = AnnotationStore(tmp_path / "ann.json", bundle.n_frames,
                            bundle.width, bundle.height)
    o = store.create_object("big", 5)
    store.set_points(o["object_id"], [[60, 60]], [1])
    store.record_preview(o["object_id"],
                         {"found": True, "score": 0.9, "mask_area": 6400})
    store.accept_mask(o["object_id"])
    rows = [{"frame_index": i,
             "tracking_direction": "forward" if i >= 5 else "backward",
             "sam_local_id": 1, "score": 0.9, "bbox_x": 20.0, "bbox_y": 20.0,
             "bbox_w": 80.0, "bbox_h": 80.0, "mask_area": 6400,
             "mask_rle": rle_for_box(20, 20, 100, 100)} for i in range(N)]
    trackstore.replace_object_rows(tmp_path / "t.db", o["object_id"], rows)
    store.record_track_result(o["object_id"], {
        "rows": N, "frames_present": N, "first_frame": 0,
        "last_frame": N - 1, "forward_ok": True, "backward_ok": True,
        "whole_video": True})
    store.set_review_status(o["object_id"], "accepted")
    tracked = tmp_path / "tracked_csvparent"
    publish_tracked_artifact(bundle, store, tmp_path / "t.db", tracked,
                             make_overlay=False)
    with pytest.raises(ArtifactError, match="trimmed-recording parent"):
        publish_fusion_artifact(tracked, tmp_path / "fusion",
                                make_demo=False)


def test_loader_rejects_tampered_counts_and_validation(tmp_path):
    tracked, out, manifest = publish(tmp_path, make_demo=False)
    mf = Path(out) / "fusion_manifest.json"

    m = json.loads(mf.read_text())
    m["counts"]["samples_total"] += 1
    mf.write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="samples_total"):
        load_fusion_artifact(out)

    m = json.loads(mf.read_text())
    m["counts"]["samples_total"] -= 1
    m["validation"][0]["ok"] = False
    mf.write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="failures"):
        load_fusion_artifact(out)


def test_loader_rejects_missing_declared_demo(tmp_path):
    tracked, out, _ = publish(tmp_path, make_demo=True)
    (Path(out) / "gaze_demo.mp4").unlink()
    with pytest.raises(ArtifactError):
        load_fusion_artifact(out)


def test_artifact_portable_after_move(tmp_path):
    tracked, out, _ = publish(tmp_path, make_demo=False)
    import shutil
    moved = tmp_path / "elsewhere" / "fusion_moved"
    moved.parent.mkdir()
    shutil.move(str(out), str(moved))
    art = load_fusion_artifact(moved)
    assert len(pd.read_parquet(art.gaze_samples_parquet_path)) == len(GAZE)


def test_cli_default_naming_and_run(tmp_path):
    assert default_out_dir(Path("/a/b/rec_tracked_objects")).name \
        == "rec_gaze_fusion"
    assert default_out_dir(Path("/a/b/other")).name == "other_gaze_fusion"
    tracked, trim, _ = make_tracked_artifact(tmp_path, GAZE, [BIG])
    out = tmp_path / "cliout"
    rc = cli_main(["--artifact", str(tracked), "--out", str(out),
                   "--no-demo"])
    assert rc == 0
    assert load_fusion_artifact(out).manifest["demo"]["present"] is False
