"""tracked_objects artifact tests: publish, validate, load, portability."""

import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from common.artifacts import ArtifactError
from object_tracking import intake
from object_tracking.annotations import AnnotationStore
from object_tracking.artifact import (load_tracked_artifact,
                                      publish_tracked_artifact)

from util import (WORLD_TS, make_trimmed_artifact, rle_for_box,
                  write_track_rows)


def build_session(tmp_path, *, ref=5, absent_head=0):
    """bundle + store + tracks dir with one accepted object."""
    src = make_trimmed_artifact(tmp_path)
    bundle = intake.open_bundle(src)
    store = AnnotationStore(tmp_path / "ann.json", bundle.n_frames,
                            bundle.width, bundle.height)
    o = store.create_object("pipette", ref)
    store.set_points(o["object_id"], [[50, 50], [90, 90]], [1, 0])
    store.record_preview(o["object_id"], {"found": True, "score": 0.9,
                                          "mask_area": 1600})
    store.accept_mask(o["object_id"])
    tracks = tmp_path / "tracks.db"
    n = bundle.n_frames
    frames = ([(f, "forward") for f in range(ref, n)]
              + [(f, "backward") for f in range(absent_head, ref)])
    write_track_rows(tracks, o["object_id"], frames)
    store.record_track_result(o["object_id"], {
        "rows": len(frames), "frames_present": len(frames),
        "first_frame": absent_head, "last_frame": n - 1,
        "forward_ok": True, "backward_ok": True, "whole_video": True})
    store.set_review_status(o["object_id"], "accepted")
    return bundle, store, tracks, o["object_id"]


def test_publish_and_reload_roundtrip(tmp_path):
    bundle, store, tracks, oid = build_session(tmp_path)
    out = tmp_path / "out" / "tracked"
    manifest = publish_tracked_artifact(bundle, store, tracks, out,
                                        make_overlay=True)
    assert manifest["artifact_type"] == "tracked_objects"
    assert manifest["parent_artifact"]["artifact_id"] == bundle.artifact_id
    assert manifest["overlay"]["present"] is True
    assert manifest["overlay"]["max_pts_dev_ms"] <= 2.0
    assert all(v["ok"] for v in manifest["validation"])
    # loader
    art = load_tracked_artifact(out)
    assert art.artifact_id == manifest["artifact_id"]
    table = pd.read_parquet(art.objects_table_path)
    assert sorted(table["frame_index"].unique()) == list(range(bundle.n_frames))
    assert set(table["object_id"]) == {oid}
    # timestamp mapping preserved
    for _, r in table.iterrows():
        assert int(r["timestamp_ns"]) == WORLD_TS[int(r["frame_index"])]
    # parent paths resolvable for gaze fusion
    pp = art.parent_paths()
    assert pp["video"].is_file() and pp["gaze"].is_file() \
        and pp["world_timestamps"].is_file()


def test_artifact_portable_after_move(tmp_path):
    bundle, store, tracks, oid = build_session(tmp_path)
    out = tmp_path / "t1"
    publish_tracked_artifact(bundle, store, tracks, out, make_overlay=False)
    moved = tmp_path / "elsewhere" / "t1_moved"
    moved.parent.mkdir()
    shutil.move(str(out), moved)
    art = load_tracked_artifact(moved)
    assert art.objects_table_path == moved / "objects.parquet"


def test_object_absent_from_part_of_clip(tmp_path):
    bundle, store, tracks, oid = build_session(tmp_path, ref=5,
                                               absent_head=3)
    out = tmp_path / "absent"
    manifest = publish_tracked_artifact(bundle, store, tracks, out,
                                        make_overlay=False)
    obj = [o for o in manifest["objects"] if o["object_id"] == oid][0]
    assert obj["presence"]["first_frame"] == 3     # absent 0..2 is fine
    assert all(v["ok"] for v in manifest["validation"])


def test_publish_gate_blocks_undecided_objects(tmp_path):
    bundle, store, tracks, oid = build_session(tmp_path)
    store.create_object("undecided", 0)
    with pytest.raises(ArtifactError, match="not publishable"):
        publish_tracked_artifact(bundle, store, tracks, tmp_path / "x",
                                 make_overlay=False)


def test_duplicate_rows_rejected_atomically(tmp_path):
    """The trackstore PK forbids duplicate (object, frame) rows; a write
    containing duplicates rolls back WITHOUT touching the previous rows."""
    import sqlite3
    from object_tracking import trackstore
    from util import track_rows
    bundle, store, tracks, oid = build_session(tmp_path)
    before = trackstore.object_stats(tracks, oid)["rows"]
    bad = track_rows([(5, "forward"), (5, "backward")])   # duplicate frame
    with pytest.raises(sqlite3.IntegrityError):
        trackstore.replace_object_rows(tracks, oid, bad)
    after = trackstore.object_stats(tracks, oid)
    assert after["rows"] == before                        # old rows intact


def test_out_of_bounds_frames_rejected(tmp_path):
    bundle, store, tracks, oid = build_session(tmp_path)
    import sqlite3
    conn = sqlite3.connect(tracks)
    conn.execute("UPDATE track_rows SET frame_index = 999 WHERE object_id = ?"
                 " AND frame_index = (SELECT MAX(frame_index) FROM track_rows"
                 " WHERE object_id = ?)", (oid, oid))
    conn.commit(); conn.close()
    with pytest.raises(ArtifactError, match="validation failed"):
        publish_tracked_artifact(bundle, store, tracks, tmp_path / "b",
                                 make_overlay=False)


def test_invalid_rle_rejected(tmp_path):
    bundle, store, tracks, oid = build_session(tmp_path)
    import sqlite3
    conn = sqlite3.connect(tracks)
    conn.execute("UPDATE track_rows SET mask_rle = ? WHERE object_id = ? AND"
                 " frame_index = (SELECT MIN(frame_index) FROM track_rows"
                 " WHERE object_id = ?)",
                 ('{"size": [120, 160], "counts": "!!!corrupt"}', oid, oid))
    conn.commit(); conn.close()
    with pytest.raises(ArtifactError, match="validation failed"):
        publish_tracked_artifact(bundle, store, tracks, tmp_path / "r",
                                 make_overlay=False)


def test_existing_destination_conflict(tmp_path):
    bundle, store, tracks, oid = build_session(tmp_path)
    out = tmp_path / "conflict"
    out.mkdir()
    with pytest.raises(ArtifactError, match="already exists"):
        publish_tracked_artifact(bundle, store, tracks, out,
                                 make_overlay=False)


def test_failed_publish_leaves_no_partial_output(tmp_path, monkeypatch):
    bundle, store, tracks, oid = build_session(tmp_path)
    import object_tracking.artifact as am

    def boom(*a, **k):
        raise RuntimeError("synthetic overlay failure")
    monkeypatch.setattr(am, "render_overlay", boom)
    out = tmp_path / "fails"
    with pytest.raises(RuntimeError):
        publish_tracked_artifact(bundle, store, tracks, out,
                                 make_overlay=True)
    assert not out.exists()
    leftovers = [p for p in out.parent.iterdir()
                 if p.name.startswith(".tracked_objects_tmp")]
    assert leftovers == []


def test_loader_rejects_symlink_escape(tmp_path):
    bundle, store, tracks, oid = build_session(tmp_path)
    out = tmp_path / "sym"
    publish_tracked_artifact(bundle, store, tracks, out, make_overlay=False)
    victim = tmp_path / "victim.parquet"
    victim.write_bytes(b"x")
    (out / "objects.parquet").unlink()
    (out / "objects.parquet").symlink_to(victim)
    with pytest.raises(ArtifactError, match="symlink escape"):
        load_tracked_artifact(out)


def test_loader_rejects_tampered_ids_and_colors(tmp_path):
    bundle, store, tracks, oid = build_session(tmp_path)
    out = tmp_path / "tamper"
    publish_tracked_artifact(bundle, store, tracks, out, make_overlay=False)
    ann = json.loads((out / "object_annotations.json").read_text())
    ann["objects"][0]["color"] = "#123456"        # break color determinism
    (out / "object_annotations.json").write_text(json.dumps(ann))
    with pytest.raises(ArtifactError, match="deterministic"):
        load_tracked_artifact(out)


def test_loader_rejects_unaccepted_table_ids(tmp_path):
    bundle, store, tracks, oid = build_session(tmp_path)
    out = tmp_path / "ids"
    publish_tracked_artifact(bundle, store, tracks, out, make_overlay=False)
    df = pd.read_parquet(out / "objects.parquet")
    df.loc[df.index[0], "object_id"] = "obj_9999"
    df.to_parquet(out / "objects.parquet", index=False)
    with pytest.raises(ArtifactError, match="not\naccepted|not accepted"):
        load_tracked_artifact(out)


def test_video_only_bundle_publishes_without_gaze_claim(tmp_path):
    from util import encode_vfr_video
    d = tmp_path / "vidonly"
    d.mkdir()
    encode_vfr_video(d / "scene.mp4")
    bundle = intake.open_bundle(d)
    store = AnnotationStore(tmp_path / "a.json", bundle.n_frames,
                            bundle.width, bundle.height)
    o = store.create_object("thing", 2)
    store.set_points(o["object_id"], [[40, 40]], [1])
    store.record_preview(o["object_id"], {"found": True})
    store.accept_mask(o["object_id"])
    tracks = tmp_path / "tr.db"
    write_track_rows(tracks, o["object_id"],
                     [(f, "forward") for f in range(2, 10)]
                     + [(f, "backward") for f in range(0, 2)])
    store.record_track_result(o["object_id"], {"rows": 10,
                                               "forward_ok": True,
                                               "backward_ok": True})
    store.set_review_status(o["object_id"], "accepted")
    out = tmp_path / "vo_out"
    manifest = publish_tracked_artifact(bundle, store, tracks, out,
                                        make_overlay=False)
    assert manifest["synchronization"]["gaze_compatible"] is False
    table = pd.read_parquet(out / "objects.parquet")
    assert table["timestamp_ns"].isna().all()      # no fabricated timestamps


def test_overlay_tolerates_trimmed_container_offset(tmp_path):
    """Regression: sources with nonzero container start (trimmed artifacts)
    get muxer-rebased overlay pts — a constant offset must pass validation."""
    from neon_trim.export import trim_recording
    from util import make_csv_export
    src = make_csv_export(tmp_path, "off_src")
    mid = tmp_path / "mid_trim"
    trim_recording(src, 2, 8, mid)              # nonzero cut -> pts offset
    bundle = intake.open_bundle(mid)
    assert bundle.container_start_s if hasattr(bundle, "container_start_s") \
        else True
    store = AnnotationStore(tmp_path / "a.json", bundle.n_frames,
                            bundle.width, bundle.height)
    o = store.create_object("x", 3)
    store.set_points(o["object_id"], [[40, 40]], [1])
    store.record_preview(o["object_id"], {"found": True})
    store.accept_mask(o["object_id"])
    tracks = tmp_path / "tr.db"
    n = bundle.n_frames
    write_track_rows(tracks, o["object_id"],
                     [(f, "forward") for f in range(3, n)]
                     + [(f, "backward") for f in range(0, 3)])
    store.record_track_result(o["object_id"], {"rows": n, "forward_ok": True,
                                               "backward_ok": True})
    store.set_review_status(o["object_id"], "accepted")
    manifest = publish_tracked_artifact(bundle, store, tracks,
                                        tmp_path / "off_out",
                                        make_overlay=True)
    assert manifest["overlay"]["present"]
    assert manifest["overlay"]["max_pts_dev_ms"] <= 2.0


def test_publish_cli_roundtrip(tmp_path):
    """The subprocess/CLI publish path produces a loadable artifact."""
    import subprocess, sys as _sys, hashlib, shutil as _sh
    bundle, store, tracks, oid = build_session(tmp_path)
    state = tmp_path / "state"
    key = hashlib.sha1(str(bundle.source_dir).encode()).hexdigest()[:16]
    wd = state / "sessions" / key
    wd.mkdir(parents=True)
    _sh.copy(store.path, wd / "annotations.json")
    _sh.copy(tracks, wd / "tracks.db")
    out = tmp_path / "cli_out"
    cli = (Path(__file__).resolve().parent.parent / "publish_cli.py")
    r = subprocess.run(
        [_sys.executable, str(cli), "--src", str(bundle.source_dir),
         "--out", str(out), "--no-overlay", "--state-dir", str(state)],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    final = json.loads(lines[-1])
    assert final.get("done") is True
    art = load_tracked_artifact(out)
    assert art.artifact_id == final["manifest"]["artifact_id"]
