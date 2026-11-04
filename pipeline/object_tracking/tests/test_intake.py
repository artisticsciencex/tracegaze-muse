"""Intake tests: input-type detection, bundle opening, authorization."""

import shutil

import pytest

from object_tracking import intake

from util import (WORLD_TS, make_csv_export, make_trimmed_artifact,
                  encode_vfr_video, write_csvs)

NATIVE_FIXTURE = ("data/neon-data/"
                  "lab_study/participant_02")


# ------------------------------------------------------------ classification

def test_classify_csv_export(tmp_path):
    d = make_csv_export(tmp_path)
    assert intake.classify_folder(d) == "csv_export"


def test_classify_trimmed_artifact(tmp_path):
    d = make_trimmed_artifact(tmp_path)
    assert intake.classify_folder(d) == "trimmed_artifact"


def test_classify_native_recording_real_fixture():
    assert intake.classify_folder(NATIVE_FIXTURE) == "native_recording"


def test_classify_video_only(tmp_path):
    d = tmp_path / "vid"
    d.mkdir()
    encode_vfr_video(d / "scene.mp4")
    assert intake.classify_folder(d) == "video_only"


def test_classify_unsupported(tmp_path):
    d = tmp_path / "junk"
    d.mkdir()
    (d / "notes.txt").write_text("x")
    assert intake.classify_folder(d) is None


def test_classify_ambiguous_videos(tmp_path):
    d = make_csv_export(tmp_path)
    (d / "second.mp4").write_bytes(b"\x00")
    # two videos + csvs: not a clean csv_export, not video_only
    assert intake.classify_folder(d) is None


def test_classify_parent_of_recordings(tmp_path):
    parent = tmp_path / "day"
    make_csv_export(parent, "r1")
    make_csv_export(parent, "r2")
    assert intake.classify_folder(parent) is None  # navigate, don't select


# ------------------------------------------------------------ open_bundle

def test_open_bundle_trimmed_artifact(tmp_path):
    d = make_trimmed_artifact(tmp_path)
    b = intake.open_bundle(d)
    assert b.source_type == "trimmed_artifact"
    assert b.sync_ok and b.gaze_compatible
    assert b.n_frames == len(WORLD_TS) == len(b.world_ts_ns)
    assert b.artifact_id and b.artifact_type == "trimmed_neon_recording"
    assert b.timestamp_ns_for_frame(0) == b.world_ts_ns[0]


def test_open_bundle_csv_export(tmp_path):
    d = make_csv_export(tmp_path)
    b = intake.open_bundle(d)
    assert b.source_type == "csv_export"
    assert b.sync_ok and b.gaze_compatible
    assert b.gaze_rows == 40
    assert b.recording_id == "rec-1"


def test_open_bundle_video_only(tmp_path):
    d = tmp_path / "vid"
    d.mkdir()
    encode_vfr_video(d / "scene.mp4")
    b = intake.open_bundle(d)
    assert b.source_type == "video_only"
    assert not b.gaze_compatible and not b.sync_ok
    assert b.timestamp_ns_for_frame(0) is None
    assert any("gaze unavailable" in w for w in b.warnings)


def test_open_bundle_native_requires_preparation():
    with pytest.raises(intake.IntakeError, match="preparation"):
        intake.open_bundle(NATIVE_FIXTURE)


def test_open_bundle_sync_mismatch_not_gaze_compatible(tmp_path):
    d = make_csv_export(tmp_path)
    # drop one world row -> count mismatch
    lines = (d / "world_timestamps.csv").read_text().splitlines()
    (d / "world_timestamps.csv").write_text("\n".join(lines[:-1]) + "\n")
    b = intake.open_bundle(d)
    assert not b.sync_ok and not b.gaze_compatible


def test_bundle_survives_moved_artifact(tmp_path):
    d = make_trimmed_artifact(tmp_path)
    moved = tmp_path / "relocated"
    shutil.move(str(d), moved)
    b = intake.open_bundle(moved)
    assert b.source_type == "trimmed_artifact" and b.sync_ok


# ------------------------------------------------------- API authorization

@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient
    from common import config as cc
    from object_tracking import server as srv
    root = tmp_path / "root"
    root.mkdir()
    make_csv_export(root, "rec1")
    (tmp_path / "outside").mkdir()
    cfg = cc.StageConfig(stage="object_tracking", port=8002,
                         browse_roots=[root], state_dir=tmp_path / "state")
    return TestClient(srv.create_app(cfg), base_url="http://localhost"), \
        root, tmp_path


def test_api_traversal_and_symlink_escape(client):
    c, root, tmp = client
    for bad in [str(root) + "/../outside", "/etc", "../../etc"]:
        assert c.get("/api/browse", params={"path": bad}).status_code == 403
    link = root / "sneaky"
    link.symlink_to(tmp / "outside")
    assert c.get("/api/browse", params={"path": str(link)}).status_code == 403
    assert c.get("/api/inspect",
                 params={"path": str(link)}).status_code == 403
    assert c.post("/api/load",
                  json={"dir": str(tmp / "outside")}).status_code == 403


def test_api_load_and_video_flow(client):
    c, root, tmp = client
    r = c.post("/api/load", json={"dir": str(root / "rec1")})
    assert r.status_code == 200
    d = r.json()
    assert d["source_type"] == "csv_export" and d["sync_ok"]
    assert d["n_frames"] == len(d["pts_s"]) == len(d["world_rel_ns"])
    v = c.get("/api/video", params={"dir": str(root / "rec1")},
              headers={"Range": "bytes=0-99"})
    assert v.status_code == 206 and len(v.content) == 100


def test_api_badges_in_browse(client):
    c, root, tmp = client
    r = c.get("/api/browse", params={"path": str(root)})
    kinds = {e["name"]: e["kind"] for e in r.json()["entries"]
             if e["is_dir"]}
    assert kinds.get("rec1") == "csv_export"


def test_api_native_inspect_real_fixture(client):
    c, root, tmp = client
    # native fixture is OUTSIDE the configured root -> 403 (authorization
    # applies to inspection too)
    assert c.get("/api/inspect",
                 params={"path": NATIVE_FIXTURE}).status_code == 403
