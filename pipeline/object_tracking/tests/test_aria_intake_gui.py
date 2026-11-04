"""Object-Tracking compatibility with Project Aria artifacts.

Covers the complete downstream contract that made a valid Aria artifact
unselectable: the semantic kind sets, the chooser/recents/inspect gates,
strict loading, publication reload-validation of the Aria parent, and the
handoff to gaze fusion. Synthetic fixtures only — no SAM3 pass, no
participant data.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from common import config as cc
from common.artifacts import ArtifactError
from object_tracking import intake, server as srv, trackstore
from object_tracking.annotations import AnnotationStore
from object_tracking.artifact import publish_tracked_artifact

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent
                       / "aria_source" / "tests"))
import autil                                             # noqa: E402

from util import make_trimmed_artifact                   # noqa: E402

PIPELINE_DIR = Path(__file__).resolve().parent.parent.parent


def aria_trimmed(tmp_path: Path, name="aria_trim") -> Path:
    """A valid trimmed_aria_recording (prepared → real trim boundary)."""
    from aria_source.trim import trim_aria_recording
    prep = autil.make_prepared_artifact(tmp_path / (name + "_prep"),
                                        n_frames=20, gaze_n=60)
    out = tmp_path / name
    trim_aria_recording(prep, 3, 16, out)
    return out


@pytest.fixture()
def client(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    cfg = cc.StageConfig(stage="object_tracking", port=8002,
                         browse_roots=[root], state_dir=tmp_path / "state")
    return TestClient(srv.create_app(cfg),
                      base_url="http://localhost"), root


# ------------------------------------------------------- 1. semantic sets

def test_kind_sets_include_both_aria_artifact_types():
    for kind in ("prepared_aria_recording", "trimmed_aria_recording"):
        assert kind in intake.CHOOSER_KINDS
        assert kind in intake.TRACKABLE_KINDS
        assert intake.KIND_BADGE_CLASS[kind] == "ready"
        assert kind in intake.KIND_LABELS
        assert intake.is_aria_kind(kind)
    # raw downloads stay preparation-required, never directly trackable
    assert "aria_download" in intake.PREPARATION_REQUIRED_KINDS
    assert "aria_download" not in intake.TRACKABLE_KINDS
    assert "aria_download" not in intake.CHOOSER_KINDS
    # NEON behavior preserved exactly
    for kind in ("trimmed_artifact", "prepared_artifact", "csv_export",
                 "video_only"):
        assert kind in intake.CHOOSER_KINDS and kind in intake.TRACKABLE_KINDS
    assert "native_recording" in intake.CHOOSER_KINDS
    assert "native_recording" not in intake.TRACKABLE_KINDS
    assert "tracked_objects" in intake.NON_INPUT_KINDS
    assert intake.GAZE_TRIMMED_PARENT_KINDS == (
        "trimmed_artifact", "trimmed_aria_recording")


def test_config_exposes_the_kind_sets_to_the_browser(client):
    c, root = client
    d = c.get("/api/config").json()
    assert d["chooser_kinds"] == list(intake.CHOOSER_KINDS)
    assert "trimmed_aria_recording" in d["chooser_kinds"]
    assert d["kind_badge_class"]["trimmed_aria_recording"] == "ready"
    assert d["kind_labels"]["trimmed_aria_recording"]


def test_frontend_derives_selectable_from_server_config():
    """No duplicated allow-list may return to the page."""
    page = (PIPELINE_DIR / "object_tracking" / "static"
            / "index.html").read_text()
    assert "CHOOSER_KINDS.includes(kind)" in page
    assert "d.chooser_kinds" in page
    assert '["trimmed_artifact","prepared_artifact","csv_export"' not in page
    assert "KIND_BADGE_CLASS[kind]" in page


# ------------------------------------------------------- 2/3. chooser API

def test_browse_classifies_aria_artifacts(client, tmp_path):
    c, root = client
    aria_trimmed(root, "aria_trim")
    autil.make_prepared_artifact(root / "aria_prep")
    make_trimmed_artifact(root, "neon_trim")
    r = c.get("/api/browse", params={"path": str(root)})
    assert r.status_code == 200
    kinds = {e["name"]: e["kind"] for e in r.json()["entries"]}
    assert kinds["aria_trim"] == "trimmed_aria_recording"
    assert kinds["aria_prep"] == "prepared_aria_recording"
    assert kinds["neon_trim"] == "trimmed_artifact", "NEON unchanged"
    for k in ("aria_trim", "aria_prep", "neon_trim"):
        assert kinds[k] in intake.CHOOSER_KINDS


def test_inspect_is_source_aware_for_aria(client, tmp_path):
    c, root = client
    art = aria_trimmed(root)
    d = c.get("/api/inspect", params={"path": str(art)}).json()
    assert d["kind"] == "trimmed_aria_recording"
    assert d["source_system"] == "project_aria"
    a = d["aria"]
    assert a["artifact_type"] == "trimmed_aria_recording"
    assert a["artifact_id"] and a["recording_id"] == autil.REC_ID
    assert a["dataset"] == autil.DATASET
    assert a["files"] == {"video": "scene.mp4", "gaze": "gaze.csv",
                          "world_timestamps": "world_timestamps.csv"}
    assert a["n_frames"] == 14 and a["world_rows"] == 14
    assert a["gaze_rows"] > 0
    assert a["timestamp_domain"] == "aria_device_time"
    assert a["validation_ok"] is True and a["validation_checks"] >= 9
    assert d.get("problems") in (None, [])
    # NEON-only terminology must be explicitly unavailable, never invented
    assert a["wearer_name"] is None
    assert a["neon_gaze_offset"] is None and a["neon_world_time"] is None
    assert "do not exist for a Project Aria recording" in \
        a["not_applicable_note"]
    assert "inspection" not in d, "must not fall back to the NEON inspector"


def test_inspect_rejects_broken_aria_artifacts(client, tmp_path):
    c, root = client
    art = aria_trimmed(root)
    mf = art / "trim_manifest.json"
    m = json.loads(mf.read_text())
    m["validation"] = [{"check": "frame count", "ok": False,
                        "detail": "mismatch"}]
    mf.write_text(json.dumps(m))
    d = c.get("/api/inspect", params={"path": str(art)}).json()
    assert d["problems"], "a failed-validation artifact must be rejected"

    # incomplete payload: no longer classified as an Aria artifact at all
    (art / "scene.mp4").unlink()
    d2 = c.get("/api/inspect", params={"path": str(art)}).json()
    assert d2["kind"] != "trimmed_aria_recording"


def test_raw_aria_download_remains_preparation_required(client, tmp_path):
    c, root = client
    dl = autil.make_download_folder(root)
    d = c.get("/api/inspect", params={"path": str(dl)}).json()
    assert d["kind"] == "aria_download"
    assert d["needs_preparation"] is True and d["problems"]
    assert d["kind"] not in intake.CHOOSER_KINDS
    r = c.post("/api/load", json={"dir": str(dl)})
    assert r.status_code >= 400
    assert "preparation" in json.dumps(r.json()["detail"]).lower()


def test_recents_marks_aria_artifacts_usable(client, tmp_path):
    c, root = client
    trimmed = aria_trimmed(root)
    prep = autil.make_prepared_artifact(root / "aria_prep")
    neon = make_trimmed_artifact(root, "neon_trim")
    dl = autil.make_download_folder(root)
    for p in (trimmed, prep, neon, dl):
        c.get("/api/inspect", params={"path": str(p)})
        c.post("/api/load", json={"dir": str(p)})
    got = {e["name"]: e for e in c.get("/api/recents").json()["recents"]}
    assert got[trimmed.name]["usable"] is True
    assert got[trimmed.name]["kind"] == "trimmed_aria_recording"
    assert got["aria_prep"]["usable"] is True
    assert got["neon_trim"]["usable"] is True, "NEON recents unchanged"
    if dl.name in got:
        assert got[dl.name]["usable"] is False


# -------------------------------------------------------- 4. strict load

def test_valid_trimmed_aria_artifact_loads(client, tmp_path):
    c, root = client
    art = aria_trimmed(root)
    r = c.post("/api/load", json={"dir": str(art)})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["source_type"] == "trimmed_aria_recording"
    assert d["n_frames"] == 14 and d["world_rows"] == 14
    assert d["sync_ok"] is True and d["gaze_compatible"] is True

    b = intake.open_bundle(art)
    assert b.artifact_type == "trimmed_aria_recording"
    assert b.recording_id == autil.REC_ID
    assert len(b.world_ts_ns) == b.n_frames

    # the one-frame/one-timestamp check is NOT weakened
    import csv as _csv
    w = art / "world_timestamps.csv"
    rows = list(_csv.reader(open(w)))
    rows.append(rows[-1][:-1] + [str(int(rows[-1][-1]) + 1)])
    with open(w, "w", newline="") as f:
        _csv.writer(f).writerows(rows)
    b2 = intake.open_bundle(art)
    assert b2.sync_ok is False and b2.gaze_compatible is False
    assert "mismatch" in b2.sync_status.lower()


def test_unauthorized_paths_still_rejected(client, tmp_path):
    c, root = client
    for bad in ("/etc", str(tmp_path / "outside"), "/etc/../etc/passwd"):
        assert c.get("/api/inspect", params={"path": bad}).status_code == 403
        assert c.post("/api/load", json={"dir": bad}).status_code == 403


# ----------------------------------------- 5/6. publication + fusion handoff

def _publish_tracked(tmp_path, parent_dir, tamper=None):
    """Publish a tracked artifact over a synthetic (mocked) track table.

    `tamper()` runs AFTER intake has opened the bundle and BEFORE
    publication, so the publish-time parent RELOAD is what is exercised.
    """
    import numpy as np
    import pycocotools.mask as mask_util
    bundle = intake.open_bundle(parent_dir)
    store = AnnotationStore(tmp_path / f"ann_{parent_dir.name}.json",
                            bundle.n_frames, bundle.width, bundle.height)
    o = store.create_object("book", 4)
    oid = o["object_id"]
    store.set_points(oid, [[40, 40]], [1])
    store.record_preview(oid, {"found": True, "score": 0.9,
                               "mask_area": 400})
    store.accept_mask(oid)
    m = np.zeros((bundle.height, bundle.width), dtype=np.uint8)
    m[20:60, 20:60] = 1
    rle = mask_util.encode(np.asfortranarray(m))
    rle_json = json.dumps({"size": [bundle.height, bundle.width],
                           "counts": rle["counts"].decode("ascii")})
    rows = [{"frame_index": i,
             "tracking_direction": "forward" if i >= 4 else "backward",
             "sam_local_id": 1, "score": 0.9, "bbox_x": 20.0,
             "bbox_y": 20.0, "bbox_w": 40.0, "bbox_h": 40.0,
             "mask_area": int(m.sum()), "mask_rle": rle_json}
            for i in range(bundle.n_frames)]
    db = tmp_path / f"t_{parent_dir.name}.db"
    trackstore.replace_object_rows(db, oid, rows)
    store.record_track_result(oid, {
        "rows": len(rows), "frames_present": len(rows), "first_frame": 0,
        "last_frame": bundle.n_frames - 1, "forward_ok": True,
        "backward_ok": True, "whole_video": True})
    store.set_review_status(oid, "accepted")
    if tamper is not None:
        tamper()
    out = tmp_path / f"{parent_dir.name}_tracked_objects"
    return publish_tracked_artifact(bundle, store, db, out,
                                    make_overlay=False), out


def test_publication_reload_validates_the_aria_parent(tmp_path):
    art = aria_trimmed(tmp_path)
    m, out = _publish_tracked(tmp_path, art)
    checks = {v["check"]: v for v in m["validation"]}
    pv = checks["parent artifact still loadable"]
    assert pv["ok"] is True
    assert "Aria loader" in pv["detail"]
    assert "aria_device_time" in pv["detail"]

    # a parent that stops validating between load and publish is caught by
    # the publish-time RELOAD (the intake-time check already passed)
    art2 = aria_trimmed(tmp_path, "aria_trim2")
    mf = art2 / "trim_manifest.json"

    def break_parent():
        m2 = json.loads(mf.read_text())
        m2["validation"] = [{"check": "frame count", "ok": False,
                             "detail": "tampered after load"}]
        mf.write_text(json.dumps(m2))

    with pytest.raises(ArtifactError, match="publish validation failed"):
        _publish_tracked(tmp_path / "second", art2, tamper=break_parent)

    # ... and an identity swap is caught too
    art3 = aria_trimmed(tmp_path, "aria_trim3")
    mf3 = art3 / "trim_manifest.json"

    def swap_id():
        m3 = json.loads(mf3.read_text())
        m3["artifact_id"] = "0" * 32
        mf3.write_text(json.dumps(m3))

    with pytest.raises(ArtifactError, match="publish validation failed"):
        _publish_tracked(tmp_path / "third", art3, tamper=swap_id)


def test_tracking_manifest_preserves_aria_provenance(tmp_path):
    art = aria_trimmed(tmp_path)
    m, out = _publish_tracked(tmp_path, art)
    parent_manifest = json.loads((art / "trim_manifest.json").read_text())
    assert m["parent"]["source_type"] == "trimmed_aria_recording"
    assert m["parent_artifact"]["artifact_id"] == \
        parent_manifest["artifact_id"]
    assert m["parent_artifact"]["artifact_type"] == "trimmed_aria_recording"
    assert m["parent_artifact"]["recording_id"] == autil.REC_ID
    sync = m["synchronization"]
    assert sync["gaze_compatible"] is True and sync["sync_ok"] is True
    assert sync["source_system"] == "project_aria"
    assert sync["timestamp_domain"] == "aria_device_time"
    # never NEON labels for an Aria parent
    blob = json.dumps(m)
    assert '"source_type": "trimmed_artifact"' not in blob
    assert "pupil_neon" not in blob
    assert m["parent"]["video_sha256"] and m["parent"]["gaze_sha256"]
    assert any(v["check"] == "source recording/artifact unchanged (sha256)"
               and v["ok"] for v in m["validation"])


def test_gaze_fusion_accepts_the_aria_tracked_artifact(tmp_path):
    from gaze_object_fusion.fusion import FusionInputs
    from gaze_object_fusion.server import inspect_tracked_dir
    art = aria_trimmed(tmp_path)
    m, tracked = _publish_tracked(tmp_path, art)
    info = inspect_tracked_dir(tracked)
    assert info["ok"] is True, info["problems"]
    assert info["parent"]["source_type"] == "trimmed_aria_recording"
    fi = FusionInputs(tracked)
    assert fi.source_system == "project_aria"
    assert fi.timestamp_domain == "aria_device_time"
    assert fi.interval_end_ns > fi.interval_start_ns


def test_neon_parent_path_is_unchanged(tmp_path):
    """The NEON publication + fusion contract must be untouched."""
    from gaze_object_fusion.fusion import FusionInputs
    neon = make_trimmed_artifact(tmp_path, "neon_trim")
    m, tracked = _publish_tracked(tmp_path, neon)
    assert m["parent"]["source_type"] == "trimmed_artifact"
    assert m["synchronization"]["source_system"] == "pupil_neon"
    assert m["synchronization"]["timestamp_domain"] == "neon_world_time"
    checks = {v["check"]: v for v in m["validation"]}
    assert checks["parent artifact still loadable"]["ok"] is True
    fi = FusionInputs(tracked)
    assert fi.source_system == "pupil_neon"


# ------------------------------------------------------ 13. server purity

def test_server_import_stays_pandas_and_pyarrow_free():
    code = (
        "import sys; sys.path.insert(0, '.')\n"
        "from pathlib import Path\n"
        "from object_tracking import server as srv\n"
        "from common import config as cc\n"
        "app = srv.create_app(cc.StageConfig(stage='object_tracking',"
        " port=8002, browse_roots=[Path('/tmp')],"
        " state_dir=Path('/tmp/ot_aria_iso')))\n"
        "import aria_source.artifact, object_tracking.intake\n"
        "bad = [m for m in ('pandas','pyarrow','projectaria_tools','torch')"
        " if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print('PURE_OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(PIPELINE_DIR),
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PURE_OK" in r.stdout
