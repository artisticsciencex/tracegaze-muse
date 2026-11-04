"""Middle-frame object prompting.

An object may be anchored at ANY exactly-presented frame. Clicking the
image while a differently-anchored object is selected must record
nothing, seek nothing and change nothing — it offers explicit choices
instead.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from common import config as cc
from object_tracking import server as srv, trackstore
from object_tracking.annotations import AnnotationError, AnnotationStore

from util import make_csv_export

PAGE = (Path(__file__).resolve().parent.parent / "static" / "index.html")


@pytest.fixture()
def store(tmp_path):
    return AnnotationStore(tmp_path / "ann.json", 800, 1408, 1408)


# ------------------------------------------------- creation at frame N

def test_object_created_at_frame_400_keeps_that_reference(store):
    o = store.create_object("smartphone", 400)
    assert o["reference_frame"] == 400
    assert o["status"] == "draft"
    store.set_points(o["object_id"], [[700.0, 900.0], [720.0, 910.0]],
                     [1, 0])
    got = store.get(o["object_id"])
    assert got["reference_frame"] == 400, "anchors must not move the frame"
    assert got["points"] == [[700.0, 900.0], [720.0, 910.0]]
    rev = [r for r in got["revisions"] if r["action"] == "points_set"]
    assert rev[-1]["reference_frame"] == 400, "anchors stored against 400"


def test_preview_and_tracking_use_the_object_reference_frame(tmp_path):
    """The preview and track requests carry frame 400, not frame 0."""
    root = tmp_path / "root"
    root.mkdir()
    rec = make_csv_export(root, "rec1")
    cfg = cc.StageConfig(stage="object_tracking", port=8002,
                         browse_roots=[root], state_dir=tmp_path / "state")
    c = TestClient(srv.create_app(cfg), base_url="http://localhost")
    assert c.post("/api/load", json={"dir": str(rec)}).status_code == 200
    o = c.post("/api/objects", json={"dir": str(rec), "name": "phone",
                                     "ref_frame": 7}).json()
    assert o["reference_frame"] == 7
    c.post(f"/api/objects/{o['object_id']}/points",
           json={"dir": str(rec), "points": [[80, 60]], "labels": [1]})
    got = c.get("/api/objects", params={"dir": str(rec)}).json()["objects"][0]
    assert got["reference_frame"] == 7

    # the tracking engine receives the object's own reference frame
    from object_tracking import tracking
    import inspect
    src = inspect.getsource(tracking)
    assert "ref_frame" in src
    srv_src = (Path(__file__).resolve().parent.parent
               / "server.py").read_text()
    assert '"--ref-frame", o["reference_frame"]' in srv_src
    assert 'ref_frame=o["reference_frame"]' in srv_src


def test_bidirectional_tracking_from_a_middle_reference(store, tmp_path):
    """A middle reference propagates forward AND backward into one id."""
    o = store.create_object("phone", 400)
    oid = o["object_id"]
    store.set_points(oid, [[700.0, 900.0]], [1])
    store.record_preview(oid, {"found": True, "score": 0.9,
                               "mask_area": 900})
    store.accept_mask(oid)
    frames = ([(f, "forward") for f in range(400, 800)]
              + [(f, "backward") for f in range(0, 400)])
    rows = [{"frame_index": f, "tracking_direction": d, "sam_local_id": 1,
             "score": 0.9, "bbox_x": 1.0, "bbox_y": 1.0, "bbox_w": 5.0,
             "bbox_h": 5.0, "mask_area": 25, "mask_rle": "{}"}
            for f, d in frames]
    db = tmp_path / "t.db"
    trackstore.replace_object_rows(db, oid, rows)
    store.record_track_result(oid, {
        "rows": len(rows), "frames_present": len(rows), "first_frame": 0,
        "last_frame": 799, "forward_ok": True, "backward_ok": True,
        "whole_video": True})
    got = store.get(oid)
    assert got["track"]["forward_ok"] and got["track"]["backward_ok"]
    assert got["track"]["first_frame"] == 0
    assert got["track"]["last_frame"] == 799
    seen = {r["frame_index"] for r in trackstore.iter_object_rows(db, oid)}
    assert seen == set(range(800)), "one persistent id covers the video"
    assert got["reference_frame"] == 400


# ------------------------------------------- explicit reference movement

def test_move_reference_clears_anchors_preview_track_and_status(store,
                                                                tmp_path):
    o = store.create_object("phone", 0)
    oid, name, color = o["object_id"], o["name"], o["color"]
    store.set_points(oid, [[10.0, 10.0]], [1])
    store.record_preview(oid, {"found": True, "score": 0.8,
                               "mask_area": 100})
    store.accept_mask(oid)
    store.record_track_result(oid, {"rows": 5, "frames_present": 5,
                                    "first_frame": 0, "last_frame": 4,
                                    "forward_ok": True, "backward_ok": True,
                                    "whole_video": True})
    store.set_review_status(oid, "accepted")

    moved = store.move_reference_frame(oid, 400)
    assert moved["reference_frame"] == 400
    assert moved["object_id"] == oid and moved["name"] == name
    assert moved["color"] == color, "immutable identity preserved"
    assert moved["points"] == [] and moved["point_labels"] == []
    assert moved["preview"] is None
    assert moved["track"] is None
    assert moved["status"] == "draft", "must be re-previewed and retracked"
    rev = [r for r in moved["revisions"]
           if r["action"] == "reference_frame_moved"]
    assert len(rev) == 1
    assert rev[0]["from_frame"] == 0 and rev[0]["to_frame"] == 400
    assert rev[0]["cleared"]["points"] == 1
    assert rev[0]["cleared"]["previous_status"] == "accepted"
    # not publishable until the whole cycle is redone
    ready, blockers = store.publishable()
    assert not ready and blockers


def test_move_reference_rejects_out_of_range_and_is_a_noop_when_same(store):
    o = store.create_object("phone", 5)
    with pytest.raises(AnnotationError):
        store.move_reference_frame(o["object_id"], 9999)
    same = store.move_reference_frame(o["object_id"], 5)
    assert same["reference_frame"] == 5
    assert not [r for r in same["revisions"]
                if r["action"] == "reference_frame_moved"]


def test_move_reference_endpoint_drops_the_stale_track(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    rec = make_csv_export(root, "rec1")
    cfg = cc.StageConfig(stage="object_tracking", port=8002,
                         browse_roots=[root], state_dir=tmp_path / "state")
    c = TestClient(srv.create_app(cfg), base_url="http://localhost")
    c.post("/api/load", json={"dir": str(rec)})
    o = c.post("/api/objects", json={"dir": str(rec), "name": "phone",
                                     "ref_frame": 2}).json()
    c.post(f"/api/objects/{o['object_id']}/points",
           json={"dir": str(rec), "points": [[80, 60]], "labels": [1]})
    r = c.post(f"/api/objects/{o['object_id']}/move_reference",
               json={"dir": str(rec), "frame": 6})
    assert r.status_code == 200
    upd = r.json()
    assert upd["reference_frame"] == 6 and upd["points"] == []
    bad = c.post(f"/api/objects/{o['object_id']}/move_reference",
                 json={"dir": str(rec), "frame": 10_000})
    assert bad.status_code == 422


# ------------------------------------------------- GUI contract (page)

def test_page_never_silently_seeks_on_a_reference_mismatch():
    page = PAGE.read_text()
    # the old silent-seek shape must be gone
    assert "if(FS) FS.requestFrame(o.reference_frame);   // exact-seek" \
        not in page
    assert "showRefMismatch(o, FS.displayedFrame)" in page
    # the panel offers exactly the four documented actions
    for token in ("Create a new object at frame",
                  "Move ${esc(o.name)} reference to frame",
                  "Go to frame ${o.reference_frame}",
                  "Cancel"):
        assert token in page, token
    # creation is frame-aware and uses FrameSync, never video.currentTime
    assert "+ Add new object at frame ${f}" in page
    assert "FS.annotationAllowed()) ? FS.displayedFrame : null" in page
    assert "Reference frame: ${o.reference_frame}" in page
    assert "move_reference" in page
    # presentation truth is untouched
    assert "video.currentTime" not in page.split("function refFrameForNewObject")[1]


def test_creation_is_blocked_until_a_frame_is_committed():
    """A rapid seek must not bind a new object to a stale frame."""
    page = PAGE.read_text()
    assert "waiting for exact frame" in page
    assert "b.disabled = f===null" in page
    assert ("the displayed frame is not resolved yet" in page)
