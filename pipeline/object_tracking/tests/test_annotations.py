"""Annotation-model tests: identity, names, colors, points, status flow."""

import pytest

from object_tracking.annotations import (AnnotationError, AnnotationStore,
                                         color_for_object_id)


@pytest.fixture()
def store(tmp_path):
    return AnnotationStore(tmp_path / "ann.json", frame_count=100,
                           frame_w=160, frame_h=120)


def test_ids_sequential_immutable_never_reused(store):
    a = store.create_object("pipette", 0)
    b = store.create_object("dish", 5)
    assert a["object_id"] == "obj_0001" and b["object_id"] == "obj_0002"
    store.delete(a["object_id"])
    c = store.create_object("rack", 9)
    assert c["object_id"] == "obj_0003"          # deleted id retired


def test_arbitrary_names_and_duplicate_rejection(store):
    store.create_object("blue_cloth", 0)
    store.create_object("Zentrifugendeckel 3000 (!)", 0)  # arbitrary
    with pytest.raises(AnnotationError, match="suggestion.*blue_cloth_2"):
        store.create_object("blue_cloth", 1)
    with pytest.raises(AnnotationError, match="non-empty"):
        store.create_object("   ", 1)


def test_rename_keeps_id_and_color(store):
    o = store.create_object("pipette", 0)
    col, oid = o["color"], o["object_id"]
    o2 = store.rename(oid, "green_pipette")
    assert o2["object_id"] == oid and o2["color"] == col
    assert o2["name"] == "green_pipette"
    with pytest.raises(AnnotationError):
        store.create_object("green_pipette", 0)   # rename holds the name


def test_deterministic_colors_stable():
    assert color_for_object_id("obj_0001") == color_for_object_id("obj_0001")
    assert color_for_object_id("obj_0001") != color_for_object_id("obj_0002")
    # regression-pin two values so cross-process stability is guaranteed
    assert color_for_object_id("obj_0001") == "#39de69"
    assert color_for_object_id("obj_0002") == "#9a39de"


def test_points_validation(store):
    o = store.create_object("x", 10)
    store.set_points(o["object_id"], [[10, 10], [150, 100]], [1, 0])
    with pytest.raises(AnnotationError, match="outside"):
        store.set_points(o["object_id"], [[200, 10]], [1])
    with pytest.raises(AnnotationError, match="label"):
        store.set_points(o["object_id"], [[10, 10]], [2])
    with pytest.raises(AnnotationError, match="mismatch"):
        store.set_points(o["object_id"], [[10, 10]], [1, 0])
    with pytest.raises(AnnotationError, match="reference frame"):
        store.set_points(o["object_id"], [[10, 10]], [1], ref_frame=100)


def test_reference_frame_per_object(store):
    a = store.create_object("a", 3)
    b = store.create_object("b", 77)
    assert store.get(a["object_id"])["reference_frame"] == 3
    assert store.get(b["object_id"])["reference_frame"] == 77
    store.set_points(a["object_id"], [[1, 1]], [1], ref_frame=42)
    assert store.get(a["object_id"])["reference_frame"] == 42


def test_mask_acceptance_gate(store):
    o = store.create_object("x", 0)
    with pytest.raises(AnnotationError, match="positive point"):
        store.accept_mask(o["object_id"])
    store.set_points(o["object_id"], [[5, 5]], [1])
    with pytest.raises(AnnotationError, match="preview"):
        store.accept_mask(o["object_id"])
    store.record_preview(o["object_id"],
                         {"found": True, "score": 0.9, "mask_area": 100})
    got = store.accept_mask(o["object_id"])
    assert got["status"] == "mask_accepted"
    # changing points invalidates the accepted mask
    store.set_points(o["object_id"], [[6, 6]], [1])
    assert store.get(o["object_id"])["status"] == "draft"
    assert store.get(o["object_id"])["preview"] is None


def test_review_status_flow_and_publish_gate(store):
    o = store.create_object("x", 0)
    store.set_points(o["object_id"], [[5, 5]], [1])
    store.record_preview(o["object_id"], {"found": True})
    store.accept_mask(o["object_id"])
    with pytest.raises(AnnotationError, match="untracked"):
        store.set_review_status(o["object_id"], "accepted")
    store.record_track_result(o["object_id"],
                              {"rows": 10, "forward_ok": True,
                               "backward_ok": True})
    ready, blockers = store.publishable()
    assert not ready and any("tracked" in b for b in blockers)
    store.set_review_status(o["object_id"], "accepted")
    ready, blockers = store.publishable()
    assert ready and not blockers
    # a second undecided object blocks publish again
    o2 = store.create_object("y", 1)
    ready, blockers = store.publishable()
    assert not ready
    store.delete(o2["object_id"])
    assert store.publishable()[0]


def test_draft_persistence_reload(tmp_path):
    p = tmp_path / "ann.json"
    s1 = AnnotationStore(p, 100, 160, 120)
    o = s1.create_object("persisted", 7)
    s1.set_points(o["object_id"], [[9, 9], [20, 20]], [1, 0])
    s2 = AnnotationStore(p, 100, 160, 120)   # fresh instance = reload
    got = s2.get(o["object_id"])
    assert got["name"] == "persisted"
    assert got["points"] == [[9.0, 9.0], [20.0, 20.0]]
    assert got["point_labels"] == [1, 0]
    assert s2.create_object("next", 0)["object_id"] == "obj_0002"


def test_revision_history_records_actions(store):
    o = store.create_object("x", 0)
    store.set_points(o["object_id"], [[5, 5]], [1])
    store.record_preview(o["object_id"], {"found": True})
    store.accept_mask(o["object_id"])
    store.rename(o["object_id"], "renamed")
    acts = [r["action"] for r in store.get(o["object_id"])["revisions"]]
    assert acts == ["created", "points_set", "previewed", "mask_accepted",
                    "renamed"]


def test_prompt_edit_invalidates_track_and_blocks_accept(store):
    """Regression (review CRITICAL): editing points after tracking must void
    the track meta so the stale result can never be accepted/published."""
    o = store.create_object("x", 5)
    store.set_points(o["object_id"], [[5, 5]], [1])
    store.record_preview(o["object_id"], {"found": True})
    store.accept_mask(o["object_id"])
    store.record_track_result(o["object_id"], {"rows": 10,
                                               "forward_ok": True,
                                               "backward_ok": True})
    store.set_points(o["object_id"], [[6, 6]], [1])   # nudge a point
    got = store.get(o["object_id"])
    assert got["status"] == "draft"
    assert got["track"] is None                        # track invalidated
    assert any(r["action"] == "track_invalidated" for r in got["revisions"])
    with pytest.raises(AnnotationError, match="untracked"):
        store.set_review_status(o["object_id"], "accepted")


def test_record_track_result_compare_and_set(store):
    """Regression: a result arriving after a concurrent edit is refused."""
    o = store.create_object("x", 5)
    store.set_points(o["object_id"], [[5, 5]], [1])
    store.record_preview(o["object_id"], {"found": True})
    store.accept_mask(o["object_id"])
    rev = len(store.get(o["object_id"])["revisions"])
    store.set_points(o["object_id"], [[7, 7]], [1])    # edit mid-flight
    with pytest.raises(AnnotationError, match="edited while tracking"):
        store.record_track_result(o["object_id"], {"rows": 1},
                                  expected_revision_count=rev)
    assert store.get(o["object_id"])["track"] is None
