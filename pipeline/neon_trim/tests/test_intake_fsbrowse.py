"""Tests for the intake layer (discovery/inspection) and fsbrowse
(authorization, listing, recents) — plus HTTP-level authorization checks.
"""

import json
import os
from pathlib import Path

import pytest

from neon_trim import config as config_mod
from neon_trim import fsbrowse, intake

T0 = 1_733_200_839_984_000_000


def make_recording(d: Path, n_world: int = 5, n_gaze: int = 7,
                   video_name: str = "scene.mp4", info: bool = True) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / video_name).write_bytes(b"\x00" * 64)      # placeholder (no header probe assertions)
    with open(d / "world_timestamps.csv", "w") as f:
        f.write("section id,recording id,timestamp [ns]\n")
        for i in range(n_world):
            f.write(f"s,r,{T0 + i * 33_000_000}\n")
    with open(d / "gaze.csv", "w") as f:
        f.write("section id,recording id,timestamp [ns],gaze x [px],gaze y [px]\n")
        for i in range(n_gaze):
            f.write(f"s,r,{T0 + i * 5_000_000},1.0,2.0\n")
    if info:
        (d / "info.json").write_text(json.dumps({
            "recording_id": "rid-1", "wearer_name": "tester",
            "template_data": {"recording_name": "rec-name"},
            "start_time": T0, "duration": 10**9}))
    return d


# ---------------------------------------------------------------------------
# intake: discovery + inspection
# ---------------------------------------------------------------------------

def test_valid_recording_discovery(tmp_path):
    rec = make_recording(tmp_path / "rec1")
    # a REAL (decodable) container so the header probe succeeds
    from test_export import _encode_vfr_video
    (rec / "scene.mp4").unlink()
    _encode_vfr_video(rec / "scene.mp4", [0, 33, 66, 100, 133])
    assert intake.looks_like_recording(rec)
    res = intake.inspect_recording(rec)
    assert res.is_recording and not res.problems
    b = res.bundle
    assert b.world_rows == 5 and b.gaze_rows == 7
    assert b.recording_id == "rid-1"
    assert b.wearer_name == "tester"
    assert b.recording_name == "rec-name"
    assert b.video_path.name == "scene.mp4"
    assert b.header_frame_count == 5 and b.width == 160
    assert res.sync_hint == "likely_ok"


def test_unreadable_video_header_disqualifies(tmp_path):
    # placeholder bytes are not a real container: header probe fails ->
    # problems non-empty and is_recording False (consistent with load)
    rec = make_recording(tmp_path / "recbad")
    res = intake.inspect_recording(rec)
    assert not res.is_recording
    assert any("cannot read video header" in p for p in res.problems)
    assert res.bundle is not None        # metadata still available for the UI


def test_count_rows_ignores_blank_lines(tmp_path):
    rec = make_recording(tmp_path / "recblank")
    with open(rec / "gaze.csv", "a") as f:
        f.write("\n\n   \n")             # trailing blank lines
    res = intake.inspect_recording(rec)
    assert res.bundle.gaze_rows == 7     # unchanged


def test_multi_video_folder_not_badged(tmp_path):
    rec = make_recording(tmp_path / "recmulti")
    (rec / "extra.mp4").write_bytes(b"\x00")
    # badge must agree with what inspect()/load will accept
    assert not intake.looks_like_recording(rec)


def test_parent_folder_with_multiple_recordings(tmp_path):
    parent = tmp_path / "participant"
    make_recording(parent / "rec_a")
    make_recording(parent / "rec_b")
    (parent / "notes").mkdir()
    res = intake.inspect_recording(parent)
    assert not res.is_recording
    names = [c["name"] for c in res.child_recordings]
    assert names == ["rec_a", "rec_b"]


def test_missing_required_files(tmp_path):
    rec = make_recording(tmp_path / "rec")
    (rec / "gaze.csv").unlink()
    assert not intake.looks_like_recording(rec)
    res = intake.inspect_recording(rec)
    assert not res.is_recording
    assert any("gaze.csv" in p for p in res.problems)


def test_ambiguous_multiple_mp4(tmp_path):
    rec = make_recording(tmp_path / "rec")
    (rec / "second.mp4").write_bytes(b"\x00" * 32)
    res = intake.inspect_recording(rec)
    assert not res.is_recording
    assert any("ambiguous" in p and "second.mp4" in p for p in res.problems)


def test_row_counts_unterminated_final_line(tmp_path):
    rec = make_recording(tmp_path / "rec")
    # strip the trailing newline: count must not change
    p = rec / "gaze.csv"
    p.write_bytes(p.read_bytes().rstrip(b"\n"))
    res = intake.inspect_recording(rec)
    assert res.bundle.gaze_rows == 7


# ---------------------------------------------------------------------------
# fsbrowse: authorization
# ---------------------------------------------------------------------------

def test_path_traversal_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret").mkdir()
    auth = fsbrowse.PathAuthorizer([root])
    assert auth.is_allowed(root)
    assert auth.is_allowed(root / "sub")
    with pytest.raises(fsbrowse.PathNotAllowed):
        auth.resolve(str(root) + "/../secret")
    with pytest.raises(fsbrowse.PathNotAllowed):
        auth.resolve(str(root / "a" / ".." / ".." / "secret"))
    with pytest.raises(fsbrowse.PathNotAllowed):
        auth.resolve("/etc/passwd")
    with pytest.raises(fsbrowse.PathNotAllowed):
        auth.resolve("relative/path")


def test_symlink_escape_rejected(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir(); outside.mkdir()
    (outside / "loot.txt").write_text("secret")
    link = root / "innocent"
    link.symlink_to(outside)
    auth = fsbrowse.PathAuthorizer([root])
    # the symlink resolves outside the root -> rejected, even though the
    # requested path is textually inside the root
    with pytest.raises(fsbrowse.PathNotAllowed):
        auth.resolve(link)
    with pytest.raises(fsbrowse.PathNotAllowed):
        auth.resolve(link / "loot.txt")


def test_symlink_inside_root_allowed(tmp_path):
    root = tmp_path / "root"
    (root / "real").mkdir(parents=True)
    link = root / "alias"
    link.symlink_to(root / "real")
    auth = fsbrowse.PathAuthorizer([root])
    assert auth.resolve(link) == (root / "real").resolve()


def test_list_dir_states_and_ordering(tmp_path):
    root = tmp_path / "root"
    make_recording(root / "b_rec")
    (root / "a_plain").mkdir(parents=True)
    (root / "z_file.txt").write_text("x")
    auth = fsbrowse.PathAuthorizer([root])
    listing = fsbrowse.list_dir(auth, root)
    assert listing.error is None
    assert listing.parent is None                       # at root boundary
    names = [e.name for e in listing.entries]
    assert names == ["a_plain", "b_rec", "z_file.txt"]  # folders first
    flags = {e.name: e.is_recording for e in listing.entries}
    assert flags["b_rec"] and not flags["a_plain"]
    # breadcrumbs inside a subdir
    sub = fsbrowse.list_dir(auth, root / "b_rec")
    assert [c["name"] for c in sub.breadcrumbs][-1] == "b_rec"
    assert sub.parent == str(root)
    # nonexistent
    missing = fsbrowse.list_dir(auth, root / "nope")
    assert missing.error == "not_found"


def test_recents_store_roundtrip(tmp_path):
    st = fsbrowse.RecentsStore(tmp_path / "state")
    assert st.recent_recordings() == []
    st.add_recording("/a"); st.add_recording("/b"); st.add_recording("/a")
    assert st.recent_recordings() == ["/a", "/b"]       # dedup, most recent first
    st.set_last_out_parent("/out")
    assert st.last_out_parent() == "/out"


# ---------------------------------------------------------------------------
# HTTP-level authorization (the same rules must hold at the API surface)
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient
    from neon_trim import server as server_mod
    root = tmp_path / "root"
    make_recording(root / "rec1")
    (tmp_path / "outside").mkdir()
    cfg = config_mod.AppConfig(browse_roots=[root],
                               state_dir=tmp_path / "state")
    app = server_mod.create_app(cfg)
    # base_url must be a host the anti-DNS-rebinding allowlist accepts
    return TestClient(app, base_url="http://localhost"), root, tmp_path


def test_api_browse_traversal_403(client):
    c, root, tmp = client
    assert c.get("/api/browse", params={"path": str(root)}).status_code == 200
    for bad in [str(root) + "/../outside", "/etc", str(tmp / "outside"),
                "../../etc/passwd"]:
        r = c.get("/api/browse", params={"path": bad})
        assert r.status_code == 403, bad


def test_api_inspect_load_video_authorized_only(client):
    c, root, tmp = client
    outside = tmp / "outside_rec"
    make_recording(outside)
    # inspect: outside root -> 403
    assert c.get("/api/inspect", params={"path": str(outside)}).status_code == 403
    # load: outside root -> 403 (never reaches the loader)
    assert c.post("/api/load", json={"dir": str(outside)}).status_code == 403
    # video: path never loaded + outside -> 403 before any 404
    assert c.get("/api/video", params={"dir": str(outside)}).status_code == 403


def test_api_export_unauthorized_output_403(client):
    c, root, tmp = client
    r = c.post("/api/export", json={
        "dir": str(root / "rec1"), "start_frame": 0, "end_frame": 1,
        "out_dir": str(tmp / "outside" / "out")})
    # rec1 was never loaded -> 404 is acceptable ordering; force-load first
    # (the placeholder video cannot load; so unauthorized-out must be
    # detectable regardless: check the authorizer path directly)
    assert r.status_code in (403, 404)
    if r.status_code == 404:
        # out_dir authorization must still hold once dir is resolvable:
        r2 = c.get("/api/out_check", params={
            "dir": str(root / "rec1"), "path": str(tmp / "outside" / "out")})
        assert r2.status_code == 200
        assert r2.json()["authorized"] is False


def test_api_out_check_inside_source_flagged(client):
    c, root, tmp = client
    rec = root / "rec1"
    r = c.get("/api/out_check", params={
        "dir": str(rec), "path": str(rec / "sub_out")})
    assert r.status_code == 200
    d = r.json()
    assert d["authorized"] is True and d["inside_source"] is True


def test_api_symlink_escape_403(client):
    c, root, tmp = client
    outside = tmp / "outside"
    link = root / "sneaky"
    link.symlink_to(outside)
    assert c.get("/api/browse", params={"path": str(link)}).status_code == 403
    assert c.get("/api/inspect", params={"path": str(link)}).status_code == 403


def test_api_load_parent_folder_lists_children(client):
    c, root, tmp = client
    parent = root / "day1"
    make_recording(parent / "r1")
    make_recording(parent / "r2")
    r = c.post("/api/load", json={"dir": str(parent)})
    assert r.status_code == 422
    d = r.json()["detail"]
    assert [x["name"] for x in d["child_recordings"]] == ["r1", "r2"]
