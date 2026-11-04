"""Frame-presentation regressions: exact-still endpoint, time-domain map,
and the JS FrameSync race suite (run under node).
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from object_tracking.timemap import FrameTimeMap

from util import OFFSETS_MS, T0, WORLD_TS, make_csv_export

NODE = shutil.which("node")


# --------------------------------------------------------------- timemap

def _fixture_map(offset_ms=16.0):
    # VFR pts WITH a nonzero container/world offset (like the real trim)
    pts = [offset_ms / 1e3 + ms / 1e3 for ms in OFFSETS_MS]
    return FrameTimeMap(pts, WORLD_TS, sync_ok=True), pts


def test_timemap_roundtrip_all_domains():
    tm, pts = _fixture_map()
    for i in range(tm.n_frames):
        assert tm.pts_for_frame(i) == pts[i]
        assert tm.frame_for_pts(pts[i]) == i
        assert tm.media_time_for_frame(i) == pts[i]
        assert tm.frame_for_media_time(pts[i]) == i
        assert tm.world_ns_for_frame(i) == WORLD_TS[i]
        assert tm.frame_for_world_ns(WORLD_TS[i]) == i
        assert tm.frame_for_world_ns(WORLD_TS[i] + 1_000_000) == i


def test_timemap_preserves_constant_offset():
    tm, pts = _fixture_map(offset_ms=16.0)
    assert tm.video_world_offset_s() == pytest.approx(0.016)
    # a mediaTime in the video domain maps by NEAREST pts — never through
    # world-relative time (mixing domains was a root-cause candidate)
    assert tm.frame_for_media_time(0.016) == 0
    assert tm.frame_for_media_time(0.0) is None      # before frame 0 - tol


def test_timemap_nearest_tolerance_vfr():
    tm, pts = _fixture_map()
    # smallest gap is 14 ms -> tolerance 7 ms
    assert tm.snap_tol_s == pytest.approx(0.007)
    assert tm.frame_for_media_time(pts[4] + 0.004) == 4
    # 9 ms past frame 4 is 5 ms before frame 5 -> nearest = 5 (in tolerance)
    assert tm.frame_for_media_time(pts[4] + 0.009) == 5
    # the middle of the 50 ms head gap is >7 ms from BOTH neighbours: refuse
    assert tm.frame_for_media_time(pts[0] + 0.020) is None
    assert tm.frame_for_pts(pts[0] + 0.020) == 0            # floor still maps


def test_timemap_video_only_recording():
    tm = FrameTimeMap([0.0, 0.033, 0.066], world_ts_ns=None, sync_ok=False)
    assert tm.world_ns_for_frame(1) is None
    assert tm.frame_for_world_ns(T0) is None


def test_timemap_rejects_bad_tables():
    with pytest.raises(ValueError):
        FrameTimeMap([])
    with pytest.raises(ValueError):
        FrameTimeMap([0.0, 0.033, 0.033])   # not strictly increasing


# --------------------------------------------------- /api/frame endpoint

@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient
    from common import config as cc
    from object_tracking import server as srv
    root = tmp_path / "root"
    root.mkdir()
    make_csv_export(root, "rec1")
    cfg = cc.StageConfig(stage="object_tracking", port=8002,
                         browse_roots=[root], state_dir=tmp_path / "state")
    c = TestClient(srv.create_app(cfg), base_url="http://localhost")
    r = c.post("/api/load", json={"dir": str(root / "rec1")})
    assert r.status_code == 200
    return c, str(root / "rec1"), tmp_path


def _decode_jpeg_mean(data: bytes) -> float:
    import io

    import av
    with av.open(io.BytesIO(data), format="mjpeg") as c:
        for frame in c.decode(video=0):
            return float(frame.to_ndarray(format="gray").mean())
    raise AssertionError("no frame in jpeg")


def test_frame_endpoint_exact_by_presentation_order(client):
    """Synthetic frames are distinct solid greys ((i*23)%255): the returned
    JPEG must BE the requested frame, not a neighbour."""
    c, d, _ = client
    for idx in (0, 1, 5, 9, 4, 2):          # includes backward jumps
        r = c.get("/api/frame", params={"dir": d, "frame_index": idx})
        assert r.status_code == 200
        assert r.headers["X-Frame-Index"] == str(idx)
        expected_pts = (OFFSETS_MS[idx]) / 1e3
        assert abs(float(r.headers["X-Frame-Media-Time"])
                   - expected_pts) < 2e-3
        assert r.headers["X-World-Timestamp-Ns"] == str(WORLD_TS[idx])
        mean = _decode_jpeg_mean(r.content)
        assert abs(mean - (idx * 23) % 255) < 6, \
            f"frame {idx}: pixel value {mean} != {(idx * 23) % 255}"


def test_frame_endpoint_cache_and_bounds(client):
    c, d, _ = client
    r1 = c.get("/api/frame", params={"dir": d, "frame_index": 3})
    assert r1.headers["X-Frame-Cache"] == "miss"
    r2 = c.get("/api/frame", params={"dir": d, "frame_index": 3})
    assert r2.headers["X-Frame-Cache"] == "hit"
    assert r2.content == r1.content
    assert c.get("/api/frame",
                 params={"dir": d, "frame_index": 999}).status_code == 400
    assert c.get("/api/frame",
                 params={"dir": d, "frame_index": -1}).status_code == 400


def test_frame_endpoint_requires_authorized_loaded_dir(client):
    c, d, tmp = client
    outside = tmp / "outside"
    outside.mkdir()
    assert c.get("/api/frame", params={
        "dir": str(outside), "frame_index": 0}).status_code == 403


def test_frame_endpoint_sequential_and_random_consistency(client):
    """Stepping 0..9 then jumping around returns identical bytes for
    revisited frames (decode-position reuse must not corrupt identity)."""
    c, d, _ = client
    first = {}
    for idx in list(range(10)) + [0, 7, 2, 9, 35 % 10]:
        r = c.get("/api/frame", params={"dir": d, "frame_index": idx})
        assert r.status_code == 200
        if idx in first:
            assert r.content == first[idx]
        else:
            first[idx] = r.content
        mean = _decode_jpeg_mean(r.content)
        assert abs(mean - (idx * 23) % 255) < 6


# ------------------------------------------------------ JS race suite

@pytest.mark.skipif(NODE is None, reason="node not available")
def test_framesync_race_suite_under_node():
    test_js = Path(__file__).parent / "framesync_races.test.js"
    r = subprocess.run([NODE, str(test_js)], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "FRAMESYNC_RACES_OK" in r.stdout
