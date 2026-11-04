"""GUI server tests: authorization, discovery, build job via subprocess,
and the pandas/pyarrow-free server-process guarantee."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from futil import WORLD_TS, make_tracked_artifact

BIG = {"name": "big", "box": (20, 20, 100, 100), "score": 0.95}
GAZE = [{"ts": WORLD_TS[i] + 1_000_000, "x": 50.5, "y": 50.5}
        for i in range(len(WORLD_TS))]


@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient
    from common import config as cc
    from gaze_object_fusion import server as srv
    root = tmp_path / "root"
    root.mkdir()
    tracked, trim, oids = make_tracked_artifact(root, GAZE, [BIG])
    cfg = cc.StageConfig(stage="gaze_object_fusion", port=8003,
                         browse_roots=[root],
                         state_dir=tmp_path / "state")
    return TestClient(srv.create_app(cfg),
                      base_url="http://localhost"), root, tracked


def test_browse_flags_tracked_artifacts_and_authorizes(client):
    c, root, tracked = client
    r = c.get("/api/browse", params={"path": str(root)})
    assert r.status_code == 200
    flags = {e["name"]: e["is_tracked_artifact"] for e in r.json()["entries"]}
    assert flags["tracked"] is True and flags["trim"] is False
    assert c.get("/api/browse",
                 params={"path": "/etc"}).status_code == 403
    assert c.get("/api/demo",
                 params={"path": "/etc/passwd"}).status_code == 403


def test_inspect_discovers_and_validates(client):
    c, root, tracked = client
    d = c.get("/api/inspect", params={"path": str(tracked)}).json()
    assert d["ok"] is True and d["problems"] == []
    assert d["artifact"]["objects_accepted"] == 1
    assert d["artifact"]["validation_ok"] is True
    assert d["parent"]["resolvable"] is True
    assert d["gaze"]["rows"] == len(GAZE)
    assert d["default_out"].endswith("_gaze_fusion")
    # a non-artifact folder is reported with a clear problem, not a 500
    d2 = c.get("/api/inspect", params={"path": str(root)}).json()
    assert d2["ok"] is False and d2["problems"]


def test_out_check_flags_conflicts(client):
    c, root, tracked = client
    d = c.get("/api/out_check", params={
        "artifact": str(tracked), "path": str(tracked) + "/x"}).json()
    assert d["inside_source"] is True
    d = c.get("/api/out_check", params={
        "artifact": str(tracked), "path": str(root / "newout")}).json()
    assert d["authorized"] and not d["exists"] and not d["inside_source"]


def test_build_job_end_to_end_and_no_overwrite(client):
    c, root, tracked = client
    out = root / "fusion_gui_out"
    r = c.post("/api/build", json={"artifact": str(tracked),
                                   "out": str(out), "demo": False})
    assert r.status_code == 200
    job = r.json()["job_id"]
    for _ in range(300):
        d = c.get(f"/api/build/{job}").json()
        if d["state"] != "running":
            break
        time.sleep(0.3)
    assert d["state"] == "done", d
    assert d["summary"]["counts"]["samples_total"] == len(GAZE)
    assert d["summary"]["counts"]["per_object"][0]["samples_hit_any"] \
        == len(GAZE)
    assert (out / "gaze_samples.csv").is_file()
    assert (out / "gaze_samples.parquet").is_file()
    # second build into the same destination is refused up front
    r = c.post("/api/build", json={"artifact": str(tracked),
                                   "out": str(out), "demo": False})
    assert r.status_code == 400
    assert "exists" in r.json()["detail"]


def test_server_process_never_imports_pandas_or_pyarrow():
    """Arrow-isolation policy holds for this GUI server too."""
    code = (
        "import sys; sys.path.insert(0, '.')\n"
        "assert 'pandas' not in sys.modules and 'pyarrow' not in sys.modules\n"
        "from gaze_object_fusion import server as srv\n"
        "from common import config as cc\n"
        "from pathlib import Path\n"
        "app = srv.create_app(cc.StageConfig(stage='gaze_object_fusion',"
        " port=8003, browse_roots=[Path('/tmp')],"
        " state_dir=Path('/tmp/gof_iso_state')))\n"
        "bad = [m for m in ('pandas', 'pyarrow') if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print('ISOLATION_OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code],
                       cwd=str(Path(__file__).resolve().parent.parent.parent),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ISOLATION_OK" in r.stdout
