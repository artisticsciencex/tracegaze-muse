"""Workbench tests: process isolation, lifecycle, and validated handoff.

Module lifecycle is exercised against a stdlib stand-in server
(fake_module.py) so the suite never starts SAM3, a GPU worker, or a real
encode; the REAL module servers are covered by their own suites plus the
standalone-usability tests at the bottom.
"""

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from common import config as cc
from workbench import config as wb_config
from workbench import procman
from workbench.server import create_app

PIPELINE_DIR = Path(__file__).resolve().parent.parent.parent
FAKE = Path(__file__).resolve().parent / "fake_module.py"
PY = sys.executable


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def fake_spec(key="neon_trim", *, port=None, extra=(), **over):
    port = port or free_port()
    base = dict(
        key=key, title=f"test {key}", stage=1, launch_name=f"test-{key}",
        port=port, executable=PY,
        args=[str(FAKE), "--port", str(port), "--module", key, *extra],
        cwd=PIPELINE_DIR, description="d", input_desc="i", output_desc="o",
        validator="none")
    base.update(over)
    return wb_config.ModuleSpec(**base)


def make_app(specs, tmp_path, *, roots=None, start_timeout=12.0):
    cfg = cc.StageConfig(stage="workbench", port=8000,
                         browse_roots=roots or [tmp_path],
                         state_dir=tmp_path / "state")
    app = create_app(cfg, specs, start_timeout=start_timeout)
    return TestClient(app, base_url="http://localhost"), app


@pytest.fixture()
def stopper():
    apps = []
    yield apps
    for app in apps:
        app.state.pm.shutdown()


# ------------------------------------------------------- process isolation

def test_workbench_imports_no_heavy_libraries():
    """A pristine interpreter running the workbench must not pull in
    torch, sam3, pandas or pyarrow — module crashes stay in modules."""
    code = (
        "import sys; sys.path.insert(0, '.')\n"
        "from pathlib import Path\n"
        "from workbench.server import create_app\n"
        "from workbench import config as wc\n"
        "from common import config as cc\n"
        "specs = wc.load_specs()\n"
        "app = create_app(cc.StageConfig(stage='workbench', port=8000,"
        " browse_roots=[Path('/tmp')],"
        " state_dir=Path('/tmp/wb_iso_state')), specs)\n"
        "bad = [m for m in ('torch','sam3','pandas','pyarrow')"
        " if m in sys.modules]\n"
        "assert not bad, bad\n"
        "assert 'neon_trim.server' not in sys.modules\n"
        "assert 'object_tracking.server' not in sys.modules\n"
        "print('ISOLATION_OK', len(specs))\n"
    )
    r = subprocess.run([PY, "-c", code], cwd=str(PIPELINE_DIR),
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ISOLATION_OK 3" in r.stdout


def test_specs_come_from_launch_json_not_guessed():
    specs = wb_config.load_specs()
    assert set(specs) == {"neon_trim", "object_tracking",
                          "gaze_object_fusion"}
    launch = json.loads(wb_config.LAUNCH_JSON.read_text())
    by_name = {c["name"]: c for c in launch["configurations"]}
    for key, spec in specs.items():
        conf = by_name[spec.launch_name]
        assert spec.cmd() == [conf["runtimeExecutable"], *conf["runtimeArgs"]]
        assert spec.port == conf["port"]
    assert [specs[k].port for k in wb_config.ORDER] == [8001, 8002, 8003]


def test_missing_launch_entry_is_reported(tmp_path):
    bad = tmp_path / "launch.json"
    bad.write_text(json.dumps({"configurations": [
        {"name": "neon-trim", "runtimeExecutable": "python",
         "runtimeArgs": ["x.py"], "port": 8001}]}))
    with pytest.raises(wb_config.WorkbenchConfigError,
                       match="object-tracking"):
        wb_config.load_specs(bad, tmp_path)


# ------------------------------------------------------ start / reuse / stop

def test_start_then_recognized_by_health_identity(tmp_path, stopper):
    spec = fake_spec()
    c, app = make_app({spec.key: spec}, tmp_path)
    stopper.append(app)
    assert c.get("/api/modules").json()["modules"][0]["status"] == "stopped"
    d = c.post("/api/start", json={"key": spec.key}).json()
    assert d["status"] == "ready" and d["owned"] is True
    assert d["health"]["module"] == spec.key and d["health"]["version"]
    assert d["reused"] is False


def test_running_module_is_reused_and_not_duplicated(tmp_path, stopper):
    """An already-healthy module (started outside the workbench) is
    reused, and no second process is ever spawned."""
    spec = fake_spec()
    ext = subprocess.Popen(spec.cmd(), cwd=str(spec.cwd),
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    try:
        for _ in range(80):
            if procman.probe_health(spec.health_url)[0]:
                break
            time.sleep(0.1)
        c, app = make_app({spec.key: spec}, tmp_path)
        stopper.append(app)
        d = c.post("/api/start", json={"key": spec.key}).json()
        assert d["status"] == "ready" and d["reused"] is True
        assert d["owned"] is False, "not our child — must not be claimed"
        assert app.state.pm.owned(spec.key) is None
        # repeated starts stay idempotent
        for _ in range(3):
            c.post("/api/start", json={"key": spec.key})
        assert ext.poll() is None
    finally:
        ext.terminate()
        ext.wait(timeout=10)


def test_owned_module_started_once_only(tmp_path, stopper):
    spec = fake_spec()
    c, app = make_app({spec.key: spec}, tmp_path)
    stopper.append(app)
    first = c.post("/api/start", json={"key": spec.key}).json()
    pid = first["pid"]
    for _ in range(3):
        again = c.post("/api/start", json={"key": spec.key}).json()
        assert again["pid"] == pid, "a second process was launched"
        assert again["reused"] is True


def test_wrong_process_on_port_is_reported_not_killed(tmp_path, stopper):
    port = free_port()
    spec = fake_spec(port=port)
    foreign = subprocess.Popen(
        [PY, str(FAKE), "--port", str(port), "--no-identity"],
        cwd=str(PIPELINE_DIR), stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    try:
        for _ in range(80):
            if procman.probe_health(spec.health_url)[0]:
                break
            time.sleep(0.1)
        c, app = make_app({spec.key: spec}, tmp_path)
        stopper.append(app)
        st = c.get("/api/modules").json()["modules"][0]
        assert st["status"] == "port_conflict"
        assert "not" in st["detail"] and str(port) in st["detail"]
        r = c.post("/api/start", json={"key": spec.key})
        assert r.status_code == 409
        assert "port" in r.json()["detail"].lower()
        time.sleep(0.4)
        assert foreign.poll() is None, "a foreign process was killed"
        # and it may not be stopped through the workbench either
        r = c.post("/api/stop", json={"key": spec.key})
        assert r.status_code == 409
        assert "not started by the workbench" in r.json()["detail"]
        assert foreign.poll() is None
    finally:
        foreign.terminate()
        foreign.wait(timeout=10)


def test_startup_timeout_is_reported_safely(tmp_path, stopper):
    spec = fake_spec(extra=["--delay", "30"])
    c, app = make_app({spec.key: spec}, tmp_path, start_timeout=2.0)
    stopper.append(app)
    r = c.post("/api/start", json={"key": spec.key})
    assert r.status_code == 502
    assert "did not become healthy" in r.json()["detail"]
    assert c.get("/api/health").json()["ok"] is True     # workbench alive


def test_child_failure_is_reported_with_diagnostics(tmp_path, stopper):
    spec = fake_spec(extra=["--exit-code", "3"])
    c, app = make_app({spec.key: spec}, tmp_path, start_timeout=8.0)
    stopper.append(app)
    r = c.post("/api/start", json={"key": spec.key})
    assert r.status_code == 502
    assert "exited during startup" in r.json()["detail"]
    logs = c.get(f"/api/logs/{spec.key}").json()["lines"]
    assert any("refusing to start" in ln for ln in logs)
    assert len(logs) <= procman.LOG_LINES
    assert c.get("/api/modules").json()["modules"][0]["status"] == "failed"


def test_workbench_survives_module_crash(tmp_path, stopper):
    spec = fake_spec()
    c, app = make_app({spec.key: spec}, tmp_path)
    stopper.append(app)
    c.post("/api/start", json={"key": spec.key})
    proc = app.state.pm.owned(spec.key)
    proc.kill()
    proc.wait(timeout=10)
    for _ in range(40):
        st = c.get("/api/modules").json()["modules"][0]
        if st["status"] != "ready":
            break
        time.sleep(0.1)
    assert st["status"] in ("failed", "stopped")
    assert c.get("/api/health").json()["module"] == "workbench"
    # and the workbench can start it again afterwards
    assert c.post("/api/start", json={"key": spec.key}).json()["status"] \
        == "ready"


def test_only_owned_children_can_be_stopped(tmp_path, stopper):
    spec = fake_spec()
    c, app = make_app({spec.key: spec}, tmp_path)
    stopper.append(app)
    c.post("/api/start", json={"key": spec.key})
    assert c.post("/api/stop", json={"key": spec.key}).json()["status"] \
        == "stopped"
    assert app.state.pm.owned(spec.key) is None


def test_busy_module_needs_explicit_force_to_stop(tmp_path, stopper):
    spec = fake_spec(extra=["--busy"])
    c, app = make_app({spec.key: spec}, tmp_path)
    stopper.append(app)
    st = c.post("/api/start", json={"key": spec.key}).json()
    assert st["status"] == "busy"
    r = c.post("/api/stop", json={"key": spec.key})
    assert r.status_code == 409 and "active job" in r.json()["detail"]
    assert app.state.pm.owned(spec.key) is not None
    assert c.post("/api/stop",
                  json={"key": spec.key, "force": True}).status_code == 200


# --------------------------------------------------------------- handoff

def _fusion_like(tmp_path, verdict_payload, *, validator="fusable"):
    return fake_spec(key="gaze_object_fusion", validator=validator,
                     input_slot="tracked_output",
                     output_slot="fusion_output",
                     extra=["--inspect-json", json.dumps(verdict_payload)])


def test_handoff_encodes_path_and_uses_destination_validation(tmp_path,
                                                              stopper):
    art = tmp_path / "rec with space_tracked_objects"
    art.mkdir()
    spec = _fusion_like(tmp_path, {"ok": True, "problems": []})
    c, app = make_app({spec.key: spec}, tmp_path)
    stopper.append(app)
    r = c.get("/go", params={"to": spec.key, "handoff": str(art)},
              follow_redirects=False)
    assert r.status_code == 303
    loc = r.headers["location"]
    assert loc.startswith(spec.url)
    assert "handoff=" in loc and "%20" in loc, "path must be URL-encoded"
    assert "home=" in loc
    # the accepted path is remembered for the stage that consumes it
    state = c.get("/api/state").json()
    assert state["tracked_output"] == str(art)


def test_handoff_rejected_when_destination_validation_fails(tmp_path,
                                                            stopper):
    art = tmp_path / "not_really_tracked"
    art.mkdir()
    spec = _fusion_like(tmp_path,
                        {"ok": False, "problems": ["no tracking_manifest"]})
    c, app = make_app({spec.key: spec}, tmp_path)
    stopper.append(app)
    r = c.get("/go", params={"to": spec.key, "handoff": str(art)},
              follow_redirects=False)
    assert r.status_code == 409
    assert "no tracking_manifest" in r.text
    assert "tracked_output" not in c.get("/api/state").json()


def test_unauthorized_handoff_path_is_rejected(tmp_path, stopper):
    spec = _fusion_like(tmp_path, {"ok": True})
    c, app = make_app({spec.key: spec}, tmp_path, roots=[tmp_path / "root"])
    (tmp_path / "root").mkdir()
    stopper.append(app)
    for bad in ("/etc", str(tmp_path / "outside"), "/etc/../etc/passwd"):
        r = c.get("/go", params={"to": spec.key, "handoff": bad},
                  follow_redirects=False)
        assert r.status_code == 403, bad
    # ... and the module was never even started for a rejected path
    assert app.state.pm.owned(spec.key) is None


def test_missing_handoff_path_reported_without_starting_navigation(
        tmp_path, stopper):
    spec = _fusion_like(tmp_path, {"ok": True})
    c, app = make_app({spec.key: spec}, tmp_path)
    stopper.append(app)
    r = c.get("/go", params={"to": spec.key,
                             "handoff": str(tmp_path / "gone")},
              follow_redirects=False)
    assert r.status_code == 409 and "does not exist" in r.text


def test_go_home_records_final_result(tmp_path, stopper):
    out = tmp_path / "rec_gaze_fusion_v3"
    out.mkdir()
    spec = fake_spec()
    c, app = make_app({spec.key: spec}, tmp_path)
    stopper.append(app)
    r = c.get("/go", params={"to": "home", "handoff": str(out),
                             "slot": "fusion_output"},
              follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert c.get("/api/state").json()["fusion_output"] == str(out)


def test_module_without_handoff_opens_plainly(tmp_path, stopper):
    spec = fake_spec()
    c, app = make_app({spec.key: spec}, tmp_path)
    stopper.append(app)
    r = c.get("/go", params={"to": spec.key}, follow_redirects=False)
    assert r.status_code == 303
    assert "handoff=" not in r.headers["location"]


def test_verdict_interpretation_matrix():
    v = wb_config.verdict
    assert v("recording", {"is_recording": True})[0] is True
    assert v("recording", {"is_recording": False, "problems": ["x"]})[0] \
        is False
    assert v("trackable", {"kind": "trimmed_artifact"})[0] is True
    assert v("trackable", {"kind": "tracked_objects"})[0] is False
    assert v("trackable", {"kind": None})[0] is False
    assert v("fusable", {"ok": True})[0] is True
    assert v("fusable", {"ok": False, "problems": ["p"]})[0] is False


# ------------------------------------------- standalone modules unaffected

@pytest.mark.parametrize("mod,key", [
    ("neon_trim", "neon_trim"),
    ("object_tracking", "object_tracking"),
    ("gaze_object_fusion", "gaze_object_fusion"),
])
def test_each_module_still_works_standalone(mod, key, tmp_path):
    """Direct use of every module (no workbench, no handoff) is unchanged,
    and each now answers with its health identity."""
    import importlib
    srv = importlib.import_module(f"{mod}.server")
    root = tmp_path / mod
    root.mkdir()
    if mod == "neon_trim":
        from neon_trim import config as nt_config
        cfg = nt_config.AppConfig(browse_roots=[root], port=8001,
                                  state_dir=tmp_path / f"{mod}_state")
    else:
        cfg = cc.StageConfig(stage=mod, port=8002, browse_roots=[root],
                             state_dir=tmp_path / f"{mod}_state")
    c = TestClient(srv.create_app(cfg), base_url="http://localhost")
    h = c.get("/api/health").json()
    assert h["ok"] is True and h["module"] == key
    assert h["version"] and h["busy"] is False
    page = c.get("/")
    assert page.status_code == 200
    assert "Pipeline Home" in page.text


def test_module_pages_use_same_tab_navigation():
    """Workbench + module pages must not auto-open browser tabs; stage
    navigation is same-tab and every module links back home."""
    pages = {
        "workbench": Path(__file__).resolve().parent.parent
        / "static" / "index.html",
        "neon_trim": PIPELINE_DIR / "neon_trim" / "static" / "index.html",
        "object_tracking": PIPELINE_DIR / "object_tracking" / "static"
        / "index.html",
        "gaze_object_fusion": PIPELINE_DIR / "gaze_object_fusion" / "static"
        / "index.html",
    }
    for name, p in pages.items():
        text = p.read_text()
        assert "window.open" not in text, f"{name} opens a browser window"
        if name != "workbench":
            assert 'id="pipelineHome"' in text, f"{name}: no home control"
            assert "← Pipeline Home" in text
            assert "/go?to=" in text, f"{name}: no next-stage handoff link"
            assert 'WB_Q.get("handoff")' in text, f"{name}: no handoff read"
    wb = pages["workbench"].read_text()
    assert "/go?to=" in wb and "location.href=" in wb
    # the only new-tab affordance anywhere is the explicit, user-clicked
    # "open demo video in a new tab" action in the fusion page
    assert pages["gaze_object_fusion"].read_text().count('target="_blank"') \
        == 1
