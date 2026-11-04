"""Small local GUI server for the gaze-fusion stage (port 8003).

Pure ORCHESTRATION: discovery/validation read manifests as plain JSON and
count CSV lines; the actual fusion + demo rendering run through the
existing CLI (`python -m gaze_object_fusion`) in a SUBPROCESS. This keeps
the long-lived FastAPI process pandas/pyarrow-free (the project's
Arrow-isolation policy) and guarantees the GUI exercises exactly the same
code path as the command line.

    ~/.venvs/neon-trim/bin/python pipeline/gaze_object_fusion/server.py
        → http://127.0.0.1:8003
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import signal
import subprocess
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from . import __version__
    from .cli import default_out_dir
except ImportError:                       # executed directly as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gaze_object_fusion import __version__
    from gaze_object_fusion.cli import default_out_dir

from common import config as common_config
from common import fsbrowse

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = APP_DIR.parent
DEFAULT_PORT = 8003
PROGRESS_RE = re.compile(r"^\[\s*([0-9.]+)%\]\s*(.+)$")
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
CHUNK = 1 << 20


class BuildBody(BaseModel):
    artifact: str
    out: str
    demo: bool = True


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception as e:
        raise HTTPException(400, f"cannot read {p.name}: {e}")


def _count_csv_rows(p: Path) -> int:
    with open(p) as f:
        return max(0, sum(1 for _ in f) - 1)


def inspect_tracked_dir(rp: Path) -> dict:
    """Pandas-free discovery + validation summary of a tracked artifact and
    its parent trimmed recording / gaze data."""
    out = {"path": str(rp), "ok": False, "problems": [], "artifact": None,
           "parent": None, "gaze": None}

    mf = rp / "tracking_manifest.json"
    if not mf.is_file():
        out["problems"].append("no tracking_manifest.json here — choose a "
                               "published tracked_objects artifact folder")
        return out
    m = _read_json(mf)
    if m.get("artifact_type") != "tracked_objects":
        out["problems"].append(
            f"artifact_type is {m.get('artifact_type')!r}, expected "
            f"tracked_objects")
        return out
    validation_ok = bool(m.get("validation")) and \
        all(v.get("ok") for v in m.get("validation", []))
    objs = [{"object_id": o.get("object_id"), "name": o.get("name"),
             "review_status": o.get("review_status"),
             "color": o.get("color")}
            for o in m.get("objects", [])]
    out["artifact"] = {
        "artifact_id": m.get("artifact_id"),
        "created_utc": m.get("created_utc"),
        "n_frames": m.get("video", {}).get("n_frames"),
        "size": f"{m.get('video', {}).get('width')}x"
                f"{m.get('video', {}).get('height')}",
        "objects": objs,
        "objects_accepted": m.get("counts", {}).get("objects_accepted"),
        "table_rows": m.get("counts", {}).get("table_rows"),
        "validation_ok": validation_ok,
        "gaze_compatible":
            m.get("synchronization", {}).get("gaze_compatible"),
    }
    if not validation_ok:
        out["problems"].append("tracked artifact validation record is not "
                               "all-ok")
    if not out["artifact"]["gaze_compatible"]:
        out["problems"].append("tracked artifact is not gaze-compatible — "
                               "nothing to fuse")

    parent = m.get("parent", {})
    pdir = Path(parent.get("source_dir", ""))
    pinfo = {"source_dir": str(pdir), "source_type":
             parent.get("source_type"), "resolvable": pdir.is_dir()}
    out["parent"] = pinfo
    # both source systems provide a trimmed parent with a validated
    # interval; neither masquerades as the other
    supported = ("trimmed_artifact", "trimmed_aria_recording")
    if parent.get("source_type") not in supported:
        out["problems"].append(
            f"parent source_type is {parent.get('source_type')!r}; fusion "
            f"needs a trimmed recording parent ({', '.join(supported)})")
    elif not pdir.is_dir():
        out["problems"].append(f"parent recording folder not found: {pdir}")
    else:
        tm = pdir / "trim_manifest.json"
        if not tm.is_file():
            out["problems"].append("parent has no trim_manifest.json")
        else:
            t = _read_json(tm)
            sel = t.get("selection", {})
            pinfo["trim_artifact_id"] = t.get("artifact_id")
            pinfo["interval_ok"] = (
                isinstance(sel.get("absolute_start_ns"), int)
                and isinstance(sel.get("absolute_end_boundary_ns"), int))
            if not pinfo["interval_ok"]:
                out["problems"].append("trim manifest lacks the validated "
                                       "interval boundaries")
            if t.get("artifact_id") != m.get("parent_artifact",
                                             {}).get("artifact_id"):
                out["problems"].append("parent artifact id mismatch between "
                                       "trim and tracking manifests")
        gaze_p = pdir / (parent.get("gaze") or "gaze.csv")
        world_p = pdir / (parent.get("world_timestamps")
                          or "world_timestamps.csv")
        ginfo = {"gaze_csv": gaze_p.name, "exists": gaze_p.is_file()}
        if gaze_p.is_file():
            ginfo["rows"] = _count_csv_rows(gaze_p)
        else:
            out["problems"].append("parent gaze.csv not found")
        if world_p.is_file():
            ginfo["world_rows"] = _count_csv_rows(world_p)
            n = out["artifact"]["n_frames"]
            if n is not None and ginfo["world_rows"] != n:
                out["problems"].append(
                    f"world rows {ginfo['world_rows']} != frames {n}")
        else:
            out["problems"].append("parent world_timestamps.csv not found")
        out["gaze"] = ginfo

    out["ok"] = not out["problems"]
    if out["ok"]:
        out["default_out"] = str(default_out_dir(rp))
    return out


def create_app(cfg: common_config.StageConfig) -> FastAPI:
    auth = fsbrowse.PathAuthorizer(cfg.browse_roots)
    recents = fsbrowse.RecentsStore(cfg.state_dir)
    JOBS: dict = {}
    JOBS_LOCK = threading.Lock()

    @asynccontextmanager
    async def _lifespan(app):
        yield
        with JOBS_LOCK:
            jobs = list(JOBS.values())
        for j in jobs:
            p = j.get("proc")
            if j["state"] == "running" and p is not None \
                    and p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    app = FastAPI(title="gaze_object_fusion", lifespan=_lifespan)
    app.state.cfg = cfg

    loopback = False
    try:
        loopback = ipaddress.ip_address(cfg.host).is_loopback
    except ValueError:
        loopback = cfg.host in ("localhost",)
    if loopback:
        from starlette.middleware.trustedhost import TrustedHostMiddleware
        app.add_middleware(TrustedHostMiddleware,
                           allowed_hosts=["localhost", "127.0.0.1", "[::1]"])

    def authorized(raw: str) -> Path:
        try:
            return auth.resolve(raw)
        except fsbrowse.PathNotAllowed as e:
            raise HTTPException(403, str(e))

    # ---------------------------------------------------------------- pages

    @app.get("/")
    def index():
        return FileResponse(APP_DIR / "static" / "index.html")

    @app.get("/api/config")
    def api_config():
        return {"roots": [str(r) for r in auth.roots],
                "version": __version__}

    # ------------------------------------------------------------- browsing

    @app.get("/api/browse")
    def api_browse(path: str = ""):
        if not path:
            if len(auth.roots) == 1:
                path = str(auth.roots[0])
            else:
                return {"path": None, "root": None, "parent": None,
                        "breadcrumbs": [],
                        "entries": [{"name": str(r), "path": str(r),
                                     "is_dir": True, "inaccessible": False}
                                    for r in auth.roots],
                        "error": None}
        rp = authorized(path)
        listing = fsbrowse.list_dir(auth, rp).as_dict()
        for e in listing["entries"]:
            e["is_tracked_artifact"] = (
                e["is_dir"]
                and (Path(e["path"]) / "tracking_manifest.json").is_file())
        return listing

    @app.get("/api/inspect")
    def api_inspect(path: str):
        rp = authorized(path)
        info = inspect_tracked_dir(rp)
        if info["ok"]:
            recents.add_recording(str(rp))
        return info

    @app.get("/api/recents")
    def api_recents():
        out = []
        for p in recents.recent_recordings():
            if auth.is_allowed(p) and Path(p).is_dir():
                out.append({"path": p, "name": Path(p).name})
        return {"recents": out}

    @app.get("/api/out_check")
    def api_out_check(artifact: str, path: str):
        src = authorized(artifact)
        try:
            rp = auth.resolve(path)
        except fsbrowse.PathNotAllowed as e:
            return {"resolved": path, "authorized": False,
                    "reason": str(e), "exists": None, "inside_source": None}
        inside = (rp == src) or (src in rp.parents)
        return {"resolved": str(rp), "authorized": True, "reason": None,
                "exists": rp.exists(), "inside_source": inside,
                "parent_exists": rp.parent.exists()}

    # ---------------------------------------------------------------- build

    @app.post("/api/build")
    def api_build(body: BuildBody):
        artifact = authorized(body.artifact)
        out_dir = authorized(body.out)
        if out_dir.exists():
            raise HTTPException(400, f"destination already exists: "
                                     f"{out_dir} — choose a new folder")
        info = inspect_tracked_dir(artifact)
        if not info["ok"]:
            raise HTTPException(400, "artifact not buildable:\n- "
                                + "\n- ".join(info["problems"]))
        job_id = uuid.uuid4().hex[:12]
        summary_path = cfg.state_dir / f"summary_{job_id}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "gaze_object_fusion",
               "--artifact", str(artifact), "--out", str(out_dir),
               "--summary-json", str(summary_path)]
        if not body.demo:
            cmd.append("--no-demo")
        job = {"id": job_id, "state": "running", "progress": 0.0,
               "phase": "starting", "error": None, "summary": None,
               "out": str(out_dir), "proc": None}
        with JOBS_LOCK:
            JOBS[job_id] = job

        def runner():
            try:
                proc = subprocess.Popen(
                    cmd, cwd=str(PIPELINE_DIR), stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True,
                    start_new_session=True)
                job["proc"] = proc
                for line in proc.stdout:
                    mt = PROGRESS_RE.match(line.strip())
                    if mt:
                        job["progress"] = float(mt.group(1)) / 100.0
                        job["phase"] = mt.group(2)
                rc = proc.wait()
                stderr_tail = proc.stderr.read()[-800:]
                if job["state"] == "cancelled":
                    return
                if rc != 0:
                    job["state"] = "error"
                    job["error"] = (stderr_tail.strip()
                                    or f"fusion subprocess exited {rc}")
                    return
                job["summary"] = json.loads(summary_path.read_text())
                job["progress"] = 1.0
                job["phase"] = "done"
                job["state"] = "done"
            except Exception as e:              # noqa: BLE001
                job["state"] = "error"
                job["error"] = str(e)
            finally:
                summary_path.unlink(missing_ok=True)

        t = threading.Thread(target=runner, daemon=True)
        job["thread"] = t
        t.start()
        return {"job_id": job_id}

    @app.get("/api/build/{job_id}")
    def api_build_status(job_id: str):
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return {k: job[k] for k in ("id", "state", "progress", "phase",
                                    "error", "summary", "out")}

    @app.post("/api/build/{job_id}/cancel")
    def api_build_cancel(job_id: str):
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        p = job.get("proc")
        if job["state"] == "running" and p is not None and p.poll() is None:
            job["state"] = "cancelled"
            job["error"] = "cancelled by user"
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return {"ok": True}

    # ----------------------------------------------------------- demo video

    @app.get("/api/demo")
    def api_demo(path: str):
        rp = authorized(path)
        if not rp.is_file() or rp.suffix != ".mp4":
            raise HTTPException(404, "demo video not found")
        # +faststart mp4; starlette FileResponse serves Range requests
        return FileResponse(rp, media_type="video/mp4")

    @app.get("/api/health")
    def api_health():
        # identity + liveness for the pipeline workbench (no table
        # libraries are imported in this process at all)
        busy = any(j["state"] == "running" for j in JOBS.values())
        return {"ok": True, "module": "gaze_object_fusion",
                "version": __version__, "busy": busy,
                "stage": "gaze_object_fusion"}

    return app


if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="gaze-fusion GUI server")
    ap.add_argument("--host", default=common_config.DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--root", action="append", default=None, metavar="DIR")
    ap.add_argument("--allow-lan", action="store_true")
    args = ap.parse_args()
    cfg = common_config.load_stage_config(
        "gaze_object_fusion", "GAZE_FUSION", DEFAULT_PORT,
        extra_roots=args.root, port=args.port, host=args.host)
    application = create_app(cfg)
    lb = False
    try:
        lb = ipaddress.ip_address(cfg.host).is_loopback
    except ValueError:
        lb = cfg.host in ("localhost",)
    if not lb and not args.allow_lan:
        ap.error(f"refusing to bind {cfg.host} without --allow-lan")
    print(f"gaze_object_fusion GUI  →  http://{cfg.host}:{cfg.port}")
    print("browse roots:", ", ".join(str(r) for r in cfg.browse_roots))
    uvicorn.run(application, host=cfg.host, port=cfg.port,
                log_level="warning")
