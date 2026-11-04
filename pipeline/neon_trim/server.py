"""FastAPI server for the NEON trim GUI.

Run (venv with fastapi/uvicorn/av/imageio-ffmpeg):

    python pipeline/neon_trim/server.py                    # http://127.0.0.1:8001
    NEON_TRIM_ROOTS=/data/neon:/mnt/other python server.py --port 8002
    python server.py --root /extra/recordings --host 0.0.0.0 --allow-lan

The server is a thin HTTP layer: path authorization lives in fsbrowse,
recording inspection in intake, sync/trim/export in core/media/export, and
the artifact contract in artifact. Single-user local tool: loaded recordings
and export jobs are in-memory. The server never writes inside a source
recording directory, and EVERY path it touches (browse, inspect, load,
stream, output) is canonicalized and checked against the configured roots.
"""

from __future__ import annotations

import ipaddress
import json
import re
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from . import artifact as artifact_mod
    from . import config as config_mod
    from . import core, fsbrowse, intake, media
    from . import export as export_mod
except ImportError:  # executed directly as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from neon_trim import artifact as artifact_mod
    from neon_trim import config as config_mod
    from neon_trim import core, fsbrowse, intake, media
    from neon_trim import export as export_mod

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
CHUNK = 1 << 20
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class LoadBody(BaseModel):
    dir: str


class ExportBody(BaseModel):
    dir: str
    start_frame: int
    end_frame: int
    out_dir: str


class AriaPrepareBody(BaseModel):
    dir: str
    out_dir: str


def _manifest_for_js(manifest):
    """Copy with absolute-ns ints stringified (they exceed JS 2^53)."""
    if manifest is None:
        return None
    m = json.loads(json.dumps(manifest))  # deep copy
    sel, sync = m.get("selection", {}), m.get("synchronization", {})
    for d, k in ((sel, "absolute_start_ns"), (sel, "absolute_end_boundary_ns"),
                 (sync, "recording_end_ns")):
        if k in d and d[k] is not None:
            d[k] = str(d[k])
    return m


def create_app(cfg: config_mod.AppConfig) -> FastAPI:
    auth = fsbrowse.PathAuthorizer(cfg.browse_roots)
    inspector = intake.RecordingInspector()
    recents = fsbrowse.RecentsStore(cfg.state_dir)
    RECS: dict = {}          # canonical dir -> LoadedRecording
    RECS_LOCK = threading.Lock()
    JOBS: dict = {}          # job_id -> job state dict
    JOBS_LOCK = threading.Lock()

    @asynccontextmanager
    async def _lifespan(app):
        yield
        # graceful shutdown: cancel running exports so ffmpeg is killed and
        # temp dirs are cleaned up instead of leaking an orphaned encode
        with JOBS_LOCK:
            jobs = list(JOBS.values())
        for j in jobs:
            if j["state"] == "running":
                j["cancel"].set()
        for j in jobs:
            t = j.get("thread")
            if t is not None and t.is_alive():
                t.join(timeout=15)

    app = FastAPI(title="neon_trim", lifespan=_lifespan)
    app.state.cfg = cfg
    app.state.recs = RECS
    app.state.jobs = JOBS

    # DNS-rebinding defense must protect EVERY consumer of this app object
    # (python server.py AND `uvicorn neon_trim.server:app`), so it lives here,
    # keyed off the configured host — not only in the __main__ launcher.
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
        """Canonicalize + authorize or raise 403 (uniform error either way)."""
        try:
            return auth.resolve(raw)
        except fsbrowse.PathNotAllowed as e:
            raise HTTPException(403, str(e))

    def get_rec(d: str):
        rec = RECS.get(str(authorized(d)))
        if rec is None:
            raise HTTPException(404, "recording not loaded — POST /api/load first")
        return rec

    # ---------------------------------------------------------------- pages

    @app.get("/")
    def index():
        return FileResponse(APP_DIR / "static" / "index.html")

    @app.get("/api/config")
    def api_config():
        return {"roots": [str(r) for r in auth.roots],
                "version": export_mod.__version__}

    # ------------------------------------------------------------- browsing

    @app.get("/api/browse")
    def api_browse(path: str = ""):
        if not path:
            if len(auth.roots) == 1:
                path = str(auth.roots[0])
            else:
                # synthetic top level listing the configured roots
                return {"path": None, "root": None, "parent": None,
                        "breadcrumbs": [],
                        "entries": [{"name": str(r), "path": str(r),
                                     "is_dir": True,
                                     "is_recording": intake.looks_like_recording(r),
                                     "inaccessible": False}
                                    for r in auth.roots],
                        "error": None}
        rp = authorized(path)
        from common import recordings as rec_dispatch

        def classify(p):
            # cheap structural badge across BOTH source systems
            aria = rec_dispatch.classify_aria_folder(p)
            if aria:
                return aria
            return "recording" if intake.looks_like_recording(p) else None

        listing = fsbrowse.list_dir(auth, rp, classifier=classify).as_dict()
        for e in listing["entries"]:
            k = e.get("kind")
            e["is_aria"] = bool(k) and (k in rec_dispatch.ARIA_KIND_LABELS)
            if e["is_aria"]:
                e["kind_label"] = rec_dispatch.ARIA_KIND_LABELS[k]
        return listing

    @app.get("/api/inspect")
    def api_inspect(path: str):
        rp = authorized(path)
        from common import recordings as rec_dispatch
        aria_kind = rec_dispatch.classify_aria_folder(rp)
        if aria_kind:
            from aria_source.detect import (ARIA_LABEL, inspect_aria_folder,
                                            validate_against_receipt)
            out = {"path": str(rp), "source_system": "project_aria",
                   "aria_kind": aria_kind,
                   "kind_label": rec_dispatch.ARIA_KIND_LABELS.get(
                       aria_kind, ARIA_LABEL),
                   "problems": [], "child_recordings": [],
                   # a prepared/trimmed Aria artifact is directly loadable;
                   # a raw download needs preparation first
                   "is_recording": aria_kind != "aria_download",
                   "needs_preparation": aria_kind == "aria_download"}
            if aria_kind == "aria_download":
                pkg = inspect_aria_folder(rp)
                out["aria"] = pkg.as_dict()
                out["kind_label"] = pkg.label
                out["problems"] = list(pkg.problems)
                out["receipt_validation"] = validate_against_receipt(pkg)
            else:
                out["aria"] = rec_dispatch.read_manifest_type(rp)[2].get(
                    "recording", {})
            return out
        return inspector.inspect(rp).as_dict()

    @app.post("/api/aria/prepare")
    def api_aria_prepare(body: AriaPrepareBody):
        """Prepare a Project Aria download folder (isolated helper)."""
        rp = authorized(body.dir)
        out_dir = authorized(body.out_dir)
        if out_dir == rp or rp in out_dir.parents:
            raise HTTPException(409, "output directory must be outside the "
                                     "Aria download folder")
        if out_dir.exists():
            raise HTTPException(409, f"output directory already exists: "
                                     f"{out_dir}")
        from aria_source import config as aria_config
        if not aria_config.aria_env_available():
            raise HTTPException(
                503, f"the Aria interpreter is not available at "
                     f"{aria_config.aria_python()} — install "
                     f"projectaria-tools there or set "
                     f"{aria_config.ENV_PYTHON}")
        job_id = uuid.uuid4().hex[:12]
        cancel = threading.Event()
        job = {"state": "running", "phase": "starting", "progress": 0.0,
               "error": None, "manifest": None, "cancel": cancel,
               "thread": None, "out_dir": str(out_dir)}
        with JOBS_LOCK:
            JOBS[job_id] = job

        def worker():
            from aria_source.prepare import prepare_aria_recording
            try:
                manifest = prepare_aria_recording(
                    rp, out_dir, cancel_event=cancel,
                    path_guard=auth.is_allowed,
                    progress_cb=lambda ph, f: job.update(
                        phase=ph, progress=round(float(f), 4)))
            except Exception as e:                       # noqa: BLE001
                # a helper crash/segfault lands here as a clear message;
                # the server itself is unaffected (native code never ran
                # inside this process)
                job["state"] = ("cancelled" if cancel.is_set()
                                else "failed")
                job["error"] = f"{type(e).__name__}: {e}"
                return
            job["manifest"] = manifest
            job["phase"] = "done"
            job["progress"] = 1.0
            job["state"] = "done"

        t = threading.Thread(target=worker, daemon=True)
        job["thread"] = t
        t.start()
        return {"job_id": job_id}

    def _valid_last_out_parent():
        lop = recents.last_out_parent()
        return lop if lop and auth.is_allowed(lop) else None

    @app.get("/api/recents")
    def api_recents():
        out = []
        for p in recents.recent_recordings():
            if not auth.is_allowed(p):
                continue                    # roots may have changed
            pp = Path(p)
            out.append({"path": p, "name": pp.name,
                        "exists": pp.is_dir(),
                        "is_recording": pp.is_dir()
                        and intake.looks_like_recording(pp)})
        return {"recents": out, "last_out_parent": _valid_last_out_parent()}

    @app.get("/api/out_check")
    def api_out_check(dir: str, path: str):
        """Pre-flight for a chosen output destination (display + warnings)."""
        src = authorized(dir)
        try:
            rp = auth.resolve(path)
        except fsbrowse.PathNotAllowed as e:
            return {"resolved": path, "authorized": False, "reason": str(e),
                    "exists": None, "inside_source": None}
        inside = (rp == src) or (src in rp.parents)
        return {"resolved": str(rp), "authorized": True, "reason": None,
                "exists": rp.exists(), "inside_source": inside,
                "parent_exists": rp.parent.exists()}

    # ------------------------------------------------------------ recording

    @app.post("/api/load")
    def api_load(body: LoadBody):
        rp = authorized(body.dir)
        # ---- Project Aria dispatch (own artifact types, own manifests) ----
        from common import recordings as rec_dispatch
        aria_kind = rec_dispatch.classify_aria_folder(rp)
        if aria_kind == "aria_download":
            raise HTTPException(422, {
                "message": "Project Aria recording — preparation required",
                "problems": ["This is a Project Aria download folder. Use "
                             "“Prepare Aria recording” to build a prepared "
                             "artifact, then trim it."],
                "aria": True, "aria_kind": aria_kind,
                "child_recordings": []})
        if aria_kind:
            from aria_source.loading import load_aria_for_trim
            try:
                rec = load_aria_for_trim(rp)
            except Exception as e:                       # noqa: BLE001
                raise HTTPException(400, f"{type(e).__name__}: {e}")
            key = str(rp)
            with RECS_LOCK:
                RECS[key] = rec
            recents.add_recording(key)
            g, w0 = rec.gaze, (rec.world.ts_ns[0] if rec.world.ts_ns else 0)
            m = rec.manifest
            return {
                "dir": key,
                "video": {"filename": Path(rec.video.path).name,
                          "width": rec.video.width,
                          "height": rec.video.height,
                          "codec": rec.video.codec,
                          "n_frames": rec.video.n_frames,
                          "duration_s": rec.video.duration_s,
                          "has_audio": rec.video.has_audio, "audio": None},
                "world_rows": rec.world.n_rows,
                "gaze": {"rows": g.n_rows,
                         "first_rel_s": ((g.ts_ns[0] - w0) / core.NS
                                         if g.ts_ns else None),
                         "last_rel_s": ((g.ts_ns[-1] - w0) / core.NS
                                        if g.ts_ns else None),
                         "first_ns": str(g.ts_ns[0]) if g.ts_ns else None,
                         "last_ns": str(g.ts_ns[-1]) if g.ts_ns else None},
                "recording_id": rec.recording_id,
                "recording_name": (m.get("recording", {}).get("task")
                                   or m.get("dataset")),
                "wearer_name": None,
                "is_trimmed_artifact":
                    rec.artifact_type == "trimmed_aria_recording",
                "rec_end_ns": str(rec.rec_end_ns),
                "sync": rec.sync.as_dict(),
                "warnings": rec.warnings,
                "pts_s": rec.video.pts_s,
                "default_out_base": str(Path(key).parent
                                        / f"{Path(key).name}_trim"),
                "last_out_parent": _valid_last_out_parent(),
                "source": {
                    "system": "project_aria",
                    "badge": ("Project Aria / "
                              + (m.get("dataset") or "recording")),
                    "artifact_type": rec.artifact_type,
                    "dataset": m.get("dataset"),
                    "task": m.get("recording", {}).get("task"),
                    "gaze_model": m.get("gaze", {}).get("model"),
                    "gaze_rows_total": m.get("gaze", {}).get("rows_total"),
                    "gaze_projection_ok": m.get("gaze", {})
                        .get("projection_ok"),
                    "timestamp_domain": m["timestamps"]["domain"],
                    "timestamp_note": m["timestamps"].get(
                        "compatibility_note"),
                    "orientation": m.get("orientation", {}).get("transform"),
                    "audio_present": m.get("audio", {}).get("present"),
                    "cache_dir": m.get("cache", {}).get("dir"),
                    "cache_reused": m.get("cache", {}).get("reused"),
                },
            }

        insp = inspector.inspect(rp)
        if not insp.is_recording:
            detail = {"message": "not a valid NEON recording folder",
                      "problems": insp.problems,
                      "child_recordings": insp.child_recordings}
            raise HTTPException(422, detail)
        try:
            rec = export_mod.load_recording(rp)
        except Exception as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}")
        key = str(rp)
        with RECS_LOCK:
            RECS[key] = rec
        recents.add_recording(key)

        b = insp.bundle
        g = rec.gaze
        w0 = rec.world.ts_ns[0] if rec.world.ts_ns else 0
        rec_name = Path(key).name
        return {
            "dir": key,
            "video": {
                "filename": rec.video.path.name,
                "width": rec.video.width, "height": rec.video.height,
                "codec": rec.video.codec,
                "n_frames": rec.video.n_frames,
                "duration_s": rec.video.duration_s,
                "has_audio": rec.video.has_audio,
                "audio": (f"{rec.video.audio_codec} {rec.video.audio_rate} Hz "
                          f"x{rec.video.audio_channels}"
                          if rec.video.has_audio else None),
            },
            "world_rows": rec.world.n_rows,
            "gaze": {
                "rows": g.n_rows,
                "first_rel_s": (g.ts_ns[0] - w0) / core.NS if g.ts_ns else None,
                "last_rel_s": (g.ts_ns[-1] - w0) / core.NS if g.ts_ns else None,
                "first_ns": str(g.ts_ns[0]) if g.ts_ns else None,  # ns as str —
                "last_ns": str(g.ts_ns[-1]) if g.ts_ns else None,  # exceeds 2^53
            },
            "recording_id": rec.info.recording_id,
            "recording_name": b.recording_name if b else None,
            "wearer_name": b.wearer_name if b else None,
            "is_trimmed_artifact": b.is_trimmed_artifact if b else False,
            "rec_end_ns": str(rec.rec_end_ns),
            "sync": rec.sync.as_dict(),
            "warnings": rec.warnings,
            "pts_s": rec.video.pts_s,
            "default_out_base": str(Path(key).parent / f"{rec_name}_trim"),
            "last_out_parent": _valid_last_out_parent(),
        }

    # -- video streaming with Range support ----------------------------------

    @app.get("/api/video")
    def api_video(dir: str, request: Request):
        rec = get_rec(dir)
        path = rec.video.path
        size = path.stat().st_size
        rng = request.headers.get("range")
        start, end = 0, size - 1
        status = 200
        if rng:
            m = RANGE_RE.match(rng.strip())
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    if m.group(2):
                        end = min(int(m.group(2)), size - 1)
                elif m.group(2):  # suffix range: last N bytes
                    start = max(0, size - int(m.group(2)))
                if start > end or start >= size:
                    return Response(status_code=416,
                                    headers={"Content-Range": f"bytes */{size}"})
                status = 206

        def gen(s=start, e=end):
            with open(path, "rb") as f:
                f.seek(s)
                left = e - s + 1
                while left > 0:
                    b = f.read(min(CHUNK, left))
                    if not b:
                        break
                    left -= len(b)
                    yield b

        headers = {"Accept-Ranges": "bytes",
                   "Content-Length": str(end - start + 1),
                   "Content-Type": "video/mp4"}
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return StreamingResponse(gen(), status_code=status, headers=headers)

    # -- selection stats ------------------------------------------------------

    @app.get("/api/gaze_count")
    def api_gaze_count(dir: str, start_frame: int, end_frame: int):
        rec = get_rec(dir)
        n = rec.video.n_frames
        if not (0 <= start_frame <= end_frame < n):
            raise HTTPException(400, f"invalid selection [{start_frame}, "
                                     f"{end_frame}] for {n} frames")
        try:
            plan = core.plan_trim(rec.world.ts_ns, rec.video.pts_s,
                                  rec.video.duration_s, start_frame, end_frame,
                                  rec.rec_end_ns)
        except ValueError as e:
            raise HTTPException(400, str(e))
        count = core.count_in_interval(rec.gaze, plan.abs_start_ns,
                                       plan.abs_end_ns)
        return {
            "gaze_count": count,
            "n_frames": plan.n_frames_selected,
            "duration_s": plan.duration_s,
            "abs_start_ns": str(plan.abs_start_ns),
            "abs_end_ns": str(plan.abs_end_ns),
            "rel_start_s": plan.rel_start_s,
            "rel_end_s": plan.rel_end_s,
        }

    # -- export ---------------------------------------------------------------

    @app.post("/api/export")
    def api_export(body: ExportBody):
        rec = get_rec(body.dir)
        if not rec.sync.ok:
            raise HTTPException(409, "export blocked — synchronization not "
                                     "established: " + rec.sync.message)
        n = rec.video.n_frames
        if not (0 <= body.start_frame <= body.end_frame < n):
            raise HTTPException(400, f"invalid selection [{body.start_frame}, "
                                     f"{body.end_frame}] for {n} frames")
        out_dir = authorized(body.out_dir)     # output is root-gated too
        if out_dir == rec.rec_dir or rec.rec_dir in out_dir.parents:
            raise HTTPException(409, "output directory must be outside the "
                                     "source recording folder")
        if out_dir.exists():
            raise HTTPException(409, f"output directory already exists: "
                                     f"{out_dir}")

        job_id = uuid.uuid4().hex[:12]
        cancel = threading.Event()
        job = {"state": "running", "phase": "starting", "progress": 0.0,
               "error": None, "manifest": None, "cancel": cancel,
               "thread": None, "out_dir": str(out_dir)}
        with JOBS_LOCK:
            JOBS[job_id] = job

        def progress_cb(phase: str, frac: float):
            job["phase"] = phase
            job["progress"] = round(float(frac), 4)

        def worker():
            try:
                if getattr(rec, "is_aria", False):
                    from aria_source.trim import trim_aria_recording
                    manifest = trim_aria_recording(
                        rec.rec_dir, body.start_frame, body.end_frame,
                        out_dir, progress_cb=progress_cb,
                        cancel_event=cancel, path_guard=auth.is_allowed)
                else:
                    manifest = export_mod.run_export(
                        rec, body.start_frame, body.end_frame, out_dir,
                        progress_cb=progress_cb, cancel_event=cancel,
                        path_guard=auth.is_allowed)
            except (export_mod.ExportCancelled, media.CutCancelled):
                # classified by TYPE: a real failure that merely races a
                # late cancel click must still be reported as failed
                job["state"] = "cancelled"
                job["error"] = "export cancelled — no output was written"
                return
            except Exception as e:
                job["state"] = "failed"
                job["error"] = f"{type(e).__name__}: {e}"
                return
            # export succeeded — nothing after this point may flip the state
            job["manifest"] = manifest
            job["phase"] = "done"
            job["progress"] = 1.0
            job["state"] = "done"
            try:
                recents.set_last_out_parent(str(out_dir.parent))
            except OSError:
                pass                      # state-file trouble ≠ export failure

        t = threading.Thread(target=worker, daemon=True)
        job["thread"] = t
        t.start()
        return {"job_id": job_id}

    @app.get("/api/export/{job_id}")
    def api_export_status(job_id: str):
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        out = {k: job[k] for k in ("state", "phase", "progress", "error",
                                   "out_dir")}
        out["manifest"] = _manifest_for_js(job["manifest"])
        return out

    @app.post("/api/export/{job_id}/cancel")
    def api_export_cancel(job_id: str):
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        job["cancel"].set()
        return {"ok": True}

    # -- artifacts -------------------------------------------------------------

    @app.get("/api/artifact")
    def api_artifact(path: str):
        """Validate + open a trimmed artifact through its manifest."""
        rp = authorized(path)
        try:
            art = artifact_mod.load_artifact(rp)
        except artifact_mod.ArtifactError as e:
            raise HTTPException(422, str(e))
        return {**art.as_dict(), "manifest": _manifest_for_js(art.manifest)}

    @app.get("/api/health")
    def api_health():
        # identity + liveness for the pipeline workbench; deliberately
        # cheap (no heavy imports, no filesystem walk)
        with JOBS_LOCK:
            busy = any(j["state"] == "running" for j in JOBS.values())
        return {"ok": True, "module": "neon_trim",
                "version": export_mod.__version__, "busy": busy,
                "loaded": list(RECS.keys()),
                "roots": [str(r) for r in auth.roots],
                "ffmpeg": media.ffmpeg_exe()}

    return app


# module-level app for `uvicorn neon_trim.server:app` (env-config defaults)
app = create_app(config_mod.load_config())


if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="NEON trim GUI server")
    ap.add_argument("--host", default=config_mod.DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=None,
                    help=f"default {config_mod.DEFAULT_PORT} "
                         f"(env NEON_TRIM_PORT)")
    ap.add_argument("--root", action="append", default=None, metavar="DIR",
                    help="additional browse root (repeatable; also env "
                         "NEON_TRIM_ROOTS, ':'-separated)")
    ap.add_argument("--allow-lan", action="store_true",
                    help="required to bind a non-loopback host: the API reads "
                         "and writes local paths with NO authentication, so "
                         "only expose it on a trusted network (prefer "
                         "`ssh -L 8001:localhost:8001 <box>` instead)")
    args = ap.parse_args()

    cfg = config_mod.load_config(extra_roots=args.root, port=args.port,
                                 host=args.host)
    application = create_app(cfg)

    loopback = False
    try:
        loopback = ipaddress.ip_address(cfg.host).is_loopback
    except ValueError:
        loopback = cfg.host in ("localhost",)
    if not loopback and not args.allow_lan:
        ap.error(f"refusing to bind {cfg.host}: this unauthenticated API can "
                 f"read recordings and write directories as your user. Use "
                 f"ssh port-forwarding, or pass --allow-lan on a trusted LAN.")
    # (TrustedHost anti-DNS-rebinding middleware is added inside create_app,
    # so it also protects `uvicorn neon_trim.server:app`.)
    if not loopback:
        print("⚠ serving without authentication on a non-loopback interface — "
              "trusted networks only")
    print(f"neon_trim GUI  →  http://{cfg.host}:{cfg.port}")
    print("browse roots:", ", ".join(str(r) for r in cfg.browse_roots))
    uvicorn.run(application, host=cfg.host, port=cfg.port, log_level="warning")
