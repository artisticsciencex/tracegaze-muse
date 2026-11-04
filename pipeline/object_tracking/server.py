"""FastAPI server for the object-tracking GUI.

Run:
    python pipeline/object_tracking/server.py            # http://127.0.0.1:8002
    OBJTRACK_ROOTS=/data/a:/data/b python server.py --port 8002
    python server.py --root /extra --host 0.0.0.0 --allow-lan

Thin HTTP layer only: authorization lives in common.fsbrowse, intake rules in
object_tracking.intake, annotation rules in annotations.py, SAM3 execution in
tracking.py, artifact rules in artifact.py. Single-user local tool; loaded
sessions and jobs are in-memory; the server never writes into source
recordings.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

# NOTE: this process must NEVER import pandas/pyarrow (libarrow segfaulted
# the server from a FastAPI thread — kernel-log verified). All Parquet work
# happens in subprocesses: table_io (import/inspect), publish_cli (publish),
# sam3_worker (GPU). Track rows live in the SQLite trackstore.
try:
    from common import config as common_config
    from common import fsbrowse, media
    from common.jobs import JobCancelled, JobManager
    from object_tracking import annotations as ann_mod
    from object_tracking import artifact as track_artifact
    from object_tracking import framesource, intake, native_prep, table_io, \
        tracking, trackstore
except ImportError:  # executed directly as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from common import config as common_config
    from common import fsbrowse, media
    from common.jobs import JobCancelled, JobManager
    from object_tracking import annotations as ann_mod
    from object_tracking import artifact as track_artifact
    from object_tracking import framesource, intake, native_prep, table_io, \
        tracking, trackstore

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
CHUNK = 1 << 20
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
MAX_MASK_FRAMES_PER_REQUEST = 90
DEFAULT_PORT = 8002


class PrepBody(BaseModel):
    dir: str
    out_dir: str


class LoadBody(BaseModel):
    dir: str


class CreateObjBody(BaseModel):
    dir: str
    name: str
    ref_frame: int
    note: str = ""


class PointsBody(BaseModel):
    dir: str
    points: list
    labels: list
    ref_frame: int | None = None


class NameBody(BaseModel):
    dir: str
    name: str


class MoveRefBody(BaseModel):
    dir: str
    frame: int


class NoteBody(BaseModel):
    dir: str
    note: str


class DirBody(BaseModel):
    dir: str


class StatusBody(BaseModel):
    dir: str
    status: str


class TrackBody(BaseModel):
    dir: str
    object_ids: list


class PublishBody(BaseModel):
    dir: str
    out_dir: str
    make_overlay: bool = True



class TrackSession:
    """Server-side state for one loaded recording."""

    def __init__(self, bundle: intake.RecordingBundle, state_dir: Path):
        self.bundle = bundle
        key = hashlib.sha1(str(bundle.source_dir).encode()).hexdigest()[:16]
        self.workdir = state_dir / "sessions" / key
        self.tracks_dir = self.workdir / "tracks"
        self.tracks_dir.mkdir(parents=True, exist_ok=True)
        self.store = ann_mod.AnnotationStore(
            self.workdir / "annotations.json", bundle.n_frames,
            bundle.width or 0, bundle.height or 0)
        self.worker: tracking.WorkerHandle | None = None
        self.lock = threading.Lock()
        self._frame_decoder: framesource.FrameDecoder | None = None

    def frame_decoder(self) -> framesource.FrameDecoder:
        with self.lock:
            if self._frame_decoder is None:
                self._frame_decoder = framesource.FrameDecoder(
                    self.bundle.video_path, self.bundle.pts_s)
            return self._frame_decoder

    @property
    def db_path(self) -> Path:
        return self.workdir / "tracks.db"

    def worker_log(self) -> Path:
        return self.workdir / "sam3_worker.log"

    def recover_orphans(self) -> list:
        """Import legacy/orphaned per-object parquet files into the SQLite
        store through the ISOLATED table helper (never read in-process).

        Two cases:
        * rows missing but annotation still has track metadata (legacy
          format migration) -> import rows only
        * completed track whose registration was lost in a server crash
          (track is null) -> import rows AND register the result, so only
          truly unrecoverable objects need a rerun
        """
        if getattr(self, "_orphans_checked", False):
            return []
        self._orphans_checked = True
        notes = []
        in_db = set(trackstore.objects_in_db(self.db_path))
        for o in self.store.objects():
            oid = o["object_id"]
            pq = self.tracks_dir / f"{oid}.parquet"
            if not pq.is_file():
                continue
            if oid in in_db:
                pq.rename(pq.with_suffix(".parquet.imported"))
                continue
            try:
                res = table_io.run_helper(
                    ["import-parquet", "--db", self.db_path,
                     "--object", oid, "--parquet", pq,
                     "--n-frames", self.bundle.n_frames,
                     "--ref-frame", o["reference_frame"]])
            except table_io.HelperError as e:
                notes.append(f"{oid} ({o['name']}): orphaned track could not "
                             f"be recovered — {e} — rerun this object")
                continue
            pq.rename(pq.with_suffix(".parquet.imported"))
            if o.get("track"):
                notes.append(f"{oid} ({o['name']}): migrated "
                             f"{res['rows']} legacy track rows")
                continue
            meta = {"rows": res["rows"],
                    "frames_present": res["frames_present"],
                    "first_frame": res["first_frame"],
                    "last_frame": res["last_frame"],
                    "forward_ok": res["forward_rows"] > 0,
                    "backward_ok": res["backward_rows"] > 0,
                    "forward_error": None, "backward_error": None,
                    "validation_problems": [],
                    "whole_video": (res["forward_rows"] > 0
                                    and res["backward_rows"] > 0),
                    "recovered": True,
                    "note": "registered from an orphaned completed track "
                            "after a server crash (validated on import)"}
            try:
                self.store.record_track_result(oid, meta)
                notes.append(f"{oid} ({o['name']}): recovered completed "
                             f"track ({res['rows']} rows) — review it")
            except ann_mod.AnnotationError as e:
                notes.append(f"{oid}: recovery blocked — {e}")
        return notes

    def ensure_worker(self, phase_cb=None) -> dict:
        with self.lock:
            if self.worker is None:
                self.worker = tracking.WorkerHandle(
                    self.bundle.video_path, self.worker_log())
            return self.worker.start(phase_cb=phase_cb)

    def stop_worker(self):
        with self.lock:
            if self.worker is not None:
                self.worker.stop()
                self.worker = None

    def drop_track(self, object_id: str):
        """Remove an object's rows (SQLite, atomic) plus any legacy
        parquet, so stale prompts can never resurface via recovery."""
        trackstore.delete_object_rows(self.db_path, object_id)
        for suffix in (".parquet", ".parquet.imported"):
            p = self.tracks_dir / f"{object_id}{suffix}"
            if p.is_file():
                p.unlink()


def create_app(cfg: common_config.StageConfig) -> FastAPI:
    auth = fsbrowse.PathAuthorizer(cfg.browse_roots)
    recents = fsbrowse.RecentsStore(cfg.state_dir)
    jobs = JobManager()
    SESSIONS: dict = {}
    SESSIONS_LOCK = threading.Lock()

    @asynccontextmanager
    async def _lifespan(app):
        yield
        jobs.shutdown()
        with SESSIONS_LOCK:
            sessions = list(SESSIONS.values())
        for s in sessions:
            s.stop_worker()               # no orphaned GPU workers

    app = FastAPI(title="object_tracking", lifespan=_lifespan)

    # DNS-rebinding defense on every consumer of this app object
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

    def get_session(d: str) -> TrackSession:
        s = SESSIONS.get(str(authorized(d)))
        if s is None:
            raise HTTPException(404, "recording not loaded — POST /api/load")
        return s

    # ------------------------------------------------------------- pages
    @app.get("/")
    def index():
        return FileResponse(APP_DIR / "static" / "index.html")

    @app.get("/framesync.js")
    def framesync_js():
        return FileResponse(APP_DIR / "static" / "framesync.js",
                            media_type="application/javascript")

    @app.get("/api/config")
    def api_config():
        gpu_ok, gpu_detail = tracking.gpu_available()
        return {"roots": [str(r) for r in auth.roots],
                # the browser derives its chooser gate from THIS, so a new
                # source system never needs a second frontend allow-list
                "chooser_kinds": list(intake.CHOOSER_KINDS),
                "trackable_kinds": list(intake.TRACKABLE_KINDS),
                "preparation_required_kinds":
                    list(intake.PREPARATION_REQUIRED_KINDS),
                "kind_labels": dict(intake.KIND_LABELS),
                "kind_badge_class": dict(intake.KIND_BADGE_CLASS),
                "version": track_artifact.__version__,
                "sam3_python": str(tracking.sam3_python()),
                "sam3_python_exists": tracking.sam3_python().exists(),
                "local_patch_present": tracking.verify_local_patch(),
                "gpu_ok": gpu_ok, "gpu_detail": gpu_detail,
                "kind_labels": intake.KIND_LABELS}

    # ---------------------------------------------------------- browsing
    @app.get("/api/browse")
    def api_browse(path: str = ""):
        if not path:
            if len(auth.roots) == 1:
                path = str(auth.roots[0])
            else:
                return {"path": None, "root": None, "parent": None,
                        "breadcrumbs": [],
                        "entries": [{"name": str(r), "path": str(r),
                                     "is_dir": True, "inaccessible": False,
                                     "is_recording": bool(
                                         intake.classify_folder(r)),
                                     "kind": intake.classify_folder(r)}
                                    for r in auth.roots],
                        "error": None}
        rp = authorized(path)
        return fsbrowse.list_dir(auth, rp,
                                 classifier=intake.classify_folder).as_dict()

    @app.get("/api/inspect")
    def api_inspect(path: str):
        rp = authorized(path)
        kind = intake.classify_folder(rp)
        out = {"path": str(rp), "kind": kind,
               "kind_label": intake.KIND_LABELS.get(kind,
                                                    "Incomplete or "
                                                    "unsupported folder")}
        if kind in intake.ARIA_KINDS:
            # Aria artifacts are inspected through THEIR OWN contract — the
            # NEON inspector would report source-inappropriate results
            # (wearer, NEON gaze offset, NEON world time). JSON/stat only:
            # no pandas, no pyarrow, no projectaria_tools in this process.
            out["source_system"] = "project_aria"
            try:
                from aria_source.artifact import load_aria_artifact
                art = load_aria_artifact(rp)
                m = art.manifest
                counts = {}
                for key, label in (("world_timestamps", "world_rows"),
                                   ("gaze", "gaze_rows")):
                    p = (art.world_timestamps_path if key == "world_timestamps"
                         else art.gaze_path)
                    try:                    # cheap line count, no parsing
                        with open(p) as f:
                            counts[label] = max(0, sum(1 for _ in f) - 1)
                    except OSError:
                        counts[label] = None
                out["aria"] = {
                    "artifact_type": art.artifact_type,
                    "artifact_id": art.artifact_id,
                    "source_system": art.source_system,
                    "dataset": m.get("dataset"),
                    "dataset_release": m.get("dataset_release"),
                    "recording_id": art.recording_id,
                    "task": (m.get("recording") or {}).get("task"),
                    "files": {"video": art.video_path.name,
                              "gaze": art.gaze_path.name,
                              "world_timestamps":
                                  art.world_timestamps_path.name},
                    "n_frames": (m.get("video") or {}).get("n_frames"),
                    "width": (m.get("video") or {}).get("width"),
                    "height": (m.get("video") or {}).get("height"),
                    "world_rows": counts.get("world_rows"),
                    "gaze_rows": counts.get("gaze_rows"),
                    "timestamp_domain": m["timestamps"]["domain"],
                    "timestamp_note":
                        m["timestamps"].get("compatibility_note"),
                    "orientation": (m.get("orientation") or {})
                        .get("transform"),
                    "gaze_model": (m.get("gaze") or {}).get("model"),
                    "audio_present": (m.get("audio") or {}).get("present"),
                    "validation_ok": art.validation_ok,
                    "validation_checks": len(m.get("validation") or []),
                    "parent_artifact": m.get("parent_artifact"),
                    # NEON-only concepts, explicitly marked unavailable
                    "wearer_name": None,
                    "neon_gaze_offset": None,
                    "neon_world_time": None,
                    "not_applicable_note":
                        "wearer, NEON gaze offset and NEON world time do "
                        "not exist for a Project Aria recording",
                }
                if not art.validation_ok:
                    out["problems"] = [
                        f"{v['check']}: {v['detail']}"
                        for v in m.get("validation", []) if not v.get("ok")]
            except Exception as e:                       # noqa: BLE001
                out["problems"] = [str(e)]
        elif kind == "aria_download":
            out["source_system"] = "project_aria"
            from aria_source.detect import (inspect_aria_folder,
                                            validate_against_receipt)
            pkg = inspect_aria_folder(rp)
            out["aria"] = pkg.as_dict()
            out["aria"]["receipt_validation"] = validate_against_receipt(pkg)
            out["needs_preparation"] = True
            out["problems"] = list(pkg.problems) or [
                "Project Aria download folder — prepare it in the "
                "Trim/Prepare stage before tracking"]
        elif kind == "native_recording":
            out["native"] = native_prep.native_overview(rp)
        elif kind == "tracked_objects":
            try:
                art = track_artifact.load_tracked_artifact(rp)
                out["tracked"] = {"artifact_id": art.artifact_id,
                                  "objects": [o["name"] for o in
                                              art.manifest.get("objects", [])
                                              if o.get("review_status")
                                              == "accepted"]}
            except Exception as e:
                out["problems"] = [str(e)]
        else:
            from neon_trim import intake as nt_intake
            insp = nt_intake.inspect_recording(rp)
            out["inspection"] = insp.as_dict()
        return out

    def _valid_last_out_parent():
        lop = recents.last_out_parent()
        return lop if lop and auth.is_allowed(lop) else None

    @app.get("/api/recents")
    def api_recents():
        out = []
        for p in recents.recent_recordings():
            if not auth.is_allowed(p):
                continue
            pp = Path(p)
            kind = intake.classify_folder(pp) if pp.is_dir() else None
            out.append({"path": p, "name": pp.name, "exists": pp.is_dir(),
                        "kind": kind,
                        # one authoritative set — Aria artifacts included
                        "usable": kind in intake.TRACKABLE_KINDS})
        return {"recents": out, "last_out_parent": _valid_last_out_parent()}

    @app.get("/api/out_check")
    def api_out_check(dir: str, path: str):
        src = authorized(dir)
        try:
            rp = auth.resolve(path)
        except fsbrowse.PathNotAllowed as e:
            return {"resolved": path, "authorized": False, "reason": str(e),
                    "exists": None, "inside_source": None}
        inside = (rp == src) or (src in rp.parents)
        return {"resolved": str(rp), "authorized": True, "reason": None,
                "exists": rp.exists(), "inside_source": inside}

    # ------------------------------------------------------- native prep

    @app.post("/api/prepare")
    def api_prepare(body: PrepBody):
        native_dir = authorized(body.dir)
        out_dir = authorized(body.out_dir)
        if not native_prep.is_native_recording(native_dir):
            raise HTTPException(422, "not a native NEON recording")
        if out_dir.exists():
            raise HTTPException(409, f"output already exists: {out_dir}")

        def run(job):
            def cb(phase, frac):
                job["phase"] = phase
                job["progress"] = round(frac, 4)
            return native_prep.prepare_native_recording(
                native_dir, out_dir, path_guard=auth.is_allowed,
                progress_cb=cb, cancel_event=job["cancel"])

        job_id = jobs.create(run, kind="prepare",
                             meta={"out_dir": str(out_dir)})
        return {"job_id": job_id}

    # ------------------------------------------------------------- load

    @app.post("/api/load")
    def api_load(body: LoadBody):
        rp = authorized(body.dir)
        try:
            bundle = intake.open_bundle(rp)
        except intake.IntakeError as e:
            kind = intake.classify_folder(rp)
            raise HTTPException(422, {"message": str(e), "kind": kind})
        except Exception as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}")
        # get-or-create: re-loading the same recording must reuse the live
        # session (one AnnotationStore per annotations.json, warm SAM3 worker
        # preserved, no orphaned GPU subprocess)
        with SESSIONS_LOCK:
            sess = SESSIONS.get(str(rp))
            if sess is None:
                sess = TrackSession(bundle, cfg.state_dir)
                SESSIONS[str(rp)] = sess
            else:
                sess.bundle = bundle          # refresh probe metadata only
        recents.add_recording(str(rp))
        recovery_notes = sess.recover_orphans()
        if recovery_notes:
            bundle.warnings.extend(recovery_notes)
        n_passes = bundle.n_frames
        return {
            "dir": str(rp),
            **bundle.summary(),
            "pts_s": bundle.pts_s,
            "world_t0_ns": (str(bundle.world_ts_ns[0])
                            if bundle.world_ts_ns else None),
            "world_rel_ns": ([t - bundle.world_ts_ns[0]
                              for t in bundle.world_ts_ns]
                             if bundle.world_ts_ns else None),
            "objects": sess.store.objects(),
            "last_out_parent": _valid_last_out_parent(),
            # fingerprint for immutable frame-URL caching (&v=): changes when
            # the video file at this path is replaced (e.g. re-trimmed)
            "video_token": (lambda st: f"{st.st_size}-{st.st_mtime_ns}")(
                bundle.video_path.stat()),
        }

    # ---------------------------------------------------- video streaming
    @app.get("/api/video")
    def api_video(dir: str, request: Request):
        sess = get_session(dir)
        path = sess.bundle.video_path
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
                elif m.group(2):
                    start = max(0, size - int(m.group(2)))
                if start > end or start >= size:
                    return Response(status_code=416, headers={
                        "Content-Range": f"bytes */{size}"})
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

    # -- exact still frames (paused-review path) -----------------------------
    @app.get("/api/frame")
    def api_frame(dir: str, frame_index: int):
        """The image that IS frame `frame_index` by presentation order,
        decoded against the authoritative PTS table (never trusts browser
        seeking). Headers carry the frame identity so the client can commit
        background + masks + readout atomically for one generation."""
        sess = get_session(dir)
        try:
            jpeg, pts, hit = sess.frame_decoder().get(frame_index)
        except framesource.FrameSourceError as e:
            raise HTTPException(400, str(e))
        wts = sess.bundle.timestamp_ns_for_frame(frame_index)
        return Response(jpeg, media_type="image/jpeg", headers={
            "X-Frame-Index": str(frame_index),
            "X-Frame-Media-Time": f"{pts:.6f}",
            "X-World-Timestamp-Ns": str(wts) if wts is not None else "null",
            "X-Frame-Cache": "hit" if hit else "miss",
            # frames are immutable per (dir, index); the client appends a
            # video fingerprint (&v=) so a re-trimmed recording at the same
            # path never serves stale cached frames
            "Cache-Control": "private, max-age=3600",
        })

    # -------------------------------------------------------- annotations

    @app.post("/api/objects")
    def api_create_object(body: CreateObjBody):
        sess = get_session(body.dir)
        try:
            return sess.store.create_object(body.name, body.ref_frame,
                                            body.note)
        except ann_mod.AnnotationError as e:
            raise HTTPException(422, str(e))

    @app.get("/api/objects")
    def api_objects(dir: str):
        sess = get_session(dir)
        return {"objects": sess.store.objects()}


    @app.post("/api/objects/{oid}/points")
    def api_points(oid: str, body: PointsBody):
        sess = get_session(body.dir)
        try:
            out = sess.store.set_points(oid, body.points, body.labels,
                                        body.ref_frame)
        except ann_mod.AnnotationError as e:
            raise HTTPException(422, str(e))
        # the old track no longer corresponds to these prompts: remove the
        # stale parquet so it can never be reviewed or published
        sess.drop_track(oid)
        return out


    @app.post("/api/objects/{oid}/move_reference")
    def api_move_reference(oid: str, body: MoveRefBody):
        """Explicit, confirmed re-anchoring to another reference frame."""
        sess = get_session(body.dir)
        try:
            out = sess.store.move_reference_frame(oid, body.frame)
        except ann_mod.AnnotationError as e:
            raise HTTPException(422, str(e))
        # the old track belonged to the old reference image
        sess.drop_track(oid)
        return out

    @app.post("/api/objects/{oid}/rename")
    def api_rename(oid: str, body: NameBody):
        sess = get_session(body.dir)
        try:
            return sess.store.rename(oid, body.name)
        except ann_mod.AnnotationError as e:
            raise HTTPException(422, str(e))


    @app.post("/api/objects/{oid}/note")
    def api_note(oid: str, body: NoteBody):
        sess = get_session(body.dir)
        try:
            return sess.store.set_note(oid, body.note)
        except ann_mod.AnnotationError as e:
            raise HTTPException(422, str(e))


    @app.post("/api/objects/{oid}/accept_mask")
    def api_accept_mask(oid: str, body: DirBody):
        sess = get_session(body.dir)
        try:
            return sess.store.accept_mask(oid)
        except ann_mod.AnnotationError as e:
            raise HTTPException(422, str(e))


    @app.post("/api/objects/{oid}/status")
    def api_status(oid: str, body: StatusBody):
        sess = get_session(body.dir)
        try:
            return sess.store.set_review_status(oid, body.status)
        except ann_mod.AnnotationError as e:
            raise HTTPException(422, str(e))

    @app.post("/api/objects/{oid}/delete")
    def api_delete(oid: str, body: DirBody):
        sess = get_session(body.dir)
        try:
            sess.store.delete(oid)
        except ann_mod.AnnotationError as e:
            raise HTTPException(422, str(e))
        sess.drop_track(oid)
        return {"ok": True}

    # ------------------------------------------------------------ worker
    @app.post("/api/worker/start")
    def api_worker_start(body: DirBody):
        sess = get_session(body.dir)
        gpu_ok, gpu_detail = tracking.gpu_available()
        if not gpu_ok:
            raise HTTPException(409, f"no GPU available: {gpu_detail}")

        def run(job):
            def phase(p):
                job["phase"] = p
            info = sess.ensure_worker(phase_cb=phase)
            return {k: info.get(k) for k in
                    ("gpus", "gpu_names", "torch", "version", "fa3_disabled")}

        return {"job_id": jobs.create(run, kind="worker_start")}

    @app.get("/api/worker/status")
    def api_worker_status(dir: str):
        sess = get_session(dir)
        w = sess.worker
        return {"running": bool(w is not None and w.alive),
                "ready": bool(w is not None and w.ready_info),
                "info": (w.ready_info if w is not None else None),
                "log": str(sess.worker_log())}

    @app.post("/api/worker/stop")
    def api_worker_stop(body: DirBody):
        sess = get_session(body.dir)
        sess.stop_worker()
        return {"ok": True}

    # ----------------------------------------------------------- preview
    @app.post("/api/objects/{oid}/preview")
    def api_preview(oid: str, body: DirBody):
        sess = get_session(body.dir)
        o = sess.store.get(oid)
        if not o["points"] or 1 not in o["point_labels"]:
            raise HTTPException(422, "add at least one positive point first")
        if sess.worker is None or not sess.worker.alive \
                or not sess.worker.ready_info:
            raise HTTPException(409, "SAM3 worker not running — start it "
                                     "first (model loads once per recording)")
        try:
            ev = sess.worker.preview(
                o["points"], o["point_labels"], o["reference_frame"],
                sess.bundle.width, sess.bundle.height)
        except tracking.TrackingError as e:
            raise HTTPException(500, str(e))
        meta = {k: ev.get(k) for k in ("found", "score", "mask_area",
                                       "bbox_x", "bbox_y", "bbox_w",
                                       "bbox_h")}
        sess.store.record_preview(oid, meta)
        return {**meta, "mask_rle": ev.get("mask_rle"),
                "object": sess.store.get(oid)}

    # ------------------------------------------------------------- track

    @app.get("/api/track/estimate")
    def api_track_estimate(dir: str, ids: str):
        sess = get_session(dir)
        oids = [i for i in ids.split(",") if i]
        n = sess.bundle.n_frames
        return {"n_objects": len(oids), "n_frames": n,
                "message": f"{len(oids)} object(s) x {n} frames = "
                           f"{len(oids)} tracking pass(es) "
                           f"(one reset session per object, forward + "
                           f"backward from its reference frame)"}

    @app.post("/api/track")
    def api_track(body: TrackBody):
        sess = get_session(body.dir)
        object_ids = list(dict.fromkeys(body.object_ids))   # dedupe, keep order
        for oid in object_ids:
            o = sess.store.get(oid)
            if o.get("deleted"):
                raise HTTPException(422, f"{oid} has been deleted")
            if o["status"] not in ("mask_accepted", "tracked", "accepted",
                                   "rejected"):
                raise HTTPException(422, f"{oid} has no accepted initial "
                                         f"mask — preview and accept first")
        if not object_ids:
            raise HTTPException(422, "no objects selected")
        if sess.worker is None or not sess.worker.alive \
                or not sess.worker.ready_info:
            raise HTTPException(409, "SAM3 worker not running")

        def run(job):
            t0 = time.time()
            results = {}
            job["meta"]["objects"] = {oid: "queued" for oid in object_ids}
            for i, oid in enumerate(object_ids):
                if job["cancel"].is_set():
                    raise JobCancelled()
                # FRESH snapshot at pass start: edits since the POST must not
                # be tracked from stale prompts
                o = sess.store.get(oid)
                if o.get("deleted") or o["status"] not in (
                        "mask_accepted", "tracked", "accepted", "rejected"):
                    results[oid] = {"skipped":
                                    "edited or deleted since request"}
                    job["meta"]["objects"][oid] = "skipped"
                    continue
                rev_count = len(o["revisions"])
                job["meta"]["objects"][oid] = "running"
                job["phase"] = (f"tracking {oid} ({o['name']}) "
                                f"[{i + 1}/{len(object_ids)}]")

                def prog(ev, i=i, oid=oid):
                    frac = min(ev.get("frames_done", 0)
                               / max(sess.bundle.n_frames, 1), 1.0)
                    job["progress"] = round((i + frac * 0.5
                                             * (2 if ev.get("direction")
                                                == "backward" else 1))
                                            / len(object_ids), 4)
                    job["phase"] = (f"tracking {oid} "
                                    f"({ev.get('direction')}) "
                                    f"[{i + 1}/{len(object_ids)}]")

                # rerun-safe: the worker replaces this object's rows in ONE
                # SQLite transaction; the previous good rows survive any
                # cancel/failure (nothing to invalidate server-side).
                ev = sess.worker.track(
                    oid, o["points"], o["point_labels"],
                    o["reference_frame"], sess.bundle.width,
                    sess.bundle.height, sess.db_path,
                    progress_cb=prog, cancel_event=job["cancel"])
                # validation reads (frame, direction) tuples from SQLite —
                # NO pandas/pyarrow in this process, ever
                pairs = trackstore.object_frames(sess.db_path, oid)
                problems = tracking.validate_track_rows(
                    pairs, object_id=oid, n_frames=sess.bundle.n_frames,
                    ref_frame=o["reference_frame"])
                meta = {k: ev.get(k) for k in
                        ("rows", "frames_present", "first_frame",
                         "last_frame", "forward_ok", "backward_ok",
                         "forward_error", "backward_error")}
                meta["validation_problems"] = problems
                meta["whole_video"] = bool(ev.get("forward_ok")
                                           and ev.get("backward_ok"))
                try:
                    # compare-and-set: refuse to attach the result if the
                    # object was edited while this pass ran
                    sess.store.record_track_result(
                        oid, meta, expected_revision_count=rev_count)
                    job["meta"]["objects"][oid] = "done"
                except ann_mod.AnnotationError as e:
                    sess.drop_track(oid)   # result no longer matches prompts
                    meta = {"skipped": str(e)}
                    job["meta"]["objects"][oid] = "stale"
                results[oid] = meta
                job["progress"] = round((i + 1) / len(object_ids), 4)
            return {"objects": results,
                    "runtime_s": round(time.time() - t0, 1)}

        job_id = jobs.create(run, kind="track",
                             cancel_exceptions=(JobCancelled,))
        return {"job_id": job_id}

    # ------------------------------------------------------------ review
    @app.get("/api/review/summary")
    def api_review_summary(dir: str):
        sess = get_session(dir)
        out = []
        for o in sess.store.objects():
            if not o.get("track"):
                continue
            pairs = trackstore.object_frames(sess.db_path, o["object_id"])
            entry = {"object_id": o["object_id"], "name": o["name"],
                     "color": o["color"], "status": o["status"],
                     "reference_frame": o["reference_frame"],
                     "track": o["track"],
                     "presence": tracking.presence_summary(
                         pairs, sess.bundle.n_frames)}
            tl = trackstore.timelines(sess.db_path, o["object_id"])
            entry["area_timeline"] = tl["area"]
            entry["score_timeline"] = tl["score"]
            out.append(entry)
        return {"objects": out}

    @app.get("/api/review/masks")
    def api_review_masks(dir: str, start: int, end: int):
        sess = get_session(dir)
        if end < start:
            raise HTTPException(400, "end < start")
        if end - start + 1 > MAX_MASK_FRAMES_PER_REQUEST:
            raise HTTPException(400, f"at most {MAX_MASK_FRAMES_PER_REQUEST} "
                                     f"frames per request")
        visible = {o["object_id"]: o["color"] for o in sess.store.objects()
                   if o.get("track")
                   and o["status"] not in ("rejected", "excluded")}
        if not visible:
            return {"start": start, "end": end, "frames": {}}
        # bounded, indexed SQLite range query — never loads all masks
        raw = trackstore.rows_for_range(sess.db_path, start, end,
                                        object_ids=list(visible))
        frames = {f: [{**e, "color": visible[e["object_id"]]}
                      for e in entries]
                  for f, entries in raw.items()}
        return {"start": start, "end": end, "frames": frames}

    # ----------------------------------------------------------- publish
    @app.get("/api/publish/check")
    def api_publish_check(dir: str):
        sess = get_session(dir)
        ready, blockers = sess.store.publishable()
        src = sess.bundle.source_dir
        return {"ready": ready, "blockers": blockers,
                "default_out": str(src.parent
                                   / f"{src.name}_tracked_objects"),
                "overlay_recommended": sess.bundle.n_frames <= 3000,
                "gaze_compatible": sess.bundle.gaze_compatible}


    @app.post("/api/publish")
    def api_publish(body: PublishBody):
        sess = get_session(body.dir)
        out_dir = authorized(body.out_dir)
        if out_dir.exists():
            raise HTTPException(409, f"output already exists: {out_dir}")
        worker_info = (sess.worker.ready_info
                       if sess.worker and sess.worker.ready_info else None)
        # publish gate checked HERE for a fast 4xx; the subprocess re-checks
        ready, blockers = sess.store.publishable()
        if not ready:
            raise HTTPException(422, "not publishable: " + "; ".join(blockers))

        def run(job):
            # ISOLATED SUBPROCESS: pyarrow segfaults were observed inside the
            # server's thread context (kernel log: libarrow.so segfault); a
            # publisher crash must never take the GUI server down.
            import subprocess as sp
            extra = {"worker_info": worker_info,
                     "runtime_s": (sess.store.objects() and None)}
            extra_path = sess.workdir / "publish_extra.json"
            extra_path.write_text(json.dumps(extra))
            cmd = [sys.executable, str(APP_DIR / "publish_cli.py"),
                   "--src", str(sess.bundle.source_dir),
                   "--out", str(out_dir),
                   "--state-dir", str(cfg.state_dir),
                   "--extra-json", str(extra_path)]
            if not body.make_overlay:
                cmd.append("--no-overlay")
            proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, text=True,
                            bufsize=1)
            manifest = None
            err = None
            try:
                for line in proc.stdout:
                    if job["cancel"].is_set():
                        proc.terminate()
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        continue
                    if "phase" in ev:
                        job["phase"] = ev["phase"]
                        job["progress"] = round(float(ev["progress"]), 4)
                    elif ev.get("done"):
                        manifest = ev["manifest"]
                    elif "error" in ev:
                        err = ev["error"]
                proc.wait(timeout=60)
            finally:
                if proc.poll() is None:
                    proc.kill()
            if job["cancel"].is_set():
                raise JobCancelled()
            if manifest is None:
                tail = proc.stderr.read()[-400:] if proc.stderr else ""
                raise RuntimeError(err or
                                   f"publisher crashed (exit {proc.returncode})"
                                   + (": " + tail if tail else ""))
            recents.set_last_out_parent(str(out_dir.parent))
            return manifest

        return {"job_id": jobs.create(run, kind="publish",
                                      cancel_exceptions=(JobCancelled,),
                                      meta={"out_dir": str(out_dir)})}

    @app.get("/api/artifact")
    def api_artifact(path: str):
        rp = authorized(path)
        # artifact loading reads objects.parquet -> ISOLATED helper process
        try:
            res = table_io.run_helper(["inspect-artifact", "--path", rp],
                                      timeout_s=300)
        except table_io.HelperError as e:
            raise HTTPException(422, str(e))
        # int-ns fields exceed JS 2^53 — stringify in the display copy
        m = res.get("manifest", {})
        for o in m.get("objects", []):
            if o.get("reference_timestamp_ns") is not None:
                o["reference_timestamp_ns"] = str(o["reference_timestamp_ns"])
        return {"artifact_dir": res["artifact_dir"],
                "artifact_id": res["artifact_id"],
                "parent_paths": res["parent_paths"],
                "parent_resolvable": res["parent_resolvable"],
                "manifest": m}

    # -------------------------------------------------------------- jobs
    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str):
        view = jobs.public_view(job_id)
        if view is None:
            raise HTTPException(404, "unknown job")
        return view

    @app.post("/api/jobs/{job_id}/cancel")
    def api_job_cancel(job_id: str):
        if not jobs.cancel(job_id):
            raise HTTPException(404, "unknown job")
        return {"ok": True}

    @app.get("/api/health")
    def api_health():
        # identity + liveness for the pipeline workbench; never touches
        # SAM3, the GPU worker, or any table library
        return {"ok": True, "module": "object_tracking",
                "version": track_artifact.__version__,
                "busy": bool(jobs.running_jobs()),
                "loaded": list(SESSIONS.keys()),
                "roots": [str(r) for r in auth.roots]}

    return app


app = create_app(common_config.load_stage_config(
    "object_tracking", "OBJTRACK", DEFAULT_PORT))


if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="object-tracking GUI server")
    ap.add_argument("--host", default=common_config.DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--root", action="append", default=None, metavar="DIR")
    ap.add_argument("--allow-lan", action="store_true",
                    help="required for a non-loopback bind (unauthenticated "
                         "API; prefer ssh port-forwarding)")
    args = ap.parse_args()
    cfg = common_config.load_stage_config(
        "object_tracking", "OBJTRACK", DEFAULT_PORT,
        extra_roots=args.root, port=args.port, host=args.host)
    application = create_app(cfg)
    lb = False
    try:
        lb = ipaddress.ip_address(cfg.host).is_loopback
    except ValueError:
        lb = cfg.host in ("localhost",)
    if not lb and not args.allow_lan:
        ap.error(f"refusing to bind {cfg.host} without --allow-lan")
    print(f"object_tracking GUI  →  http://{cfg.host}:{cfg.port}")
    print("browse roots:", ", ".join(str(r) for r in cfg.browse_roots))
    uvicorn.run(application, host=cfg.host, port=cfg.port,
                log_level="warning")
