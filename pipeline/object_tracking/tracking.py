"""Tracking engine: SAM3 worker lifecycle + per-object bidirectional runs.

The engine is worker-agnostic: it talks to a :class:`WorkerHandle` (a real
subprocess in the GPU env) or any object with the same ``request``/``events``
surface (tests inject a fake). All merge/validation rules live here — not in
HTTP routes, not in the worker.

One pass per object (correctness-first): every object runs in its own reset
SAM3 session pass; the cost is objects x frames and is surfaced to the user
before launch. Bidirectional: forward (ref -> end) then backward (ref-1 -> 0)
with independent per-direction status; the reference frame appears exactly
once (forward pass).
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from common.jobs import JobCancelled

SAM3_PYTHON_DEFAULT = Path.home() / ".venvs" / "sam3" / "bin" / "python"
WORKER_SCRIPT = Path(__file__).resolve().parent / "sam3_worker.py"
VENDORED_INFERENCE = (Path(__file__).resolve().parent.parent.parent
                      / "sam3-main" / "sam3" / "model"
                      / "sam3_video_inference.py")


class TrackingError(RuntimeError):
    pass


def sam3_python() -> Path:
    return Path(os.environ.get("OBJTRACK_SAM3_PYTHON", SAM3_PYTHON_DEFAULT))


def verify_local_patch() -> bool:
    """The point-prompt LOCAL PATCH must be present in the vendored SAM3."""
    try:
        src = VENDORED_INFERENCE.read_text()
    except OSError:
        return False
    return "LOCAL PATCH" in src and src.count("_THIS_FRAME_HAS_OUTPUTS_") >= 2


def gpu_available() -> tuple:
    """(ok, detail) via nvidia-smi — checked before any real run."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return False, out.stderr.strip() or "nvidia-smi failed"
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return bool(lines), "; ".join(lines)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"nvidia-smi unavailable: {e}"


class WorkerHandle:
    """One SAM3 worker subprocess bound to one video (kept alive between
    previews so the model loads once)."""

    def __init__(self, video_path: Path, log_path: Path,
                 score_thresh: float = 0.5, version: str = "sam3",
                 python: Optional[Path] = None,
                 worker_script: Optional[Path] = None,
                 require_patch: bool = True,
                 init_timeout_s: float = 1800.0):
        self.video_path = Path(video_path)
        self.log_path = Path(log_path)
        self.score_thresh = score_thresh
        self.version = version
        self._python = Path(python) if python else sam3_python()
        self._worker_script = Path(worker_script) if worker_script \
            else WORKER_SCRIPT
        self._require_patch = require_patch
        self._init_timeout_s = init_timeout_s
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()   # one in-flight command at a time
        self._q: Optional[queue.Queue] = None   # stdout events (reader thread)
        self.ready_info: Optional[dict] = None

    # ------------------------------------------------------------ lifecycle
    def start(self, phase_cb: Optional[Callable[[str], None]] = None) -> dict:
        if self._proc is not None and self._proc.poll() is None \
                and self.ready_info:
            return self.ready_info
        if not self._python.exists():
            raise TrackingError(
                f"SAM3 python not found at {self._python} — set "
                f"OBJTRACK_SAM3_PYTHON to a torch+sam3 environment")
        if self._require_patch and not verify_local_patch():
            raise TrackingError(
                "the required LOCAL PATCH is missing from the vendored "
                "sam3_video_inference.py (point-prompt propagation would "
                "fail) — re-apply it per pipeline/README.md section 6")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = open(self.log_path, "ab")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self._proc = subprocess.Popen(
            [str(self._python), str(self._worker_script)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=logf,
            text=True, bufsize=1, start_new_session=True, env=env)
        # dedicated reader thread -> queue: deadlines can fire even while the
        # worker is silent (a blocking readline would ignore them), and EOF
        # is delivered as a sentinel
        self._q = queue.Queue()

        def _reader(proc=self._proc, q=self._q):
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    q.put(json.loads(line))
                except ValueError:
                    continue
            q.put(None)                      # EOF sentinel
        threading.Thread(target=_reader, daemon=True).start()
        self._send({"cmd": "init", "video": str(self.video_path),
                    "version": self.version,
                    "score_thresh": self.score_thresh})
        deadline = time.monotonic() + self._init_timeout_s
        for ev in self._events(deadline=deadline):
            if ev.get("event") == "phase" and phase_cb:
                phase_cb(ev["phase"])
            elif ev.get("event") == "ready":
                self.ready_info = ev
                return ev
            elif ev.get("event") == "error":
                raise TrackingError(f"worker init failed: {ev['error']}")
        raise TrackingError("worker exited during init — see "
                            + str(self.log_path))

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _send(self, obj: dict):
        if not self.alive:
            raise TrackingError("SAM3 worker is not running")
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _events(self, deadline: Optional[float] = None):
        """Yield stdout events (from the reader queue) until EOF/deadline.

        A deadline expiry means the single-in-flight protocol is DESYNCED
        (a late event would be attributed to the next command), so callers
        must fail-stop the worker — _timeout() does that and raises.
        """
        while True:
            timeout = None
            if deadline is not None:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    self._timeout()
            try:
                ev = self._q.get(timeout=timeout)
            except queue.Empty:
                self._timeout()
                continue  # unreachable (timeout raises)
            if ev is None:
                return  # worker exited
            yield ev

    def _timeout(self):
        """Timeout = protocol desync: kill the worker so no stale event can
        ever be attributed to a later command, then raise."""
        self.stop(grace_s=0.5)
        raise TrackingError(
            "timed out waiting for the SAM3 worker — the worker was stopped "
            "to prevent response desync; start it again to continue "
            f"(log: {self.log_path})")

    def cancel_current(self):
        """Async cancel of an in-flight track command."""
        if self.alive:
            try:
                self._proc.stdin.write(json.dumps({"cmd": "cancel"}) + "\n")
                self._proc.stdin.flush()
            except OSError:
                pass

    def stop(self, grace_s: float = 8.0):
        """Graceful shutdown, then SIGKILL of the whole process group so no
        SAM3 multi-GPU child worker survives."""
        proc = self._proc
        self._proc = None
        self.ready_info = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                    proc.stdin.flush()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=grace_s)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

    # ------------------------------------------------------------ commands
    def preview(self, points, labels, ref_frame, width, height,
                timeout_s: float = 300.0) -> dict:
        with self._lock:
            self._send({"cmd": "preview", "points": points, "labels": labels,
                        "ref_frame": ref_frame, "width": width,
                        "height": height})
            deadline = time.monotonic() + timeout_s
            for ev in self._events(deadline=deadline):
                if ev.get("event") == "preview":
                    return ev
                if ev.get("event") == "error":
                    raise TrackingError(ev["error"])
        self.stop(grace_s=0.5)
        raise TrackingError("worker exited during preview — see "
                            + str(self.log_path))

    def track(self, object_id, points, labels, ref_frame, width, height,
              db_path: Path,
              progress_cb: Optional[Callable[[dict], None]] = None,
              cancel_event: Optional[threading.Event] = None,
              timeout_s: float = 4 * 3600.0) -> dict:
        with self._lock:
            self._send({"cmd": "track", "object_id": object_id,
                        "points": points, "labels": labels,
                        "ref_frame": ref_frame, "width": width,
                        "height": height, "db_path": str(db_path)})
            deadline = time.monotonic() + timeout_s
            watcher_stop = threading.Event()
            if cancel_event is not None:
                def watch():
                    while not watcher_stop.is_set():
                        if cancel_event.is_set():
                            self.cancel_current()
                            return
                        time.sleep(0.3)
                threading.Thread(target=watch, daemon=True).start()
            try:
                for ev in self._events(deadline=deadline):
                    kind = ev.get("event")
                    # correlation: only accept events for THIS object; a
                    # mismatched id means protocol desync -> fail-stop
                    if kind in ("track_progress", "track_done",
                                "track_cancelled") \
                            and ev.get("object_id") != object_id:
                        self.stop(grace_s=0.5)
                        raise TrackingError(
                            f"worker response desync: got event for "
                            f"{ev.get('object_id')!r} while tracking "
                            f"{object_id!r} — worker stopped")
                    if kind == "track_progress" and progress_cb:
                        progress_cb(ev)
                    elif kind == "track_done":
                        return ev
                    elif kind == "track_cancelled":
                        raise JobCancelled("tracking cancelled")
                    elif kind == "error":
                        raise TrackingError(ev["error"])
            finally:
                watcher_stop.set()
        self.stop(grace_s=0.5)
        raise TrackingError("worker exited during tracking — see "
                            + str(self.log_path))


# --------------------------------------------------------------------------
# merge + validation of per-object results (worker-agnostic, unit-tested)
# --------------------------------------------------------------------------

def _normalize_rows(rows) -> list:
    """Accept list[(frame, direction)], list[dict], or a DataFrame-like and
    return [(frame_index, direction)]. The SERVER always passes plain
    tuples/dicts — DataFrames appear only in the isolated publication
    subprocess (pandas must never be imported in the server process)."""
    if hasattr(rows, "itertuples"):        # DataFrame (subprocess-side only)
        return [(int(r.frame_index), str(r.tracking_direction))
                for r in rows.itertuples(index=False)]
    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append((int(r["frame_index"]), str(r["tracking_direction"])))
        else:
            out.append((int(r[0]), str(r[1])))
    return out


def validate_track_rows(rows, *, object_id: str, n_frames: int,
                        ref_frame: int) -> list:
    """Structural validation of one object's rows (pandas-free)."""
    pairs = _normalize_rows(rows)
    problems = []
    if not pairs:
        return problems                     # legitimate: object never found
    frames = [f for f, _ in pairs]
    if min(frames) < 0 or max(frames) >= n_frames:
        problems.append(f"{object_id}: frame_index outside 0..{n_frames - 1}")
    n_dup = len(frames) - len(set(frames))
    if n_dup:
        # one sam_local_id per pass -> a duplicated frame means the forward/
        # backward merge produced incompatible duplicates
        problems.append(
            f"{object_id}: {n_dup} duplicate frame rows after "
            f"direction merge (incompatible forward/backward results)")
    if any(d not in ("forward", "backward") for _, d in pairs):
        problems.append(f"{object_id}: invalid tracking_direction values")
    if any(f < ref_frame for f, d in pairs if d == "forward"):
        problems.append(f"{object_id}: forward rows before reference frame")
    if any(f >= ref_frame for f, d in pairs if d == "backward"):
        problems.append(f"{object_id}: backward rows at/after reference frame")
    return problems


def presence_summary(rows, n_frames: int) -> dict:
    """Presence/coverage/gap summary used by the review UI + manifest
    (pandas-free; accepts the same row shapes as validate_track_rows)."""
    pairs = _normalize_rows(rows)
    if not pairs:
        return {"frames_present": 0, "first_frame": None, "last_frame": None,
                "coverage_of_span": None, "gaps": [], "n_gaps": 0}
    frames = sorted({f for f, _ in pairs})
    first, last = frames[0], frames[-1]
    span = last - first + 1
    gaps = []
    prev = first
    for f in frames[1:]:
        if f > prev + 1:
            gaps.append([prev + 1, f - 1])
        prev = f
    return {"frames_present": len(frames), "first_frame": first,
            "last_frame": last,
            "coverage_of_span": round(len(frames) / span, 4),
            "gaps": gaps[:50], "n_gaps": len(gaps)}
