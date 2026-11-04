"""Background job manager shared by pipeline servers.

FastAPI-free. Each job runs a callable in a daemon thread with a progress
dict, a cancel Event, and typed-cancellation classification. Server shutdown
cancels running jobs and joins their threads so no orphaned subprocess/GPU
work survives (job functions must honour their cancel event and kill any
child processes they spawned — see object_tracking.tracking for the SAM3
subprocess pattern).
"""

from __future__ import annotations

import threading
import uuid
from typing import Callable, Optional


class JobCancelled(RuntimeError):
    """Raise from a job function to mark the job as cancelled (not failed)."""


class JobManager:
    def __init__(self):
        self._jobs: dict = {}
        self._lock = threading.Lock()

    def create(self, fn: Callable[[dict], object], *,
               kind: str = "job", meta: Optional[dict] = None,
               cancel_exceptions: tuple = ()) -> str:
        """Start `fn(job)` in a thread. `fn` receives the live job dict and
        should update job['phase'] / job['progress'] and honour
        job['cancel'] (a threading.Event). Its return value becomes
        job['result']. Exceptions in `cancel_exceptions` (plus JobCancelled)
        mark the job cancelled; anything else marks it failed.
        """
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id, "kind": kind, "state": "queued", "phase": "queued",
            "progress": 0.0, "error": None, "result": None,
            "cancel": threading.Event(), "thread": None,
            "meta": dict(meta or {}),
        }
        with self._lock:
            self._jobs[job_id] = job

        def runner():
            job["state"] = "running"
            try:
                result = fn(job)
            except JobCancelled:
                job["state"] = "cancelled"
                job["error"] = "cancelled"
                return
            except cancel_exceptions:
                job["state"] = "cancelled"
                job["error"] = "cancelled"
                return
            except Exception as e:  # noqa: BLE001 — reported to the user
                job["state"] = "failed"
                job["error"] = f"{type(e).__name__}: {e}"
                return
            job["result"] = result
            job["progress"] = 1.0
            job["state"] = "done"

        t = threading.Thread(target=runner, daemon=True)
        job["thread"] = t
        t.start()
        return job_id

    def get(self, job_id: str) -> Optional[dict]:
        return self._jobs.get(job_id)

    def public_view(self, job_id: str, result_transform=None) -> Optional[dict]:
        job = self.get(job_id)
        if job is None:
            return None
        result = job["result"]
        if result is not None and result_transform is not None:
            result = result_transform(result)
        return {"id": job["id"], "kind": job["kind"], "state": job["state"],
                "phase": job["phase"], "progress": job["progress"],
                "error": job["error"], "result": result, "meta": job["meta"]}

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        job["cancel"].set()
        return True

    def running_jobs(self) -> list:
        with self._lock:
            return [j for j in self._jobs.values() if j["state"] in
                    ("queued", "running")]

    def shutdown(self, join_timeout: float = 20.0) -> None:
        """Cancel everything and join threads (server-shutdown cleanup)."""
        for job in self.running_jobs():
            job["cancel"].set()
        for job in self.running_jobs():
            t = job.get("thread")
            if t is not None and t.is_alive():
                t.join(timeout=join_timeout)
