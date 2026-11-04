"""Child-process manager: start / reuse / observe / stop module servers.

Rules enforced here (they are the reason this file exists):

- a module is REUSED when the configured port already answers with the
  expected module identity — never started twice
- a foreign process on a configured port is reported as a port conflict
  and NEVER killed
- only children the workbench started itself may be stopped, and a module
  reporting `busy` needs an explicit force (so an active tracking job is
  never terminated silently)
- a module crash is observed, logged (bounded), and reported — it can
  never propagate into the workbench process

Only the standard library is imported here: no HTTP client library, no
computation library, no module server.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

LOG_LINES = 200               # bounded diagnostic buffer per module
HEALTH_TIMEOUT = 1.5
START_TIMEOUT = 40.0
STOP_GRACE = 8.0

STOPPED = "stopped"
STARTING = "starting"
READY = "ready"
BUSY = "busy"
FAILED = "failed"
PORT_CONFLICT = "port_conflict"


class PortConflict(RuntimeError):
    pass


class StartFailed(RuntimeError):
    pass


class NotOwned(RuntimeError):
    pass


class ModuleBusy(RuntimeError):
    pass


def probe_health(url: str, timeout: float = HEALTH_TIMEOUT):
    """GET a health endpoint. Returns (payload | None, note).

    Never raises: a refused connection, a timeout, a non-JSON body and an
    HTTP error are all ordinary observations here.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            raw = r.read(64_000)
        return json.loads(raw.decode("utf-8", "replace")), "ok"
    except urllib.error.HTTPError as e:
        return None, f"http {e.code}"
    except urllib.error.URLError as e:
        return None, f"unreachable ({e.reason})"
    except json.JSONDecodeError:
        return None, "non-JSON response"
    except Exception as e:                              # noqa: BLE001
        return None, f"probe error ({type(e).__name__})"


class ProcessManager:
    def __init__(self, specs: dict, *, start_timeout: float = START_TIMEOUT):
        self.specs = specs
        self.start_timeout = start_timeout
        self._procs = {}                 # key -> Popen (workbench-owned only)
        self._logs = {k: deque(maxlen=LOG_LINES) for k in specs}
        self._errors = {}                # key -> last error string
        self._lock = threading.RLock()

    # ------------------------------------------------------------- logging
    def log(self, key: str, line: str) -> None:
        self._logs.setdefault(key, deque(maxlen=LOG_LINES)).append(
            f"{time.strftime('%H:%M:%S')} {line.rstrip()}"[:800])

    def logs(self, key: str) -> list:
        return list(self._logs.get(key, ()))

    def _pump(self, key: str, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stdout:                    # merged stderr
                self.log(key, line)
        except Exception:                               # noqa: BLE001
            pass
        finally:
            rc = proc.poll()
            if rc not in (None, 0):
                self.log(key, f"[process exited with code {rc}]")

    # -------------------------------------------------------------- status
    def owned(self, key: str):
        p = self._procs.get(key)
        if p is not None and p.poll() is not None:
            return None
        return p

    def status(self, key: str) -> dict:
        spec = self.specs[key]
        payload, note = probe_health(spec.health_url)
        proc = self._procs.get(key)
        alive = proc is not None and proc.poll() is None
        out = {"key": key, "port": spec.port, "url": spec.url,
               "owned": bool(alive), "pid": proc.pid if alive else None,
               "detail": "", "health": None,
               "error": self._errors.get(key)}
        if payload is not None:
            module = payload.get("module")
            if module != key:
                out["status"] = PORT_CONFLICT
                out["detail"] = (
                    f"port {spec.port} is answering, but it is not "
                    f"{key} (identity: {module or 'unknown'}). The "
                    f"workbench will not touch a process it does not own — "
                    f"free the port or run {key} elsewhere.")
                return out
            out["health"] = payload
            out["status"] = BUSY if payload.get("busy") else READY
            out["detail"] = (f"{key} {payload.get('version', '?')} "
                             f"{'busy' if payload.get('busy') else 'ready'}"
                             + ("" if alive else " (not started by the "
                                                 "workbench — reused)"))
            return out
        if alive:
            out["status"] = STARTING
            out["detail"] = f"started (pid {proc.pid}), waiting for health"
            return out
        if proc is not None:                            # exited
            out["status"] = FAILED
            out["detail"] = (self._errors.get(key)
                             or f"process exited with code {proc.poll()}")
            return out
        out["status"] = STOPPED
        out["detail"] = note if note != "ok" else "not running"
        return out

    def statuses(self) -> dict:
        return {k: self.status(k) for k in self.specs}

    # --------------------------------------------------------------- start
    def start(self, key: str, *, timeout: float = None) -> dict:
        """Start the module unless it is already usable. Idempotent."""
        spec = self.specs[key]
        timeout = self.start_timeout if timeout is None else timeout
        with self._lock:
            st = self.status(key)
            if st["status"] in (READY, BUSY):
                st["reused"] = True
                return st
            if st["status"] == PORT_CONFLICT:
                raise PortConflict(st["detail"])
            if st["status"] == STARTING:
                # someone already launched it — wait, never spawn a second
                return self._await_health(key, timeout, spawned=False)
            self._errors.pop(key, None)
            self.log(key, f"[workbench] launching: {' '.join(spec.cmd())}")
            try:
                proc = subprocess.Popen(
                    spec.cmd(), cwd=str(spec.cwd),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, start_new_session=True,
                    env={**os.environ, **spec.env})
            except (OSError, ValueError) as e:
                self._errors[key] = f"cannot launch {spec.executable}: {e}"
                self.log(key, f"[workbench] {self._errors[key]}")
                raise StartFailed(self._errors[key])
            self._procs[key] = proc
            threading.Thread(target=self._pump, args=(key, proc),
                             daemon=True).start()
        return self._await_health(key, timeout, spawned=True)

    def _await_health(self, key: str, timeout: float, *,
                      spawned: bool) -> dict:
        spec = self.specs[key]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            proc = self._procs.get(key)
            if proc is not None and proc.poll() is not None:
                tail = " | ".join(self.logs(key)[-6:])
                self._errors[key] = (
                    f"{key} exited during startup (code {proc.poll()}). "
                    f"Last output: {tail or 'none'}")
                raise StartFailed(self._errors[key])
            payload, _ = probe_health(spec.health_url, timeout=1.0)
            if payload is not None:
                if payload.get("module") == key:
                    st = self.status(key)
                    st["reused"] = not spawned
                    return st
                self._errors[key] = (
                    f"port {spec.port} answered with a different service "
                    f"(identity: {payload.get('module') or 'unknown'})")
                raise PortConflict(self._errors[key])
            time.sleep(0.35)
        tail = " | ".join(self.logs(key)[-6:])
        self._errors[key] = (
            f"{key} did not become healthy within {timeout:.0f}s. "
            f"Last output: {tail or 'none'}")
        raise StartFailed(self._errors[key])

    # ---------------------------------------------------------------- stop
    def stop(self, key: str, *, force: bool = False) -> dict:
        with self._lock:
            proc = self.owned(key)
            if proc is None:
                st = self.status(key)
                if st["status"] in (READY, BUSY, PORT_CONFLICT):
                    raise NotOwned(
                        f"the process serving port {self.specs[key].port} "
                        f"was not started by the workbench — stop it where "
                        f"it was started")
                return self.status(key)
            st = self.status(key)
            if st["status"] == BUSY and not force:
                raise ModuleBusy(
                    f"{key} reports an active job — stopping now would "
                    f"abort it. Confirm to stop anyway.")
            self.log(key, "[workbench] stopping (SIGTERM to process group)")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=STOP_GRACE)
            except subprocess.TimeoutExpired:
                self.log(key, "[workbench] grace expired — SIGKILL")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            self._procs.pop(key, None)
            self._errors.pop(key, None)
            return self.status(key)

    def shutdown(self) -> None:
        """Terminate every owned child (called on workbench shutdown)."""
        for key in list(self._procs):
            try:
                self.stop(key, force=True)
            except Exception as e:                      # noqa: BLE001
                self.log(key, f"[workbench] shutdown issue: {e}")
