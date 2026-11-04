"""Run the isolated Aria helper as a subprocess and parse its protocol.

The helper is native-code-bearing (VRS/calibration). This module is what
server processes are allowed to import: it only uses subprocess + json.
Every abnormal termination — nonzero exit, signal, segfault, missing
result — becomes a clear `AriaHelperError`, never an exception escaping
from native code inside the caller.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Callable, Optional

from . import config as aria_config

PREFIX = "@@ARIA@@ "
LOG_LINES = 200


class AriaHelperError(RuntimeError):
    def __init__(self, message, *, kind="helper_failed", log=None):
        super().__init__(message)
        self.kind = kind
        self.log = list(log or [])


class AriaHelperCancelled(AriaHelperError):
    def __init__(self, message="cancelled", **kw):
        super().__init__(message, kind="cancelled", **kw)


def _signal_name(rc: int) -> str:
    try:
        return signal.Signals(-rc).name
    except (ValueError, TypeError):
        return f"signal {-rc}"


def run_helper(args: list, *,
               progress: Optional[Callable[[str, float], None]] = None,
               cancel: Optional[Callable[[], bool]] = None,
               timeout: float = 3600.0,
               python: Optional[Path] = None) -> dict:
    """Run `helper_cli.py <args>` and return its `result` event payload."""
    py = Path(python or aria_config.aria_python())
    if not py.is_file():
        raise AriaHelperError(
            f"the Aria interpreter is not available at {py}. Create it and "
            f"install projectaria-tools, or set "
            f"{aria_config.ENV_PYTHON}.", kind="env_missing")
    cmd = [str(py), str(aria_config.helper_path()), *args]
    log = deque(maxlen=LOG_LINES)
    result = {}
    error = {}

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    except OSError as e:
        raise AriaHelperError(f"cannot launch the Aria helper: {e}",
                              kind="launch_failed")

    def pump():
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                # the prefix may be preceded by logger colour escapes
                at = line.find(PREFIX)
                if at >= 0:
                    try:
                        ev = json.loads(line[at + len(PREFIX):])
                    except json.JSONDecodeError:
                        log.append(line[:400])
                        continue
                    kind = ev.get("event")
                    if kind == "progress" and progress:
                        progress(ev.get("phase", ""),
                                 float(ev.get("frac", 0.0)))
                    elif kind == "result":
                        result.update(ev)
                    elif kind == "error":
                        error.update(ev)
                else:
                    if line.strip():
                        log.append(line[:400])
        except Exception:                                # noqa: BLE001
            pass

    t = threading.Thread(target=pump, daemon=True)
    t.start()

    killed_for_cancel = False
    waited = 0.0
    step = 0.25
    while True:
        try:
            proc.wait(timeout=step)
            break
        except subprocess.TimeoutExpired:
            waited += step
            if cancel and cancel():
                killed_for_cancel = True
                _terminate(proc)
                break
            if waited > timeout:
                _terminate(proc)
                raise AriaHelperError(
                    f"the Aria helper exceeded {timeout:.0f}s and was "
                    f"stopped", kind="timeout", log=list(log))
    t.join(timeout=5)
    rc = proc.poll()

    if killed_for_cancel:
        raise AriaHelperCancelled("Aria preparation cancelled by request",
                                  log=list(log))
    if error:
        raise AriaHelperError(
            f"{error.get('kind', 'error')}: {error.get('message', '')}",
            kind="helper_error", log=list(log))
    if rc is not None and rc < 0:
        raise AriaHelperError(
            f"the Aria helper terminated on {_signal_name(rc)} — the "
            f"native VRS/calibration code crashed. No output was "
            f"published; source files and cached results are "
            f"untouched.", kind="crashed", log=list(log))
    if rc != 0:
        raise AriaHelperError(
            f"the Aria helper exited with code {rc}", kind="exit_nonzero",
            log=list(log))
    if not result:
        raise AriaHelperError(
            "the Aria helper produced no result", kind="no_result",
            log=list(log))
    result["helper_log_tail"] = list(log)[-20:]
    return result


def _terminate(proc) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def probe_vrs(vrs_path, **kw) -> dict:
    return run_helper(["probe", "--vrs", str(vrs_path)], **kw)
