"""Publish a tracked-objects artifact from the command line / a subprocess.

    python -m object_tracking.publish_cli --src <recording_dir> \
        --out <artifact_dir> [--no-overlay] [--state-dir DIR] [--extra-json F]

The server runs THIS in a subprocess rather than publishing in-process:
pyarrow was observed to segfault inside the FastAPI thread context (verified
kernel log: `segfault ... in libarrow.so`), while the identical publish in a
fresh process succeeds. Subprocess isolation also means a publisher crash can
never take the GUI server down, and it doubles as the pipeline/CLI entry
point for publication.

Protocol on stdout (JSON lines): {"phase": str, "progress": float} updates,
then {"done": true, "manifest": {...}} or {"error": str}. SIGTERM cancels;
the atomic-publish cleanup runs on the way out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import threading
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from object_tracking import intake
from object_tracking.annotations import AnnotationStore
from object_tracking.artifact import publish_tracked_artifact


def default_state_dir() -> Path:
    return Path.home() / ".cache" / "object_tracking"


def session_workdir(state_dir: Path, src_dir: Path) -> Path:
    key = hashlib.sha1(str(src_dir).encode()).hexdigest()[:16]
    return Path(state_dir) / "sessions" / key


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="publish tracked_objects artifact")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--state-dir", default=str(default_state_dir()))
    ap.add_argument("--extra-json", default=None,
                    help="JSON file with worker_info/score_thresh/runtime_s")
    args = ap.parse_args(argv)

    cancel = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: cancel.set())

    def emit(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    try:
        src = Path(args.src).resolve()
        bundle = intake.open_bundle(src)
        workdir = session_workdir(Path(args.state_dir), src)
        store = AnnotationStore(workdir / "annotations.json", bundle.n_frames,
                                bundle.width or 0, bundle.height or 0)
        extra = {}
        if args.extra_json:
            extra = json.loads(Path(args.extra_json).read_text())
        manifest = publish_tracked_artifact(
            bundle, store, workdir / "tracks.db", Path(args.out),
            worker_info=extra.get("worker_info"),
            score_thresh=extra.get("score_thresh", 0.5),
            runtime_s=extra.get("runtime_s"),
            make_overlay=not args.no_overlay,
            progress_cb=lambda p, f: emit({"phase": p, "progress": f}),
            cancel_event=cancel)
        emit({"done": True, "manifest": manifest})
        return 0
    except Exception as e:  # noqa: BLE001 — reported over the protocol
        emit({"error": f"{type(e).__name__}: {e}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
