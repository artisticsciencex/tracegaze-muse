"""Isolated Arrow/Parquet helper — the ONLY sanctioned bridge between the
GUI server and Pandas/PyArrow.

pyarrow segfaulted the FastAPI process twice (kernel log:
``segfault ... in libarrow.so.2500``), so every Parquet touch now happens in
a short-lived subprocess launched through :func:`run_helper`. A native crash
(SIGSEGV / exit 139) in this helper produces a clear job error and leaves
the server running.

Subcommands (JSON result on the last stdout line):

  import-parquet   validate an (orphaned) per-object parquet and import it
                   ATOMICALLY into the session SQLite store
  inspect-artifact load + fully validate a tracked_objects artifact
                   (the /api/artifact endpoint runs this)

This module's TOP LEVEL must stay pandas-free: the server imports it for
:func:`run_helper`; heavy imports live inside the subcommand functions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HELPER_PATH = Path(__file__).resolve()


class HelperError(RuntimeError):
    pass


def run_helper(args: list, timeout_s: float = 600.0) -> dict:
    """Run this module as a subprocess; return the parsed JSON result.

    Maps native crashes to a precise, non-fatal error message — the GUI
    server must survive any helper death.
    """
    cmd = [sys.executable, str(HELPER_PATH)] + [str(a) for a in args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise HelperError(f"table helper timed out after {timeout_s:.0f}s")
    if proc.returncode != 0:
        if proc.returncode in (-11, 139):
            raise HelperError(
                "table helper crashed with a segmentation fault (exit 139) "
                "in native code — the GUI server is unaffected; retry, or "
                "rerun the object if the input file is corrupt")
        tail = (proc.stderr or "").strip()[-400:]
        # a JSON error line beats a stderr tail
        for line in reversed((proc.stdout or "").splitlines()):
            try:
                obj = json.loads(line)
                if "error" in obj:
                    raise HelperError(obj["error"])
            except ValueError:
                continue
        raise HelperError(f"table helper failed (exit {proc.returncode})"
                          + (f": {tail}" if tail else ""))
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if "error" in obj:
            raise HelperError(obj["error"])
        return obj
    raise HelperError("table helper produced no result")


# ------------------------------------------------------------- subcommands

def _cmd_import_parquet(args) -> dict:
    """Validate + atomically import one object's parquet into SQLite."""
    import pandas as pd                      # subprocess only

    from object_tracking import trackstore
    from object_tracking.tracking import validate_track_rows

    df = pd.read_parquet(args.parquet)
    required = {"frame_index", "tracking_direction", "sam_local_id", "score",
                "bbox_x", "bbox_y", "bbox_w", "bbox_h", "mask_area",
                "mask_rle"}
    missing = required - set(df.columns)
    if missing:
        return {"error": f"parquet lacks columns: {sorted(missing)}"}
    rows = df.to_dict("records")
    problems = validate_track_rows(
        rows, object_id=args.object, n_frames=args.n_frames,
        ref_frame=args.ref_frame)
    if problems:
        return {"error": "orphaned track failed validation: "
                         + "; ".join(problems)}
    n = trackstore.replace_object_rows(args.db, args.object, rows)
    frames = sorted({int(r["frame_index"]) for r in rows})
    dirs = {d for r in rows for d in [r["tracking_direction"]]}
    return {"ok": True, "rows": n, "frames_present": len(frames),
            "first_frame": frames[0] if frames else None,
            "last_frame": frames[-1] if frames else None,
            "forward_rows": sum(1 for r in rows
                                if r["tracking_direction"] == "forward"),
            "backward_rows": sum(1 for r in rows
                                 if r["tracking_direction"] == "backward"),
            "directions": sorted(dirs)}


def _cmd_inspect_artifact(args) -> dict:
    from object_tracking.artifact import load_tracked_artifact  # pandas inside

    art = load_tracked_artifact(args.path)
    parents = {k: (str(v) if v else None)
               for k, v in art.parent_paths().items()}
    resolvable = {k: (v is not None and Path(v).is_file())
                  for k, v in parents.items()}
    return {"ok": True, "artifact_dir": str(art.artifact_dir),
            "artifact_id": art.artifact_id,
            "parent_paths": parents, "parent_resolvable": resolvable,
            "manifest": art.manifest}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="isolated parquet/artifact helper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("import-parquet")
    p1.add_argument("--db", required=True)
    p1.add_argument("--object", required=True)
    p1.add_argument("--parquet", required=True)
    p1.add_argument("--n-frames", type=int, required=True)
    p1.add_argument("--ref-frame", type=int, required=True)
    p2 = sub.add_parser("inspect-artifact")
    p2.add_argument("--path", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "import-parquet":
            result = _cmd_import_parquet(args)
        else:
            result = _cmd_inspect_artifact(args)
    except Exception as e:  # noqa: BLE001 — protocol-reported
        result = {"error": f"{type(e).__name__}: {e}"}
    sys.stdout.write(json.dumps(result) + "\n")
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
