"""CLI: fuse gaze with a tracked_objects artifact.

    ~/.venvs/neon-trim/bin/python -m gaze_object_fusion \
        --artifact /path/to/<rec>_tracked_objects [--out DIR] \
        [--sanity-gap-ms 200] [--no-demo]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def default_out_dir(tracked_dir: Path) -> Path:
    if tracked_dir.is_file():          # manifest path given -> its directory
        tracked_dir = tracked_dir.parent
    name = tracked_dir.name
    if name.endswith("_tracked_objects"):
        name = name[: -len("_tracked_objects")]
    return tracked_dir.parent / f"{name}_gaze_fusion"


def main(argv=None) -> int:
    from .artifact import publish_fusion_artifact
    from .timecells import DEFAULT_SANITY_GAP_MS

    ap = argparse.ArgumentParser(
        prog="gaze_object_fusion",
        description="Fuse NEON gaze samples with tracked object masks "
                    "(timestamp-defined frame cells; deterministic hit "
                    "ordering; time-faithful demo video).")
    ap.add_argument("--artifact", required=True,
                    help="tracked_objects artifact directory")
    ap.add_argument("--out", default=None,
                    help="output directory (default: sibling "
                         "<name>_gaze_fusion; refuses to overwrite)")
    ap.add_argument("--sanity-gap-ms", type=float,
                    default=DEFAULT_SANITY_GAP_MS,
                    help="world-timestamp gap REPORTING threshold (never "
                         "drops samples; default %(default)s)")
    ap.add_argument("--no-demo", action="store_true",
                    help="skip the demonstration video")
    ap.add_argument("--summary-json", default=None,
                    help="also write the run summary as JSON to this path "
                         "(used by the GUI orchestrator)")
    args = ap.parse_args(argv)

    tracked = Path(args.artifact).expanduser().resolve()
    out = (Path(args.out).expanduser().resolve() if args.out
           else default_out_dir(tracked))

    def progress(phase, frac):
        print(f"[{frac * 100:5.1f}%] {phase}", flush=True)

    from common.artifacts import ArtifactError
    from .timecells import TimeCellError
    try:
        manifest = publish_fusion_artifact(
            tracked, out, sanity_gap_ms=args.sanity_gap_ms,
            make_demo=not args.no_demo, progress_cb=progress)
    except (ArtifactError, TimeCellError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    summary = {
        "published": manifest["output_directory"],
        "artifact_id": manifest["artifact_id"],
        "counts": manifest["counts"],
        "demo": manifest["demo"],
        "anomalous_gaps": len(
            manifest["assignment_rule"]["anomalous_gaps"]),
    }
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
