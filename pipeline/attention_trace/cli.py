"""CLI: build an attention trace from a gaze-fusion artifact.

    ~/.venvs/neon-trim/bin/python -m attention_trace \
        --fusion /path/to/<rec>_gaze_fusion [--out DIR] [--quiet]

Run from `pipeline/` (or with `pipeline/` on PYTHONPATH). Refuses to
overwrite an existing output directory; the parent fusion artifact is
only ever read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def default_out_dir(fusion_dir: Path) -> Path:
    if fusion_dir.is_file():
        fusion_dir = fusion_dir.parent
    name = fusion_dir.name
    for suffix in ("_gaze_fusion", "_fusion"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return fusion_dir.parent / f"{name}_attention_trace"


def main(argv=None) -> int:
    from common.artifacts import ArtifactError
    from .artifact import publish_attention_trace

    ap = argparse.ArgumentParser(
        prog="attention_trace",
        description="Build an auditable relational attention trace "
                    "(episodes, transitions, per-object summary) from a "
                    "validated gaze-fusion artifact.")
    ap.add_argument("--fusion", required=True,
                    help="gaze_object_fusion artifact directory")
    ap.add_argument("--out", default=None,
                    help="output directory (default: sibling "
                         "<name>_attention_trace; never overwritten)")
    ap.add_argument("--quiet", action="store_true",
                    help="only print the final summary")
    args = ap.parse_args(argv)

    fusion = Path(args.fusion).expanduser().resolve()
    out = (Path(args.out).expanduser().resolve() if args.out
           else default_out_dir(fusion))
    if out.exists():
        print(f"error: output directory already exists: {out}",
              file=sys.stderr)
        return 1

    def progress(phase, frac):
        if not args.quiet:
            print(f"[{frac * 100:5.1f}%] {phase}", flush=True)

    try:
        m = publish_attention_trace(fusion, out, progress_cb=progress)
    except ArtifactError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:                                   # noqa: BLE001
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    c = m["counts"]
    print(f"\npublished → {m['output_directory']}")
    print(f"  parent fusion artifact : "
          f"{m['parent_artifact']['artifact_id']}")
    print(f"  timestamp domain       : {m['source']['timestamp_domain']}")
    print(f"  eligible / supported   : {c['eligible_samples']} / "
          f"{c['supported_samples']} of {c['authoritative_rows']} rows")
    print(f"  episodes / transitions : {c['episodes']} / {c['transitions']}")
    print(f"  objects attended       : {c['objects_attended']}")
    print(f"  supported duration     : {c['supported_duration_s']:.3f} s "
          f"(no-hit {c['no_hit_duration_ns'] / 1e9:.3f} s)")
    print(f"  transition types       : "
          f"{json.dumps(c['transition_types'], sort_keys=True)}")
    ok = sum(1 for v in m["validation"] if v["ok"])
    print(f"  validation             : {ok}/{len(m['validation'])} checks "
          f"passed")
    for v in m["validation"]:
        if not v["ok"]:
            print(f"    x {v['check']}: {v['detail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
