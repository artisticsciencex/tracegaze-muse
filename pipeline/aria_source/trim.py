"""Trim a prepared Aria recording into a `trimmed_aria_recording`.

The user-facing interaction is identical to the verified NEON trimmer —
an INCLUSIVE `[start_frame, end_frame]` selection — and the interval math
and frame-accurate cut are the SAME verified source-agnostic code
(`neon_trim.core.plan_trim` / `common.media.run_cut`), reused rather than
reimplemented. Only the manifest differs: it declares the Aria artifact
type, the Aria timestamp domain, and Aria provenance.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from common.artifacts import ArtifactError, AtomicPublisher, sha256_file
from common.media import decode_all_pts, probe_video, run_cut

from . import (ARIA_TRIM_MANIFEST, SOURCE_SYSTEM, TRIMMED_ARIA_TYPE,
               __version__)
from .artifact import (ARIA_SCHEMA_VERSION, TIMESTAMP_DOMAIN_NOTE,
                       load_prepared_aria)


def trim_aria_recording(
        prepared_dir, start_frame: int, end_frame: int, out_dir, *,
        progress_cb: Optional[Callable[[str, float], None]] = None,
        cancel_event=None,
        path_guard: Optional[Callable[[Path], bool]] = None) -> dict:
    """Inclusive [start_frame, end_frame] selection → published manifest."""
    from neon_trim.core import (load_timestamp_csv, plan_trim,
                                rows_in_interval, trim_csv)

    def report(phase, frac):
        if progress_cb:
            progress_cb(phase, frac)

    def cancelled():
        return cancel_event is not None and cancel_event.is_set()

    art = load_prepared_aria(prepared_dir)
    report("loading prepared recording", 0.02)
    world = load_timestamp_csv(art.world_timestamps_path)
    gaze = load_timestamp_csv(art.gaze_path)
    vp = probe_video(art.video_path)
    n = len(world.ts_ns)
    if vp.n_frames != n:
        raise ArtifactError(
            f"prepared artifact is inconsistent: {vp.n_frames} video frames "
            f"vs {n} frame timestamps")

    src_hashes = {"video": sha256_file(art.video_path),
                  "gaze": sha256_file(art.gaze_path),
                  "world_timestamps": sha256_file(
                      art.world_timestamps_path)}

    # the recording's exclusive end boundary: last frame + median interval
    gaps = [b - a for a, b in zip(world.ts_ns, world.ts_ns[1:])]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 33_000_000
    rec_end_ns = world.ts_ns[-1] + median_gap

    plan = plan_trim(world.ts_ns, vp.pts_s, vp.duration_s,
                     start_frame, end_frame, rec_end_ns)
    report("cutting scene video", 0.08)

    pub = AtomicPublisher(Path(out_dir), source_dir=art.artifact_dir,
                          path_guard=path_guard,
                          tmp_prefix=".trimmed_aria_tmp_")
    try:
        tmp = pub.tmp_dir
        enc = run_cut(art.video_path, tmp / "scene.mp4",
                      plan.cut_start_s, plan.cut_end_s,
                      has_audio=vp.has_audio,
                      container_start_s=vp.container_start_s,
                      progress_cb=lambda f: report("cutting scene video",
                                                   0.08 + 0.6 * f),
                      cancel_event=cancel_event)
        if cancelled():
            raise ArtifactError("trim cancelled")

        report("trimming timestamp and gaze tables", 0.72)
        n_world = trim_csv(world, plan.abs_start_ns, plan.abs_end_ns,
                           tmp / "world_timestamps.csv")
        n_gaze = trim_csv(gaze, plan.abs_start_ns, plan.abs_end_ns,
                          tmp / "gaze.csv")
        lo, hi = rows_in_interval(gaze, plan.abs_start_ns, plan.abs_end_ns)

        report("validating", 0.85)
        validation = []

        def check(name, ok, detail=""):
            validation.append({"check": name, "ok": bool(ok),
                               "detail": detail})

        out_pts = decode_all_pts(tmp / "scene.mp4")
        out_world = load_timestamp_csv(tmp / "world_timestamps.csv")
        check("selected frame count exported",
              len(out_pts) == plan.n_frames_selected,
              f"{len(out_pts)} video frames vs "
              f"{plan.n_frames_selected} selected")
        check("one frame timestamp per exported frame",
              n_world == len(out_pts),
              f"{n_world} timestamp rows vs {len(out_pts)} frames")
        check("exported timestamps are the selected ones, in order",
              out_world.ts_ns == world.ts_ns[start_frame:end_frame + 1],
              f"[{out_world.ts_ns[0]}, {out_world.ts_ns[-1]}] ns")
        check("frame timestamps strictly increasing",
              all(b > a for a, b in zip(out_world.ts_ns,
                                        out_world.ts_ns[1:])),
              "Aria DEVICE_TIME, exact integers")
        check("gaze rows lie inside the half-open interval",
              all(plan.abs_start_ns <= t < plan.abs_end_ns
                  for t in gaze.ts_ns[lo:hi]),
              f"{n_gaze} rows in [{plan.abs_start_ns}, "
              f"{plan.abs_end_ns})")
        check("gaze timestamps preserved exactly",
              [t for t in gaze.ts_ns[lo:hi]]
              == load_timestamp_csv(tmp / "gaze.csv").ts_ns,
              "original integer nanoseconds copied verbatim")
        out_probe = probe_video(tmp / "scene.mp4")
        check("orientation and dimensions preserved",
              (out_probe.width == vp.width
               and out_probe.height == vp.height),
              f"{out_probe.width}x{out_probe.height} "
              f"({art.manifest['orientation']['transform']})")
        # relative PTS spacing must mirror the selected device-time spacing
        rel_dev = [(t - out_world.ts_ns[0]) / 1e9 for t in out_world.ts_ns]
        rel_pts = [p - out_pts[0] for p in out_pts]
        max_dev = max((abs(a - b) for a, b in zip(rel_pts, rel_dev)),
                      default=0.0)
        check("video PTS follow the selected device-time spacing",
              max_dev <= 0.002, f"max deviation {max_dev * 1e3:.3f} ms")
        rehash = {"video": sha256_file(art.video_path),
                  "gaze": sha256_file(art.gaze_path),
                  "world_timestamps": sha256_file(
                      art.world_timestamps_path)}
        check("parent artifact unchanged (sha256)", rehash == src_hashes,
              "prepared scene/gaze/world re-hashed identical")

        failed = [v for v in validation if not v["ok"]]
        if failed:
            raise ArtifactError("trim validation failed:\n" + "\n".join(
                f"  x {v['check']}: {v['detail']}" for v in failed))

        pm = art.manifest
        manifest = {
            "schema_version": ARIA_SCHEMA_VERSION,
            "artifact_type": TRIMMED_ARIA_TYPE,
            "artifact_id": uuid.uuid4().hex,
            "source_system": SOURCE_SYSTEM,
            "dataset": pm.get("dataset"),
            "dataset_release": pm.get("dataset_release"),
            "created_utc": datetime.now(timezone.utc)
                .isoformat(timespec="seconds"),
            "tool": {"name": "aria_source", "version": __version__},
            "files": {"manifest": ARIA_TRIM_MANIFEST,
                      "video": "scene.mp4", "gaze": "gaze.csv",
                      "world_timestamps": "world_timestamps.csv"},
            "parent_artifact": {
                "artifact_id": art.artifact_id,
                "artifact_type": art.artifact_type,
                "artifact_dir": str(art.artifact_dir),
                "files_sha256": src_hashes,
            },
            "recording": dict(pm.get("recording", {})),
            "selection": {
                "first_frame": plan.start_frame,
                "last_frame": plan.end_frame,
                "n_frames": plan.n_frames_selected,
                "absolute_start_ns": plan.abs_start_ns,
                "absolute_end_boundary_ns": plan.abs_end_ns,
                "interval_semantics":
                    "[start_frame_time, time_after_last_frame) — both "
                    "endpoint frames included; the UI selection is "
                    "inclusive, the interval is half-open",
                "selected_duration_s": plan.duration_s,
                "relative_start_s": plan.rel_start_s,
                "relative_end_s": plan.rel_end_s,
            },
            "counts": {"source_frames": n,
                       "output_frames": len(out_pts),
                       "source_gaze_rows": len(gaze.ts_ns),
                       "retained_gaze_rows": n_gaze},
            "video": {"n_frames": len(out_pts),
                      "width": out_probe.width, "height": out_probe.height,
                      "duration_s": out_probe.duration_s,
                      "has_audio": out_probe.has_audio},
            "rgb_stream": dict(pm.get("rgb_stream", {})),
            "timestamps": {
                "domain": pm["timestamps"]["domain"],
                "units": pm["timestamps"]["units"],
                "first_ns": out_world.ts_ns[0],
                "last_ns": out_world.ts_ns[-1],
                "compatibility_note": TIMESTAMP_DOMAIN_NOTE,
            },
            "orientation": dict(pm.get("orientation", {})),
            "gaze": {**{k: v for k, v in pm.get("gaze", {}).items()
                        if k in ("model", "model_reason",
                                 "projection_method", "fallback_depth_m",
                                 "not_provided_note")},
                     "rows_retained": n_gaze},
            "audio": dict(pm.get("audio", {})),
            "environment": dict(pm.get("environment", {})),
            "video_encoding": enc,
            "validation": validation,
            "output_directory": str(pub.out_dir),
        }
        (tmp / ARIA_TRIM_MANIFEST).write_text(json.dumps(manifest, indent=2)
                                              + "\n")
        report("publishing", 0.97)
        pub.publish()
        report("done", 1.0)
        return manifest
    finally:
        pub.cleanup()
