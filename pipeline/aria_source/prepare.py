"""Prepare an Aria download folder into a `prepared_aria_recording`.

Server-safe: this module orchestrates only — cache extraction is stdlib
zipfile, and everything native runs in the isolated helper subprocess.
"""

from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from common.artifacts import ArtifactError, AtomicPublisher, sha256_file

from . import (ARIA_PREP_MANIFEST, PREPARED_ARIA_TYPE, SOURCE_SYSTEM,
               __version__)
from . import cache as cache_mod
from . import config as aria_config
from .artifact import ARIA_SCHEMA_VERSION, TIMESTAMP_DOMAIN_NOTE
from .detect import (PKG_META, PKG_MPS, PKG_PREVIEW, PKG_VRS,
                     inspect_aria_folder, validate_against_receipt)
from .helper import AriaHelperError, run_helper

DATASET_DEFAULT = "Project Aria"
GAZE_PERSONALIZED = "personalized"
GAZE_GENERAL = "general"


def _first_last_us(csv_path: Path) -> tuple:
    """(first, last, rows) tracking timestamps in µs — exact ints."""
    first = last = None
    rows = 0
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            v = (r.get("tracking_timestamp_us") or "").strip()
            if not v:
                continue
            t = int(v)
            first = t if first is None else first
            last = t
            rows += 1
    return first, last, rows


def choose_gaze_model(cache: cache_mod.RecordingCache,
                      device_time_range_ns: tuple) -> dict:
    """Prefer personalized gaze; fall back to general with a recorded
    reason. `device_time_range_ns` is the VRS RGB (first, last)."""
    t0, t1 = device_time_range_ns
    candidates = []
    for model in (GAZE_PERSONALIZED, GAZE_GENERAL):
        p = cache.gaze_csv(model)
        entry = {"model": model, "path": p, "present": p.is_file()}
        if entry["present"]:
            try:
                first, last, rows = _first_last_us(p)
                entry.update(rows=rows, first_us=first, last_us=last)
                if not rows:
                    entry["problem"] = "file is empty"
                else:
                    # temporal compatibility with the VRS RGB stream
                    overlap = (first * 1000 <= t1 and last * 1000 >= t0)
                    entry["overlaps_recording"] = overlap
                    if not overlap:
                        entry["problem"] = (
                            f"gaze span [{first * 1000}, {last * 1000}] ns "
                            f"does not overlap the RGB device-time span "
                            f"[{t0}, {t1}]")
            except (OSError, ValueError) as e:
                entry["problem"] = f"unparseable: {e}"
        else:
            entry["problem"] = "not present in the MPS archive"
        candidates.append(entry)

    for entry in candidates:
        if entry["present"] and "problem" not in entry:
            reason = (
                "personalized eye gaze is present, non-empty, parseable "
                "and temporally compatible with the VRS recording"
                if entry["model"] == GAZE_PERSONALIZED else
                "personalized eye gaze unusable — fell back to the general "
                "model: " + (candidates[0].get("problem") or "unavailable"))
            return {"model": entry["model"], "path": entry["path"],
                    "reason": reason, "rows": entry["rows"],
                    "candidates": [{k: (str(v) if isinstance(v, Path) else v)
                                    for k, v in c.items()}
                                   for c in candidates]}
    raise ArtifactError(
        "no usable eye-gaze model: "
        + "; ".join(f"{c['model']}: {c.get('problem')}" for c in candidates))


def prepare_aria_recording(
        folder, out_dir, *,
        cache_root: Optional[Path] = None,
        progress_cb: Optional[Callable[[str, float], None]] = None,
        cancel_event=None,
        frame_limit: int = 0,
        hash_sources: bool = True,
        path_guard: Optional[Callable[[Path], bool]] = None) -> dict:
    """Full preparation. Returns the published prep manifest."""
    def report(phase, frac):
        if progress_cb:
            progress_cb(phase, frac)

    def cancelled():
        return cancel_event is not None and cancel_event.is_set()

    folder = Path(folder)
    pkg = inspect_aria_folder(folder)
    if not pkg.ok:
        raise ArtifactError("not a usable Aria recording folder:\n  - "
                            + "\n  - ".join(pkg.problems))

    report("validating source package", 0.01)
    validation = list(validate_against_receipt(pkg))
    failed = [v for v in validation if not v["ok"]]
    if failed:
        raise ArtifactError(
            "download validation failed:\n" + "\n".join(
                f"  x {v['check']}: {v['detail']}" for v in failed))

    # source identity BEFORE any work — proven unchanged again afterwards
    source_stat = {k: {"path": str(p), "bytes": p.stat().st_size,
                       "mtime_ns": p.stat().st_mtime_ns}
                   for k, p in sorted(pkg.files.items())}

    cache = cache_mod.RecordingCache(pkg.recording_id,
                                     cache_root or aria_config.cache_dir())
    report("preparing cache", 0.03)
    cache_state = cache.ensure(
        pkg, progress=lambda ph, f: report(ph, 0.03 + 0.17 * f),
        cancel=cancelled, hash_sources=hash_sources)
    validation.append({
        "check": "working cache ready",
        "ok": True,
        "detail": ("reused an existing valid cache entry" if
                   cache_state.get("reused") else
                   f"rebuilt ({cache_state.get('reason')})")})
    if cancelled():
        raise ArtifactError("preparation cancelled")

    report("probing VRS RGB stream", 0.21)
    probe = run_helper(["probe", "--vrs", str(cache.vrs)],
                       progress=lambda ph, f: report(ph, 0.21),
                       cancel=cancelled)
    ts_range = (probe["first_device_time_ns"], probe["last_device_time_ns"])
    gaps = probe.get("gap_report", {})
    validation.append({
        "check": "RGB device timestamps strictly increasing",
        "ok": bool(gaps.get("strictly_increasing", True)),
        "detail": f"{probe['n_frames']} frames, median gap "
                  f"{gaps.get('median_gap_ms')} ms"})

    report("selecting gaze model", 0.23)
    choice = choose_gaze_model(cache, ts_range)
    validation.append({"check": "eye-gaze model selected", "ok": True,
                       "detail": f"{choice['model']}: {choice['reason']}"})

    pub = AtomicPublisher(Path(out_dir), source_dir=folder,
                          path_guard=path_guard,
                          tmp_prefix=".prepared_aria_tmp_")
    try:
        tmp = pub.tmp_dir
        report("extracting frames and projecting gaze", 0.25)
        res = run_helper(
            ["prepare", "--vrs", str(cache.vrs),
             "--eye-gaze", str(choice["path"]),
             "--gaze-model", choice["model"],
             "--recording-id", str(pkg.recording_id),
             "--out", str(tmp),
             *(["--frame-limit", str(frame_limit)] if frame_limit else [])],
            progress=lambda ph, f: report(ph, 0.25 + 0.65 * f),
            cancel=cancelled)

        report("validating prepared artifact", 0.92)
        scene = tmp / "scene.mp4"
        world = tmp / "world_timestamps.csv"
        gaze = tmp / "gaze.csv"
        for p in (scene, world, gaze):
            if not p.is_file():
                raise ArtifactError(f"helper did not produce {p.name}")

        from neon_trim.core import load_timestamp_csv
        wts = load_timestamp_csv(world)
        n_ts = len(wts.ts_ns)
        validation.append({
            "check": "one world_timestamps row per RGB frame",
            "ok": n_ts == res["n_frames"],
            "detail": f"{n_ts} rows vs {res['n_frames']} frames"})
        validation.append({
            "check": "frame timestamps strictly increasing",
            "ok": all(b > a for a, b in zip(wts.ts_ns, wts.ts_ns[1:])),
            "detail": f"[{wts.ts_ns[0]}, {wts.ts_ns[-1]}] ns "
                      f"(Aria DEVICE_TIME)"})

        from common.media import decode_all_pts, probe_video
        vp = probe_video(scene)
        pts = decode_all_pts(scene)
        validation.append({
            "check": "scene.mp4 frame count matches the timestamp table",
            "ok": len(pts) == n_ts,
            "detail": f"{len(pts)} decoded frames vs {n_ts} timestamps"})
        validation.append({
            "check": "scene.mp4 dimensions match the exported frames",
            "ok": (vp.width == res["width"] and vp.height == res["height"]),
            "detail": f"{vp.width}x{vp.height}"})
        # video PTS spacing must mirror the device-time spacing
        dev_rel = [(t - wts.ts_ns[0]) / 1e9 for t in wts.ts_ns]
        max_dev = max((abs(a - b) for a, b in zip(pts, dev_rel)),
                      default=0.0)
        validation.append({
            "check": "video PTS follow the RGB device-time spacing",
            "ok": max_dev <= 0.002,
            "detail": f"max deviation {max_dev * 1e3:.3f} ms"})

        n_gaze = sum(1 for _ in open(gaze)) - 1
        validation.append({
            "check": "gaze rows written", "ok": n_gaze > 0,
            "detail": f"{n_gaze} rows "
                      f"({res['gaze_projection_ok']} projected inside the "
                      f"image, {res['gaze_projection_outside_image']} "
                      f"outside, {res['gaze_projection_failed']} without a "
                      f"projection)"})

        report("verifying sources unchanged", 0.95)
        after = {k: {"path": str(p), "bytes": p.stat().st_size,
                     "mtime_ns": p.stat().st_mtime_ns}
                 for k, p in sorted(pkg.files.items())}
        validation.append({
            "check": "source files untouched",
            "ok": after == source_stat,
            "detail": "size+mtime identical for every downloaded source"})

        failed = [v for v in validation if not v["ok"]]
        if failed:
            raise ArtifactError("preparation validation failed:\n"
                                + "\n".join(f"  x {v['check']}: {v['detail']}"
                                            for v in failed))
        if cancelled():
            raise ArtifactError("preparation cancelled")

        meta = {}
        meta_path = cache.dir / cache_mod.METADATA
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                meta = {}

        cache_manifest = cache.read_manifest() or {}
        manifest = {
            "schema_version": ARIA_SCHEMA_VERSION,
            "artifact_type": PREPARED_ARIA_TYPE,
            "artifact_id": uuid.uuid4().hex,
            "source_system": SOURCE_SYSTEM,
            "dataset": pkg.dataset or DATASET_DEFAULT,
            "dataset_release": pkg.release,
            "source_layout": pkg.layout,
            "created_utc": datetime.now(timezone.utc)
                .isoformat(timespec="seconds"),
            "tool": {"name": "aria_source", "version": __version__},
            "files": {"manifest": ARIA_PREP_MANIFEST,
                      "video": "scene.mp4", "gaze": "gaze.csv",
                      "world_timestamps": "world_timestamps.csv"},
            "recording": {
                "recording_id": pkg.recording_id,
                "source_dir": str(folder),
                "task": (meta.get("campaign_task_name") or
                         (f"AEA script {meta['script']}"
                          if meta.get("script") is not None else None)),
                "aea_location": meta.get("location"),
                "aea_script": meta.get("script"),
                "aea_sequence": meta.get("sequence"),
                "aea_recording": meta.get("recording"),
                "capture_time_utc_start": meta.get("capture_time_utc_start"),
                "capture_time_utc_end": meta.get("capture_time_utc_end"),
                "reported_duration_s": meta.get("duration"),
                "notes": meta.get("notes"),
                "profile_name": meta.get("profile_name"),
            },
            "sources": {k: {**v,
                            "sha256": (cache_manifest.get("source_sha256")
                                       or {}).get(k)}
                        for k, v in source_stat.items()},
            "video": {"n_frames": res["n_frames"],
                      "n_frames_available": res["n_frames_available"],
                      "width": res["width"], "height": res["height"],
                      "duration_s": res["duration_s"],
                      "has_audio": bool(res["audio_present"]),
                      "time_base": res["video_time_base"]},
            "rgb_stream": {"stream_id": res["rgb_stream_id"],
                           "label": res["rgb_stream_label"]},
            "timestamps": {
                "domain": res["timestamp_domain"],
                "units": res["timestamp_units"],
                "first_ns": res["first_device_time_ns"],
                "last_ns": res["last_device_time_ns"],
                "compatibility_note": TIMESTAMP_DOMAIN_NOTE,
                "gap_report": res["gap_report"],
            },
            "orientation": {"transform": res["orientation"],
                            "detail": res["orientation_detail"]},
            "gaze": {
                "model": res["gaze_model"],
                "model_reason": choice["reason"],
                "model_candidates": choice["candidates"],
                "rows_total": res["gaze_rows_total"],
                "projection_ok": res["gaze_projection_ok"],
                "projection_outside_image":
                    res["gaze_projection_outside_image"],
                "projection_failed": res["gaze_projection_failed"],
                "depth_measured": res["gaze_depth_measured"],
                "depth_fallback": res["gaze_depth_fallback"],
                "fallback_depth_m": res["fallback_depth_m"],
                "projection_method":
                    "projectaria_tools get_gaze_vector_reprojection with the "
                    "device calibration and the CW90-rotated RGB camera "
                    "calibration (make_upright=True) — no approximate "
                    "pinhole model, no image-size scaling",
                "not_provided_note":
                    "NEON-specific worn/fixation/blink fields are NOT "
                    "invented for Aria; they are absent and read as null "
                    "downstream",
            },
            "audio": {"present": bool(res["audio_present"]),
                      "reason": res["audio_note"]},
            "cache": {"dir": str(cache.dir),
                      "reused": bool(cache_state.get("reused")),
                      "selected_members":
                          cache_manifest.get("selected_members"),
                      "omitted_note": cache_manifest.get("omitted_note")},
            "environment": res["versions"],
            "omitted": {
                "note": "the VRS recording, MPS archive and preview video "
                        "are NOT copied into this artifact; they remain in "
                        "the download folder and the working cache",
                "preview_video_used": False,
                "preview_video_note":
                    "the preview MP4 was not used as the canonical video: "
                    "scene.mp4 is generated directly from VRS RGB frames, "
                    "so frame-for-frame correspondence with the timestamp "
                    "table is guaranteed by construction",
            },
            "validation": validation,
            "output_directory": str(pub.out_dir),
        }
        (tmp / ARIA_PREP_MANIFEST).write_text(json.dumps(manifest, indent=2)
                                              + "\n")
        report("publishing", 0.98)
        pub.publish()
        report("done", 1.0)
        return manifest
    finally:
        pub.cleanup()
