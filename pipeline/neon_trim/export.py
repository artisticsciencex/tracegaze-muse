"""Validated, atomic export of a trimmed NEON recording.

Output layout (deliberately NOT a complete NEON export — auxiliary streams
are omitted, never copied stale):

    <out_dir>/
        <source video name>            trimmed re-encoded MP4 (audio retained)
        gaze.csv                       rows with ts in [start, end), verbatim
        world_timestamps.csv           rows start_frame..end_frame, verbatim
        trim_manifest.json             provenance + sync + validation record

Atomicity: everything is written into a hidden temp dir next to the final
destination and os.rename()d into place only after ALL validations pass.
A failed or cancelled export leaves no partial final output.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from . import __version__
from . import artifact as artifact_mod
from . import core, media

OMITTED_FILES = ["fixations.csv", "saccades.csv", "blinks.csv", "imu.csv",
                 "3d_eye_states.csv", "events.csv", "labels.csv", "info.json",
                 "scene_camera.json"]


class ExportError(RuntimeError):
    pass


class ExportCancelled(ExportError):
    pass


@dataclass
class LoadedRecording:
    """Everything the server/exporter needs about a source recording."""
    rec_dir: Path
    video: media.VideoProbe
    world: core.TimestampCsv
    gaze: core.TimestampCsv
    info: core.RecordingInfo
    sync: core.SyncCheck
    rec_end_ns: int
    warnings: list = field(default_factory=list)


def load_recording(rec_dir: Path) -> LoadedRecording:
    rec_dir = Path(rec_dir).resolve()
    if not rec_dir.is_dir():
        raise ExportError(f"not a directory: {rec_dir}")
    for name in ("world_timestamps.csv", "gaze.csv"):
        if not (rec_dir / name).exists():
            raise ExportError(f"{name} not found in {rec_dir} — not a NEON export?")
    video_path = media.find_scene_video(rec_dir)
    video = media.probe_video(video_path)
    world = core.load_timestamp_csv(rec_dir / "world_timestamps.csv")
    gaze = core.load_timestamp_csv(rec_dir / "gaze.csv")
    info = core.load_recording_info(rec_dir)
    sync = core.check_sync(video.pts_s, world.ts_ns)
    rec_end = core.recording_end_ns(info, world.ts_ns)

    warnings = list(info.warnings) + list(video.warnings)
    if not world.sorted_non_decreasing:
        warnings.append("world_timestamps.csv is not sorted")
    if not gaze.sorted_non_decreasing:
        warnings.append("gaze.csv timestamps are not sorted (interval filtering "
                        "falls back to a linear scan)")
    return LoadedRecording(rec_dir=rec_dir, video=video, world=world, gaze=gaze,
                           info=info, sync=sync, rec_end_ns=rec_end,
                           warnings=warnings)


# ---------------------------------------------------------------------------

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _check(results: list, name: str, ok: bool, detail: str) -> bool:
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    return ok


def run_export(rec: LoadedRecording, start_frame: int, end_frame: int,
               out_dir: Path,
               progress_cb: Optional[Callable[[str, float], None]] = None,
               cancel_event: Optional[threading.Event] = None,
               path_guard: Optional[Callable[[Path], bool]] = None) -> dict:
    """Execute a validated, atomic trim export. Returns the manifest dict.

    progress_cb(phase, fraction 0..1) is called during the video encode and
    at phase transitions. path_guard (optional) is re-checked against the
    fully-resolved destination at claim time, so an authorization decision
    made earlier (e.g. by the HTTP layer) cannot be bypassed by swapping a
    parent directory for a symlink in between. Raises ExportError (with a
    diagnostic message) on any failure — in which case no final output
    directory exists.
    """
    def report(phase: str, frac: float):
        if progress_cb is not None:
            progress_cb(phase, frac)

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    # -- preconditions -------------------------------------------------------
    if not rec.sync.ok:
        raise ExportError(
            "synchronization could not be established — export blocked.\n"
            + rec.sync.message)
    if start_frame > end_frame:
        raise ExportError(f"start frame {start_frame} is after end frame {end_frame}")

    out_dir = Path(out_dir).resolve()
    src_dir = rec.rec_dir
    if out_dir == src_dir or src_dir in out_dir.parents:
        raise ExportError("output directory must be outside the source recording folder")
    # record parents we are about to create so a failed export can undo them
    created_parents = []
    p = out_dir.parent
    while not p.exists():
        created_parents.append(p)          # deepest first
        p = p.parent
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    # re-resolve THROUGH the now-existing parent (follows any symlink swapped
    # in since the caller authorized the path) and re-check authorization —
    # keeps the authorization decision and the write on the same inode path
    try:
        parent_real = out_dir.parent.resolve(strict=True)
    except OSError as e:
        raise ExportError(f"cannot resolve output parent: {e}") from None
    out_dir = parent_real / out_dir.name
    if path_guard is not None and not (path_guard(parent_real)
                                       and path_guard(out_dir)):
        raise ExportError(
            "output directory escaped the authorized location between "
            "request and write (symlinked parent?) — export refused")
    if out_dir == src_dir or src_dir in out_dir.parents:
        raise ExportError("output directory must be outside the source recording folder")
    # atomically CLAIM the output dir: a concurrent export to the same path
    # fails here in milliseconds instead of after a full duplicate encode
    try:
        out_dir.mkdir()
    except FileExistsError:
        raise ExportError(f"output directory already exists: {out_dir} — "
                          f"refusing to overwrite") from None

    # everything below runs with the out_dir claim held: the finally block
    # undoes the claim (and any created parents) unless we actually publish
    tmp_dir = out_dir.parent / f".neon_trim_tmp_{uuid.uuid4().hex[:8]}"
    validation: list = []
    published = False
    try:
        plan = core.plan_trim(rec.world.ts_ns, rec.video.pts_s,
                              rec.video.duration_s, start_frame, end_frame,
                              rec.rec_end_ns)

        src_video = rec.video.path
        src_files = {"video": src_video, "gaze": rec.gaze.path,
                     "world": rec.world.path}
        report("hashing sources", 0.0)
        hashes_before = {k: _sha256(p) for k, p in src_files.items()}

        tmp_dir.mkdir()
        if cancelled():
            raise ExportCancelled("export cancelled")

        # -- 1) video cut ----------------------------------------------------
        report("encoding video", 0.0)
        out_video = tmp_dir / src_video.name
        enc = media.run_cut(
            src_video, out_video, plan.cut_start_s, plan.cut_end_s,
            has_audio=rec.video.has_audio,
            container_start_s=rec.video.container_start_s,
            progress_cb=lambda f: report("encoding video", f * 0.7),
            cancel_event=cancel_event)

        # -- 2) CSV trims ------------------------------------------------------
        report("trimming CSVs", 0.72)
        n_gaze = core.trim_csv(rec.gaze, plan.abs_start_ns, plan.abs_end_ns,
                               tmp_dir / "gaze.csv")
        n_world = core.trim_csv(rec.world, plan.abs_start_ns, plan.abs_end_ns,
                                tmp_dir / "world_timestamps.csv")
        if cancelled():
            raise ExportCancelled("export cancelled")

        # -- 3) validations ----------------------------------------------------
        report("validating output", 0.8)
        _validate(rec, plan, tmp_dir, out_video, n_gaze, n_world, validation,
                  effective_cut_start_s=enc.get("effective_cut_start_s",
                                                plan.cut_start_s))
        if cancelled():
            raise ExportCancelled("export cancelled")

        report("re-hashing sources", 0.93)
        hashes_after = {k: _sha256(p) for k, p in src_files.items()}
        _check(validation, "source files byte-for-byte unchanged",
               hashes_after == hashes_before,
               "sha256 of video/gaze/world before vs after export: "
               + ("identical" if hashes_after == hashes_before else "MISMATCH"))

        failed = [v for v in validation if not v["ok"]]
        if failed:
            raise ExportError(
                "validation failed — export aborted, no output written:\n"
                + "\n".join(f"  ✗ {v['check']}: {v['detail']}" for v in failed))

        # -- 4) manifest -------------------------------------------------------
        manifest = _build_manifest(rec, plan, enc, n_gaze, n_world,
                                   out_dir, validation, hashes_before)
        (tmp_dir / "trim_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n")

        # -- 5) atomic publish -------------------------------------------------
        report("finalizing", 0.98)
        if cancelled():
            raise ExportCancelled("export cancelled")
        try:
            # atomically replaces the (empty) claimed out_dir on POSIX
            os.rename(tmp_dir, out_dir)
        except OSError as e:
            raise ExportError(
                f"could not publish to {out_dir} (was something written into it "
                f"during the export?): {e}") from None
        published = True
        report("done", 1.0)
        return manifest
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if not published:
            # undo the claimed (still empty) out_dir and any parents we created
            try:
                out_dir.rmdir()
            except OSError:
                pass
            for p in created_parents:
                try:
                    p.rmdir()          # only removes if empty
                except OSError:
                    break


# ---------------------------------------------------------------------------

def _validate(rec: LoadedRecording, plan: core.TrimPlan, tmp_dir: Path,
              out_video: Path, n_gaze: int, n_world: int, results: list,
              effective_cut_start_s: Optional[float] = None) -> None:
    """Run all output validations, appending to `results` (does not raise)."""
    # the timestamp rebase point ffmpeg actually used (container start_time
    # compensated) — every duration/offset expectation must derive from it
    cut0 = (effective_cut_start_s if effective_cut_start_s is not None
            else plan.cut_start_s)
    # video: real decode of the trimmed output
    out_pts = media.decode_all_pts(out_video)
    n_expected = plan.n_frames_selected
    _check(results, "output frame count == selected frame count",
           len(out_pts) == n_expected,
           f"decoded {len(out_pts)} frames, expected {n_expected} "
           f"(frames {plan.start_frame}..{plan.end_frame} inclusive)")
    _check(results, "output video PTS strictly increasing",
           all(b > a for a, b in zip(out_pts, out_pts[1:])),
           f"{len(out_pts)} PTS checked")

    if len(out_pts) == n_expected and n_expected >= 2:
        # relative spacing must match the source frames' spacing (VFR preserved)
        src = rec.video.pts_s[plan.start_frame:plan.end_frame + 1]
        src_gaps = [b - a for a, b in zip(src[:-1], src[1:])]
        out_gaps = [b - a for a, b in zip(out_pts[:-1], out_pts[1:])]
        max_gap_err = max(abs(a - b) for a, b in zip(src_gaps, out_gaps))
        _check(results, "output frame timing matches source (VFR preserved)",
               max_gap_err <= 0.002,
               f"max inter-frame interval deviation {max_gap_err*1e3:.3f} ms (tol 2 ms)")
    if len(out_pts) == n_expected and n_expected >= 1:
        # the first output frame must sit exactly where the cut geometry puts
        # it: pts[start] - effective cut start (= half the pre-start PTS gap,
        # container start_time compensated; 0 for S=0 on a pristine source).
        # Comparing against the exact expectation stays correct for arbitrary
        # VFR gaps (a fixed cap would spuriously reject starts after a long
        # dropped-frames gap).
        expected_off = plan.rel_start_s - cut0
        start_off = out_pts[0]
        _check(results, "output starts on the selected frame",
               abs(start_off - expected_off) <= 0.002,
               f"first output frame at +{start_off*1e3:.2f} ms vs expected "
               f"+{expected_off*1e3:.2f} ms on the trimmed timeline "
               f"(uniform cut offset; frame identity verified by count + spacing)")

    # world rows: must be exactly the selected frame rows
    w_out = core.load_timestamp_csv(tmp_dir / "world_timestamps.csv")
    _check(results, "world rows == selected frames",
           n_world == n_expected and w_out.n_rows == n_expected,
           f"{w_out.n_rows} rows written, expected {n_expected}")
    if w_out.n_rows == n_expected:
        exact = (w_out.ts_ns[0] == rec.world.ts_ns[plan.start_frame]
                 and w_out.ts_ns[-1] == rec.world.ts_ns[plan.end_frame])
        _check(results, "world endpoint timestamps exact",
               exact, "first/last world timestamps equal source rows "
                      f"{plan.start_frame}/{plan.end_frame}")
    _check(results, "world timestamps strictly increasing",
           core.validate_monotonic(w_out.ts_ns, strict=True), f"{w_out.n_rows} rows")

    # gaze rows: verbatim, inside interval, none missed
    g_out = core.load_timestamp_csv(tmp_dir / "gaze.csv")
    _check(results, "gaze rows monotonic non-decreasing",
           core.validate_monotonic(g_out.ts_ns, strict=False), f"{g_out.n_rows} rows")
    _check(results, "every retained gaze timestamp inside interval",
           core.validate_within(g_out.ts_ns, plan.abs_start_ns, plan.abs_end_ns),
           f"interval [{plan.abs_start_ns}, {plan.abs_end_ns}) ns")
    expected_gaze = core.count_in_interval(rec.gaze, plan.abs_start_ns, plan.abs_end_ns)
    _check(results, "no in-interval gaze row missed or extra retained",
           g_out.n_rows == expected_gaze == n_gaze,
           f"retained {g_out.n_rows}, source rows in interval {expected_gaze}")
    _check(results, "empty gaze interval reported" if expected_gaze == 0
           else "gaze present in interval", True,
           f"{expected_gaze} gaze samples in the selected interval"
           + (" — WARNING: selection contains no gaze data" if expected_gaze == 0 else ""))

    # audio
    if rec.video.has_audio:
        out_probe = media.probe_video(out_video)
        ok_audio = out_probe.has_audio
        detail = "audio stream present"
        if ok_audio and out_probe.audio_duration_s is not None and len(out_pts) >= 1:
            vid_dur = plan.cut_end_s - cut0
            drift = abs(out_probe.audio_duration_s - vid_dur)
            ok_audio = drift <= 0.15
            detail = (f"audio {out_probe.audio_duration_s:.3f}s vs cut "
                      f"{vid_dur:.3f}s (|Δ|={drift*1e3:.0f} ms, tol 150 ms; "
                      f"same-input single-pass cut keeps A/V aligned)")
        _check(results, "audio retained and aligned", ok_audio, detail)


def _build_manifest(rec: LoadedRecording, plan: core.TrimPlan, enc: dict,
                    n_gaze: int, n_world: int, out_dir: Path,
                    validation: list, src_hashes: dict) -> dict:
    med_iv = None
    if len(rec.world.ts_ns) >= 2:
        med_iv = statistics.median(
            b - a for a, b in zip(rec.world.ts_ns[:-1], rec.world.ts_ns[1:])) / 1e6
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        # -- versioned downstream contract (see artifact.py) -----------------
        "schema_version": artifact_mod.SCHEMA_VERSION,
        "artifact_type": artifact_mod.ARTIFACT_TYPE,
        "artifact_id": uuid.uuid4().hex,
        "parent_artifact": artifact_mod.parent_artifact_ref(rec.rec_dir),
        "created_utc": created,
        # paths RELATIVE to the artifact directory: the artifact can be moved
        "files": {
            "video": rec.video.path.name,
            "gaze": "gaze.csv",
            "world_timestamps": "world_timestamps.csv",
            "manifest": artifact_mod.MANIFEST_NAME,
        },
        "tool": {"name": "neon_trim", "version": __version__,
                 "created_utc": created},
        "source": {
            "directory": str(rec.rec_dir),   # absolute: provenance only
            "directory_note": "absolute source path, provenance only — all "
                              "'files' entries are relative to this artifact",
            "recording_id": rec.info.recording_id,
            "video_filename": rec.video.path.name,
            "video_sha256": src_hashes["video"],
            "gaze_sha256": src_hashes["gaze"],
            "world_timestamps_sha256": src_hashes["world"],
        },
        "selection": {
            "first_frame": plan.start_frame,
            "last_frame": plan.end_frame,
            "relative_start_s": round(plan.rel_start_s, 6),
            "relative_end_s": round(plan.rel_end_s, 6),
            "absolute_start_ns": plan.abs_start_ns,
            "absolute_end_boundary_ns": plan.abs_end_ns,
            "interval_semantics": "[start_frame_time, time_after_last_frame) — "
                                  "both endpoint frames included",
            "selected_duration_s": round(plan.duration_s, 6),
        },
        "counts": {
            "original_frames": rec.video.n_frames,
            "output_frames": plan.n_frames_selected,
            "original_world_rows": rec.world.n_rows,
            "output_world_rows": n_world,
            "original_gaze_rows": rec.gaze.n_rows,
            "retained_gaze_rows": n_gaze,
        },
        "video_encoding": enc,
        "synchronization": {
            "method": core.SYNC_METHOD,
            "description": (
                "Scene MP4 is variable-frame-rate; container nominal fps is "
                "unreliable. Decoded video frame i maps 1:1 to "
                "world_timestamps.csv row i (verified: equal counts and "
                "constant per-index PTS-vs-world offset). Absolute time of "
                "frame i is its world timestamp [ns]. Gaze rows are kept iff "
                "their timestamp lies in the half-open absolute interval; no "
                "fixed gaze-per-frame ratio is assumed."),
            **rec.sync.as_dict(),
            "median_frame_interval_ms": round(med_iv, 3) if med_iv else None,
            "recording_end_ns": rec.rec_end_ns,
        },
        "omitted_files": {
            "files": [f for f in OMITTED_FILES if (rec.rec_dir / f).exists()],
            "note": ("These NEON streams were intentionally NOT copied: this "
                     "output is a trimmed subset (video + gaze + world "
                     "timestamps only), not a complete NEON export. Trim them "
                     "separately if needed — do not reuse stale copies."),
        },
        "validation": validation,
        "output_directory": str(out_dir),
    }


# ---------------------------------------------------------------------------
# library / CLI entry point
# ---------------------------------------------------------------------------

def trim_recording(rec_dir, start_frame: int, end_frame: int, out_dir,
                   progress_cb: Optional[Callable[[str, float], None]] = None,
                   cancel_event: Optional[threading.Event] = None) -> dict:
    """One-call trim: load + validate sync + export. Returns the manifest.

    This is the single shared implementation behind all three invocation
    modes (GUI server, Python library, future CLI/orchestrator). Progress
    and cancellation use plain callables/threading.Event — nothing
    FastAPI-specific.

        from neon_trim.export import trim_recording
        from neon_trim.artifact import load_artifact
        manifest = trim_recording("/data/rec01", 1500, 2399, "/data/rec01_trim")
        art = load_artifact(manifest["output_directory"])
    """
    rec = load_recording(rec_dir)
    return run_export(rec, start_frame, end_frame, out_dir,
                      progress_cb=progress_cb, cancel_event=cancel_event)
