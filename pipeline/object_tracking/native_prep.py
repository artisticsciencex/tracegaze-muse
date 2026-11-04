"""Native NEON recording -> canonical prepared artifact.

Uses the official ``pupil-labs-neon-recording`` library for native stream
access and temporal synchronization — native binary files are NEVER
hand-decoded in this production path.

Empirical basis (participant_02 native vs the matching participant_a Cloud CSV
export, recording 0000abcd-c979-4a9d-ae0d-3afc97d426ba — same recording):

* native scene: 6,403 decoded video frames == 6,403 `.time` entries (1:1);
  MP4 PTS match the `.time` sidecar within 0.44 ms spread -> the standard
  index-1:1 sync check applies directly.
* Cloud export: 6,433 world rows = native 6,403 + 30 synthetic PAD frames at
  a 50 ms grid covering recording-start -> first camera frame (~1.474 s), and
  the real frames are re-timestamped with ±5.5 µs jitter. Mapping between
  native and export is therefore index-based (native i == export i+pad),
  never timestamp-equality-based. Preparation does NOT reproduce Cloud's pad
  frames: the prepared artifact contains only real camera frames.
* native gaze (library ``gaze`` == the device's gaze_200hz stream) is
  IDENTICAL to the Cloud export gaze (timestamps exactly equal; coordinates
  within CSV rounding ±0.0005 px). The recorded ``gaze_offset`` in info.json
  is ALREADY APPLIED inside this stream — applying it again produces ~56 px
  error. Cloud additionally trims gaze samples past recording end; we do the
  same and record it.

The prepared folder is deliberately shaped like a NEON CSV export
(scene mp4 + gaze.csv + world_timestamps.csv + info.json) so every existing
stage (trimming, tracking, gaze fusion) consumes it unchanged, plus a
``prep_manifest.json`` documenting provenance and validation.
"""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from common import media
from common.artifacts import ArtifactError, AtomicPublisher, sha256_file
from neon_trim import core as nt_core

from . import __version__
from .intake import NATIVE_SCENE_MP4, NATIVE_SCENE_TIME, PREP_MANIFEST

PREP_ARTIFACT_TYPE = "prepared_neon_recording"
PREP_SCHEMA_VERSION = "1.0"

GAZE_OFFSET_PROVENANCE = (
    "info.json gaze_offset is already incorporated in the device 200 Hz gaze "
    "stream exposed by pupil-labs-neon-recording (verified against the "
    "matching Cloud CSV export: coordinates equal within CSV rounding; "
    "re-applying the offset produces ~offset-sized error). Recorded here for "
    "provenance only — NOT re-applied. Post-recording Cloud/Player gaze "
    "corrections, if any were made after this preparation, are NOT included.")


class PrepError(RuntimeError):
    pass


def is_native_recording(d: Path) -> bool:
    d = Path(d)
    return ((d / "info.json").is_file() and (d / NATIVE_SCENE_MP4).is_file()
            and (d / NATIVE_SCENE_TIME).is_file())


def native_overview(d: Path) -> dict:
    """Cheap metadata + stream inventory for the GUI (no heavy decode)."""
    d = Path(d)
    if not is_native_recording(d):
        raise PrepError(f"not a native NEON recording: {d}")
    info = json.loads((d / "info.json").read_text())
    streams = sorted(p.name for p in d.iterdir()
                     if p.suffix in (".raw", ".time", ".mp4", ".bin"))
    wearer = None
    if (d / "wearer.json").is_file():
        try:
            wearer = json.loads((d / "wearer.json").read_text()).get("name")
        except (OSError, ValueError):
            pass
    return {
        "recording_id": info.get("recording_id"),
        "start_time": info.get("start_time"),
        "duration": info.get("duration"),
        "wearer_name": wearer or info.get("wearer_name"),
        "gaze_offset": info.get("gaze_offset"),
        "app_version": info.get("app_version"),
        "streams": streams,
        "explanation": (
            "This is a raw device recording. Canonical preparation extracts "
            "the scene video, synchronized scene timestamps, and 200 Hz gaze "
            "into a portable export-shaped artifact (the raw folder is never "
            "modified). Trimming before tracking is recommended."),
    }


def prepare_native_recording(
        native_dir: Path, out_dir: Path, *,
        path_guard: Optional[Callable[[Path], bool]] = None,
        progress_cb: Optional[Callable[[str, float], None]] = None,
        cancel_event: Optional[threading.Event] = None) -> dict:
    """Create a canonical prepared artifact from a native recording.

    Returns the prep manifest. Atomic: failures leave no partial output.
    The native source is opened read-only and hash-verified unchanged.
    """
    def report(phase: str, frac: float):
        if progress_cb is not None:
            progress_cb(phase, frac)

    def cancelled():
        return cancel_event is not None and cancel_event.is_set()

    native_dir = Path(native_dir).resolve()
    if not is_native_recording(native_dir):
        raise PrepError(f"not a native NEON recording: {native_dir}")

    report("opening native recording", 0.02)
    import pupil_labs.neon_recording as nr    # official library, lazy import

    rec = nr.open(str(native_dir))
    rec_id = rec.id
    start_ns, duration_ns = int(rec.start_time), int(rec.duration)
    scene = rec.scene
    gaze = rec.gaze

    # authoritative synchronized timestamps from the official library
    scene_ts = [int(t) for t in scene.time]
    gaze_ts = [int(t) for t in gaze.time]
    gaze_x = [float(v) for v in gaze.data["point_x"]]
    gaze_y = [float(v) for v in gaze.data["point_y"]]
    if not all(b > a for a, b in zip(scene_ts, scene_ts[1:])):
        raise PrepError("native scene timestamps are not strictly increasing")

    # Cloud-compatible end trim: drop gaze past recording end (documented).
    # Guarded: a missing/zero/short info.json duration must never silently
    # truncate gaze — recording_end_ns falls back to last scene ts + median
    # frame interval when the declared end is implausible.
    end_ns = nt_core.recording_end_ns(
        nt_core.RecordingInfo(start_time_ns=start_ns,
                              duration_ns=duration_ns), scene_ts)
    n_gaze_all = len(gaze_ts)
    keep = [i for i, t in enumerate(gaze_ts) if t < end_ns]
    n_trimmed_tail = n_gaze_all - len(keep)

    src_video = native_dir / NATIVE_SCENE_MP4
    report("hashing native source", 0.05)
    src_hashes = {
        "video": sha256_file(src_video),
        "scene_time": sha256_file(native_dir / NATIVE_SCENE_TIME),
    }

    report("validating video against timestamps", 0.10)
    probe = media.probe_video(src_video)
    if cancelled():
        raise PrepError("preparation cancelled")
    sync = nt_core.check_sync(probe.pts_s, scene_ts)
    sync_dict = sync.as_dict()
    if probe.n_frames != len(scene_ts):
        raise PrepError(
            f"native scene video decodes to {probe.n_frames} frames but the "
            f"synchronized scene timestamps have {len(scene_ts)} entries — "
            f"no reliable 1:1 mapping; refusing to publish a gaze-compatible "
            f"artifact. (Never silently forced.)")
    if not sync.ok:
        raise PrepError(
            "native video PTS do not align with synchronized scene "
            "timestamps — refusing to publish a gaze-compatible artifact.\n"
            + sync.message)

    artifact_id = uuid.uuid4().hex
    pub = AtomicPublisher(out_dir, source_dir=native_dir,
                          path_guard=path_guard,
                          tmp_prefix=".neon_prep_tmp_")
    try:
        tmp = pub.tmp_dir
        # 1) scene video: safe copy (source is never modified; artifact is
        # self-contained; symlinks would violate artifact containment rules)
        report("copying scene video", 0.15)
        out_video = tmp / "scene.mp4"
        shutil.copyfile(src_video, out_video)
        if cancelled():
            raise PrepError("preparation cancelled")

        # 2) canonical CSVs (Cloud-export column layout; int ns preserved)
        report("writing world_timestamps.csv", 0.60)
        section = artifact_id  # documented: synthesized section id
        with open(tmp / "world_timestamps.csv", "w") as f:
            f.write("section id,recording id,timestamp [ns]\n")
            for t in scene_ts:
                f.write(f"{section},{rec_id},{t}\n")
        report("writing gaze.csv", 0.66)
        with open(tmp / "gaze.csv", "w") as f:
            f.write("section id,recording id,timestamp [ns],"
                    "gaze x [px],gaze y [px]\n")
            for i in keep:
                f.write(f"{section},{rec_id},{gaze_ts[i]},"
                        f"{gaze_x[i]:.6f},{gaze_y[i]:.6f}\n")

        # 3) info.json subset (compatible with existing intake)
        info_src = json.loads((native_dir / "info.json").read_text())
        (tmp / "info.json").write_text(json.dumps({
            "recording_id": rec_id,
            "start_time": start_ns,
            "duration": duration_ns,
            "wearer_name": info_src.get("wearer_name"),
            "gaze_offset": info_src.get("gaze_offset"),
            "data_format_version": info_src.get("data_format_version"),
            "prepared_by": "object_tracking.native_prep",
        }, indent=2))

        # 4) validations
        report("validating prepared artifact", 0.72)
        validation = []

        def check(name, ok, detail):
            validation.append({"check": name, "ok": bool(ok),
                               "detail": detail})

        out_probe = media.probe_video(out_video)
        check("copied video decodes to native frame count",
              out_probe.n_frames == len(scene_ts),
              f"{out_probe.n_frames} frames vs {len(scene_ts)} timestamps")
        w = nt_core.load_timestamp_csv(tmp / "world_timestamps.csv")
        check("world csv int-ns roundtrip exact", w.ts_ns == scene_ts,
              f"{w.n_rows} rows")
        g = nt_core.load_timestamp_csv(tmp / "gaze.csv")
        check("gaze csv int-ns roundtrip exact",
              g.ts_ns == [gaze_ts[i] for i in keep], f"{g.n_rows} rows")
        check("gaze trimmed to recording end",
              all(t < end_ns for t in g.ts_ns),
              f"{n_trimmed_tail} tail sample(s) past recording end dropped "
              f"(matches Cloud-export behavior)")
        resync = nt_core.check_sync(out_probe.pts_s, w.ts_ns)
        check("prepared video<->timestamp sync verified", resync.ok,
              resync.message)
        check("native source unchanged (sha256)",
              sha256_file(src_video) == src_hashes["video"]
              and sha256_file(native_dir / NATIVE_SCENE_TIME)
              == src_hashes["scene_time"],
              "video + scene .time re-hashed identical")
        failed = [v for v in validation if not v["ok"]]
        if failed:
            raise PrepError("preparation validation failed:\n" + "\n".join(
                f"  x {v['check']}: {v['detail']}" for v in failed))
        if cancelled():
            raise PrepError("preparation cancelled")

        manifest = {
            "schema_version": PREP_SCHEMA_VERSION,
            "artifact_type": PREP_ARTIFACT_TYPE,
            "artifact_id": artifact_id,
            "parent_artifact": None,
            "created_utc": datetime.now(timezone.utc)
                .isoformat(timespec="seconds"),
            "tool": {"name": "object_tracking.native_prep",
                     "version": __version__,
                     "native_library": "pupil-labs-neon-recording"},
            "files": {"video": "scene.mp4", "gaze": "gaze.csv",
                      "world_timestamps": "world_timestamps.csv",
                      "info": "info.json", "manifest": PREP_MANIFEST},
            "source": {
                "directory": str(native_dir),
                "directory_note": "absolute native-source path, provenance "
                                  "only — files above are artifact-relative",
                "recording_id": rec_id,
                "native_video_sha256": src_hashes["video"],
                "native_scene_time_sha256": src_hashes["scene_time"],
            },
            "counts": {
                "scene_frames": len(scene_ts),
                "world_rows": len(scene_ts),
                "gaze_rows_native": n_gaze_all,
                "gaze_rows_written": len(keep),
                "gaze_tail_samples_dropped_past_recording_end": n_trimmed_tail,
            },
            "synchronization": {
                "method": nt_core.SYNC_METHOD,
                **sync_dict,
                "note": ("native scene .time entries map 1:1 to decoded "
                         "video frames; Cloud exports of this recording "
                         "additionally prepend synthetic 50 ms-grid pad "
                         "frames from recording start to the first camera "
                         "frame and re-timestamp real frames at µs scale — "
                         "this prepared artifact contains ONLY real camera "
                         "frames with native timestamps."),
                "gaze_offset": info_src.get("gaze_offset"),
                "gaze_offset_provenance": GAZE_OFFSET_PROVENANCE,
            },
            "omitted_streams": {
                "files": sorted(p.name for p in native_dir.iterdir()
                                if p.suffix in (".raw", ".time", ".bin")
                                and not p.name.startswith("Neon Scene")),
                "note": "IMU, eye state, eye video, events, calibration and "
                        "other native streams were intentionally not "
                        "converted; prepare them separately if needed.",
            },
            "validation": validation,
            "output_directory": str(pub.out_dir),
        }
        (tmp / PREP_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
        report("publishing", 0.98)
        pub.publish()
        report("done", 1.0)
        return manifest
    finally:
        pub.cleanup()
