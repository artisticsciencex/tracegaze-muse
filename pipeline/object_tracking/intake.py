"""Input-type detection and the canonical RecordingBundle for tracking.

Four supported input types (plus recognition of already-tracked artifacts):

| kind                | structural signature (stat-level)                    |
|---------------------|------------------------------------------------------|
| trimmed_artifact    | trim_manifest.json + video + gaze + world csvs       |
| prepared_artifact   | prep_manifest.json + video + gaze + world csvs       |
| csv_export          | video + gaze.csv + world_timestamps.csv (no manifest)|
| native_recording    | info.json + 'Neon Scene Camera v1 ps1.mp4' + .time   |
| tracked_objects     | tracking_manifest.json                               |
| video_only          | exactly one video, no gaze/world/native signature    |

`classify_folder` is stat-only (safe for directory listings); `open_bundle`
performs full validation (video probe + authoritative sync check) and returns
a :class:`RecordingBundle`. Bundles never load the gaze table into memory —
only row counts and the timestamp column bounds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from common import media
from neon_trim import artifact as trim_artifact
from neon_trim import core as nt_core
from neon_trim import intake as nt_intake

PREP_MANIFEST = "prep_manifest.json"
TRACK_MANIFEST = "tracking_manifest.json"

KIND_LABELS = {
    "trimmed_artifact": "Trimmed NEON artifact — ready",
    "prepared_artifact": "Prepared NEON recording — ready",
    "csv_export": "NEON CSV export — ready",
    "native_recording": "Native NEON recording — preparation required",
    "tracked_objects": "Tracked-object artifact",
    "video_only": "Video only — gaze unavailable",
    # Project Aria artifacts (own types — never disguised as NEON)
    "prepared_aria_recording": "Prepared Aria recording — ready",
    "trimmed_aria_recording": "Trimmed Aria artifact — ready",
    "aria_download": "Project Aria / Reading in the Wild recording",
}

ARIA_KINDS = ("prepared_aria_recording", "trimmed_aria_recording")

# --------------------------------------------------------------------------
# Authoritative semantic kind sets.
#
# These exist so the server, the folder chooser and the artifact layer stop
# repeating ad-hoc string lists that drift apart (an outdated frontend
# allow-list is exactly what made valid Aria artifacts unselectable). Any
# consumer that needs "can this folder be chosen / loaded / tracked" asks
# here; the browser receives CHOOSER_KINDS through /api/config rather than
# hard-coding its own copy.
# --------------------------------------------------------------------------

#: folders a user may pick in the chooser as a tracking INPUT
CHOOSER_KINDS = (
    "trimmed_artifact", "prepared_artifact", "csv_export",
    "native_recording", "video_only",
    "prepared_aria_recording", "trimmed_aria_recording",
)

#: folders `open_bundle()` can load directly (no preparation step)
TRACKABLE_KINDS = (
    "trimmed_artifact", "prepared_artifact", "csv_export", "video_only",
    "prepared_aria_recording", "trimmed_aria_recording",
)

#: folders that must be prepared before they can be tracked
PREPARATION_REQUIRED_KINDS = ("native_recording", "aria_download")

#: trimmed parents that carry a validated interval for gaze fusion
GAZE_TRIMMED_PARENT_KINDS = ("trimmed_artifact", "trimmed_aria_recording")

#: kinds recognised but never usable as a tracking input
NON_INPUT_KINDS = ("tracked_objects",)

#: chooser badge styling class per kind (kept beside the sets it belongs to)
KIND_BADGE_CLASS = {
    "trimmed_artifact": "ready", "prepared_artifact": "ready",
    "csv_export": "ready", "native_recording": "prep",
    "tracked_objects": "tracked", "video_only": "videoonly",
    "prepared_aria_recording": "ready", "trimmed_aria_recording": "ready",
    "aria_download": "prep",
}


def is_aria_kind(kind) -> bool:
    return kind in ARIA_KINDS

NATIVE_SCENE_MP4 = "Neon Scene Camera v1 ps1.mp4"
NATIVE_SCENE_TIME = "Neon Scene Camera v1 ps1.time"


class IntakeError(RuntimeError):
    pass


def classify_folder(d: Path) -> Optional[str]:
    """Cheap stat-only classification for listing badges. None = unsupported."""
    d = Path(d)
    try:
        if not d.is_dir():
            return None
        if (d / TRACK_MANIFEST).is_file():
            return "tracked_objects"
        # Aria artifacts share the manifest FILENAMES with NEON, so they
        # are dispatched first by manifest artifact_type (never by name)
        from common import recordings as rec_dispatch
        aria_kind = rec_dispatch.classify_aria_folder(d)
        if aria_kind:
            return aria_kind
        has_csvs = ((d / "gaze.csv").is_file()
                    and (d / "world_timestamps.csv").is_file())
        if (d / trim_artifact.MANIFEST_NAME).is_file() and has_csvs:
            return "trimmed_artifact"
        if (d / PREP_MANIFEST).is_file() and has_csvs:
            return "prepared_artifact"
        if ((d / "info.json").is_file() and (d / NATIVE_SCENE_MP4).is_file()
                and (d / NATIVE_SCENE_TIME).is_file()):
            return "native_recording"
        vids = [p for p in d.iterdir()
                if p.is_file() and p.suffix.lower() in media.VIDEO_EXTS]
        if has_csvs and len(vids) == 1:
            return "csv_export"
        if len(vids) == 1 and not has_csvs:
            return "video_only"
        return None
    except OSError:
        return None


@dataclass
class RecordingBundle:
    """Canonical recording representation consumed by tracking (and shaped to
    be reusable by trimming / gaze fusion / orchestration)."""
    source_dir: Path
    source_type: str                      # one of KIND_LABELS keys
    recording_id: Optional[str]
    artifact_id: Optional[str]            # when source is an artifact
    artifact_type: Optional[str]
    parent_artifact: Optional[dict]
    video_path: Path
    gaze_path: Optional[Path]             # None for video_only / unprepared native
    world_path: Optional[Path]
    # metadata
    wearer_name: Optional[str] = None
    recording_name: Optional[str] = None
    duration_s: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    n_frames: int = 0
    gaze_rows: Optional[int] = None
    world_rows: Optional[int] = None
    has_audio: bool = False
    # timing (authoritative, from full probe + world csv)
    pts_s: list = field(default_factory=list)         # frame idx -> seconds
    world_ts_ns: list = field(default_factory=list)   # frame idx -> abs int ns
    container_start_s: float = 0.0
    # sync
    sync_ok: bool = False
    sync_status: str = "unknown"
    gaze_compatible: bool = False         # may this feed gaze fusion?
    gaze_offset_provenance: Optional[str] = None
    warnings: list = field(default_factory=list)

    def timestamp_ns_for_frame(self, frame_index: int) -> Optional[int]:
        # only the VERIFIED 1:1 mapping may produce absolute timestamps
        if not self.sync_ok:
            return None
        if self.world_ts_ns and 0 <= frame_index < len(self.world_ts_ns):
            return self.world_ts_ns[frame_index]
        return None

    def summary(self) -> dict:
        return {
            "source_dir": str(self.source_dir),
            "source_type": self.source_type,
            "source_type_label": KIND_LABELS.get(self.source_type,
                                                 self.source_type),
            "recording_id": self.recording_id,
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "parent_artifact": self.parent_artifact,
            "video": self.video_path.name,
            "wearer_name": self.wearer_name,
            "recording_name": self.recording_name,
            "duration_s": self.duration_s,
            "width": self.width, "height": self.height,
            "n_frames": self.n_frames,
            "gaze_rows": self.gaze_rows,
            "world_rows": self.world_rows,
            "has_audio": self.has_audio,
            "sync_ok": self.sync_ok,
            "sync_status": self.sync_status,
            "gaze_compatible": self.gaze_compatible,
            "gaze_offset_provenance": self.gaze_offset_provenance,
            "warnings": self.warnings,
        }


def _count_gaze_rows(path: Path) -> Optional[int]:
    return nt_intake._count_rows(path)


def open_bundle(d: Path) -> RecordingBundle:
    """Fully open + validate a folder as a tracking input.

    Raises IntakeError with a precise diagnostic for unsupported folders or
    native recordings (which require preparation first).
    """
    d = Path(d).resolve()
    kind = classify_folder(d)
    aria_meta = None
    if kind is None:
        insp = nt_intake.inspect_recording(d)
        raise IntakeError(
            "unsupported folder: " + ("; ".join(insp.problems)
                                      or "no recognizable recording content"))
    if kind == "native_recording":
        raise IntakeError(
            "native NEON recording — canonical preparation is required "
            "before tracking (use the Prepare recording action)")
    if kind == "tracked_objects":
        raise IntakeError(
            "this folder is already a tracked-object artifact; choose its "
            "parent recording to run a new tracking session")
    if kind == "aria_download":
        raise IntakeError(
            "Project Aria download folder — preparation is required before "
            "tracking (prepare it in the Trim/Prepare stage first)")

    if kind in ARIA_KINDS:
        from common import recordings as rec_dispatch
        lp = rec_dispatch.logical_paths(d)     # full Aria contract check
        video_path = lp["video"]
        gaze_path, world_path = lp["gaze"], lp["world_timestamps"]
        rec_id = lp["recording_id"]
        artifact_id, artifact_type = lp["artifact_id"], lp["artifact_type"]
        parent = lp["parent_artifact"]
        sync_meta = {"gaze_offset_provenance":
                     "Project Aria MPS eye gaze projected with the official "
                     "device calibration; no offset re-applied"}
        aria_meta = lp
    elif kind == "trimmed_artifact":
        art = trim_artifact.load_artifact(d)      # full contract validation
        video_path = art.video_path
        gaze_path, world_path = art.gaze_path, art.world_timestamps_path
        rec_id = art.source_recording_id
        artifact_id, artifact_type = art.artifact_id, "trimmed_neon_recording"
        parent = art.manifest.get("parent_artifact")
        sync_meta = art.manifest.get("synchronization", {})
    elif kind == "prepared_artifact":
        pm = json.loads((d / PREP_MANIFEST).read_text())
        video_path = media.find_scene_video(d)
        gaze_path, world_path = d / "gaze.csv", d / "world_timestamps.csv"
        rec_id = pm.get("source", {}).get("recording_id")
        artifact_id = pm.get("artifact_id")
        artifact_type = pm.get("artifact_type")
        parent = pm.get("parent_artifact")
        sync_meta = pm.get("synchronization", {})
    elif kind == "csv_export":
        video_path = media.find_scene_video(d)
        gaze_path, world_path = d / "gaze.csv", d / "world_timestamps.csv"
        info = nt_core.load_recording_info(d)
        rec_id = info.recording_id
        artifact_id = artifact_type = parent = None
        sync_meta = {}
    else:  # video_only
        video_path = media.find_scene_video(d)
        gaze_path = world_path = None
        rec_id = artifact_id = artifact_type = parent = None
        sync_meta = {}

    probe = media.probe_video(video_path)
    bundle = RecordingBundle(
        source_dir=d, source_type=kind,
        recording_id=rec_id, artifact_id=artifact_id,
        artifact_type=artifact_type, parent_artifact=parent,
        video_path=video_path, gaze_path=gaze_path, world_path=world_path,
        duration_s=probe.duration_s, width=probe.width, height=probe.height,
        n_frames=probe.n_frames, has_audio=probe.has_audio,
        pts_s=probe.pts_s, container_start_s=probe.container_start_s,
        warnings=list(probe.warnings),
    )

    # wearer / recording-name metadata (best effort)
    if aria_meta is None:
        insp = nt_intake.inspect_recording(d)
        if insp.bundle is not None:
            bundle.wearer_name = insp.bundle.wearer_name
            bundle.recording_name = insp.bundle.recording_name
    else:
        rm = aria_meta.get("recording_meta") or {}
        bundle.recording_name = rm.get("task") or aria_meta.get("dataset")

    if kind == "video_only":
        bundle.sync_ok = False
        bundle.gaze_compatible = False
        bundle.sync_status = ("video only — no NEON timestamps or gaze; "
                              "tracking allowed, gaze fusion unavailable")
        bundle.warnings.append("Video only — gaze unavailable")
        return bundle

    world = nt_core.load_timestamp_csv(world_path)
    bundle.world_rows = world.n_rows
    bundle.world_ts_ns = world.ts_ns
    bundle.gaze_rows = _count_gaze_rows(gaze_path)
    sync = nt_core.check_sync(probe.pts_s, world.ts_ns)
    bundle.sync_ok = sync.ok
    bundle.sync_status = sync.message
    bundle.gaze_compatible = sync.ok
    bundle.gaze_offset_provenance = sync_meta.get("gaze_offset_provenance")
    if not sync.ok:
        bundle.warnings.append(
            "synchronization not established — tracking may proceed but the "
            "result cannot be published as gaze-compatible")
    return bundle
