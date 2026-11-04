"""Load a prepared/trimmed Aria artifact for the Trim/Prepare GUI.

Exposes the same fields the verified NEON trim GUI already consumes
(video probe, frame-timestamp table, gaze table, 1:1 sync check), so the
timeline, frame stepping and inclusive [start, end] selection work
unchanged — without the Aria artifact pretending to be a NEON recording.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from common import media

from .artifact import load_aria_artifact


@dataclass
class AriaTrimSource:
    rec_dir: Path
    manifest: dict
    video: object                    # common.media.VideoProbe
    world: object                    # neon_trim.core.TimestampCsv
    gaze: object                     # neon_trim.core.TimestampCsv
    sync: object                     # neon_trim.core.SyncCheck
    rec_end_ns: int
    artifact_type: str
    recording_id: str
    warnings: list = field(default_factory=list)
    is_aria: bool = True


def load_aria_for_trim(d) -> AriaTrimSource:
    from neon_trim.core import check_sync, load_timestamp_csv

    art = load_aria_artifact(d)
    m = art.manifest
    video = media.probe_video(art.video_path)
    world = load_timestamp_csv(art.world_timestamps_path)
    gaze = load_timestamp_csv(art.gaze_path)
    sync = check_sync(video.pts_s, world.ts_ns)

    gaps = [b - a for a, b in zip(world.ts_ns, world.ts_ns[1:])]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 33_000_000
    rec_end_ns = world.ts_ns[-1] + median_gap

    warnings = list(video.warnings)
    g = m.get("gaze", {})
    if g.get("projection_failed"):
        warnings.append(
            f"{g['projection_failed']} gaze samples could not be projected "
            f"into the RGB image (outside the camera field of view)")
    if g.get("projection_outside_image"):
        warnings.append(
            f"{g['projection_outside_image']} gaze samples project outside "
            f"the image bounds (kept unclamped)")
    if not m.get("audio", {}).get("present"):
        warnings.append("audio not preserved: "
                        + str(m.get("audio", {}).get("reason", "")))
    irr = (m.get("timestamps", {}).get("gap_report") or {}).get(
        "irregular_count")
    if irr:
        warnings.append(f"{irr} irregular RGB frame intervals reported")

    return AriaTrimSource(
        rec_dir=Path(art.artifact_dir), manifest=m, video=video, world=world,
        gaze=gaze, sync=sync, rec_end_ns=rec_end_ns,
        artifact_type=art.artifact_type,
        recording_id=str(art.recording_id or ""), warnings=warnings)
