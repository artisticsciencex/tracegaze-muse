"""Aria artifact contracts: prepared_aria_recording / trimmed_aria_recording.

Deliberately SEPARATE artifact types from the NEON contract — an Aria
recording never claims to be a NEON one. For first-version downstream
compatibility the payload filenames match what tracking/fusion already
consume (`scene.mp4`, `gaze.csv`, `world_timestamps.csv`), and the
manifest states plainly that `world_timestamps.csv` is a COMPATIBILITY
FILENAME holding Aria DEVICE_TIME RGB-frame timestamps, not NEON world
time.
"""

from __future__ import annotations

import json
from pathlib import Path

from common.artifacts import (ArtifactError, read_manifest, require,
                              resolve_declared_file)

from . import (ARIA_PREP_MANIFEST, ARIA_TRIM_MANIFEST, PREPARED_ARIA_TYPE,
               SOURCE_SYSTEM, TRIMMED_ARIA_TYPE)

ARIA_SCHEMA_VERSION = "1.0"

TIMESTAMP_DOMAIN_NOTE = (
    "world_timestamps.csv is a COMPATIBILITY FILENAME: each row is the "
    "capture timestamp of the corresponding scene.mp4 frame in Aria "
    "DEVICE_TIME (exact integer nanoseconds, RGB stream 214-1/camera-rgb). "
    "It is NOT Pupil Labs NEON world time and the two must never be "
    "compared across source systems."
)


class AriaArtifact:
    """A loaded prepared/trimmed Aria artifact."""

    def __init__(self, artifact_dir: Path, manifest: dict, files: dict):
        self.artifact_dir = artifact_dir
        self.manifest = manifest
        self.video_path = files["video"]
        self.gaze_path = files["gaze"]
        self.world_timestamps_path = files["world_timestamps"]

    @property
    def artifact_type(self) -> str:
        return self.manifest["artifact_type"]

    @property
    def artifact_id(self) -> str:
        return self.manifest["artifact_id"]

    @property
    def recording_id(self):
        return self.manifest.get("recording", {}).get("recording_id")

    @property
    def source_system(self) -> str:
        return self.manifest.get("source_system", SOURCE_SYSTEM)

    @property
    def validation_ok(self) -> bool:
        v = self.manifest.get("validation", [])
        return bool(v) and all(i.get("ok") for i in v)

    def as_dict(self) -> dict:
        return {"artifact_dir": str(self.artifact_dir),
                "artifact_id": self.artifact_id,
                "artifact_type": self.artifact_type,
                "source_system": self.source_system,
                "recording_id": self.recording_id,
                "video_path": str(self.video_path),
                "gaze_path": str(self.gaze_path),
                "world_timestamps_path": str(self.world_timestamps_path),
                "validation_ok": self.validation_ok}


def _load(path, manifest_name: str, artifact_type: str) -> AriaArtifact:
    art_dir, manifest = read_manifest(path, manifest_name)
    where = str(art_dir / manifest_name)
    require(manifest, where, artifact_type=artifact_type, schema_major="1")
    if manifest.get("source_system") != SOURCE_SYSTEM:
        raise ArtifactError(f"{where}: source_system must be "
                            f"{SOURCE_SYSTEM!r}")
    files_decl = manifest.get("files")
    if not isinstance(files_decl, dict):
        raise ArtifactError(f"{where}: missing files section")
    files = {k: resolve_declared_file(art_dir, rel, key=k, where=where)
             for k, rel in files_decl.items()}
    for key in ("video", "gaze", "world_timestamps"):
        if key not in files:
            raise ArtifactError(f"{where}: files must declare {key}")
    ts = manifest.get("timestamps", {})
    if ts.get("domain") != "aria_device_time":
        raise ArtifactError(f"{where}: timestamps.domain must be "
                            f"'aria_device_time' (got {ts.get('domain')!r})")
    if ts.get("units") != "nanoseconds":
        raise ArtifactError(f"{where}: timestamps.units must be "
                            f"'nanoseconds'")
    if not manifest.get("validation"):
        raise ArtifactError(f"{where}: no validation record")
    if any(not v.get("ok") for v in manifest["validation"]):
        raise ArtifactError(f"{where}: validation record contains failures")
    return AriaArtifact(art_dir, manifest, files)


def load_prepared_aria(path) -> AriaArtifact:
    return _load(path, ARIA_PREP_MANIFEST, PREPARED_ARIA_TYPE)


def load_trimmed_aria(path) -> AriaArtifact:
    return _load(path, ARIA_TRIM_MANIFEST, TRIMMED_ARIA_TYPE)


def load_aria_artifact(path) -> AriaArtifact:
    """Load either Aria artifact type (whichever manifest is present)."""
    p = Path(path)
    if (p / ARIA_TRIM_MANIFEST).is_file():
        return load_trimmed_aria(p)
    if (p / ARIA_PREP_MANIFEST).is_file():
        return load_prepared_aria(p)
    raise ArtifactError(f"{p}: no {ARIA_PREP_MANIFEST} or "
                        f"{ARIA_TRIM_MANIFEST}")


def classify_aria_artifact(d) -> str:
    """Cheap structural classification for folder listings (stat only)."""
    d = Path(d)
    try:
        has_payload = ((d / "scene.mp4").is_file()
                       and (d / "gaze.csv").is_file()
                       and (d / "world_timestamps.csv").is_file())
        if not has_payload:
            return ""
        for name, kind in ((ARIA_TRIM_MANIFEST, TRIMMED_ARIA_TYPE),
                           (ARIA_PREP_MANIFEST, PREPARED_ARIA_TYPE)):
            p = d / name
            if p.is_file():
                try:
                    m = json.loads(p.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if m.get("artifact_type") == kind:
                    return kind
    except OSError:
        return ""
    return ""
