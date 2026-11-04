"""Versioned artifact contract for trimmed NEON recordings.

`trim_manifest.json` (schema 2.x) is the downstream contract: future pipeline
stages (SAM3 tracking, gaze fusion) consume a trimmed recording through
:func:`load_artifact` instead of being handed loose video/CSV paths.

Contract essentials (see export._build_manifest for the full field set):

* ``schema_version``       "2.0" (major bump = breaking change)
* ``artifact_type``        "trimmed_neon_recording"
* ``artifact_id``          unique id of THIS artifact
* ``parent_artifact``      the source artifact's id when the source was
                           itself a trimmed artifact (re-trim chains)
* ``files``                paths RELATIVE to the artifact directory, so the
                           whole directory can be moved/renamed safely;
                           absolute paths appear only under ``source`` as
                           clearly-labelled provenance
* ``validation``           the export-time validation record
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = "2.0"
ARTIFACT_TYPE = "trimmed_neon_recording"
MANIFEST_NAME = "trim_manifest.json"


class ArtifactError(RuntimeError):
    pass


@dataclass
class TrimmedRecordingArtifact:
    """A validated, opened trimmed-recording artifact."""
    artifact_dir: Path
    manifest: dict
    video_path: Path
    gaze_path: Path
    world_timestamps_path: Path

    @property
    def artifact_id(self) -> str:
        return self.manifest["artifact_id"]

    @property
    def source_recording_id(self) -> Optional[str]:
        return self.manifest.get("source", {}).get("recording_id")

    @property
    def selection(self) -> dict:
        return self.manifest.get("selection", {})

    @property
    def sync_method(self) -> Optional[str]:
        return self.manifest.get("synchronization", {}).get("method")

    @property
    def validation_ok(self) -> bool:
        v = self.manifest.get("validation", [])
        return bool(v) and all(item.get("ok") for item in v)

    def as_dict(self) -> dict:
        return {
            "artifact_dir": str(self.artifact_dir),
            "artifact_id": self.artifact_id,
            "video_path": str(self.video_path),
            "gaze_path": str(self.gaze_path),
            "world_timestamps_path": str(self.world_timestamps_path),
            "validation_ok": self.validation_ok,
        }


def parent_artifact_ref(source_dir: Path) -> Optional[dict]:
    """If `source_dir` is itself a trimmed artifact, reference it as parent."""
    mp = Path(source_dir) / MANIFEST_NAME
    if not mp.is_file():
        return None
    try:
        m = json.loads(mp.read_text())
    except (OSError, ValueError):
        return None
    if m.get("artifact_type") != ARTIFACT_TYPE:
        return None
    aid = m.get("artifact_id")
    if not isinstance(aid, str) or not aid:
        return {"artifact_id": None,
                "artifact_type": ARTIFACT_TYPE,
                "manifest": MANIFEST_NAME,
                "note": "source has a trim manifest but no valid artifact_id "
                        "(corrupt or hand-edited); parent chain unresolvable"}
    return {"artifact_id": aid,
            "artifact_type": ARTIFACT_TYPE,
            "manifest": MANIFEST_NAME,
            "note": "the source of this trim was itself a trimmed artifact"}


def load_artifact(path) -> TrimmedRecordingArtifact:
    """Open + validate a trimmed artifact from its directory or manifest path.

    Checks: manifest parses; artifact_type matches; schema major version is
    supported; all declared files exist relative to the artifact directory;
    the export-time validation record exists. Raises ArtifactError with a
    precise reason otherwise.
    """
    p = Path(path).expanduser()
    manifest_path = p if p.is_file() else p / MANIFEST_NAME
    art_dir = manifest_path.parent
    if not manifest_path.is_file():
        raise ArtifactError(f"no {MANIFEST_NAME} found at {p}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except ValueError as e:
        raise ArtifactError(f"{manifest_path}: invalid JSON: {e}") from None
    except OSError as e:
        raise ArtifactError(f"{manifest_path}: cannot read manifest: {e}") from None

    atype = manifest.get("artifact_type")
    if atype != ARTIFACT_TYPE:
        raise ArtifactError(
            f"{manifest_path}: artifact_type is {atype!r}, expected "
            f"{ARTIFACT_TYPE!r}")
    sv = str(manifest.get("schema_version", ""))
    if not sv.startswith("2."):
        raise ArtifactError(
            f"{manifest_path}: unsupported schema_version {sv!r} "
            f"(this loader supports 2.x)")
    if "artifact_id" not in manifest:
        raise ArtifactError(f"{manifest_path}: missing artifact_id")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ArtifactError(f"{manifest_path}: missing 'files' section")
    resolved = {}
    art_real = os.path.realpath(art_dir)
    for key in ("video", "gaze", "world_timestamps"):
        rel = files.get(key)
        if not rel or not isinstance(rel, str):
            raise ArtifactError(
                f"{manifest_path}: files.{key} missing or not a string path "
                f"(got {type(rel).__name__})")
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ArtifactError(
                f"{manifest_path}: files.{key} must be a plain relative "
                f"path inside the artifact directory, got {rel!r}")
        fp = art_dir / rel_path
        # resolve symlinks: a declared file must REALLY live inside the
        # artifact directory (no symlink escape to arbitrary files)
        fp_real = os.path.realpath(fp)
        if fp_real != art_real and not fp_real.startswith(art_real + os.sep):
            raise ArtifactError(
                f"{manifest_path}: files.{key} resolves outside the artifact "
                f"directory (symlink escape): {rel!r}")
        if not fp.is_file():
            raise ArtifactError(
                f"artifact incomplete: {rel!r} not found in {art_dir}")
        resolved[key] = fp

    if not manifest.get("validation"):
        raise ArtifactError(f"{manifest_path}: no validation record")

    return TrimmedRecordingArtifact(
        artifact_dir=art_dir, manifest=manifest,
        video_path=resolved["video"], gaze_path=resolved["gaze"],
        world_timestamps_path=resolved["world_timestamps"])
