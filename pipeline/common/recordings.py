"""Generic artifact dispatch across source systems (NEON, Project Aria).

Downstream stages consume ONE logical recording interface:

    scene video · one authoritative timestamp per scene frame ·
    timestamped 2D gaze in scene-video pixels · provenance

This module decides which source system a folder belongs to and resolves
that interface, without either system pretending to be the other. It is
server-safe: manifest JSON + stat only — no pandas, no PyAV, and never
projectaria_tools (native Aria work happens in the isolated helper).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

SYSTEM_NEON = "pupil_neon"
SYSTEM_ARIA = "project_aria"

# Aria artifact kinds (their manifests carry the authoritative type)
ARIA_KINDS = ("prepared_aria_recording", "trimmed_aria_recording")

ARIA_KIND_LABELS = {
    "prepared_aria_recording": "Prepared Aria recording — ready",
    "trimmed_aria_recording": "Trimmed Aria artifact — ready",
    "aria_download": "Project Aria source recording",
}

# NEON and Aria artifacts share the manifest FILENAMES (prep_manifest.json /
# trim_manifest.json) for downstream compatibility, so classification must
# read `artifact_type` — never the filename alone.
_MANIFESTS = ("trim_manifest.json", "prep_manifest.json")


def read_manifest_type(d: Path) -> tuple:
    """(artifact_type, source_system, manifest) for a folder, or ("","",{})."""
    d = Path(d)
    for name in _MANIFESTS:
        p = d / name
        if not p.is_file():
            continue
        try:
            m = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(m, dict):
            continue
        at = m.get("artifact_type") or ""
        sys_ = m.get("source_system") or (SYSTEM_ARIA if at in ARIA_KINDS
                                          else SYSTEM_NEON)
        if at:
            return at, sys_, m
    return "", "", {}


def classify_aria_folder(d) -> str:
    """Aria kinds only: artifact type, `aria_download`, or "" (not Aria).

    Stat + small-JSON reads; safe to call from a folder chooser.
    """
    d = Path(d)
    at, sys_, _ = read_manifest_type(d)
    if at in ARIA_KINDS and sys_ == SYSTEM_ARIA:
        payload = ((d / "scene.mp4").is_file()
                   and (d / "gaze.csv").is_file()
                   and (d / "world_timestamps.csv").is_file())
        return at if payload else ""
    try:
        from aria_source.detect import looks_like_aria_folder
    except ImportError:
        return ""
    return "aria_download" if looks_like_aria_folder(d) else ""


def is_aria_artifact(d) -> bool:
    return classify_aria_folder(d) in ARIA_KINDS


def source_system_of(d) -> str:
    """Which source system produced this folder (best effort)."""
    kind = classify_aria_folder(d)
    return SYSTEM_ARIA if kind else SYSTEM_NEON


def aria_selection_interval(manifest: dict) -> Optional[tuple]:
    """Validated half-open interval of a trimmed Aria artifact, or None."""
    sel = manifest.get("selection") or {}
    a, b = sel.get("absolute_start_ns"), sel.get("absolute_end_boundary_ns")
    if isinstance(a, int) and isinstance(b, int) and b > a:
        return a, b
    return None


def logical_paths(d) -> dict:
    """Canonical trio + provenance for an Aria artifact folder.

    Raises ArtifactError (via the Aria loader) when the folder is not a
    valid Aria artifact — validation is never weakened here.
    """
    from aria_source.artifact import load_aria_artifact
    art = load_aria_artifact(d)
    m = art.manifest
    return {
        "source_system": SYSTEM_ARIA,
        "artifact_type": art.artifact_type,
        "artifact_id": art.artifact_id,
        "recording_id": art.recording_id,
        "video": art.video_path,
        "gaze": art.gaze_path,
        "world_timestamps": art.world_timestamps_path,
        "parent_artifact": m.get("parent_artifact"),
        "timestamp_domain": m["timestamps"]["domain"],
        "timestamp_note": m["timestamps"].get("compatibility_note", ""),
        "orientation": m.get("orientation", {}),
        "gaze_meta": m.get("gaze", {}),
        "recording_meta": m.get("recording", {}),
        "dataset": m.get("dataset"),
        "manifest": m,
    }
