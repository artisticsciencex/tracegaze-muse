"""Lightweight, purely structural detection of an Aria download folder.

Stat + filename inspection only (plus small-JSON reads). NO archive
extraction, NO projectaria_tools — servers call this to classify a folder
in a folder chooser; native inspection belongs in the isolated helper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

RECEIPT = "download_receipt.json"
PKG_VRS = "recording_vrs"
PKG_MPS = "mps"
PKG_META = "metadata_json"
PKG_PREVIEW = "video_main_rgb"

LAYOUT_RITW = "reading_in_the_wild_archives"
LAYOUT_AEA = "aria_everyday_activities_direct_vrs"

# Source roles are canonical even though public datasets package them
# differently. Keep these suffixes explicit: accepting every .zip as MPS
# would make a SLAM/point-cloud-only download look preparation-ready.
_SOURCE_SUFFIXES = (
    ("_recording_vrs.zip", PKG_VRS, LAYOUT_RITW, "zip"),
    ("_mps.zip", PKG_MPS, LAYOUT_RITW, "zip"),
    ("_metadata_json.zip", PKG_META, LAYOUT_RITW, "zip"),
    ("_main_recording.vrs", PKG_VRS, LAYOUT_AEA, "vrs"),
    ("_mps_eye_gaze.zip", PKG_MPS, LAYOUT_AEA, "zip"),
    ("_annotations.zip", PKG_META, LAYOUT_AEA, "zip"),
    # Shared by both layouts; VRS/MPS roles determine the layout.
    ("_preview_rgb.mp4", PKG_PREVIEW, "", "mp4"),
)

ARIA_LABEL = "Project Aria recording"


def _match_source_file(name: str):
    lower = name.lower()
    for suffix, key, layout, container in _SOURCE_SUFFIXES:
        if lower.endswith(suffix):
            return suffix, key, layout, container
    return None


def _parse_source_prefix(prefix: str) -> tuple[str, str, str]:
    """Return dataset, release and recording id from a downloader prefix."""
    parts = prefix.split("_", 2)
    return tuple(parts) if len(parts) == 3 else ("", "", "")


def _dataset_label(dataset: str) -> str:
    return {
        "ReadingInTheWild": "Reading in the Wild",
        "AriaEverydayActivities": "Aria Everyday Activities",
    }.get(dataset, dataset or "Project Aria")


@dataclass
class AriaPackage:
    """A structurally-detected Aria recording download folder."""
    path: Path
    recording_id: str = ""
    dataset: str = ""
    release: str = ""
    layout: str = ""
    files: dict = field(default_factory=dict)      # package -> Path
    file_formats: dict = field(default_factory=dict)  # package -> zip/vrs/mp4
    receipt: Optional[dict] = None
    problems: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def receipt_expected(self) -> bool:
        return self.layout == LAYOUT_RITW

    @property
    def label(self) -> str:
        return f"Project Aria / {_dataset_label(self.dataset)} recording"

    def as_dict(self) -> dict:
        return {
            "path": str(self.path), "recording_id": self.recording_id,
            "dataset": self.dataset, "release": self.release,
            "layout": self.layout,
            "files": {k: str(v) for k, v in self.files.items()},
            "file_formats": dict(self.file_formats),
            "sizes": {k: (v.stat().st_size if v.is_file() else None)
                      for k, v in self.files.items()},
            "has_receipt": self.receipt is not None,
            "receipt_expected": self.receipt_expected,
            "problems": self.problems, "ok": self.ok,
            "label": self.label,
        }


def looks_like_aria_folder(d) -> bool:
    """Cheap check for folder listings/badges (stat only)."""
    d = Path(d)
    try:
        if not d.is_dir():
            return False
        if (d / RECEIPT).is_file():
            return True
        # Receipt-less AEA folders still count when source-role naming matches.
        return any(_match_source_file(p.name) for p in d.iterdir()
                   if p.is_file())
    except OSError:
        return False


def inspect_aria_folder(d) -> AriaPackage:
    """Structural inspection: which packages are present and consistent."""
    d = Path(d)
    pkg = AriaPackage(path=d)
    if not d.is_dir():
        pkg.problems.append("not a directory")
        return pkg

    layouts = set()
    identities = []
    for p in sorted(d.iterdir()):
        if not p.is_file():
            continue
        match = _match_source_file(p.name)
        if not match:
            continue
        suffix, key, layout, container = match
        if key in pkg.files:
            pkg.problems.append(
                f"multiple files claim the {key} role: "
                f"{pkg.files[key].name}, {p.name}")
            continue
        pkg.files[key] = p
        pkg.file_formats[key] = container
        if layout:
            layouts.add(layout)
        identities.append(_parse_source_prefix(p.name[:-len(suffix)]))

    if len(layouts) > 1:
        pkg.problems.append(
            "mixed incompatible Aria package layouts in one folder")
    elif layouts:
        pkg.layout = next(iter(layouts))

    valid_ids = [x for x in identities if all(x)]
    if valid_ids:
        pkg.dataset, pkg.release, pkg.recording_id = valid_ids[0]
        if any(x != valid_ids[0] for x in valid_ids[1:]):
            pkg.problems.append(
                "source filenames describe different dataset/recording ids")

    receipt_path = d / RECEIPT
    if receipt_path.is_file():
        try:
            pkg.receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            pkg.problems.append(f"{RECEIPT} unreadable: {e}")
        else:
            pkg.dataset = pkg.receipt.get("dataset") or pkg.dataset
            pkg.release = pkg.receipt.get("release") or pkg.release
            pkg.recording_id = (pkg.receipt.get("sequence_id")
                                or pkg.recording_id)

    if PKG_VRS not in pkg.files:
        pkg.problems.append(
            "no supported VRS source — expected *_recording_vrs.zip "
            "(Reading in the Wild) or *_main_recording.vrs "
            "(Aria Everyday Activities)")
    if PKG_MPS not in pkg.files:
        pkg.problems.append(
            "no supported eye-gaze source — expected *_mps.zip "
            "(Reading in the Wild) or *_mps_eye_gaze.zip "
            "(Aria Everyday Activities)")
    if not pkg.recording_id:
        pkg.problems.append("cannot determine the recording id")
    return pkg


def validate_against_receipt(pkg: AriaPackage) -> list:
    """Compare present files with `download_receipt.json` (size only —
    hashing multi-GB archives happens in preparation, where it is
    progress-reported)."""
    out = []
    if pkg.receipt is None:
        expected = pkg.receipt_expected
        out.append({
            "check": "download receipt present",
            "ok": not expected,
            "detail": (
                "no download_receipt.json — required for the Reading in "
                "the Wild downloader layout"
                if expected else
                "this Aria Everyday Activities layout does not provide a "
                "download receipt; exact source sizes and SHA-256 hashes "
                "will be recorded during preparation"),
            "verification": "receipt" if expected else "sha256_on_prepare",
        })
        return out
    listed = {f.get("filename"): f for f in pkg.receipt.get("files", [])}
    out.append({"check": "download receipt present", "ok": True,
                "detail": f"{len(listed)} files listed"})
    for key, path in sorted(pkg.files.items()):
        info = listed.get(path.name)
        if info is None:
            out.append({"check": f"{key} listed in receipt", "ok": False,
                        "detail": f"{path.name} is not in the receipt"})
            continue
        size = path.stat().st_size
        ok = size == info.get("bytes")
        out.append({"check": f"{key} size matches receipt", "ok": ok,
                    "detail": f"{path.name}: {size} bytes "
                              f"(receipt {info.get('bytes')})"})
    missing = [n for n in listed if n not in {p.name
                                              for p in pkg.files.values()}]
    if missing:
        out.append({"check": "all receipt files present", "ok": False,
                    "detail": "missing: " + ", ".join(sorted(missing))})
    else:
        out.append({"check": "all receipt files present", "ok": True,
                    "detail": ""})
    return out
