"""Recording intake: discovery + fast structural inspection of NEON folders.

This is the reusable layer between "a path on disk" and the expensive,
authoritative loading in :mod:`export` (full packet sweep + CSV parse +
sync check). Nothing here loads gaze data into memory or decodes video
packets — inspection must stay cheap enough to run while the user merely
highlights folders in the browser.

Two levels:

* :func:`looks_like_recording` — stat-only (for directory-listing badges).
* :class:`RecordingInspector` / :func:`inspect_recording` — structural
  inspection producing a :class:`RecordingBundle` (or problems): file
  discovery + ambiguity detection, video HEADER metadata (no packet demux),
  newline-counted CSV row counts, info.json metadata, and a structural
  sync hint comparing the container's header frame count with the world-row
  count. The authoritative sync check still runs in export.load_recording()
  after the user actually selects the recording.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REQUIRED_CSVS = ("gaze.csv", "world_timestamps.csv")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
OPTIONAL_META = ("info.json", "scene_camera.json")


def _video_candidates(d: Path) -> list:
    try:
        return sorted(p for p in d.iterdir()
                      if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    except PermissionError:
        return []


def looks_like_recording(d: Path) -> bool:
    """Stat-only quick check used for listing badges (no file reads).

    Requires exactly ONE video so the badge agrees with what inspect()/
    load_recording will actually accept (multiple videos = ambiguous).
    """
    try:
        return (all((d / n).is_file() for n in REQUIRED_CSVS)
                and len(_video_candidates(d)) == 1)
    except OSError:
        return False


def _count_rows(path: Path) -> Optional[int]:
    """Data-row count = non-blank line count minus header.

    Skips whitespace-only lines to match core.load_timestamp_csv semantics
    (a trailing blank line must not fabricate a phantom row / a spurious
    sync-hint mismatch). Buffered binary iteration keeps this fast.
    """
    try:
        n = 0
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    n += 1
        return max(0, n - 1)             # minus header
    except OSError:
        return None


@dataclass
class RecordingBundle:
    """Canonical description of a source NEON recording (paths + metadata).

    Contains resolved paths and cheap metadata only — no gaze samples, no
    packet lists. export.load_recording() consumes `rec_dir` to produce the
    heavyweight LoadedRecording when the user actually selects it.
    """
    rec_dir: Path
    video_path: Path
    gaze_path: Path
    world_path: Path
    info_path: Optional[Path] = None
    scene_camera_path: Optional[Path] = None
    recording_id: Optional[str] = None
    recording_name: Optional[str] = None
    wearer_name: Optional[str] = None
    duration_s: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    has_audio: Optional[bool] = None
    header_frame_count: Optional[int] = None   # container header, not decoded
    world_rows: Optional[int] = None
    gaze_rows: Optional[int] = None
    is_trimmed_artifact: bool = False          # has a trim_manifest.json

    def as_dict(self) -> dict:
        d = dict(vars(self))
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d


@dataclass
class InspectionResult:
    path: str
    is_recording: bool
    problems: list = field(default_factory=list)     # human-readable strings
    bundle: Optional[RecordingBundle] = None
    child_recordings: list = field(default_factory=list)  # [{name, path}]
    sync_hint: str = "unknown"   # 'likely_ok' | 'count_mismatch' | 'unknown'
    sync_hint_detail: str = ""

    def as_dict(self) -> dict:
        return {
            "path": self.path, "is_recording": self.is_recording,
            "problems": self.problems,
            "bundle": self.bundle.as_dict() if self.bundle else None,
            "child_recordings": self.child_recordings,
            "sync_hint": self.sync_hint,
            "sync_hint_detail": self.sync_hint_detail,
        }


class RecordingInspector:
    """Discovers and structurally validates NEON recording folders."""

    def inspect(self, rec_dir: Path) -> InspectionResult:
        rec_dir = Path(rec_dir)
        res = InspectionResult(path=str(rec_dir), is_recording=False)
        if not rec_dir.is_dir():
            res.problems.append("not a directory")
            return res

        vids = _video_candidates(rec_dir)
        missing = [n for n in REQUIRED_CSVS if not (rec_dir / n).is_file()]
        if not vids:
            res.problems.append("no scene video (*.mp4) found")
        elif len(vids) > 1:
            res.problems.append(
                "ambiguous: multiple video files found: "
                + ", ".join(v.name for v in vids))
        for n in missing:
            res.problems.append(f"missing required file: {n}")

        if res.problems:
            # maybe it is a PARENT of recordings — surface the children
            res.child_recordings = self.find_child_recordings(rec_dir)
            return res

        bundle = RecordingBundle(
            rec_dir=rec_dir, video_path=vids[0],
            gaze_path=rec_dir / "gaze.csv",
            world_path=rec_dir / "world_timestamps.csv",
            is_trimmed_artifact=(rec_dir / "trim_manifest.json").is_file(),
        )
        for name, attr in (("info.json", "info_path"),
                           ("scene_camera.json", "scene_camera_path")):
            p = rec_dir / name
            if p.is_file():
                setattr(bundle, attr, p)

        bundle.world_rows = _count_rows(bundle.world_path)
        bundle.gaze_rows = _count_rows(bundle.gaze_path)
        self._read_info(bundle)
        self._probe_video_header(bundle, res)

        # structural sync hint (header only — the authoritative check runs
        # on selection via export.load_recording)
        if bundle.header_frame_count and bundle.world_rows is not None:
            if bundle.header_frame_count == bundle.world_rows:
                res.sync_hint = "likely_ok"
                res.sync_hint_detail = (
                    f"header frame count ({bundle.header_frame_count}) == "
                    f"world rows ({bundle.world_rows}); full validation runs "
                    f"on selection")
            else:
                res.sync_hint = "count_mismatch"
                res.sync_hint_detail = (
                    f"container header reports {bundle.header_frame_count} "
                    f"frames but world_timestamps.csv has {bundle.world_rows} "
                    f"rows — full validation will decide")
        else:
            res.sync_hint_detail = ("container header has no frame count; "
                                    "full validation runs on selection")

        # header-probe problems (unreadable container) also disqualify, so
        # is_recording <=> no problems on every path; keep the bundle so the
        # UI can still show whatever metadata was discovered
        res.is_recording = not res.problems
        res.bundle = bundle
        return res

    def find_child_recordings(self, d: Path, limit: int = 200) -> list:
        out = []
        try:
            for c in sorted(d.iterdir(), key=lambda p: p.name.lower()):
                if len(out) >= limit:
                    break
                try:
                    if c.is_dir() and not c.name.startswith(".") \
                            and looks_like_recording(c):
                        out.append({"name": c.name, "path": str(c)})
                except PermissionError:
                    continue
        except (PermissionError, OSError):
            pass
        return out

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _read_info(bundle: RecordingBundle) -> None:
        if bundle.info_path is None:
            return
        try:
            data = json.loads(bundle.info_path.read_text())
        except (OSError, ValueError):
            return
        rid = data.get("recording_id")
        bundle.recording_id = rid if isinstance(rid, str) else None
        wn = data.get("wearer_name")
        bundle.wearer_name = wn if isinstance(wn, str) else None
        tmpl = data.get("template_data") or {}
        rn = tmpl.get("recording_name") if isinstance(tmpl, dict) else None
        bundle.recording_name = rn if isinstance(rn, str) else None

    @staticmethod
    def _probe_video_header(bundle: RecordingBundle,
                            res: InspectionResult) -> None:
        """Header-only probe: stream metadata without demuxing packets."""
        try:
            import av
            with av.open(str(bundle.video_path)) as c:
                if not c.streams.video:
                    res.problems.append(
                        f"{bundle.video_path.name}: no video stream")
                    return
                v = c.streams.video[0]
                bundle.width = v.codec_context.width
                bundle.height = v.codec_context.height
                bundle.header_frame_count = int(v.frames) if v.frames else None
                if v.duration is not None:
                    bundle.duration_s = float(v.duration * v.time_base)
                elif c.duration is not None:
                    bundle.duration_s = c.duration / av.time_base
                bundle.has_audio = bool(c.streams.audio)
        except Exception as e:              # unreadable/corrupt container
            res.problems.append(
                f"cannot read video header of {bundle.video_path.name}: {e}")


def inspect_recording(rec_dir: Path) -> InspectionResult:
    """Module-level convenience wrapper."""
    return RecordingInspector().inspect(rec_dir)
