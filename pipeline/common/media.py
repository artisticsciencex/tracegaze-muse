"""Media probing (PyAV) and frame-accurate cutting (ffmpeg) — shared service.

Canonical implementation promoted from neon_trim (neon_trim.media re-exports
from here; all its tests exercise this code).

Probing uses a packet-level sweep: for this VFR H.264 content each video
packet carries the presentation timestamp of exactly one frame, so sorting
packet PTS yields the display-order timestamp list without a full decode
(verified against a full decode on the test fixture: identical count and
values). Output validation after an export DOES fully decode the (short)
result, which is the strong guarantee where it matters.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional

import av
import imageio_ffmpeg

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def find_scene_video(rec_dir: Path) -> Path:
    """The scene video = the single video file in the recording folder."""
    rec_dir = Path(rec_dir)
    vids = sorted(p for p in rec_dir.iterdir()
                  if p.suffix.lower() in VIDEO_EXTS and p.is_file())
    if not vids:
        raise FileNotFoundError(f"no video file found in {rec_dir}")
    if len(vids) > 1:
        raise ValueError(
            f"multiple video files in {rec_dir}: {[v.name for v in vids]} — "
            f"cannot determine the scene video")
    return vids[0]


@dataclass
class VideoProbe:
    path: Path
    n_frames: int
    pts_s: list                       # presentation timestamps, seconds, sorted
    time_base: Fraction
    duration_s: float                 # video stream duration (container fallback)
    width: int
    height: int
    codec: str
    pix_fmt: Optional[str]
    has_audio: bool
    audio_codec: Optional[str] = None
    audio_rate: Optional[int] = None
    audio_channels: Optional[int] = None
    audio_duration_s: Optional[float] = None
    container_start_s: float = 0.0    # ffmpeg's -ss reference point (nonzero
                                      # for previously trimmed outputs)
    warnings: list = field(default_factory=list)


def probe_video(path: Path) -> VideoProbe:
    """Packet-level probe: stream metadata + per-frame presentation timestamps."""
    path = Path(path)
    with av.open(str(path)) as c:
        if not c.streams.video:
            raise ValueError(f"{path.name}: no video stream")
        v = c.streams.video[0]
        tb = v.time_base
        warnings = []

        duration_s = None
        if v.duration is not None:
            duration_s = float(v.duration * tb)
        elif c.duration is not None:
            duration_s = c.duration / av.time_base
            warnings.append("video stream has no duration; using container duration")

        audio = c.streams.audio[0] if c.streams.audio else None
        a_dur = None
        if audio is not None and audio.duration is not None:
            a_dur = float(audio.duration * audio.time_base)

        pts_ticks = []
        n_none = 0
        for pkt in c.demux(v):
            if pkt.size == 0:
                continue  # PyAV end-of-stream flush sentinel, not a frame
            if pkt.pts is None:
                n_none += 1
                continue
            pts_ticks.append(pkt.pts)
        if n_none:
            warnings.append(f"{n_none} video packet(s) had no PTS and were ignored")
        pts_ticks.sort()  # decode order -> presentation order (B-frames)
        pts_s = [float(p * tb) for p in pts_ticks]
        if duration_s is None:
            # last resort: extrapolate one median interval past the last pts
            if len(pts_s) >= 2:
                iv = sorted(b - a for a, b in zip(pts_s[:-1], pts_s[1:]))
                duration_s = pts_s[-1] + iv[len(iv) // 2]
            else:
                duration_s = (pts_s[-1] if pts_s else 0.0)
            warnings.append("no duration in stream/container; extrapolated from PTS")

        container_start_s = (c.start_time / av.time_base
                             if c.start_time is not None else 0.0)

        return VideoProbe(
            path=path, n_frames=len(pts_s), pts_s=pts_s, time_base=tb,
            duration_s=duration_s, container_start_s=container_start_s,
            width=v.codec_context.width, height=v.codec_context.height,
            codec=v.codec_context.name, pix_fmt=v.codec_context.pix_fmt,
            has_audio=audio is not None,
            audio_codec=(audio.codec_context.name if audio else None),
            audio_rate=(audio.codec_context.sample_rate if audio else None),
            audio_channels=(audio.codec_context.channels if audio else None),
            audio_duration_s=a_dur,
            warnings=warnings,
        )


def decode_all_pts(path: Path) -> list:
    """Fully decode a video and return every frame's PTS in seconds (display order).

    Used to validate exported outputs (short), where a real decode — not a
    packet sweep — is the ground truth for "these frames exist and play".
    """
    out = []
    with av.open(str(path)) as c:
        v = c.streams.video[0]
        v.thread_type = "AUTO"
        tb = v.time_base
        for frame in c.decode(v):
            t = frame.pts if frame.pts is not None else frame.dts
            out.append(float(t * tb) if t is not None else float("nan"))
    return out


# ---------------------------------------------------------------------------
# Frame-accurate cut (re-encode)
# ---------------------------------------------------------------------------

ENCODE_SETTINGS = {
    "video_codec": "libx264",
    "crf": 18,
    "preset": "veryfast",
    "pix_fmt": "yuv420p",
    "fps_mode": "passthrough (VFR timestamps preserved)",
    # -enc_time_base is ESSENTIAL: without it ffmpeg rescales frames into a
    # guessed-fps encoder time base and silently quantizes/corrupts the VFR
    # timestamps (verified: duplicate + shuffled PTS on a synthetic clip).
    "enc_time_base": "1:90000",
    "video_track_timescale": 90000,
    "audio_codec": "aac",
    "audio_bitrate": "160k",
    "container_flags": "+faststart",
}


class CutError(RuntimeError):
    pass


class CutCancelled(CutError):
    """The cut was aborted by the caller's cancel_event (not a failure)."""


def run_cut(src: Path, dst: Path, cut_start_s: float, cut_end_s: float,
            has_audio: bool,
            container_start_s: float = 0.0,
            progress_cb: Optional[Callable[[float], None]] = None,
            cancel_event: Optional[threading.Event] = None) -> dict:
    """Re-encode [cut_start_s, cut_end_s) of src into dst (frame-accurate).

    cut_start_s/cut_end_s are in the FILE'S pts domain (as probed). ffmpeg's
    input-side ``-ss`` however seeks relative to the container start_time
    (nonzero for previously trimmed outputs — verified: forgetting this shifts
    the cut by start_time and selects the wrong frames on a re-trim), so the
    argument passed to ffmpeg is ``cut_start - container_start``. Boundaries
    are frame-midpoints chosen by core.plan_trim, so µs rounding cannot move
    a frame across the cut. Returns the encode-settings dict including
    ``effective_cut_start_s`` (the exact rebase point of output timestamps).
    Raises CutError on failure/cancel.
    """
    src, dst = Path(src), Path(dst)
    # clamp: content cannot start before the container start
    eff_start = max(cut_start_s, container_start_s)
    ss_arg = eff_start - container_start_s
    duration = cut_end_s - eff_start
    if duration <= 0:
        raise CutError(f"non-positive cut duration ({duration:.6f}s)")

    progress_path = dst.with_suffix(".progress.txt")
    cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-y",
           "-ss", f"{ss_arg:.6f}",
           "-i", str(src),
           "-t", f"{duration:.6f}",
           "-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "0:a:0", "-c:a", ENCODE_SETTINGS["audio_codec"],
                "-b:a", ENCODE_SETTINGS["audio_bitrate"]]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", ENCODE_SETTINGS["video_codec"],
            "-crf", str(ENCODE_SETTINGS["crf"]),
            "-preset", ENCODE_SETTINGS["preset"],
            "-pix_fmt", ENCODE_SETTINGS["pix_fmt"],
            "-fps_mode", "passthrough",
            "-enc_time_base", ENCODE_SETTINGS["enc_time_base"],
            "-video_track_timescale", str(ENCODE_SETTINGS["video_track_timescale"]),
            "-movflags", "+faststart",
            "-progress", str(progress_path),
            str(dst)]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True)
    stderr_buf = []

    def _drain():
        for line in proc.stderr:
            stderr_buf.append(line)
    t = threading.Thread(target=_drain, daemon=True)
    t.start()

    try:
        while proc.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                proc.kill()
                proc.wait()
                raise CutCancelled("export cancelled")
            if progress_cb is not None:
                frac = _read_progress_fraction(progress_path, duration)
                if frac is not None:
                    progress_cb(min(frac, 1.0))
            time.sleep(0.2)
    finally:
        t.join(timeout=5)
        try:
            progress_path.unlink(missing_ok=True)
        except OSError:
            pass

    if proc.returncode != 0:
        tail = "".join(stderr_buf[-25:]).strip()
        raise CutError(f"ffmpeg exited with code {proc.returncode}:\n{tail}")
    if progress_cb is not None:
        progress_cb(1.0)
    # provenance string: strip the "-progress <path>" pair and use the output
    # basename, so the documented command is replayable without side effects
    disp = list(cmd)
    i = disp.index("-progress")
    del disp[i:i + 2]
    disp[-1] = os.path.basename(str(dst))
    return {"command": " ".join(disp),
            "effective_cut_start_s": eff_start, **ENCODE_SETTINGS}


def _read_progress_fraction(progress_path: Path, duration_s: float) -> Optional[float]:
    """Parse ffmpeg -progress output (key=value lines) into a 0..1 fraction."""
    try:
        text = progress_path.read_text()
    except OSError:
        return None
    out_us = None
    for line in reversed(text.splitlines()):
        if line.startswith("out_time_us="):
            try:
                out_us = int(line.split("=", 1)[1])
            except ValueError:
                pass
            break
        if line.startswith("out_time_ms="):  # older ffmpeg: microseconds despite name
            try:
                out_us = int(line.split("=", 1)[1])
            except ValueError:
                pass
            break
    if out_us is None or duration_s <= 0:
        return None
    return max(0.0, out_us / 1e6 / duration_s)
