"""Explicit, tested conversions between the four time domains:

    frame index  <->  decoded video PTS (seconds, file domain)
                 <->  HTML mediaTime (browser presentation clock)
                 <->  world timestamp (int nanoseconds)

Domain facts (empirically established on the fixtures):

* the decoded-frame PTS table is AUTHORITATIVE for frame identity — never a
  nominal FPS. For our sources the packet-PTS sweep equals decoded-frame PTS
  (verified by full decode), and intake re-verifies it per recording.
* trimmed artifacts carry a nonzero constant container start (e.g. 16 ms in
  the current fixture): frame 0's PTS is 0.016, and the verified video<->world
  offset is likewise constant. These offsets are PRESERVED, never zeroed.
* HTML mediaTime for MP4 equals the file PTS domain in the supported browser
  (Chromium); mapping uses NEAREST-PTS with a half-min-gap tolerance so a
  few-ms browser clock skew can never flip the frame index. Callers should
  treat a larger residual as "unmappable" rather than guessing.
* world timestamps are int ns (beyond float64 integer range) and map by
  INDEX: world_ts[i] belongs to decoded frame i (the verified 1:1 mapping).
"""

from __future__ import annotations

import bisect
from typing import Optional


class FrameTimeMap:
    def __init__(self, pts_s: list, world_ts_ns: Optional[list] = None,
                 sync_ok: bool = False):
        if not pts_s:
            raise ValueError("empty PTS table")
        if any(b <= a for a, b in zip(pts_s, pts_s[1:])):
            raise ValueError("PTS table must be strictly increasing")
        self.pts_s = list(pts_s)
        self.world_ts_ns = list(world_ts_ns) if world_ts_ns else None
        self.sync_ok = bool(sync_ok)
        gaps = [b - a for a, b in zip(pts_s, pts_s[1:])]
        self.min_gap_s = min(gaps) if gaps else 1 / 30
        # half the smallest real frame interval: the largest tolerance that
        # can never map a timestamp across a frame boundary
        self.snap_tol_s = self.min_gap_s / 2

    @property
    def n_frames(self) -> int:
        return len(self.pts_s)

    # ------------------------------------------------ frame <-> decoded PTS
    def pts_for_frame(self, i: int) -> float:
        if not (0 <= i < self.n_frames):
            raise IndexError(f"frame {i} outside 0..{self.n_frames - 1}")
        return self.pts_s[i]

    def frame_for_pts(self, t: float, *, nearest: bool = True) -> int:
        """Nearest (default) or floor mapping of a file-domain PTS."""
        i = bisect.bisect_right(self.pts_s, t) - 1
        i = max(0, min(i, self.n_frames - 1))
        if nearest and i + 1 < self.n_frames:
            if abs(self.pts_s[i + 1] - t) < abs(t - self.pts_s[i]):
                return i + 1
        return i

    # --------------------------------------------- frame <-> HTML mediaTime
    def media_time_for_frame(self, i: int) -> float:
        """mediaTime equals the file PTS domain for MP4 in the supported
        browser (kept as its own conversion so a per-browser calibration
        offset has one place to live if ever needed)."""
        return self.pts_for_frame(i)

    def frame_for_media_time(self, media_time: float) -> Optional[int]:
        """Presented mediaTime -> frame, or None when the residual exceeds
        the half-frame tolerance (caller must not guess)."""
        i = self.frame_for_pts(media_time, nearest=True)
        if abs(media_time - self.pts_s[i]) > self.snap_tol_s + 1e-9:
            return None
        return i

    # ------------------------------------------------ frame <-> world time
    def world_ns_for_frame(self, i: int) -> Optional[int]:
        if not self.sync_ok or self.world_ts_ns is None:
            return None
        if not (0 <= i < len(self.world_ts_ns)):
            return None
        return self.world_ts_ns[i]

    def frame_for_world_ns(self, t_ns: int) -> Optional[int]:
        """Floor-to-frame over world timestamps (the gaze-fusion rule)."""
        if not self.sync_ok or not self.world_ts_ns:
            return None
        i = bisect.bisect_right(self.world_ts_ns, int(t_ns)) - 1
        return max(0, min(i, len(self.world_ts_ns) - 1))

    # ------------------------------------------------------------ constant
    def video_world_offset_s(self) -> Optional[float]:
        """The verified constant (video PTS - world-relative) offset, e.g.
        +0.016 s for the current trimmed fixture. Preserved explicitly."""
        if not self.sync_ok or not self.world_ts_ns:
            return None
        rel0 = 0.0
        return self.pts_s[0] - rel0  # world_rel[0] is 0 by construction
