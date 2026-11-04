"""Exact still-frame decoding for the paused-review path.

`GET /api/frame?dir=...&frame_index=N` must return the frame that IS frame N
by presentation order — the browser's own MP4 seeking cannot be trusted to
have presented it (root cause of the frame/mask desync).

Isolation decision: this uses PyAV (libav) IN-PROCESS. Unlike pyarrow —
which segfaulted the server twice and is fully subprocess-isolated — libav
has run in this server on every recording load (probe) and in neon_trim's
in-thread export validation for the project's whole life with a clean
record. A per-session lock serializes access to the decoder context (libav
contexts are not thread-safe), and any decoder exception surfaces as a job
error, never touching other sessions. If libav ever shows instability the
`FrameDecoder` API is process-agnostic and can move behind table_io-style
isolation without touching callers.

Correctness rules:
* frame identity = position in the authoritative decoded-PTS table
  (bundle.pts_s), never a guessed FPS
* seeks land on the keyframe at/before the target, then decode FORWARD
  matching frame PTS against the table (nearest within half min-gap)
* sequential stepping reuses the open decode position instead of re-seeking
  (never decodes from frame zero unnecessarily)
* bounded LRU of encoded JPEGs for recently requested frames
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
from typing import Optional


class FrameSourceError(RuntimeError):
    pass


class FrameDecoder:
    CACHE_FRAMES = 48          # ~100 KB JPEG each -> bounded at ~5 MB
    ROLL_FORWARD_LIMIT = 60    # sequential window before we re-seek

    def __init__(self, video_path: Path, pts_s: list):
        import av
        self.video_path = Path(video_path)
        self.pts_s = list(pts_s)
        if not self.pts_s:
            raise FrameSourceError("empty PTS table")
        gaps = [b - a for a, b in zip(pts_s, pts_s[1:])] or [1 / 30]
        self._tol = min(gaps) / 2
        self._lock = threading.Lock()
        self._cache: OrderedDict = OrderedDict()   # idx -> (jpeg, pts)
        self._container = av.open(str(self.video_path))
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._tb: Fraction = self._stream.time_base
        self._decoder = None       # active forward-decode generator
        self._next_idx = None      # frame index the generator will yield next

    # ------------------------------------------------------------ internal
    def _frame_index_for_pts(self, pts_ticks) -> Optional[int]:
        if pts_ticks is None:
            return None
        t = float(pts_ticks * self._tb)
        import bisect
        i = bisect.bisect_right(self.pts_s, t + 1e-9) - 1
        for cand in (i, i + 1):
            if 0 <= cand < len(self.pts_s) \
                    and abs(self.pts_s[cand] - t) <= self._tol:
                return cand
        return None

    def _encode_jpeg(self, frame) -> bytes:
        import av
        ctx = av.CodecContext.create("mjpeg", "w")
        ctx.width = frame.width
        ctx.height = frame.height
        ctx.pix_fmt = "yuvj420p"
        ctx.time_base = Fraction(1, 30)
        vf = frame.reformat(format="yuvj420p")
        vf.pts = None
        packets = ctx.encode(vf) + ctx.encode(None)
        return b"".join(bytes(p) for p in packets)

    def _start_decode_at(self, idx: int):
        """Seek to the keyframe at/before frame idx and start decoding."""
        target_ticks = int(round(self.pts_s[idx] / float(self._tb)))
        self._container.seek(target_ticks, stream=self._stream,
                             backward=True, any_frame=False)
        self._decoder = self._container.decode(self._stream)
        self._next_idx = None      # unknown until the first yielded frame

    # ------------------------------------------------------------- public
    def get(self, idx: int) -> tuple:
        """(jpeg_bytes, pts_s, cache_hit) for exact frame idx."""
        if not (0 <= idx < len(self.pts_s)):
            raise FrameSourceError(
                f"frame_index {idx} outside 0..{len(self.pts_s) - 1}")
        with self._lock:
            if idx in self._cache:
                jpeg, pts = self._cache[idx]
                self._cache.move_to_end(idx)
                return jpeg, pts, True
            # reuse the open decode position for forward stepping; anything
            # else gets an indexed seek (never a decode-from-zero)
            if (self._decoder is None or self._next_idx is None
                    or not (0 <= idx - self._next_idx
                            <= self.ROLL_FORWARD_LIMIT)):
                self._start_decode_at(idx)
            jpeg_out = None
            pts_out = None
            for frame in self._decoder:
                fidx = self._frame_index_for_pts(frame.pts)
                if fidx is None:
                    continue
                self._next_idx = fidx + 1
                if fidx < idx:
                    continue
                if fidx > idx:
                    # target not produced (unexpected) — precise diagnostic
                    raise FrameSourceError(
                        f"decoder skipped frame {idx} (got {fidx}); "
                        f"PTS table/container mismatch")
                jpeg_out = self._encode_jpeg(frame)
                pts_out = self.pts_s[fidx]
                break
            if jpeg_out is None:
                raise FrameSourceError(
                    f"could not decode frame {idx} (stream ended early)")
            self._cache[idx] = (jpeg_out, pts_out)
            while len(self._cache) > self.CACHE_FRAMES:
                self._cache.popitem(last=False)
            return jpeg_out, pts_out, False

    def close(self):
        with self._lock:
            self._decoder = None
            try:
                self._container.close()
            except Exception:
                pass
