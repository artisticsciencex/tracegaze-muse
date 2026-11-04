"""Timestamp-defined frame cells: the exact gaze→frame assignment rule.

Rule (fixed by the stage specification — do NOT replace with a global
frame-distance threshold such as 16 or 20 ms):

- Frame i of the parent recording owns the half-open time cell
  ``[b[i], b[i+1])``.
- Interior boundaries are integer midpoints between adjacent
  ``world_timestamps.csv`` frame timestamps:
  ``b[i] = (world[i-1] + world[i]) // 2`` for ``0 < i < n_frames``
  (floor division on exact int nanoseconds; documented, deterministic).
- The first and final cells are bounded by the VALIDATED parent trim
  interval: ``b[0] = interval_start_ns`` (= the first frame's world
  timestamp under the trim semantics ``[start_frame_time,
  time_after_last_frame)``) and ``b[n] = interval_end_ns``.
- A gaze sample with timestamp ``ts`` belongs to frame i iff
  ``b[i] <= ts < b[i+1]``. Samples outside ``[b[0], b[n])`` receive
  ``no_matching_frame``.
- The signed offset ``ts - world[i]`` is preserved per sample
  (``frame_offset_ns`` exact int, ``frame_offset_ms`` for reporting).
- An anomalous gap between adjacent world timestamps larger than the
  documented synchronization-sanity threshold is REPORTED in validation;
  in-range samples are still assigned by the rule above, never silently
  dropped.

Everything here is pure-Python exact integer arithmetic on nanosecond
timestamps (values exceed 2**53 — floats must never touch them).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Optional, Sequence

# Sanity threshold for the gap between ADJACENT world timestamps.
# Basis: the NEON scene camera is nominally ~30 Hz; across the verified
# recordings the observed VFR intervals span ~13.6-50 ms (see
# pipeline/README and the neon_trim investigation). 200 ms = 4x the largest
# normal interval — a gap beyond it indicates dropped scene frames or a
# synchronization defect worth surfacing, while any smaller VFR jitter is
# ordinary. The threshold only triggers REPORTING; assignment is unchanged.
DEFAULT_SANITY_GAP_MS = 200.0

ASSIGNED = "assigned"
NO_MATCHING_FRAME = "no_matching_frame"


class TimeCellError(ValueError):
    pass


@dataclass(frozen=True)
class Assignment:
    frame_index: Optional[int]     # None <=> no_matching_frame
    offset_ns: Optional[int]       # signed ts - world[frame]; None if no frame
    status: str                    # ASSIGNED | NO_MATCHING_FRAME


class FrameCells:
    """Frame cells for one trimmed recording.

    Parameters
    ----------
    world_ts_ns : one int-ns timestamp per decoded video frame (index 1:1,
        the verified NEON sync model), strictly increasing.
    interval_start_ns / interval_end_ns : the validated parent trim
        interval ``[start, end)`` from the trim artifact manifest
        (``selection.absolute_start_ns`` / ``selection.absolute_end_boundary_ns``).
    sanity_gap_ms : documented synchronization-sanity threshold (reporting
        only — see module docstring).
    """

    def __init__(self, world_ts_ns: Sequence[int], interval_start_ns: int,
                 interval_end_ns: int, *,
                 sanity_gap_ms: float = DEFAULT_SANITY_GAP_MS):
        if len(world_ts_ns) == 0:
            raise TimeCellError("world_ts_ns is empty")
        w = []
        for t in world_ts_ns:
            if isinstance(t, bool) or not isinstance(t, int):
                # numpy int64 etc. must be converted by the CALLER via int();
                # silently accepting floats would corrupt >2**53 timestamps.
                raise TimeCellError(
                    f"world timestamp {t!r} is {type(t).__name__}, not int")
            w.append(t)
        for a, b in zip(w, w[1:]):
            if b <= a:
                raise TimeCellError(
                    f"world timestamps not strictly increasing: {a} -> {b}")
        if not isinstance(interval_start_ns, int) \
                or not isinstance(interval_end_ns, int):
            raise TimeCellError("interval boundaries must be exact ints")
        if interval_start_ns > w[0]:
            raise TimeCellError(
                f"interval_start_ns {interval_start_ns} is after the first "
                f"frame timestamp {w[0]} — trim interval must contain all "
                f"frames")
        if interval_end_ns <= w[-1]:
            raise TimeCellError(
                f"interval_end_ns {interval_end_ns} is not after the last "
                f"frame timestamp {w[-1]}")
        self.world = w
        self.n_frames = len(w)
        self.interval_start_ns = interval_start_ns
        self.interval_end_ns = interval_end_ns
        self.sanity_gap_ms = float(sanity_gap_ms)

        # boundaries: b[0]=interval start, interior midpoints, b[n]=interval end
        b = [interval_start_ns]
        for i in range(1, self.n_frames):
            b.append((w[i - 1] + w[i]) // 2)
        b.append(interval_end_ns)
        # strictly increasing by construction (world strictly increasing and
        # interval bounds validated above) — asserted for defense in depth
        for x, y in zip(b, b[1:]):
            if y <= x:
                raise TimeCellError(f"degenerate cell boundary {x} -> {y}")
        self.boundaries = b

    # ------------------------------------------------------------- assign
    def assign(self, ts_ns: int) -> Assignment:
        if isinstance(ts_ns, bool) or not isinstance(ts_ns, int):
            raise TimeCellError(
                f"gaze timestamp {ts_ns!r} is {type(ts_ns).__name__}, not int")
        i = bisect_right(self.boundaries, ts_ns) - 1
        if i < 0 or i >= self.n_frames:
            return Assignment(None, None, NO_MATCHING_FRAME)
        return Assignment(i, ts_ns - self.world[i], ASSIGNED)

    def cell_bounds(self, frame_index: int) -> tuple:
        """Half-open ``[start, end)`` cell of one frame."""
        if not (0 <= frame_index < self.n_frames):
            raise TimeCellError(f"frame {frame_index} out of range")
        return (self.boundaries[frame_index],
                self.boundaries[frame_index + 1])

    # ------------------------------------------------------- diagnostics
    def anomalous_gaps(self) -> list:
        """World-timestamp gaps exceeding the sanity threshold (report-only)."""
        thr_ns = int(self.sanity_gap_ms * 1e6)
        out = []
        for i in range(1, self.n_frames):
            gap = self.world[i] - self.world[i - 1]
            if gap > thr_ns:
                out.append({
                    "between_frames": [i - 1, i],
                    "gap_ms": gap / 1e6,
                    "world_ts_ns": [self.world[i - 1], self.world[i]],
                    "note": "reported per sanity threshold "
                            f"{self.sanity_gap_ms} ms; samples inside this "
                            "gap are still assigned by the midpoint rule, "
                            "not dropped",
                })
        return out

    def gap_stats_ms(self) -> dict:
        gaps = [(b - a) / 1e6 for a, b in zip(self.world, self.world[1:])]
        if not gaps:
            return {"n_gaps": 0}
        return {"n_gaps": len(gaps), "min_ms": min(gaps),
                "max_ms": max(gaps),
                "median_ms": sorted(gaps)[len(gaps) // 2]}
