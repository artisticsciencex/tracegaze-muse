"""Core timestamp-mapping and CSV-trimming logic for NEON recordings.

Pure stdlib — no numpy/pandas/av — so it is import-light and reusable by the
gaze-fusion pipeline (`gaze_fusion.load_neon_gaze` can build on
:func:`frame_index_for_ns` / :class:`TimestampCsv`).

Synchronization model (established empirically on the test fixture
``2024-01-01_00-00-00-0000abcd_participant_a``, documented in README):

* The scene MP4 is **variable frame rate**. Its container-level "nominal fps"
  (12.5 in the fixture) is meaningless; the stream is ~30 Hz with irregular
  intervals (13.6–50 ms observed).
* The decoded video has exactly one frame per ``world_timestamps.csv`` row
  (6,433 == 6,433 in the fixture) and the decoded PTS of frame *i* equals
  ``(world_ts[i] - world_ts[0])`` to within PTS quantization (1/90000 s;
  measured max deviation 0.000 ms).
* Therefore the authoritative mapping is **index 1:1**: decoded frame *i* ↔
  world row *i*, with absolute time ``world_ts[i]`` (int nanoseconds).
  :func:`check_sync` verifies this per recording — count equality plus a
  constant per-index (pts − world) offset with tight spread — and callers must
  block export when it fails rather than guess.

All absolute NEON timestamps are handled as Python ``int`` nanoseconds:
values are ~1.7e18, beyond float64's 2^53 exact-integer range, so parsing
them as float would corrupt them by O(100 ns).

Selection semantics: a selection [start_frame, end_frame] is inclusive of
both endpoint frames and is represented internally as the half-open absolute
interval ``[world_ts[start], time_after_last_frame)`` where
``time_after_last_frame`` is ``world_ts[end+1]``, or the recording end for
the final frame.
"""

from __future__ import annotations

import bisect
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

NS = 1_000_000_000


# ---------------------------------------------------------------------------
# CSV loading (line-verbatim)
# ---------------------------------------------------------------------------

@dataclass
class TimestampCsv:
    """A NEON CSV loaded with raw lines preserved verbatim.

    ``lines[i]`` (including its original line ending) corresponds to
    ``ts_ns[i]``. Writing back any subset of lines preserves headers, column
    order, numeric precision, and IDs byte-for-byte.
    """

    path: Path
    header: str                      # header line, verbatim (with line ending)
    lines: list                      # data lines, verbatim (with line endings)
    ts_ns: list                      # int ns, parallel to `lines`
    ts_col: int                      # column index of "timestamp [ns]"
    sorted_non_decreasing: bool = True

    @property
    def n_rows(self) -> int:
        return len(self.lines)


def load_timestamp_csv(path: Path, ts_col_name: str = "timestamp [ns]") -> TimestampCsv:
    """Load a NEON CSV keeping every line verbatim; parse only the ns column."""
    path = Path(path)
    with open(path, "r", newline="") as f:
        raw = f.read()
    all_lines = raw.splitlines(keepends=True)
    if not all_lines:
        raise ValueError(f"{path.name}: file is empty")
    header = all_lines[0]
    cols = [c.strip() for c in header.rstrip("\r\n").split(",")]
    try:
        ts_col = cols.index(ts_col_name)
    except ValueError:
        raise ValueError(
            f"{path.name}: no '{ts_col_name}' column. Columns found: {cols}"
        ) from None
    lines, ts_ns = [], []
    for ln in all_lines[1:]:
        if not ln.strip():
            continue  # tolerate blank trailing line(s)
        fields = ln.rstrip("\r\n").split(",")
        try:
            ts = int(fields[ts_col])
        except (IndexError, ValueError):
            raise ValueError(
                f"{path.name}: unparsable '{ts_col_name}' in line: {ln[:120]!r}"
            ) from None
        lines.append(ln)
        ts_ns.append(ts)
    non_dec = all(b >= a for a, b in zip(ts_ns, ts_ns[1:]))
    return TimestampCsv(path=path, header=header, lines=lines, ts_ns=ts_ns,
                        ts_col=ts_col, sorted_non_decreasing=non_dec)


# ---------------------------------------------------------------------------
# Recording metadata
# ---------------------------------------------------------------------------

@dataclass
class RecordingInfo:
    recording_id: Optional[str] = None
    start_time_ns: Optional[int] = None
    duration_ns: Optional[int] = None
    warnings: list = field(default_factory=list)

    @property
    def end_time_ns(self) -> Optional[int]:
        if self.start_time_ns is not None and self.duration_ns is not None:
            return self.start_time_ns + self.duration_ns
        return None


def load_recording_info(rec_dir: Path) -> RecordingInfo:
    """Read info.json (recording id, start, duration). Missing pieces become warnings."""
    info = RecordingInfo()
    p = Path(rec_dir) / "info.json"
    if not p.exists():
        info.warnings.append("info.json not found — recording end estimated from world timestamps")
        return info
    try:
        data = json.loads(p.read_text())
    except Exception as e:  # malformed json
        info.warnings.append(f"info.json unreadable ({e}) — recording end estimated from world timestamps")
        return info
    rid = data.get("recording_id")
    if isinstance(rid, str):
        info.recording_id = rid
    else:
        info.warnings.append("info.json has no recording_id")
    st, du = data.get("start_time"), data.get("duration")
    if isinstance(st, int):
        info.start_time_ns = st
    if isinstance(du, int):
        info.duration_ns = du
    if info.end_time_ns is None:
        info.warnings.append("info.json lacks integer start_time/duration — recording end estimated from world timestamps")
    return info


def recording_end_ns(info: RecordingInfo, world_ts_ns: list) -> int:
    """Absolute time just after the last frame = end of the recording.

    Prefer info.json start_time+duration (matches events.csv recording.end).
    Fallback: last world timestamp + median frame interval.
    """
    end = info.end_time_ns
    if end is not None and world_ts_ns and end > world_ts_ns[-1]:
        return end
    if len(world_ts_ns) >= 2:
        med = int(statistics.median(
            b - a for a, b in zip(world_ts_ns[:-1], world_ts_ns[1:])))
        return world_ts_ns[-1] + max(med, 1)
    return world_ts_ns[-1] + 1 if world_ts_ns else 0


# ---------------------------------------------------------------------------
# Synchronization check (video PTS  vs  world_timestamps)
# ---------------------------------------------------------------------------

@dataclass
class SyncCheck:
    ok: bool
    method: str
    n_video_frames: int
    n_world_rows: int
    offset_s: float = 0.0        # constant (pts - rel_world) offset, seconds
    spread_s: float = 0.0        # max(d) - min(d): must be ~0 for a rigid mapping
    message: str = ""

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "method": self.method,
            "n_video_frames": self.n_video_frames, "n_world_rows": self.n_world_rows,
            "offset_ms": round(self.offset_s * 1e3, 4),
            "spread_ms": round(self.spread_s * 1e3, 4),
            "message": self.message,
        }


SYNC_METHOD = "world_timestamps_index_1to1"


def check_sync(video_pts_s: list, world_ts_ns: list,
               spread_tol_s: float = 0.005,
               offset_tol_s: float = 0.5) -> SyncCheck:
    """Verify the 1:1 index mapping between decoded video frames and world rows.

    Requires: equal counts; strictly increasing on both sides; and
    d_i = pts_i - (world_i - world_0)/1e9 constant across i (spread <= tol).
    A constant offset is allowed — it arises when a previously trimmed output
    is re-loaded (the cut point sits mid-way in the PTS gap before the first
    kept frame, i.e. up to half that gap) — because the rigid index mapping is
    unaffected by it; the SPREAD is what guarantees the mapping. offset_tol_s
    is only a loose sanity bound (it must exceed half the largest plausible
    dropped-frames gap, hence 0.5 s, not a frame interval).
    """
    nv, nw = len(video_pts_s), len(world_ts_ns)
    base = dict(method=SYNC_METHOD, n_video_frames=nv, n_world_rows=nw)
    if nv == 0 or nw == 0:
        return SyncCheck(ok=False, message="no video frames or no world timestamps", **base)
    if nv != nw:
        return SyncCheck(ok=False, message=(
            f"frame-count mismatch: video decodes to {nv} frames but "
            f"world_timestamps.csv has {nw} rows. No reliable 1:1 mapping — "
            f"export blocked. (Do NOT assume one world row per decoded frame.)"), **base)
    if any(b <= a for a, b in zip(world_ts_ns, world_ts_ns[1:])):
        return SyncCheck(ok=False, message="world timestamps are not strictly increasing", **base)
    if any(b <= a for a, b in zip(video_pts_s, video_pts_s[1:])):
        return SyncCheck(ok=False, message="video PTS are not strictly increasing", **base)
    w0 = world_ts_ns[0]
    d = [video_pts_s[i] - (world_ts_ns[i] - w0) / NS for i in range(nv)]
    dmin, dmax = min(d), max(d)
    spread = dmax - dmin
    offset = statistics.median(d)
    if spread > spread_tol_s:
        return SyncCheck(ok=False, offset_s=offset, spread_s=spread, message=(
            f"PTS↔world deviation is not a constant offset "
            f"(spread {spread*1e3:.2f} ms > {spread_tol_s*1e3:.0f} ms tolerance) — "
            f"the 1:1 index mapping is unreliable; export blocked."), **base)
    if abs(offset) > offset_tol_s:
        return SyncCheck(ok=False, offset_s=offset, spread_s=spread, message=(
            f"constant PTS↔world offset {offset*1e3:.1f} ms exceeds "
            f"{offset_tol_s*1e3:.0f} ms sanity bound; export blocked."), **base)
    return SyncCheck(ok=True, offset_s=offset, spread_s=spread,
                     message=f"1:1 mapping verified: {nv} frames, "
                             f"offset {offset*1e3:.3f} ms, spread {spread*1e3:.3f} ms",
                     **base)


# ---------------------------------------------------------------------------
# Selection / trim plan
# ---------------------------------------------------------------------------

@dataclass
class TrimPlan:
    start_frame: int
    end_frame: int                  # inclusive
    n_frames_selected: int
    # absolute half-open interval [abs_start_ns, abs_end_ns)
    abs_start_ns: int
    abs_end_ns: int
    # video-relative times of the endpoint frames (= source PTS, seconds)
    rel_start_s: float
    rel_end_s: float
    # ffmpeg cut boundaries (video-relative seconds, half-open [cut_start, cut_end))
    cut_start_s: float
    cut_end_s: float

    @property
    def duration_s(self) -> float:
        return (self.abs_end_ns - self.abs_start_ns) / NS


def plan_trim(world_ts_ns: list, video_pts_s: list, video_duration_s: float,
              start_frame: int, end_frame: int, rec_end_ns: int) -> TrimPlan:
    """Compute the trim plan for an inclusive frame selection.

    ffmpeg boundaries use midpoints between neighbouring frame PTS so that
    float/µs rounding inside ffmpeg can never include frame start-1 or
    frame end+1, nor drop frame start / frame end. Frames are >=10 ms apart
    (min 13.63 ms in the fixture) while ffmpeg's seek granularity is 1 µs.
    """
    n = len(world_ts_ns)
    if n == 0 or len(video_pts_s) != n:
        raise ValueError("plan_trim requires a verified 1:1 sync (equal counts)")
    if not (0 <= start_frame <= end_frame < n):
        raise ValueError(
            f"invalid selection [{start_frame}, {end_frame}] for {n} frames "
            f"(need 0 <= start <= end <= {n - 1})")

    abs_start = world_ts_ns[start_frame]
    abs_end = world_ts_ns[end_frame + 1] if end_frame + 1 < n else rec_end_ns
    if abs_end <= abs_start:
        raise ValueError("degenerate interval: end boundary <= start boundary")

    cut_start = 0.0 if start_frame == 0 else \
        (video_pts_s[start_frame - 1] + video_pts_s[start_frame]) / 2.0
    if end_frame + 1 < n:
        cut_end = (video_pts_s[end_frame] + video_pts_s[end_frame + 1]) / 2.0
    else:
        # keep the full tail (last frame's display time + trailing audio)
        cut_end = max(video_duration_s, video_pts_s[-1] + 1e-3)

    return TrimPlan(
        start_frame=start_frame, end_frame=end_frame,
        n_frames_selected=end_frame - start_frame + 1,
        abs_start_ns=abs_start, abs_end_ns=abs_end,
        rel_start_s=video_pts_s[start_frame], rel_end_s=video_pts_s[end_frame],
        cut_start_s=cut_start, cut_end_s=cut_end,
    )


# ---------------------------------------------------------------------------
# Interval queries + CSV trimming
# ---------------------------------------------------------------------------

def rows_in_interval(csv: TimestampCsv, start_ns: int, end_ns: int) -> tuple:
    """(lo, hi) row-index range with start_ns <= ts < end_ns.

    Uses bisect when timestamps are sorted; falls back to a linear scan
    returning an explicit index list otherwise.
    """
    if csv.sorted_non_decreasing:
        lo = bisect.bisect_left(csv.ts_ns, start_ns)
        hi = bisect.bisect_left(csv.ts_ns, end_ns)
        return lo, hi
    idx = [i for i, t in enumerate(csv.ts_ns) if start_ns <= t < end_ns]
    if not idx:
        return 0, 0
    if idx != list(range(idx[0], idx[-1] + 1)):
        raise ValueError(
            f"{csv.path.name}: timestamps are unsorted and rows inside the "
            f"interval are non-contiguous — refusing to trim")
    return idx[0], idx[-1] + 1


def count_in_interval(csv: TimestampCsv, start_ns: int, end_ns: int) -> int:
    lo, hi = rows_in_interval(csv, start_ns, end_ns)
    return hi - lo


def trim_csv(csv: TimestampCsv, start_ns: int, end_ns: int, out_path: Path) -> int:
    """Write header + rows with start_ns <= ts < end_ns, lines verbatim.

    Returns the number of data rows written. A final newline is ensured so
    the output is a well-formed text file even if the source's last line
    (when it is dropped) carried the only terminator.
    """
    lo, hi = rows_in_interval(csv, start_ns, end_ns)
    kept = csv.lines[lo:hi]
    out_path = Path(out_path)
    with open(out_path, "w", newline="") as f:
        f.write(csv.header if csv.header.endswith(("\n", "\r")) else csv.header + "\n")
        for ln in kept:
            f.write(ln)
        if kept and not kept[-1].endswith(("\n", "\r")):
            f.write("\n")
    return len(kept)


# ---------------------------------------------------------------------------
# Gaze -> frame mapping (for reuse by gaze_fusion)
# ---------------------------------------------------------------------------

def frame_index_for_ns(world_ts_ns: list, t_ns: int) -> int:
    """Frame index whose display interval contains t_ns.

    Frame i spans [world_ts[i], world_ts[i+1]). Samples before the first
    frame clip to 0; samples at/after the last timestamp clip to the last
    frame. This is the same floor-to-frame rule gaze_fusion needs.
    """
    if not world_ts_ns:
        raise ValueError("empty world timestamps")
    i = bisect.bisect_right(world_ts_ns, t_ns) - 1
    return max(0, min(i, len(world_ts_ns) - 1))


# ---------------------------------------------------------------------------
# Validation helpers (shared by export + tests)
# ---------------------------------------------------------------------------

def validate_monotonic(ts: list, *, strict: bool) -> bool:
    if strict:
        return all(b > a for a, b in zip(ts, ts[1:]))
    return all(b >= a for a, b in zip(ts, ts[1:]))


def validate_within(ts: list, start_ns: int, end_ns: int) -> bool:
    return all(start_ns <= t < end_ns for t in ts)
