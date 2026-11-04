"""Unit tests for neon_trim.core — synthetic timestamps, no media files.

Covers (per spec):
  * exact boundary inclusion/exclusion of gaze samples
  * missing gaze at the beginning of the recording
  * empty gaze intervals
  * irregular scene-frame timing (VFR)
  * the observed video/world-timestamp relationship + mismatch blocking
  * CSV verbatim preservation (headers, precision, IDs)
  * interval semantics [start_frame_time, time_after_last_frame)
"""

import textwrap
from pathlib import Path

import pytest

from neon_trim import core

NS = core.NS
T0 = 1_733_200_839_984_000_000  # realistic NEON absolute origin (~1.7e18: > 2^53)


# ---------------------------------------------------------------------------
# fixtures: a tiny synthetic recording with IRREGULAR frame timing
# ---------------------------------------------------------------------------

# 8 frames with irregular intervals (ms): 50, 50, 33, 33, 14, 40, 33
FRAME_OFFSETS_MS = [0, 50, 100, 133, 166, 180, 220, 253]
WORLD_TS = [T0 + int(ms * 1e6) for ms in FRAME_OFFSETS_MS]
REC_END = WORLD_TS[-1] + 33_000_000  # 33 ms after last frame
VIDEO_PTS = [ms / 1000.0 for ms in FRAME_OFFSETS_MS]
VIDEO_DUR = VIDEO_PTS[-1] + 0.033


def write_world_csv(path: Path, ts_list) -> Path:
    lines = ["section id,recording id,timestamp [ns]"]
    for t in ts_list:
        lines.append(f"sec-1,rec-1,{t}")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_gaze_csv(path: Path, ts_list) -> Path:
    lines = ["section id,recording id,timestamp [ns],gaze x [px],gaze y [px],worn"]
    for i, t in enumerate(ts_list):
        # deliberately quirky float formatting to verify verbatim preservation
        lines.append(f"sec-1,rec-1,{t},903.4230,740.56800,1.0")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def world_csv(tmp_path):
    return core.load_timestamp_csv(write_world_csv(tmp_path / "world_timestamps.csv", WORLD_TS))


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def test_load_preserves_lines_verbatim(tmp_path):
    p = write_gaze_csv(tmp_path / "gaze.csv", [T0, T0 + 5_000_000])
    csv = core.load_timestamp_csv(p)
    assert csv.n_rows == 2
    assert csv.lines[0] == "sec-1,rec-1," + str(T0) + ",903.4230,740.56800,1.0\n"
    assert csv.ts_ns == [T0, T0 + 5_000_000]
    assert csv.header.startswith("section id,recording id,timestamp [ns]")


def test_load_int_precision_beyond_float53(tmp_path):
    # 1.7e18 + 1 ns must survive exactly (float64 would round it away)
    t = 1_733_200_839_984_000_001
    p = write_world_csv(tmp_path / "w.csv", [t])
    csv = core.load_timestamp_csv(p)
    assert csv.ts_ns[0] == t
    assert float(t) != t  # sanity: float64 really cannot represent it


def test_load_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="timestamp"):
        core.load_timestamp_csv(p)


def test_load_crlf_preserved(tmp_path):
    p = tmp_path / "crlf.csv"
    p.write_bytes(b"section id,recording id,timestamp [ns]\r\nsec,rec," + str(T0).encode() + b"\r\n")
    csv = core.load_timestamp_csv(p)
    assert csv.lines[0].endswith("\r\n")
    out = tmp_path / "out.csv"
    core.trim_csv(csv, T0, T0 + 1, out)
    assert out.read_bytes() == p.read_bytes()  # byte-identical round trip


# ---------------------------------------------------------------------------
# sync check (the observed relationship + mismatch cases)
# ---------------------------------------------------------------------------

def test_sync_ok_irregular_vfr():
    s = core.check_sync(VIDEO_PTS, WORLD_TS)
    assert s.ok
    assert s.n_video_frames == s.n_world_rows == 8
    assert abs(s.offset_s) < 1e-9 and s.spread_s < 1e-9


def test_sync_blocks_on_count_mismatch():
    # the exact failure mode the spec warns about: container/world disagree
    s = core.check_sync(VIDEO_PTS[:-1], WORLD_TS)
    assert not s.ok
    assert "mismatch" in s.message and "blocked" in s.message


def test_sync_allows_constant_offset():
    # a re-loaded trimmed output: all pts shifted by a constant ~17 ms
    shifted = [p + 0.0167 for p in VIDEO_PTS]
    s = core.check_sync(shifted, WORLD_TS)
    assert s.ok
    assert abs(s.offset_s - 0.0167) < 1e-6


def test_sync_blocks_on_drift():
    # non-constant deviation (e.g. wrong fps assumption) must block.
    # 5% drift over the 253 ms synthetic clip -> 12.6 ms spread > 5 ms tol.
    # (Scaled to the real 215 s fixture, even 0.01% drift would exceed it.)
    drifted = [p * 1.05 for p in VIDEO_PTS]
    s = core.check_sync(drifted, WORLD_TS)
    assert not s.ok and "spread" in s.message


def test_sync_blocks_on_unsorted_world():
    ts = list(WORLD_TS)
    ts[3], ts[4] = ts[4], ts[3]
    s = core.check_sync(VIDEO_PTS, ts)
    assert not s.ok


def test_sync_fixture_numbers_from_real_recording():
    """Regression-pin the investigated real-recording relationship:
    6433 decoded frames == 6433 world rows, pts == rel world exactly."""
    n = 6433
    world = [T0 + int(i * 33_344_444) for i in range(n)]
    pts = [(w - T0) / NS for w in world]
    s = core.check_sync(pts, world)
    assert s.ok and s.n_video_frames == n


# ---------------------------------------------------------------------------
# trim plan / interval semantics
# ---------------------------------------------------------------------------

def test_plan_interval_is_halfopen_including_both_endpoint_frames():
    p = core.plan_trim(WORLD_TS, VIDEO_PTS, VIDEO_DUR, 2, 4, REC_END)
    assert p.abs_start_ns == WORLD_TS[2]          # start of first included frame
    assert p.abs_end_ns == WORLD_TS[5]            # start of frame AFTER last included
    assert p.n_frames_selected == 3
    assert p.rel_start_s == VIDEO_PTS[2] and p.rel_end_s == VIDEO_PTS[4]


def test_plan_last_frame_uses_recording_end():
    p = core.plan_trim(WORLD_TS, VIDEO_PTS, VIDEO_DUR, 6, 7, REC_END)
    assert p.abs_end_ns == REC_END
    assert p.cut_end_s >= VIDEO_DUR               # keeps the full tail


def test_plan_cut_boundaries_are_midpoints():
    p = core.plan_trim(WORLD_TS, VIDEO_PTS, VIDEO_DUR, 2, 4, REC_END)
    assert p.cut_start_s == pytest.approx((VIDEO_PTS[1] + VIDEO_PTS[2]) / 2)
    assert p.cut_end_s == pytest.approx((VIDEO_PTS[4] + VIDEO_PTS[5]) / 2)
    # midpoints exclude neighbours and include endpoints
    assert VIDEO_PTS[1] < p.cut_start_s < VIDEO_PTS[2]
    assert VIDEO_PTS[4] < p.cut_end_s < VIDEO_PTS[5]


def test_plan_start_zero_cuts_from_zero():
    p = core.plan_trim(WORLD_TS, VIDEO_PTS, VIDEO_DUR, 0, 3, REC_END)
    assert p.cut_start_s == 0.0


def test_plan_single_frame_selection():
    p = core.plan_trim(WORLD_TS, VIDEO_PTS, VIDEO_DUR, 5, 5, REC_END)
    assert p.n_frames_selected == 1
    assert p.abs_start_ns == WORLD_TS[5] and p.abs_end_ns == WORLD_TS[6]


def test_plan_rejects_bad_selection():
    with pytest.raises(ValueError):
        core.plan_trim(WORLD_TS, VIDEO_PTS, VIDEO_DUR, 5, 3, REC_END)
    with pytest.raises(ValueError):
        core.plan_trim(WORLD_TS, VIDEO_PTS, VIDEO_DUR, 0, 8, REC_END)
    with pytest.raises(ValueError):
        core.plan_trim(WORLD_TS, VIDEO_PTS, VIDEO_DUR, -1, 3, REC_END)


# ---------------------------------------------------------------------------
# gaze trimming: exact boundaries, missing head, empty intervals
# ---------------------------------------------------------------------------

def test_gaze_exact_boundary_inclusion_exclusion(tmp_path):
    start, end = WORLD_TS[2], WORLD_TS[5]
    gaze_ts = [start - 1, start, start + 1, end - 1, end, end + 1]
    csv = core.load_timestamp_csv(write_gaze_csv(tmp_path / "g.csv", gaze_ts))
    out = tmp_path / "out.csv"
    n = core.trim_csv(csv, start, end, out)
    kept = core.load_timestamp_csv(out)
    # half-open: start included, end excluded
    assert n == 3
    assert kept.ts_ns == [start, start + 1, end - 1]


def test_gaze_missing_at_beginning(tmp_path):
    # gaze starts 2.37 s after frame 0 (as in the real fixture):
    # selecting from frame 0 must simply retain nothing before first gaze
    gaze_ts = [t for t in (WORLD_TS[4] + 1_000, WORLD_TS[5], WORLD_TS[6])]
    csv = core.load_timestamp_csv(write_gaze_csv(tmp_path / "g.csv", gaze_ts))
    n = core.trim_csv(csv, WORLD_TS[0], WORLD_TS[6], tmp_path / "out.csv")
    assert n == 2  # the two samples before WORLD_TS[6]; no phantom rows


def test_gaze_empty_interval_allowed(tmp_path):
    gaze_ts = [WORLD_TS[6], WORLD_TS[7]]  # all gaze late in the recording
    csv = core.load_timestamp_csv(write_gaze_csv(tmp_path / "g.csv", gaze_ts))
    out = tmp_path / "out.csv"
    n = core.trim_csv(csv, WORLD_TS[0], WORLD_TS[2], out)
    assert n == 0
    kept = core.load_timestamp_csv(out)
    assert kept.n_rows == 0 and kept.header == csv.header


def test_count_in_interval_matches_trim(tmp_path):
    gaze_ts = [WORLD_TS[0] + i * 5_000_000 for i in range(50)]
    csv = core.load_timestamp_csv(write_gaze_csv(tmp_path / "g.csv", gaze_ts))
    for s, e in [(0, 7), (2, 4), (5, 5), (0, 0)]:
        p = core.plan_trim(WORLD_TS, VIDEO_PTS, VIDEO_DUR, s, e, REC_END)
        n = core.trim_csv(csv, p.abs_start_ns, p.abs_end_ns, tmp_path / "o.csv")
        assert n == core.count_in_interval(csv, p.abs_start_ns, p.abs_end_ns)
        out = core.load_timestamp_csv(tmp_path / "o.csv")
        assert core.validate_within(out.ts_ns, p.abs_start_ns, p.abs_end_ns)


def test_world_trim_rows_equal_selected_frames(tmp_path, world_csv):
    p = core.plan_trim(WORLD_TS, VIDEO_PTS, VIDEO_DUR, 2, 5, REC_END)
    out = tmp_path / "w.csv"
    n = core.trim_csv(world_csv, p.abs_start_ns, p.abs_end_ns, out)
    kept = core.load_timestamp_csv(out)
    assert n == p.n_frames_selected == 4
    assert kept.ts_ns == WORLD_TS[2:6]
    assert kept.lines == world_csv.lines[2:6]     # verbatim rows


def test_unsorted_gaze_falls_back_to_scan(tmp_path):
    ts = [WORLD_TS[3], WORLD_TS[2], WORLD_TS[4]]  # unsorted but contiguous inside
    csv = core.load_timestamp_csv(write_gaze_csv(tmp_path / "g.csv", ts))
    assert not csv.sorted_non_decreasing
    n = core.count_in_interval(csv, WORLD_TS[2], WORLD_TS[5])
    assert n == 3


# ---------------------------------------------------------------------------
# gaze -> frame mapping (reuse target for gaze_fusion)
# ---------------------------------------------------------------------------

def test_frame_index_for_ns_floor_semantics():
    assert core.frame_index_for_ns(WORLD_TS, WORLD_TS[0]) == 0
    assert core.frame_index_for_ns(WORLD_TS, WORLD_TS[3]) == 3
    assert core.frame_index_for_ns(WORLD_TS, WORLD_TS[3] + 1) == 3
    assert core.frame_index_for_ns(WORLD_TS, WORLD_TS[4] - 1) == 3
    assert core.frame_index_for_ns(WORLD_TS, WORLD_TS[0] - 10**9) == 0    # clip head
    assert core.frame_index_for_ns(WORLD_TS, WORLD_TS[-1] + 10**9) == 7   # clip tail


# ---------------------------------------------------------------------------
# recording end
# ---------------------------------------------------------------------------

def test_recording_end_prefers_info(tmp_path):
    info = core.RecordingInfo(start_time_ns=WORLD_TS[0], duration_ns=REC_END - WORLD_TS[0])
    assert core.recording_end_ns(info, WORLD_TS) == REC_END


def test_recording_end_fallback_median(tmp_path):
    end = core.recording_end_ns(core.RecordingInfo(), WORLD_TS)
    assert end > WORLD_TS[-1]
    # median interval of the synthetic series is 33 ms
    assert end - WORLD_TS[-1] == pytest.approx(33_000_000, abs=2_000_000)


def test_recording_end_ignores_bogus_info():
    # info end before the last frame must be ignored
    info = core.RecordingInfo(start_time_ns=WORLD_TS[0], duration_ns=1)
    assert core.recording_end_ns(info, WORLD_TS) > WORLD_TS[-1]
