"""Eligibility and temporal-support policy (exact integer nanoseconds).

Kept separate from the trace construction so the rules are readable,
testable and quotable in the paper without reading the algorithm.
"""

from __future__ import annotations

from typing import Optional

from . import NO_TRACKED_OBJECT

# fusion's authoritative per-sample status vocabulary
STATUS_ON_OBJECT = "on_object"
STATUS_BACKGROUND = "background"
STATUS_INVALID = "invalid_gaze"
STATUS_OUTSIDE = "outside_frame"
STATUS_NO_FRAME = "no_matching_frame"

#: statuses that can support attention. `background` is a VALID, in-frame
#: sample that hit no accepted mask — it supports attention on the
#: explicit no-tracked-object target, not on "the background".
SUPPORTING_STATUSES = (STATUS_ON_OBJECT, STATUS_BACKGROUND)

#: statuses that break support and therefore split episodes
BREAKING_STATUSES = (STATUS_INVALID, STATUS_OUTSIDE, STATUS_NO_FRAME)

GAP_FACTOR = 3.0
MIN_GAP_NS = 50_000_000            # 50 ms
MAX_GAP_NS = 250_000_000           # 250 ms


def is_missing(v) -> bool:
    """True for None, float NaN and pandas NA (nullable Int64/Float64).

    The `x != x` NaN idiom raises on pd.NA, so every null check in this
    package goes through here.
    """
    if v is None:
        return True
    try:
        import pandas as pd
        res = pd.isna(v)
    except Exception:                                    # noqa: BLE001
        return False
    return bool(res) if not hasattr(res, "__len__") else bool(all(res))


def median_int(values: list) -> int:
    """Exact integer median (lower mean for even counts — deterministic)."""
    if not values:
        return 0
    s = sorted(values)
    n = len(s)
    return int(s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) // 2)


def gap_policy(sample_ts_ns: list) -> dict:
    """max_gap_ns = clamp(3 x median positive valid interval, 50, 250 ms).

    The median is taken over POSITIVE intervals between consecutive
    ELIGIBLE samples, so a recording's own sampling rate sets the limit.
    """
    gaps = [b - a for a, b in zip(sample_ts_ns, sample_ts_ns[1:]) if b > a]
    med = median_int(gaps)
    raw = int(GAP_FACTOR * med)
    max_gap = min(max(raw, MIN_GAP_NS), MAX_GAP_NS) if med else MAX_GAP_NS
    return {
        "median_interval_ns": med,
        "median_interval_ms": med / 1e6,
        "raw_limit_ns": raw,
        "max_gap_ns": int(max_gap),
        "max_gap_ms": max_gap / 1e6,
        "half_cap_ns": int(max_gap) // 2,
        "n_intervals": len(gaps),
        "rule": (f"clamp({GAP_FACTOR:g} x median positive valid sampling "
                 f"interval, {MIN_GAP_NS / 1e6:g} ms, "
                 f"{MAX_GAP_NS / 1e6:g} ms)"),
    }


def eligible_rows(fused) -> list:
    """Rows that may support attention, in authoritative table order.

    Requires: assigned to a frame, gaze_valid, coordinates in frame (all
    implied by the status vocabulary, checked explicitly anyway) and a
    supporting fusion status.
    """
    out = []
    n = len(fused)
    has_valid = "gaze_valid" in fused.columns
    for i in range(n):
        status = fused["gaze_status"].iat[i]
        if status not in SUPPORTING_STATUSES:
            continue
        fi = fused["frame_index"].iat[i]
        if is_missing(fi):
            continue
        if has_valid and not bool(fused["gaze_valid"].iat[i]):
            continue
        ts = int(fused["timestamp_ns"].iat[i])
        pid = fused["primary_object_id"].iat[i]
        pname = fused["primary_object_name"].iat[i]
        if status == STATUS_ON_OBJECT:
            if is_missing(pid):
                # on_object without a primary id would be a fusion
                # inconsistency; refuse rather than invent a target
                raise ValueError(
                    f"row {i}: gaze_status=on_object without "
                    f"primary_object_id — inconsistent fusion artifact")
            target = str(pid)
            target_name = None if is_missing(pname) else str(pname)
        else:
            target, target_name = NO_TRACKED_OBJECT, None
        score = fused["primary_track_score"].iat[i] \
            if "primary_track_score" in fused.columns else None
        offset = fused["frame_offset_ms"].iat[i] \
            if "frame_offset_ms" in fused.columns else None
        out.append({
            "row": i,
            "gaze_sample_index": int(fused["gaze_sample_index"].iat[i]),
            "ts_ns": ts,
            "frame_index": int(fi),
            "status": status,
            "target_id": target,
            "target_name": target_name,
            "track_score": (None if is_missing(score) else float(score)),
            "frame_offset_ms": (None if is_missing(offset)
                                else float(offset)),
        })
    out.sort(key=lambda r: (r["ts_ns"], r["gaze_sample_index"]))
    return out


def breaking_rows(fused) -> list:
    """Timestamps of samples whose status BREAKS support.

    These rows carry no attention, but they are evidence that at that
    instant we do not know what was attended: they take part in the
    midpoint tiling so their own territory becomes UNSUPPORTED, and they
    split episodes even when the same target resumes immediately.
    """
    out = []
    for i in range(len(fused)):
        if fused["gaze_status"].iat[i] in BREAKING_STATUSES:
            out.append({"row": i, "ts_ns": int(fused["timestamp_ns"].iat[i]),
                        "status": fused["gaze_status"].iat[i],
                        "breaking": True})
    return out


def support_intervals(rows: list, interval_start_ns: int,
                      interval_end_ns: int, max_gap_ns: int,
                      breaks: Optional[list] = None) -> tuple:
    """Half-open, NON-OVERLAPPING support interval per eligible sample.

    Boundaries are midpoints to the neighbouring sample in the COMBINED
    timeline of eligible and breaking samples, each side capped at
    `max_gap_ns // 2` and clipped to the validated fusion interval.

    Both boundaries use the SAME floor midpoint, so neighbouring
    intervals meet exactly for either parity of the inter-sample gap (an
    asymmetric floor/ceil pair overlaps by 1 ns on odd gaps).

    A gap longer than `max_gap_ns` leaves its middle unsupported, and a
    breaking sample's own midpoint territory is likewise unsupported —
    `break_before` marks the eligible sample that follows one.

    Returns (supported, dropped) where `dropped` records any eligible
    sample whose support collapsed to zero length (never silent).
    """
    cap = int(max_gap_ns) // 2
    combined = sorted(list(rows) + list(breaks or []),
                      key=lambda r: (r["ts_ns"], r.get("row", 0)))
    out, dropped = [], []
    for k, r in enumerate(combined):
        if r.get("breaking"):
            continue                       # evidence of a gap, not of gaze
        t = r["ts_ns"]
        prev_t = combined[k - 1]["ts_ns"] if k > 0 else None
        next_t = combined[k + 1]["ts_ns"] if k + 1 < len(combined) else None
        start = t - cap if prev_t is None else max(t - cap, (prev_t + t) // 2)
        end = t + cap if next_t is None else min(t + cap, (t + next_t) // 2)
        start = max(start, int(interval_start_ns))
        end = min(end, int(interval_end_ns))
        if end <= start:
            dropped.append({**r, "reason": "support collapsed to zero "
                                           "length after clipping"})
            continue
        out.append({**r, "support_start_ns": start, "support_end_ns": end,
                    "break_before": bool(k > 0
                                         and combined[k - 1].get("breaking"))})
    return out, dropped
