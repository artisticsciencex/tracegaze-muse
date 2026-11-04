"""Presentation-only display timeline for the demonstration video.

TWO LAYERS, never mixed:

1. ANALYTICAL (authoritative) — `gaze_samples.csv` / `.parquet`, the exact
   23-column contract. Raw timestamps, raw coordinates, raw
   `primary_object_id`. Nothing here is interpolated, fabricated or
   replaced, and nothing in this module is ever written back into it.

2. PRESENTATION (this module) — one display position per VIDEO FRAME so
   the demo does not blink when the gaze rate is lower than the frame
   rate (the real Aria clip: 20 Hz video, 10 Hz gaze → every second frame
   had no sample and therefore no cursor). Linear interpolation between
   bracketing valid samples is explicitly permitted FOR DISPLAY ONLY and
   is disclosed on the video and in the manifest. Presentation hits never
   contribute to dwell time.
"""

from __future__ import annotations

import math
from bisect import bisect_left

METHOD_EXACT = "exact"
METHOD_INTERP = "linear_interpolation"
METHOD_HOLD = "nearest_hold"
METHOD_NONE = "no_support"

MIN_GAP_MS = 50.0
MAX_GAP_MS = 250.0
GAP_FACTOR = 3.0

PRESENTATION_COLUMNS = [
    "frame_index", "frame_timestamp_ns", "display_gaze_x_px",
    "display_gaze_y_px", "presentation_method", "left_gaze_sample_index",
    "right_gaze_sample_index", "interpolation_alpha",
    "presentation_primary_object_id", "presentation_primary_object_name",
    "presentation_cursor_visible",
]

SIDECAR_NOTE = (
    "This table reproduces gaze_demo.mp4 presentation and contains "
    "interpolated visualization values. It is not an analytical gaze table."
)


def median(values: list) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return float(s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)


def gap_policy(sample_ts_ns: list) -> dict:
    """Adaptive interpolation/support limit from the observed gaze rate.

        max_gap = clamp(3 * median_interval, 50 ms, 250 ms)
    """
    gaps = [b - a for a, b in zip(sample_ts_ns, sample_ts_ns[1:]) if b > a]
    med_ns = median(gaps)
    raw_ns = GAP_FACTOR * med_ns
    lo, hi = MIN_GAP_MS * 1e6, MAX_GAP_MS * 1e6
    max_gap_ns = int(min(max(raw_ns, lo), hi)) if med_ns else int(hi)
    return {"median_interval_ns": int(med_ns),
            "median_interval_ms": med_ns / 1e6,
            "max_interpolation_gap_ns": max_gap_ns,
            "max_interpolation_gap_ms": max_gap_ns / 1e6,
            "rule": f"clamp({GAP_FACTOR:g} x median interval, "
                    f"{MIN_GAP_MS:g} ms, {MAX_GAP_MS:g} ms)",
            "n_intervals": len(gaps)}


def usable_samples(fused) -> list:
    """Raw rows eligible to SUPPORT a display position: assigned to a
    frame, valid gaze, coordinates inside the image. Invalid, outside or
    unassigned rows break interpolation — they are never crossed."""
    out = []
    for i in range(len(fused)):
        if fused["assignment"].iat[i] != "assigned":
            continue
        if not bool(fused["gaze_valid"].iat[i]):
            continue
        if not bool(fused["coords_in_frame"].iat[i]):
            continue
        x = float(fused["gaze_x_px"].iat[i])
        y = float(fused["gaze_y_px"].iat[i])
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        fi = fused["frame_index"].iat[i]
        out.append({"row": i, "ts": int(fused["gaze_ts_ns"].iat[i]),
                    "x": x, "y": y,
                    "frame": None if fi is None or fi != fi else int(fi)})
    out.sort(key=lambda s: s["ts"])
    return out


def build_timeline(fused, frame_ts_ns: list, *, masks_by_frame=None,
                   mask_hit=None) -> tuple:
    """One display row per video frame. Returns (rows, policy).

    `mask_hit(frame_index, x, y)` -> ordered hit list for the presentation
    highlight (the SAME deterministic ordering the analytical layer uses),
    evaluated against that exact frame's masks.
    """
    samples = usable_samples(fused)
    ts = [s["ts"] for s in samples]
    # the sample each frame OWNS: smallest |offset| to the frame timestamp,
    # ties resolved to the earlier timestamp (same rule as the analytical
    # per-frame selection)
    by_frame = {}
    for s in samples:
        if s["frame"] is None or not (0 <= s["frame"] < len(frame_ts_ns)):
            continue
        cur = by_frame.get(s["frame"])
        d = abs(s["ts"] - int(frame_ts_ns[s["frame"]]))
        if cur is None or (d, s["ts"]) < (cur[1], cur[0]["ts"]):
            by_frame[s["frame"]] = (s, d)
    by_frame = {k: v[0] for k, v in by_frame.items()}
    policy = gap_policy(ts)
    max_gap = policy["max_interpolation_gap_ns"]
    hold = max_gap // (2 * int(GAP_FACTOR)) if policy["median_interval_ns"] \
        else 0
    if policy["median_interval_ns"]:
        hold = policy["median_interval_ns"] // 2      # half a normal interval
    rows = []
    for fi, T in enumerate(frame_ts_ns):
        T = int(T)
        rec = {"frame_index": fi, "frame_timestamp_ns": T,
               "display_gaze_x_px": None, "display_gaze_y_px": None,
               "presentation_method": METHOD_NONE,
               "left_gaze_sample_index": None,
               "right_gaze_sample_index": None,
               "interpolation_alpha": None,
               "presentation_primary_object_id": None,
               "presentation_primary_object_name": None,
               "presentation_cursor_visible": False}
        if samples:
            # EXACT = a raw sample the analytical layer assigned to THIS
            # frame (the verified per-frame rule: smallest |offset|, ties
            # to the earlier timestamp). Interpolation only ever fills
            # frames that own no sample at all.
            own = by_frame.get(fi)
            j = bisect_left(ts, T)
            if own is not None:
                s = own
                rec.update(display_gaze_x_px=s["x"], display_gaze_y_px=s["y"],
                           presentation_method=METHOD_EXACT,
                           left_gaze_sample_index=s["row"],
                           right_gaze_sample_index=s["row"],
                           interpolation_alpha=0.0,
                           presentation_cursor_visible=True)
            else:
                left = samples[j - 1] if j - 1 >= 0 else None
                right = samples[j] if j < len(samples) else None
                if left and right and (right["ts"] - left["ts"]) <= max_gap:
                    span = right["ts"] - left["ts"]
                    alpha = (T - left["ts"]) / span if span else 0.0
                    rec.update(
                        display_gaze_x_px=left["x"] + alpha
                        * (right["x"] - left["x"]),
                        display_gaze_y_px=left["y"] + alpha
                        * (right["y"] - left["y"]),
                        presentation_method=METHOD_INTERP,
                        left_gaze_sample_index=left["row"],
                        right_gaze_sample_index=right["row"],
                        interpolation_alpha=alpha,
                        presentation_cursor_visible=True)
                else:
                    # boundary: hold the nearest sample for at most half a
                    # normal interval; never bridge a long gap
                    cand = None
                    if left is None and right is not None:
                        cand = ("right", right, right["ts"] - T)
                    elif right is None and left is not None:
                        cand = ("left", left, T - left["ts"])
                    if cand and 0 <= cand[2] <= hold:
                        s = cand[1]
                        rec.update(display_gaze_x_px=s["x"],
                                   display_gaze_y_px=s["y"],
                                   presentation_method=METHOD_HOLD,
                                   left_gaze_sample_index=s["row"],
                                   right_gaze_sample_index=s["row"],
                                   interpolation_alpha=None,
                                   presentation_cursor_visible=True)
        if rec["presentation_cursor_visible"] and mask_hit is not None:
            hits = mask_hit(fi, rec["display_gaze_x_px"],
                            rec["display_gaze_y_px"])
            if hits:
                rec["presentation_primary_object_id"] = hits[0]["object_id"]
                rec["presentation_primary_object_name"] = \
                    hits[0]["object_name"]
        rows.append(rec)
    policy["nearest_hold_limit_ns"] = int(hold)
    policy["nearest_hold_limit_ms"] = hold / 1e6
    return rows, policy


def method_counts(rows: list) -> dict:
    out = {METHOD_EXACT: 0, METHOD_INTERP: 0, METHOD_HOLD: 0,
           METHOD_NONE: 0}
    for r in rows:
        out[r["presentation_method"]] = out.get(
            r["presentation_method"], 0) + 1
    out["frames_with_cursor"] = sum(
        1 for r in rows if r["presentation_cursor_visible"])
    out["frames_total"] = len(rows)
    return out


def write_presentation_csv(rows: list, path) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# {SIDECAR_NOTE}"])
        w.writerow(PRESENTATION_COLUMNS)
        for r in rows:
            w.writerow([
                r["frame_index"], r["frame_timestamp_ns"],
                "" if r["display_gaze_x_px"] is None
                else f"{r['display_gaze_x_px']:.6f}",
                "" if r["display_gaze_y_px"] is None
                else f"{r['display_gaze_y_px']:.6f}",
                r["presentation_method"],
                "" if r["left_gaze_sample_index"] is None
                else r["left_gaze_sample_index"],
                "" if r["right_gaze_sample_index"] is None
                else r["right_gaze_sample_index"],
                "" if r["interpolation_alpha"] is None
                else f"{r['interpolation_alpha']:.6f}",
                r["presentation_primary_object_id"] or "",
                r["presentation_primary_object_name"] or "",
                "true" if r["presentation_cursor_visible"] else "false",
            ])


def read_presentation_csv(path) -> list:
    """Reader that skips the leading disclosure comment line."""
    import csv
    with open(path, newline="") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return list(csv.DictReader(lines))
