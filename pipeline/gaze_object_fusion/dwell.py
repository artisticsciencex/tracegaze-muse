"""Analytical accumulated-gaze (dwell) time — RAW EVIDENCE ONLY.

Durations shown by the demo's progress bars come from authoritative raw
gaze samples and the raw `primary_object_id`. Presentation-interpolated
positions (see `presentation.py`) NEVER contribute a millisecond here.

Support-interval rule, per valid, assigned, in-frame raw sample i:

    support_i = [ max(t_i - cap, mid(t_{i-1}, t_i)) ,
                  min(t_i + cap, mid(t_i, t_{i+1})) )   clipped to the
                  validated trim interval

`mid` is the exact integer midpoint to the neighbouring VALID sample and
`cap = max_gap / 2` (the same adaptive limit the presentation layer uses),
so a long stretch of missing data is never silently treated as continuous
gaze: its unsupported middle is simply left unattributed.

Attribution: `on_object` → its raw `primary_object_id`; valid
`background` → the explicit no-hit bucket; invalid / outside-frame /
no-matching-frame rows → nothing. Overlapping masks never double-count:
only the deterministic primary object receives the interval.
"""

from __future__ import annotations

NO_HIT_ID = "__no_tracked_object__"
NO_HIT_NAME = "No tracked-object hit"
NO_HIT_COLOR = "#8a94a3"
TOP_N = 8
OTHER_GROUP = "Other tracked objects"

SUMMARY_COLUMNS = [
    "object_id", "object_name", "object_color", "primary_gaze_time_s",
    "share_of_valid_supported_gaze", "primary_sample_count",
    "first_primary_hit_timestamp_ns", "last_primary_hit_timestamp_ns",
    "display_rank", "included_in_top_eight", "aggregation_group",
]


def support_intervals(fused, interval_start_ns: int, interval_end_ns: int,
                      max_gap_ns: int) -> list:
    """[{start_ns, end_ns, target_id, ts}] for every attributable sample."""
    rows = []
    for i in range(len(fused)):
        if fused["assignment"].iat[i] != "assigned":
            continue
        if not bool(fused["gaze_valid"].iat[i]):
            continue
        if not bool(fused["coords_in_frame"].iat[i]):
            continue
        pid = fused["primary_object_id"].iat[i]
        target = (NO_HIT_ID if (pid is None or pid != pid)      # NaN-safe
                  else str(pid))
        rows.append({"ts": int(fused["gaze_ts_ns"].iat[i]),
                     "target": target})
    rows.sort(key=lambda r: r["ts"])
    cap = max_gap_ns // 2
    out = []
    for k, r in enumerate(rows):
        t = r["ts"]
        prev_t = rows[k - 1]["ts"] if k > 0 else None
        next_t = rows[k + 1]["ts"] if k + 1 < len(rows) else None
        start = t - cap if prev_t is None else max(t - cap,
                                                   (prev_t + t) // 2)
        end = t + cap if next_t is None else min(t + cap,
                                                 (t + next_t + 1) // 2)
        start = max(start, interval_start_ns)
        end = min(end, interval_end_ns)
        if end > start:
            out.append({"start_ns": start, "end_ns": end,
                        "target_id": r["target"], "ts": t})
    return out


def accumulate(intervals: list, frame_ts_ns: list, interval_end_ns: int,
               objects: dict) -> dict:
    """Per-object totals and the per-frame cumulative curve.

    Cumulative values for frame N integrate support only up to the END
    BOUNDARY of frame N, so the last frame's totals equal the final
    summary exactly.
    """
    totals, counts, first_hit, last_hit = {}, {}, {}, {}
    for iv in intervals:
        tid = iv["target_id"]
        totals[tid] = totals.get(tid, 0) + (iv["end_ns"] - iv["start_ns"])
        counts[tid] = counts.get(tid, 0) + 1
        first_hit.setdefault(tid, iv["ts"])
        last_hit[tid] = iv["ts"]

    # frame end boundaries: next frame's timestamp, last = interval end
    bounds = [int(frame_ts_ns[i + 1]) for i in range(len(frame_ts_ns) - 1)]
    bounds.append(int(interval_end_ns))

    order = sorted(totals, key=lambda k: (-totals[k],
                                          k == NO_HIT_ID, k))
    ranked = [k for k in order if k != NO_HIT_ID]
    top = ranked[:TOP_N]
    rest = ranked[TOP_N:]

    # cumulative curve: coverage of every support interval clipped to each
    # frame's END boundary. Monotonic by construction, and the last entry
    # equals the final totals exactly.
    ivs = sorted(intervals, key=lambda x: x["start_ns"])
    cum = []
    settled = {}                 # intervals fully consumed by earlier bounds
    idx = 0
    open_ivs = []
    for b in bounds:
        while idx < len(ivs) and ivs[idx]["start_ns"] < b:
            open_ivs.append(ivs[idx])
            idx += 1
        snap = dict(settled)
        still_open = []
        for iv in open_ivs:
            covered = min(iv["end_ns"], b) - iv["start_ns"]
            if covered > 0:
                snap[iv["target_id"]] = snap.get(iv["target_id"], 0) + covered
            if iv["end_ns"] > b:
                still_open.append(iv)
            else:                       # finished: fold into the baseline
                settled[iv["target_id"]] = settled.get(iv["target_id"], 0) \
                    + max(0, iv["end_ns"] - iv["start_ns"])
        open_ivs = still_open
        cum.append(snap)

    valid_total = sum(totals.values())
    return {"totals_ns": totals, "counts": counts,
            "first_hit_ns": first_hit, "last_hit_ns": last_hit,
            "order": order, "top": top, "other": rest,
            "cumulative_ns_by_frame": cum,
            "valid_supported_ns": valid_total,
            "bar_scale_ns": max([totals[k] for k in ranked], default=0)}


def summary_rows(acc: dict, objects: dict) -> list:
    """`object_dwell_summary.csv` rows (one per accepted object + no-hit)."""
    out = []
    valid = acc["valid_supported_ns"] or 1
    for rank, oid in enumerate(acc["order"], start=1):
        meta = objects.get(oid, {})
        is_nohit = oid == NO_HIT_ID
        out.append({
            "object_id": oid,
            "object_name": (NO_HIT_NAME if is_nohit
                            else meta.get("name", oid)),
            "object_color": (NO_HIT_COLOR if is_nohit
                             else meta.get("color", "#ffffff")),
            "primary_gaze_time_s": acc["totals_ns"].get(oid, 0) / 1e9,
            "share_of_valid_supported_gaze":
                acc["totals_ns"].get(oid, 0) / valid,
            "primary_sample_count": acc["counts"].get(oid, 0),
            "first_primary_hit_timestamp_ns": acc["first_hit_ns"].get(oid),
            "last_primary_hit_timestamp_ns": acc["last_hit_ns"].get(oid),
            "display_rank": rank,
            "included_in_top_eight": (not is_nohit and oid in acc["top"]),
            "aggregation_group": (
                "no_hit" if is_nohit
                else ("top_eight" if oid in acc["top"] else OTHER_GROUP)),
        })
    return out


def write_summary_csv(rows: list, path) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({
                **r,
                "primary_gaze_time_s": f"{r['primary_gaze_time_s']:.6f}",
                "share_of_valid_supported_gaze":
                    f"{r['share_of_valid_supported_gaze']:.6f}",
                "first_primary_hit_timestamp_ns":
                    "" if r["first_primary_hit_timestamp_ns"] is None
                    else r["first_primary_hit_timestamp_ns"],
                "last_primary_hit_timestamp_ns":
                    "" if r["last_primary_hit_timestamp_ns"] is None
                    else r["last_primary_hit_timestamp_ns"],
                "included_in_top_eight":
                    "true" if r["included_in_top_eight"] else "false",
            })


def panel_rows(acc: dict, objects: dict) -> list:
    """Fixed bar order/scale for the video panel: top-8 by FINAL dwell,
    then an aggregated `Other tracked objects` row when needed."""
    rows = []
    for oid in acc["top"]:
        meta = objects.get(oid, {})
        rows.append({"key": oid, "name": meta.get("name", oid),
                     "object_id": oid,
                     "color": meta.get("color", "#ffffff"),
                     "final_ns": acc["totals_ns"].get(oid, 0),
                     "aggregate": None})
    if acc["other"]:
        rows.append({"key": OTHER_GROUP, "name": OTHER_GROUP,
                     "object_id": f"{len(acc['other'])} objects",
                     "color": "#6f7a88",
                     "final_ns": sum(acc["totals_ns"].get(o, 0)
                                     for o in acc["other"]),
                     "aggregate": list(acc["other"])})
    return rows
