"""Episode / transition / summary construction.

An EPISODE is a maximal run of consecutive eligible samples that share a
target and whose support intervals touch or overlap. Nothing is smoothed,
merged across a break, or relabelled by duration: a 40 ms episode stays a
40 ms episode.

A TRANSITION connects two consecutive supported episodes. Its
`unsupported_gap_ns` is the time between them that no sample supported —
zero when they abut, positive when a long gap or a breaking sample lay in
between.
"""

from __future__ import annotations

from . import NO_TRACKED_OBJECT, NO_TRACKED_OBJECT_NAME
from .policy import STATUS_ON_OBJECT

EPISODE_COLUMNS = [
    "episode_id", "episode_index", "target_id", "target_name",
    "is_tracked_object", "start_timestamp_ns", "end_timestamp_ns",
    "duration_ns", "duration_s", "first_gaze_sample_index",
    "last_gaze_sample_index", "first_frame_index", "last_frame_index",
    "sample_count", "mean_frame_offset_ms", "median_frame_offset_ms",
    "mean_track_score", "gaze_sample_indices", "frame_indices",
]

TRANSITION_COLUMNS = [
    "transition_id", "transition_index", "from_episode_id", "to_episode_id",
    "from_object_id", "to_object_id", "from_object_name", "to_object_name",
    "transition_timestamp_ns", "unsupported_gap_ns", "unsupported_gap_s",
    "transition_type", "from_episode_duration_ns", "to_episode_duration_ns",
]

SUMMARY_COLUMNS = [
    "object_id", "object_name", "object_color", "is_tracked_object",
    "total_attention_ns", "total_attention_s", "share_of_supported_time",
    "episode_count", "revisit_count", "sample_count",
    "first_attention_timestamp_ns", "last_attention_timestamp_ns",
    "longest_episode_ns", "mean_episode_ns", "display_rank",
]

TYPE_OBJ_OBJ = "object_to_object"
TYPE_OBJ_NOHIT = "object_to_no_hit"
TYPE_NOHIT_OBJ = "no_hit_to_object"
TYPE_NOHIT_NOHIT = "no_hit_to_no_hit"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def build_episodes(supported: list) -> list:
    """Maximal same-target runs with touching support. Deterministic."""
    episodes = []
    cur = None
    for s in supported:
        touching = (cur is not None
                    and s["target_id"] == cur["target_id"]
                    and s["support_start_ns"] <= cur["end_timestamp_ns"]
                    # a breaking sample (blink / outside frame / unassigned)
                    # always starts a new episode, however short it was
                    and not s.get("break_before"))
        if not touching:
            if cur is not None:
                episodes.append(cur)
            cur = {
                "target_id": s["target_id"],
                "target_name": s["target_name"],
                "start_timestamp_ns": s["support_start_ns"],
                "end_timestamp_ns": s["support_end_ns"],
                "duration_ns": s["support_end_ns"] - s["support_start_ns"],
                "first_gaze_sample_index": s["gaze_sample_index"],
                "last_gaze_sample_index": s["gaze_sample_index"],
                "first_frame_index": s["frame_index"],
                "last_frame_index": s["frame_index"],
                "gaze_sample_indices": [s["gaze_sample_index"]],
                "frame_indices": [s["frame_index"]],
                "_offsets": [s["frame_offset_ms"]],
                "_scores": [s["track_score"]],
                "_status": s["status"],
            }
            continue
        # extend: support intervals are non-overlapping and ordered, so
        # the added duration is exactly this sample's own support
        cur["duration_ns"] += s["support_end_ns"] - s["support_start_ns"]
        cur["end_timestamp_ns"] = s["support_end_ns"]
        cur["last_gaze_sample_index"] = s["gaze_sample_index"]
        cur["last_frame_index"] = s["frame_index"]
        cur["gaze_sample_indices"].append(s["gaze_sample_index"])
        cur["frame_indices"].append(s["frame_index"])
        cur["_offsets"].append(s["frame_offset_ms"])
        cur["_scores"].append(s["track_score"])
    if cur is not None:
        episodes.append(cur)

    out = []
    for i, e in enumerate(episodes):
        tracked = e["target_id"] != NO_TRACKED_OBJECT
        out.append({
            "episode_id": f"ep_{i + 1:05d}",
            "episode_index": i,
            "target_id": e["target_id"],
            "target_name": (e["target_name"] if tracked
                            else NO_TRACKED_OBJECT_NAME),
            "is_tracked_object": tracked,
            "start_timestamp_ns": e["start_timestamp_ns"],
            "end_timestamp_ns": e["end_timestamp_ns"],
            "duration_ns": e["duration_ns"],
            "duration_s": e["duration_ns"] / 1e9,
            "first_gaze_sample_index": e["first_gaze_sample_index"],
            "last_gaze_sample_index": e["last_gaze_sample_index"],
            "first_frame_index": e["first_frame_index"],
            "last_frame_index": e["last_frame_index"],
            "sample_count": len(e["gaze_sample_indices"]),
            "mean_frame_offset_ms": _mean(e["_offsets"]),
            "median_frame_offset_ms": _median(e["_offsets"]),
            "mean_track_score": (_mean(e["_scores"]) if tracked else None),
            "gaze_sample_indices": list(e["gaze_sample_indices"]),
            "frame_indices": list(e["frame_indices"]),
        })
    return out


def transition_type(from_tracked: bool, to_tracked: bool) -> str:
    if from_tracked and to_tracked:
        return TYPE_OBJ_OBJ
    if from_tracked and not to_tracked:
        return TYPE_OBJ_NOHIT
    if not from_tracked and to_tracked:
        return TYPE_NOHIT_OBJ
    return TYPE_NOHIT_NOHIT


def build_transitions(episodes: list) -> list:
    """One transition per consecutive episode pair."""
    out = []
    for i, (a, b) in enumerate(zip(episodes, episodes[1:])):
        gap = b["start_timestamp_ns"] - a["end_timestamp_ns"]
        out.append({
            "transition_id": f"tr_{i + 1:05d}",
            "transition_index": i,
            "from_episode_id": a["episode_id"],
            "to_episode_id": b["episode_id"],
            "from_object_id": a["target_id"],
            "to_object_id": b["target_id"],
            "from_object_name": a["target_name"],
            "to_object_name": b["target_name"],
            # the moment attention left the previous episode
            "transition_timestamp_ns": a["end_timestamp_ns"],
            "unsupported_gap_ns": max(0, gap),
            "unsupported_gap_s": max(0, gap) / 1e9,
            "transition_type": transition_type(a["is_tracked_object"],
                                               b["is_tracked_object"]),
            "from_episode_duration_ns": a["duration_ns"],
            "to_episode_duration_ns": b["duration_ns"],
        })
    return out


def build_summary(episodes: list, objects: dict) -> list:
    """Per-target totals. `revisit_count` counts RETURNS after leaving —
    an object attended in one uninterrupted episode has 0 revisits."""
    agg = {}
    for e in episodes:
        a = agg.setdefault(e["target_id"], {
            "total_ns": 0, "episodes": 0, "samples": 0,
            "first_ns": e["start_timestamp_ns"],
            "last_ns": e["end_timestamp_ns"], "longest_ns": 0,
            "name": e["target_name"],
            "tracked": e["is_tracked_object"]})
        a["total_ns"] += e["duration_ns"]
        a["episodes"] += 1
        a["samples"] += e["sample_count"]
        a["first_ns"] = min(a["first_ns"], e["start_timestamp_ns"])
        a["last_ns"] = max(a["last_ns"], e["end_timestamp_ns"])
        a["longest_ns"] = max(a["longest_ns"], e["duration_ns"])

    # every ACCEPTED object appears, including ones never attended: a
    # zero-dwell object is a finding, not an absence
    for oid, meta in (objects or {}).items():
        if oid in agg or meta.get("review_status") not in (None, "accepted"):
            continue
        agg[oid] = {"total_ns": 0, "episodes": 0, "samples": 0,
                    "first_ns": None, "last_ns": None, "longest_ns": 0,
                    "name": meta.get("name") or oid, "tracked": True}

    supported_total = sum(a["total_ns"] for a in agg.values()) or 1
    order = sorted(agg, key=lambda k: (-agg[k]["total_ns"],
                                       k == NO_TRACKED_OBJECT, k))
    rows = []
    for rank, oid in enumerate(order, start=1):
        a = agg[oid]
        meta = objects.get(oid, {})
        rows.append({
            "object_id": oid,
            "object_name": (NO_TRACKED_OBJECT_NAME if not a["tracked"]
                            else (meta.get("name") or a["name"] or oid)),
            "object_color": (meta.get("color") if a["tracked"] else None),
            "is_tracked_object": a["tracked"],
            "total_attention_ns": a["total_ns"],
            "total_attention_s": a["total_ns"] / 1e9,
            "share_of_supported_time": a["total_ns"] / supported_total,
            "episode_count": a["episodes"],
            # returns after leaving, NOT the sample count
            "revisit_count": max(0, a["episodes"] - 1),
            "sample_count": a["samples"],
            "first_attention_timestamp_ns": a["first_ns"],
            "last_attention_timestamp_ns": a["last_ns"],
            "longest_episode_ns": a["longest_ns"],
            "mean_episode_ns": (a["total_ns"] // a["episodes"]
                                if a["episodes"] else 0),
            "display_rank": rank,
        })
    return rows


def check_no_double_counting(episodes: list) -> tuple:
    """Episodes must tile time without overlapping."""
    problems = []
    for a, b in zip(episodes, episodes[1:]):
        if b["start_timestamp_ns"] < a["end_timestamp_ns"]:
            problems.append(
                f"{a['episode_id']} ends {a['end_timestamp_ns']} but "
                f"{b['episode_id']} starts {b['start_timestamp_ns']}")
    for e in episodes:
        span = e["end_timestamp_ns"] - e["start_timestamp_ns"]
        if e["duration_ns"] > span:
            problems.append(
                f"{e['episode_id']} duration {e['duration_ns']} exceeds its "
                f"span {span}")
    return (not problems), problems
