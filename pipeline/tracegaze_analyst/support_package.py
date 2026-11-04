"""Bounded evidence package for retrospective everyday-activity analysis.

Recording + researcher question -> a self-contained package of
deterministic facts, candidate intervals, episodes, transitions and a few
timestamped scene frames, each linked back to the TraceGaze artifacts it
came from. Nothing here calls a model, and nothing re-decides a gaze
assignment: fusion's `primary_object_id` and attention_trace's episodes
are used as published.

Candidate intervals are chosen by fixed rules (longest tracked-object
episodes, densest switching windows, longest no-hit stretch) so that
"which moments deserve closer examination" starts from a reproducible
shortlist rather than from the model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .facts import build_fact_catalog, load_trace

SCHEMA_VERSION = "tracegaze-support-package-0.1"
NO_HIT = "__no_tracked_object__"

#: project convention (gaze_fusion.py --nearest-within-px 25), used here
#: only to flag boundary-near samples; it is not a measured accuracy bound
PROXIMITY_PX = 25
_CROP_PX = 64               # boundary search radius around the gaze pixel

PHASE_S = 10.0              # deterministic timeline bins for "how it unfolded"
SWITCH_WINDOW_S = 3.0
SWITCH_STEP_S = 0.5
SUSTAIN_PAD_S = 1.0
MIN_GAP_EPISODE_S = 0.5


def _r3(x: float) -> float:
    return round(float(x), 3)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ids(cell) -> list:
    return json.loads(cell) if isinstance(cell, str) else list(cell)


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

def load_sources(trace_dir: str | Path) -> dict:
    """Trace + its parent fusion, tracked objects and scene video.

    Each parent is located through the manifests and checked against the
    hashes the child recorded, so frames and masks provably come from the
    recording the trace was computed on.
    """
    trace = load_trace(trace_dir)
    tm = trace["manifest"]
    fusion_dir = Path(tm["parent_artifact"]["artifact_dir"])
    fm = json.loads((fusion_dir / "fusion_manifest.json").read_text())
    if fm["artifact_id"] != tm["parent_artifact"]["artifact_id"]:
        raise ValueError("fusion artifact id does not match the trace parent")
    samples_path = fusion_dir / fm["files"]["gaze_samples_parquet"]
    if _sha256(samples_path) != tm["parent_artifact"]["files_sha256"]["gaze_samples_parquet"]:
        raise ValueError("gaze_samples.parquet changed since the trace was built")
    tracked_dir = Path(fm["parent_artifact"]["artifact_dir"])
    objects_path = tracked_dir / "objects.parquet"
    if _sha256(objects_path) != fm["parent_artifact"]["objects_table_sha256"]:
        raise ValueError("objects.parquet changed since fusion")
    rec = fm["parent_recording"]
    scene = Path(rec["source_dir"]) / rec["video"]
    if _sha256(scene) != rec["hashes_sha256"]["video"]:
        raise ValueError("scene video changed since fusion")
    demo = fusion_dir / fm["files"]["demo"]
    trim_manifest = Path(rec["source_dir"]) / "trim_manifest.json"
    return {
        "trace": trace,
        "fusion_dir": fusion_dir,
        "fusion_manifest": fm,
        "samples_path": samples_path,
        "tracked_dir": tracked_dir,
        "objects_path": objects_path,
        "scene_video": scene,
        "demo_video": demo,
        "trim_manifest": (json.loads(trim_manifest.read_text())
                          if trim_manifest.is_file() else None),
    }


def _load_samples(path: Path):
    import pandas as pd
    return pd.read_parquet(path)


# --------------------------------------------------------------------------
# boundary proximity (deterministic; never changes an assignment)
# --------------------------------------------------------------------------

def _decode_rle(rle_json: str):
    import numpy as np
    import pycocotools.mask as mask_util
    r = json.loads(rle_json)
    return mask_util.decode({"size": r["size"],
                             "counts": r["counts"].encode("ascii")}
                            ).astype(np.uint8)


def _dist_to_zero(crop, cy: int, cx: int) -> float:
    """Euclidean distance from (cy, cx) to the nearest zero pixel of crop."""
    import cv2
    if crop.all():
        return float(_CROP_PX)
    return float(cv2.distanceTransform(crop, cv2.DIST_L2, 5)[cy, cx])


def boundary_proximity(samples, objects_path: Path, accepted_ids: list,
                       proximity_px: int = PROXIMITY_PX) -> dict:
    """Per eligible sample: alternative targets within `proximity_px`.

    For an on-object sample the alternatives are other accepted masks and
    the no-hit region (leaving every mask); for a no-hit sample they are
    the accepted masks. Distances are measured from the gaze pixel used by
    fusion (floor of the gaze coordinate) inside a +/-64 px window, so
    anything farther reports 64 ("at least").
    """
    import numpy as np
    import pandas as pd
    objs = pd.read_parquet(objects_path,
                           columns=["frame_index", "object_id", "mask_rle"])
    objs = objs[objs["object_id"].isin(set(accepted_ids))]
    by_frame = {int(f): sub for f, sub in objs.groupby("frame_index")}
    out = {}
    for row in samples.itertuples(index=False):
        if row.gaze_status not in ("on_object", "background"):
            continue
        f = int(row.frame_index)
        px, py = int(row.gaze_pixel_x), int(row.gaze_pixel_y)
        masks = {}
        for r in by_frame.get(f, pd.DataFrame()).itertuples(index=False):
            masks[r.object_id] = _decode_rle(r.mask_rle)
        if not masks:
            continue
        h, w = next(iter(masks.values())).shape
        y0, y1 = max(0, py - _CROP_PX), min(h, py + _CROP_PX + 1)
        x0, x1 = max(0, px - _CROP_PX), min(w, px + _CROP_PX + 1)
        cy, cx = py - y0, px - x0
        primary = row.primary_object_id if row.gaze_status == "on_object" else NO_HIT
        dists = {}
        union = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for oid, m in masks.items():
            crop = np.ascontiguousarray(m[y0:y1, x0:x1])
            union |= crop
            if oid == primary:
                continue
            # distance to the nearest pixel INSIDE this mask
            dists[oid] = 0.0 if crop[cy, cx] else _dist_to_zero(1 - crop, cy, cx)
        if primary != NO_HIT:
            # distance to the nearest pixel covered by no accepted mask
            dists[NO_HIT] = _dist_to_zero(union, cy, cx)
        near = sorted((d, oid) for oid, d in dists.items() if d <= proximity_px)
        nearest = min(dists.items(), key=lambda kv: kv[1]) if dists else (None, None)
        out[int(row.gaze_sample_index)] = {
            "assigned_target": primary,
            "alternatives_within_px": [oid for _, oid in near],
            "nearest_alternative": nearest[0],
            "nearest_alternative_px": (None if nearest[1] is None
                                       else round(nearest[1], 1)),
            "boundary_near": bool(near),
        }
    return out


# --------------------------------------------------------------------------
# candidate intervals (fixed rules)
# --------------------------------------------------------------------------

def _episodes(trace) -> list:
    eps = []
    for e in trace["episodes"]:
        eps.append({"episode_id": e["episode_id"], "target_id": e["target_id"],
                    "target_name": e["target_name"],
                    "start_ns": int(e["start_timestamp_ns"]),
                    "end_ns": int(e["end_timestamp_ns"]),
                    "sample_count": int(e["sample_count"]),
                    "samples": _ids(e["gaze_sample_indices"]),
                    "frames": _ids(e["frame_indices"])})
    eps.sort(key=lambda e: e["start_ns"])
    return eps


def select_intervals(trace) -> list[dict]:
    """Candidate moments by fixed rules, in priority order.

    1. the two densest switching windows (transitions per 3 s);
    2. the two longest tracked-object episodes, +/-1 s of context;
    3. the longest no-hit stretch (>= 0.5 s), +/-1 s of context.
    Intervals are unions of WHOLE episodes inside the window and never
    overlap; context padding shrinks rather than overlap an earlier pick.
    """
    t0, t1 = (int(x) for x in trace["manifest"]["source"]["validated_interval_ns"])
    eps = _episodes(trace)
    trans = [int(t["transition_timestamp_ns"]) for t in trace["transitions"]]
    chosen: list[dict] = []

    def add(kind, rule, s, e, anchor=None) -> bool:
        if anchor is not None:
            for c in chosen:
                if c["end_ns"] <= anchor["start_ns"]:
                    s = max(s, c["end_ns"])
                elif c["start_ns"] >= anchor["end_ns"]:
                    e = min(e, c["start_ns"])
                else:
                    return False                  # anchor already covered
        whole = [x for x in eps
                 if x["start_ns"] >= max(s, t0) and x["end_ns"] <= min(e, t1)]
        if not whole:
            return False
        s, e = whole[0]["start_ns"], whole[-1]["end_ns"]
        if any(s < c["end_ns"] and e > c["start_ns"] for c in chosen):
            return False
        chosen.append({"kind": kind, "selection_rule": rule, "start_ns": s,
                       "end_ns": e,
                       "anchor_episode_id": anchor and anchor["episode_id"]})
        return True

    win, step = int(SWITCH_WINDOW_S * 1e9), int(SWITCH_STEP_S * 1e9)
    windows = []
    s = t0
    while s + win <= t1:
        n = sum(1 for ts in trans if s <= ts < s + win)
        k = len({e["target_id"] for e in eps
                 if e["end_ns"] > s and e["start_ns"] < s + win})
        windows.append((-n, -k, s))
        s += step
    windows.sort()
    picked = 0
    for rank, (neg_n, neg_k, s) in enumerate(windows, 1):
        if picked == 2:
            break
        picked += add(
            "switching",
            f"densest switching: {SWITCH_WINDOW_S:g}-s window at "
            f"{(s - t0) / 1e9:g} s ranks {rank} of {len(windows)} windows "
            f"(step {SWITCH_STEP_S:g} s) with {-neg_n} transitions across "
            f"{-neg_k} targets; whole episodes inside the window",
            s, s + win)

    pad = int(SUSTAIN_PAD_S * 1e9)
    tracked = sorted((e for e in eps if e["target_id"] != NO_HIT),
                     key=lambda e: (-(e["end_ns"] - e["start_ns"]), e["start_ns"]))
    for rank, ep in enumerate(tracked[:2], 1):
        add("sustained",
            f"longest tracked-object episodes: {ep['episode_id']} ranks {rank} "
            f"by duration; up to {SUSTAIN_PAD_S:g} s of whole-episode context "
            f"on each side",
            ep["start_ns"] - pad, ep["end_ns"] + pad, ep)

    gaps = sorted((e for e in eps if e["target_id"] == NO_HIT
                   and (e["end_ns"] - e["start_ns"]) >= MIN_GAP_EPISODE_S * 1e9),
                  key=lambda e: (-(e["end_ns"] - e["start_ns"]), e["start_ns"]))
    for ep in gaps[:1]:
        add("no_hit",
            f"longest stretch with no accepted tracked-object hit "
            f"({ep['episode_id']}); up to {SUSTAIN_PAD_S:g} s of whole-episode "
            f"context on each side",
            ep["start_ns"] - pad, ep["end_ns"] + pad, ep)

    chosen.sort(key=lambda c: c["start_ns"])
    for i, c in enumerate(chosen, 1):
        c["interval_id"] = f"iv_{i:02d}"
    return chosen


def _overlap(a0, a1, b0, b1) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def interval_measures(eps, trans_rows, prox, s_ns, e_ns) -> dict:
    """Deterministic totals for [s_ns, e_ns) from episode support intervals."""
    per_target: dict[str, int] = {}
    inside = [e for e in eps if e["end_ns"] > s_ns and e["start_ns"] < e_ns]
    for e in inside:
        ov = _overlap(e["start_ns"], e["end_ns"], s_ns, e_ns)
        if ov:
            per_target[e["target_id"]] = per_target.get(e["target_id"], 0) + ov
    samples = [i for e in inside for i in e["samples"]]
    n_trans = sum(1 for t in trans_rows
                  if s_ns < int(t["transition_timestamp_ns"]) < e_ns)
    return {
        "duration_ns": e_ns - s_ns,
        "per_target_ns": per_target,
        "episode_ids": [e["episode_id"] for e in inside],
        "episode_count": len(inside),
        "single_sample_episode_count": sum(1 for e in inside if e["sample_count"] == 1),
        "transition_count": n_trans,
        "sample_count": len(samples),
        "boundary_near_sample_count": sum(1 for i in samples
                                          if prox.get(i, {}).get("boundary_near")),
    }


# --------------------------------------------------------------------------
# facts
# --------------------------------------------------------------------------

def _fact(fid, metric, value, unit, ref, **context):
    f = {"fact_id": fid, "metric": metric, "value": value, "unit": unit,
         "source_ref": ref}
    if context:
        f["context"] = context
    return f


def _round_fact(f: dict) -> dict:
    g = {k: v for k, v in f.items() if k != "evidence_refs"}
    g["source_ref"] = f["evidence_refs"][0]
    if isinstance(g["value"], float):
        g["value"] = _r3(g["value"])
    return g


_OBJECT_FACT_SUFFIXES = ("total_attention_s", "share_of_supported_time",
                         "episode_count", "revisit_count", "sample_count")


def build_facts(trace, intervals, eps, prox, names, t0) -> list[dict]:
    base = [_round_fact(f) for f in build_fact_catalog(trace)]
    facts = [f for f in base if f["fact_id"].startswith("session.")]
    facts += [f for f in base if f["fact_id"].startswith("object.")
              and f["fact_id"].rsplit(".", 1)[1] in _OBJECT_FACT_SUFFIXES]
    for f in facts:          # the trace's legacy column name, plainly worded
        if f["fact_id"].endswith("total_attention_s"):
            f["metric"] = "accumulated gaze-supported time"
    pairs = [f for f in base if f["fact_id"].startswith("transition_pair.")]
    pairs.sort(key=lambda f: f["context"]["transition_pair_rank"])
    facts += pairs[:15]

    by_target: dict[str, list] = {}
    for e in eps:
        by_target.setdefault(e["target_id"], []).extend(e["samples"])
    for r in trace["summary"]:
        oid = r["object_id"]
        facts.append(_fact(
            f"object.{oid}.boundary_near_sample_count",
            f"gaze samples on this target with another target within {PROXIMITY_PX} px",
            sum(1 for i in by_target.get(oid, []) if prox.get(i, {}).get("boundary_near")),
            "samples", "boundary_proximity.json", object_id=oid,
            object_name=names.get(oid, oid)))

    all_samples = [i for e in eps for i in e["samples"]]
    facts.append(_fact("session.single_sample_episode_count",
                       "episodes supported by a single gaze sample",
                       sum(1 for e in eps if e["sample_count"] == 1), "episodes",
                       "attention_episodes.csv#sample_count=1"))
    facts.append(_fact("session.boundary_near_sample_count",
                       f"gaze samples with another target within {PROXIMITY_PX} px",
                       sum(1 for i in all_samples if prox.get(i, {}).get("boundary_near")),
                       "samples", "boundary_proximity.json"))

    for iv in intervals:
        m = interval_measures(eps, trace["transitions"], prox,
                              iv["start_ns"], iv["end_ns"])
        iv["measures"] = m
        p = f"interval.{iv['interval_id']}"
        ref = (f"attention_episodes.csv#start>={iv['start_ns']}"
               f"&end<={iv['end_ns']}")
        ctx = {"interval_id": iv["interval_id"]}
        facts += [
            _fact(f"{p}.duration_s", "interval duration",
                  _r3(m["duration_ns"] / 1e9), "seconds", ref, **ctx),
            _fact(f"{p}.episode_count", "episodes in interval",
                  m["episode_count"], "episodes", ref, **ctx),
            _fact(f"{p}.transition_count", "transitions in interval",
                  m["transition_count"], "transitions", ref, **ctx),
            _fact(f"{p}.single_sample_episode_count",
                  "single-sample episodes in interval",
                  m["single_sample_episode_count"], "episodes", ref, **ctx),
            _fact(f"{p}.sample_count", "gaze samples in interval",
                  m["sample_count"], "samples", ref, **ctx),
            _fact(f"{p}.boundary_near_sample_count",
                  f"samples in interval with another target within {PROXIMITY_PX} px",
                  m["boundary_near_sample_count"], "samples",
                  "boundary_proximity.json", **ctx),
        ]
        for oid, ns in sorted(m["per_target_ns"].items(), key=lambda kv: -kv[1]):
            facts.append(_fact(f"{p}.target.{oid}.time_s",
                               "gaze-supported time in interval",
                               _r3(ns / 1e9), "seconds", ref,
                               interval_id=iv["interval_id"], object_id=oid,
                               object_name=names.get(oid, oid)))

    t_end = int(trace["manifest"]["source"]["validated_interval_ns"][1])
    k = 0
    s = t0
    while s < t_end:
        e = min(s + int(PHASE_S * 1e9), t_end)
        k += 1
        m = interval_measures(eps, trace["transitions"], prox, s, e)
        pid = f"phase.p{k}"
        label = f"{(s - t0) / 1e9:g}-{(e - t0) / 1e9:g} s"
        ref = f"attention_episodes.csv#overlap[{s},{e})"
        facts.append(_fact(f"{pid}.transition_count", f"transitions in {label}",
                           m["transition_count"], "transitions", ref,
                           phase_id=pid, phase_window=label))
        for oid, ns in sorted(m["per_target_ns"].items(), key=lambda kv: -kv[1]):
            facts.append(_fact(f"{pid}.target.{oid}.time_s",
                               f"gaze-supported time in {label}",
                               _r3(ns / 1e9), "seconds", ref, phase_id=pid,
                               phase_window=label, object_id=oid,
                               object_name=names.get(oid, oid)))
        s = e
    return facts


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------

def select_frames(iv, eps, samples_by_index) -> list[int]:
    """2-3 frames evenly spread over the interval, each carrying a sample."""
    inside = [i for e in eps if e["episode_id"] in iv["measures"]["episode_ids"]
              for i in e["samples"]]
    if not inside:
        return []
    n = 2 if iv["measures"]["duration_ns"] < 3.5e9 else 3
    picks = []
    for k in range(n):
        t = iv["start_ns"] + (k + 0.5) / n * iv["measures"]["duration_ns"]
        best = min(inside, key=lambda i: abs(samples_by_index[i]["timestamp_ns"] - t))
        if best not in picks:
            picks.append(best)
    return picks


def _draw_ring(img, x: float, y: float, radius: int):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    for r, w, colour in ((radius + 3, 7, (0, 0, 0)), (radius, 4, (60, 255, 90))):
        d.ellipse([x - r, y - r, x + r, y + r], outline=colour, width=w)


def first_frame_pts_s(video: Path) -> float:
    """Presentation time of the first frame; players seek on this timeline."""
    import av
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        pts = min(p.pts for p in c.demux(s) if p.pts is not None)
        return round(float(pts * s.time_base), 6)


def extract_frames(video: Path, wanted: dict, out_dir: Path,
                   width: int = 1024) -> dict:
    """wanted: frame_index -> (gaze_x, gaze_y). Returns frame_index -> meta."""
    import av
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {}
    with av.open(str(video)) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i in wanted:
                img = frame.to_image()
                gx, gy = wanted[i]
                _draw_ring(img, gx, gy, PROXIMITY_PX)
                scale = width / img.width
                img = img.resize((width, round(img.height * scale)))
                path = out_dir / f"fr_{i:04d}.jpg"
                img.save(path, quality=90)
                done[i] = {"file": f"frames/{path.name}", "sha256": _sha256(path),
                           "width": img.width, "height": img.height}
            if len(done) == len(wanted):
                break
    missing = set(wanted) - set(done)
    if missing:
        raise ValueError(f"could not decode frames {sorted(missing)}")
    return done


# --------------------------------------------------------------------------
# package
# --------------------------------------------------------------------------

def build_support_package(trace_dir, question: str, context: dict,
                          out_dir) -> dict:
    out = Path(out_dir).expanduser().resolve()
    if out.exists():
        raise ValueError(f"output directory already exists: {out}")
    src = load_sources(trace_dir)
    trace, fm = src["trace"], src["fusion_manifest"]
    tm = trace["manifest"]
    t0 = int(tm["source"]["validated_interval_ns"][0])
    names = {r["object_id"]: r["object_name"] for r in trace["summary"]}
    accepted = [o["object_id"] for o in fm["objects"]]

    samples = _load_samples(src["samples_path"])
    prox = boundary_proximity(samples, src["objects_path"], accepted)
    by_index = {int(r.gaze_sample_index): {
        "timestamp_ns": int(r.timestamp_ns), "frame_index": int(r.frame_index),
        "frame_timestamp_ns": int(r.frame_timestamp_ns),
        "gaze_x_px": float(r.gaze_x_px), "gaze_y_px": float(r.gaze_y_px),
        "status": r.gaze_status,
        "primary_object_id": (None if r.gaze_status != "on_object"
                              else r.primary_object_id)}
        for r in samples.itertuples(index=False)}

    eps = _episodes(trace)
    intervals = select_intervals(trace)
    facts = build_facts(trace, intervals, eps, prox, names, t0)

    rel = lambda ns: _r3((ns - t0) / 1e9)          # noqa: E731
    wanted, frame_meta = {}, []
    for iv in intervals:
        for si in select_frames(iv, eps, by_index):
            s = by_index[si]
            wanted[s["frame_index"]] = (s["gaze_x_px"], s["gaze_y_px"])
            p = prox.get(si, {})
            target = s["primary_object_id"] or NO_HIT
            frame_meta.append({
                "frame_id": f"fr_{s['frame_index']:04d}",
                "frame_index": s["frame_index"],
                "interval_id": iv["interval_id"],
                "time_s": rel(s["timestamp_ns"]),
                "frame_time_s": round((s["frame_timestamp_ns"] - t0) / 1e9, 6),
                "gaze_sample_index": si,
                "assigned_target_id": target,
                "assigned_target_name": names.get(target, target),
                "alternatives_within_px": p.get("alternatives_within_px", []),
                "nearest_alternative_px": p.get("nearest_alternative_px"),
            })
    files = extract_frames(src["scene_video"], wanted, out / "frames")
    for f in frame_meta:
        f.update(files[f["frame_index"]])

    ep_rows = []
    for e in eps:
        near = sum(1 for i in e["samples"] if prox.get(i, {}).get("boundary_near"))
        ep_rows.append({
            "episode_id": e["episode_id"], "target_id": e["target_id"],
            "start_s": rel(e["start_ns"]),
            "duration_s": _r3((e["end_ns"] - e["start_ns"]) / 1e9),
            "sample_count": e["sample_count"],
            "boundary_near_samples": near,
            "first_frame_index": e["frames"][0], "last_frame_index": e["frames"][-1],
        })
    in_any = {eid for iv in intervals for eid in iv["measures"]["episode_ids"]}
    trans_rows = [{
        "transition_id": t["transition_id"],
        "from_episode_id": t["from_episode_id"], "to_episode_id": t["to_episode_id"],
        "from_object_id": t["from_object_id"], "to_object_id": t["to_object_id"],
        "time_s": rel(int(t["transition_timestamp_ns"])),
        "unsupported_gap_s": _r3(float(t["unsupported_gap_s"])),
        "transition_type": t["transition_type"],
    } for t in trace["transitions"]
        if t["from_episode_id"] in in_any and t["to_episode_id"] in in_any]

    iv_rows = []
    for iv in intervals:
        m = iv["measures"]
        iv_rows.append({
            "interval_id": iv["interval_id"], "kind": iv["kind"],
            "selection_rule": iv["selection_rule"],
            "start_s": rel(iv["start_ns"]), "end_s": rel(iv["end_ns"]),
            "start_timestamp_ns": iv["start_ns"], "end_timestamp_ns": iv["end_ns"],
            "first_frame_index": min(next(e for e in eps if e["episode_id"] == eid)["frames"][0]
                                     for eid in m["episode_ids"]),
            "last_frame_index": max(next(e for e in eps if e["episode_id"] == eid)["frames"][-1]
                                    for eid in m["episode_ids"]),
            "episode_ids": m["episode_ids"],
            "frame_ids": [f["frame_id"] for f in frame_meta
                          if f["interval_id"] == iv["interval_id"]],
            "fact_ids": [f["fact_id"] for f in facts
                         if f.get("context", {}).get("interval_id") == iv["interval_id"]],
        })

    counts = fm["counts"]
    video = fm["video"]
    gaze_meta = (src["trim_manifest"] or {}).get("gaze", {})
    objects = []
    for r in sorted(trace["summary"], key=lambda r: int(r["display_rank"])):
        objects.append({"object_id": r["object_id"], "name": r["object_name"],
                        "color": r["object_color"] or "#8a94a3",
                        "is_tracked_object": r["is_tracked_object"] == "true"})

    package = {
        "schema_version": SCHEMA_VERSION,
        "question": question.strip(),
        "recording": {
            "recording_id": fm["parent_artifact"]["recording_id"],
            "source_system": tm["source"]["system"],
            "timestamp_domain": tm["source"]["timestamp_domain"],
            "interval_timestamp_ns": [t0, int(tm["source"]["validated_interval_ns"][1])],
            "duration_s": _r3(tm["source"]["validated_interval_duration_ns"] / 1e9),
            "times_are": "seconds from the start of this clip",
            "artifact_ids": {
                "trimmed_recording": fm["parent_artifact"]["grandparent_artifact_id"],
                "tracked_objects": fm["parent_artifact"]["artifact_id"],
                "gaze_fusion": fm["artifact_id"],
                "attention_trace": tm["artifact_id"],
            },
        },
        "context": context,
        "data_quality": {
            "gaze_samples": counts["samples_total"],
            "gaze_samples_valid_in_frame": counts["gaze_status"]["on_object"]
                                           + counts["gaze_status"]["background"],
            "gaze_median_interval_ms": tm["support_policy"]["median_interval_ms"],
            "gaze_model": gaze_meta.get("model", "unknown"),
            "gaze_model_note": gaze_meta.get("model_reason", ""),
            "video": f"{video['width']}x{video['height']}, {video['n_frames']} frames "
                     f"over {video['duration_s']} s",
            "accepted_tracked_objects": len(accepted),
            "unsupported_gap_total_s": _r3(sum(float(t["unsupported_gap_s"])
                                               for t in trace["transitions"])),
            "notes": [
                "Episodes, transitions and durations are TraceGaze deterministic "
                "outputs; each gaze sample supports at most ~0.1 s here.",
                "A single-sample episode lasts one gaze sample; rapid switches "
                "made of such episodes may reflect gaze near a mask boundary "
                "rather than a change of what the person looked at.",
                f"boundary_near: another target (object, or the no-hit region) "
                f"lies within {PROXIMITY_PX} px of the gaze pixel. "
                f"{PROXIMITY_PX} px is a project convention, not a measured "
                f"gaze-accuracy bound.",
                "'No accepted tracked-mask hit' (__no_tracked_object__) means the "
                "gaze pixel hit none of the reviewed object masks; it may lie on an "
                "unlabeled object and does not mean empty space.",
            ],
        },
        "definitions": {
            "episode": tm["support_policy"]["episode_rule"],
            "transition": "link between consecutive episodes, with the "
                          "unsupported gap between them",
            "revisit": tm["support_policy"]["revisit_rule"],
            "support": tm["support_policy"]["support_interval"],
            "overlap_resolution": tm["evidence_policy"]["overlap_resolution"],
        },
        "objects": objects,
        "facts": facts,
        "intervals": iv_rows,
        "episodes": ep_rows,
        "transitions": trans_rows,
        "frames": frame_meta,
        "source_refs": {
            "attention_trace_dir": str(trace["root"]),
            "gaze_fusion_dir": str(src["fusion_dir"]),
            "tracked_objects_dir": str(src["tracked_dir"]),
            "scene_video": str(src["scene_video"]),
            "scene_video_sha256": fm["parent_recording"]["hashes_sha256"]["video"],
            "tracegaze_visualization_video": str(src["demo_video"]),
            "tracegaze_visualization_sha256": fm["files_sha256"]["demo"],
            "gaze_samples_sha256": tm["parent_artifact"]["files_sha256"]["gaze_samples_parquet"],
            "video_first_frame_pts_s": {
                "scene_video": first_frame_pts_s(src["scene_video"]),
                "tracegaze_visualization_video": first_frame_pts_s(src["demo_video"]),
            },
            "frame_index_to_time": "frame i starts at video time frame_time_s + "
                                   "video_first_frame_pts_s (player timeline)",
        },
    }
    (out / "support_package.json").write_text(
        json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "boundary_proximity.json").write_text(json.dumps({
        "proximity_px": PROXIMITY_PX, "search_radius_px": _CROP_PX,
        "rule": "alternatives = other accepted masks (and the no-hit region for "
                "on-object samples) within proximity_px of the fusion gaze pixel",
        "samples": {str(k): v for k, v in sorted(prox.items())},
    }, indent=1) + "\n", encoding="utf-8")
    return package
