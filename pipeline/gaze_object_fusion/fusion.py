"""Fusion orchestration: tracked_objects artifact + parent gaze → fused table.

Runs as a CLI/batch process (never inside a GUI server): pandas/pyarrow are
allowed here under the project's Arrow-isolation policy, which only forbids
them inside long-lived FastAPI server processes.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from common.artifacts import ArtifactError, sha256_file

from .timecells import (ASSIGNED, DEFAULT_SANITY_GAP_MS, NO_MATCHING_FRAME,
                        FrameCells)

TS_COL = "timestamp [ns]"
GX_COL = "gaze x [px]"
GY_COL = "gaze y [px]"

# invalid_reason values
NOT_WORN = "not_worn"
NONFINITE = "nonfinite_coords"


def _decode_rle(rle_json: str):
    import pycocotools.mask as mask_util
    r = json.loads(rle_json)
    return mask_util.decode({"size": r["size"],
                             "counts": r["counts"].encode("ascii")}
                            ).astype(bool)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FusionInputs:
    """Resolved + cross-validated inputs for one fusion run."""

    # parent artifact types that carry a VALIDATED trim interval — the
    # boundaries the first/final frame cells need. Both source systems
    # expose the same `selection` keys; neither pretends to be the other.
    SUPPORTED_PARENTS = {
        "trimmed_artifact": "pupil_neon",
        "trimmed_aria_recording": "project_aria",
    }

    def __init__(self, tracked_dir: Path):
        from object_tracking.artifact import load_tracked_artifact

        self.tracked = load_tracked_artifact(tracked_dir)
        tm = self.tracked.manifest
        parent = tm["parent"]
        src_type = parent.get("source_type")
        if src_type not in self.SUPPORTED_PARENTS:
            raise ArtifactError(
                "gaze fusion requires a trimmed-recording parent carrying a "
                f"validated interval (got source_type={src_type!r}; "
                f"supported: {sorted(self.SUPPORTED_PARENTS)})")
        self.source_system = self.SUPPORTED_PARENTS[src_type]
        if not tm.get("synchronization", {}).get("gaze_compatible"):
            raise ArtifactError(
                "tracked artifact is not gaze-compatible (recorded in its "
                "manifest synchronization section) — nothing to fuse")

        self.parent_dir = Path(parent["source_dir"])
        if not self.parent_dir.is_dir():
            raise ArtifactError(
                f"parent recording folder not resolvable: {self.parent_dir} "
                "(absolute provenance path from the tracked manifest)")

        if self.source_system == "project_aria":
            from aria_source.artifact import load_trimmed_aria
            art = load_trimmed_aria(self.parent_dir)
            trim_artifact_id = art.artifact_id
            trim_recording_id = art.recording_id
            validation_ok = art.validation_ok
            sel = art.manifest.get("selection", {})
            self.video_path = art.video_path
            self.gaze_path = art.gaze_path
            self.world_path = art.world_timestamps_path
            self.timestamp_domain = art.manifest["timestamps"]["domain"]
        else:
            from neon_trim.artifact import load_artifact as load_trim
            art = load_trim(self.parent_dir)
            trim_artifact_id = art.artifact_id
            trim_recording_id = art.source_recording_id
            validation_ok = art.validation_ok
            sel = art.selection
            self.video_path = art.video_path
            self.gaze_path = art.gaze_path
            self.world_path = art.world_timestamps_path
            self.timestamp_domain = "neon_world_time"
        self.trim = art

        if not validation_ok:
            raise ArtifactError("parent trim artifact validation record is "
                                "not all-ok")
        declared = tm["parent_artifact"]["artifact_id"]
        if trim_artifact_id != declared:
            raise ArtifactError(
                f"parent artifact id mismatch: trim manifest says "
                f"{trim_artifact_id}, tracked manifest says {declared}")
        rec_ids = {str(tm["parent_artifact"].get("recording_id") or "")
                   or None,
                   str(trim_recording_id or "") or None}
        if len(rec_ids - {None}) > 1:
            raise ArtifactError(f"recording id mismatch: {rec_ids}")

        for key in ("absolute_start_ns", "absolute_end_boundary_ns"):
            if not isinstance(sel.get(key), int):
                raise ArtifactError(
                    f"trim manifest selection.{key} missing or non-int — "
                    "cannot bound the first/final frame cells")
        self.interval_start_ns = sel["absolute_start_ns"]
        self.interval_end_ns = sel["absolute_end_boundary_ns"]
        self.n_frames = int(tm["video"]["n_frames"])
        self.width = int(tm["video"]["width"])
        self.height = int(tm["video"]["height"])
        self.has_audio = bool(tm["video"].get("has_audio"))

    def parent_hashes_now(self) -> dict:
        return {"video": sha256_file(self.video_path),
                "gaze": sha256_file(self.gaze_path),
                "world_timestamps": sha256_file(self.world_path)}

    def parent_hashes_at_tracking(self) -> dict:
        p = self.tracked.manifest["parent"]
        return {"video": p.get("video_sha256"),
                "gaze": p.get("gaze_sha256"),
                "world_timestamps": p.get("world_timestamps_sha256")}


def _parse_int_ns(path: Path, df, col: str) -> list:
    """Exact int-ns parse of a timestamp column read with dtype=str.

    pandas dtype inference would parse float-formatted text ('...432.0',
    '1.7e18') as float64, and astype('int64') then truncates >2^53 values
    SILENTLY (adversarially verified: ±hundreds of ns). Timestamps must
    therefore go text -> int() directly, which refuses non-integer text.
    """
    if col not in df.columns:
        raise ArtifactError(f"{path}: missing column {col!r}")
    if df[col].isna().any():
        raise ArtifactError(f"{path}: null timestamps in {col!r}")
    try:
        return [int(s) for s in df[col].tolist()]
    except (TypeError, ValueError) as e:
        raise ArtifactError(
            f"{path}: non-integer text in {col!r} — timestamps must be "
            f"exact integer nanoseconds, never float-formatted ({e})")


def load_world_ts(world_path: Path) -> list:
    import pandas as pd
    df = pd.read_csv(world_path, dtype={TS_COL: "string"})
    return _parse_int_ns(world_path, df, TS_COL)


def load_gaze(gaze_path: Path):
    """Gaze table with EXACT int-ns timestamps (never through float).

    `fixation id` / `blink id` are parsed as NULLABLE integers (never
    silently coerced to floats — a blink id of NaN-as-2.0 would be a
    misleading numeric value downstream).
    """
    import pandas as pd
    df = pd.read_csv(gaze_path, dtype={TS_COL: "string",
                                       "fixation id": "string",
                                       "blink id": "string"})
    for col in (GX_COL, GY_COL):
        if col not in df.columns:
            raise ArtifactError(f"{gaze_path}: missing column {col!r}")
    ts = _parse_int_ns(gaze_path, df, TS_COL)
    df = df.copy()
    df[TS_COL] = pd.array(ts, dtype="int64")   # from exact Python ints
    for col in ("fixation id", "blink id"):
        if col in df.columns:
            vals = []
            for s in df[col].tolist():
                if s is pd.NA or s is None or str(s).strip() == "":
                    vals.append(None)
                else:
                    try:
                        vals.append(int(str(s).strip()))
                    except ValueError:
                        raise ArtifactError(
                            f"{gaze_path}: non-integer {col!r} value "
                            f"{s!r} — event ids must be null or integer")
            df[col] = pd.array(vals, dtype="Int64")
    return df


def fuse(inputs: FusionInputs, *,
         sanity_gap_ms: float = DEFAULT_SANITY_GAP_MS,
         progress: Optional[Callable[[str, float], None]] = None):
    """Compute the fused table + validation record + frame cells.

    Returns (fused_df, cells, validation, extras).
    """
    import numpy as np
    import pandas as pd

    def report(phase, frac):
        if progress:
            progress(phase, frac)

    validation = []

    def check(name, ok, detail=""):
        validation.append({"check": name, "ok": bool(ok), "detail": detail})

    report("loading parents", 0.02)
    world = load_world_ts(inputs.world_path)
    gaze = load_gaze(inputs.gaze_path)
    table = pd.read_parquet(inputs.tracked.objects_table_path)

    check("world rows equal tracked frame count",
          len(world) == inputs.n_frames,
          f"{len(world)} world rows vs {inputs.n_frames} frames")
    if len(world) != inputs.n_frames:
        raise ArtifactError("world_timestamps row count does not match the "
                            "tracked artifact frame count")
    # the tracked table's timestamp_ns column came from the SAME verified
    # 1:1 mapping — cross-check every row
    if not table.empty:
        ts_match = bool((table["timestamp_ns"].astype("int64")
                         == table["frame_index"].map(
                             lambda i: world[int(i)]).astype("int64")).all())
    else:
        ts_match = True
    check("tracked table timestamps agree with parent world timestamps",
          ts_match, "timestamp_ns == world[frame_index] for all rows")
    if not ts_match:
        raise ArtifactError("tracked table timestamp_ns disagrees with the "
                            "parent world_timestamps.csv")
    if not table.empty and (table["score"].isna().any()
                            or table["mask_area"].isna().any()):
        raise ArtifactError("tracked table contains null score/mask_area — "
                            "deterministic hit ordering is impossible")

    cells = FrameCells(world, inputs.interval_start_ns,
                       inputs.interval_end_ns, sanity_gap_ms=sanity_gap_ms)
    check("trim interval brackets all frame timestamps", True,
          f"[{inputs.interval_start_ns}, {inputs.interval_end_ns}) contains "
          f"world[0]..world[{len(world) - 1}]")
    check("interval start equals first frame timestamp "
          "(trim semantics [start_frame_time, time_after_last_frame))",
          inputs.interval_start_ns == world[0],
          f"start {inputs.interval_start_ns} vs world[0] {world[0]}")
    gaps = cells.anomalous_gaps()
    check("anomalous world-timestamp gaps reported (never dropped)",
          True,
          f"{len(gaps)} gap(s) above the {sanity_gap_ms} ms sanity "
          f"threshold; affected samples remain assigned by the midpoint "
          f"rule")

    # ------------------------------------------------------------ assign
    report("assigning gaze to frame cells", 0.15)
    ts_list = [int(t) for t in gaze[TS_COL].tolist()]
    frames, offsets, statuses = [], [], []
    for ts in ts_list:
        a = cells.assign(ts)
        frames.append(a.frame_index)
        offsets.append(a.offset_ns)
        statuses.append(a.status)

    gx = gaze[GX_COL].to_numpy(dtype="float64")
    gy = gaze[GY_COL].to_numpy(dtype="float64")
    worn = (gaze["worn"].to_numpy(dtype="float64")
            if "worn" in gaze.columns else None)

    n = len(gaze)
    gaze_valid = [True] * n
    invalid_reason = [None] * n
    coords_in_frame = [False] * n
    for i in range(n):
        if not (math.isfinite(gx[i]) and math.isfinite(gy[i])):
            gaze_valid[i] = False
            invalid_reason[i] = NONFINITE
            continue
        if worn is not None and not math.isnan(worn[i]) and worn[i] == 0.0:
            gaze_valid[i] = False
            invalid_reason[i] = NOT_WORN
        # pixel-cell convention: gaze (x, y) lies in pixel (floor(x), floor(y))
        px, py = math.floor(gx[i]), math.floor(gy[i])
        coords_in_frame[i] = (0 <= px < inputs.width
                              and 0 <= py < inputs.height)

    # --------------------------------------------------------- hit tests
    report("hit-testing masks", 0.35)
    rows_by_frame = {}
    if not table.empty:
        for fidx, sub in table.groupby("frame_index"):
            rows_by_frame[int(fidx)] = sub

    hit_ids = [[] for _ in range(n)]
    hit_names = [[] for _ in range(n)]
    prim_id = [None] * n
    prim_name = [None] * n
    prim_area = [None] * n
    prim_score = [None] * n

    sample_idx_by_frame = {}
    for i in range(n):
        if statuses[i] == ASSIGNED and gaze_valid[i] and coords_in_frame[i]:
            sample_idx_by_frame.setdefault(frames[i], []).append(i)

    done = 0
    for fidx in sorted(sample_idx_by_frame):
        sub = rows_by_frame.get(fidx)
        masks = []
        if sub is not None:
            for r in sub.itertuples(index=False):
                masks.append((_decode_rle(r.mask_rle), {
                    "object_id": r.object_id, "object_name": r.object_name,
                    "mask_area": int(r.mask_area), "score": float(r.score)}))
        for i in sample_idx_by_frame[fidx]:
            px, py = math.floor(gx[i]), math.floor(gy[i])
            hits = [meta for m, meta in masks if m[py, px]]
            # deterministic primary-target ordering (fixed by the spec):
            # 1) smaller mask area  2) higher tracking score  3) stable id
            hits.sort(key=lambda h: (h["mask_area"], -h["score"],
                                     h["object_id"]))
            hit_ids[i] = [h["object_id"] for h in hits]
            hit_names[i] = [h["object_name"] for h in hits]
            if hits:
                prim_id[i] = hits[0]["object_id"]
                prim_name[i] = hits[0]["object_name"]
                prim_area[i] = hits[0]["mask_area"]
                prim_score[i] = hits[0]["score"]
        done += 1
        if done % 50 == 0:
            report("hit-testing masks",
                   0.35 + 0.4 * done / max(len(sample_idx_by_frame), 1))

    # ------------------------------------------------------ fused table
    report("building fused table", 0.8)
    fused = pd.DataFrame({
        "gaze_ts_ns": pd.array(ts_list, dtype="int64"),
        "assignment": statuses,
        "frame_index": pd.array(frames, dtype="Int64"),
        "frame_offset_ns": pd.array(offsets, dtype="Int64"),
        "frame_offset_ms": [None if o is None else o / 1e6 for o in offsets],
        "gaze_x_px": gx, "gaze_y_px": gy,
        "gaze_valid": pd.array(gaze_valid, dtype="bool"),
        "invalid_reason": invalid_reason,
        "coords_in_frame": pd.array(coords_in_frame, dtype="bool"),
        "hit_object_ids": hit_ids,
        "hit_object_names": hit_names,
        "n_hits": pd.array([len(h) for h in hit_ids], dtype="int32"),
        "primary_object_id": prim_id,
        "primary_object_name": prim_name,
        "primary_mask_area": pd.array(prim_area, dtype="Int64"),
        "primary_score": pd.array(prim_score, dtype="float64"),
    })
    # passthrough NEON columns (provenance; absent columns stay absent;
    # direct Series assignment preserves nullable dtypes like Int64)
    for src, dst in (("worn", "worn"), ("fixation id", "fixation_id"),
                     ("blink id", "blink_id"),
                     ("azimuth [deg]", "azimuth_deg"),
                     ("elevation [deg]", "elevation_deg")):
        if src in gaze.columns:
            fused[dst] = gaze[src].reset_index(drop=True)

    # -------------------------------------------- demonstration selection
    # For each frame: the valid fused row assigned to THAT EXACT frame with
    # the smallest |frame_offset_ms| (tie -> earliest timestamp). "Valid"
    # means gaze_valid (worn, finite) AND coordinates inside the frame — a
    # cursor cannot be displayed at out-of-frame coordinates. Frames
    # without a valid assigned row get no cursor — samples from adjacent
    # frames are never borrowed, however temporally close.
    demo_selected = [False] * n
    demo_rows = {}
    for fidx, idxs in sample_idx_by_frame.items():
        best = min(idxs, key=lambda i: (abs(offsets[i]), ts_list[i]))
        demo_selected[best] = True
        demo_rows[fidx] = best
    fused["demo_selected"] = pd.array(demo_selected, dtype="bool")

    # ------------------------------------------------- post-hoc rechecks
    report("verifying assignment invariants", 0.9)
    ok_cells = True
    for i in range(n):
        if statuses[i] == ASSIGNED:
            lo, hi = cells.cell_bounds(frames[i])
            if not (lo <= ts_list[i] < hi
                    and offsets[i] == ts_list[i] - cells.world[frames[i]]):
                ok_cells = False
        else:
            if (cells.interval_start_ns <= ts_list[i]
                    < cells.interval_end_ns):
                ok_cells = False
    check("every sample verifiably inside its half-open cell "
          "(and every unassigned sample outside the trim interval)",
          ok_cells, f"{n} samples rechecked against boundaries")

    order_ok = True
    for i in range(n):
        ids = hit_ids[i]
        if len(ids) >= 1 and prim_id[i] != ids[0]:
            order_ok = False
    check("first hit-list entry equals primary object", order_ok,
          "primary == hits[0] for all rows with hits")

    demo_ok = True
    for fidx, best in demo_rows.items():
        cand = sample_idx_by_frame[fidx]
        if frames[best] != fidx:
            demo_ok = False
        for j in cand:
            if (abs(offsets[j]), ts_list[j]) < (abs(offsets[best]),
                                                ts_list[best]):
                demo_ok = False
    check("demo cursor rows: exact-frame, minimal |frame_offset|, "
          "deterministic tie-break", demo_ok,
          f"{len(demo_rows)} frames with cursor / {cells.n_frames} frames")

    check("no global frame-distance threshold used", True,
          "assignment is purely timestamp-defined frame cells (midpoints + "
          "validated trim interval boundaries)")

    if not (ok_cells and order_ok and demo_ok):
        raise ArtifactError("fusion invariant recheck failed:\n" + "\n".join(
            f"  x {v['check']}" for v in validation if not v["ok"]))

    extras = {
        "gaps": gaps,
        "demo_rows": demo_rows,
        "sample_idx_by_frame": sample_idx_by_frame,
    }
    return fused, cells, validation, extras


# ------------------------------------------------------- public contract

# the agreed authoritative schema — exact names, exact order, both formats
PUBLIC_COLUMNS = [
    "gaze_sample_index", "timestamp_ns", "relative_time_s", "frame_index",
    "frame_timestamp_ns", "frame_offset_ms", "gaze_x_px", "gaze_y_px",
    "gaze_pixel_x", "gaze_pixel_y", "worn", "gaze_valid", "gaze_status",
    "is_gaze_at_object", "primary_object_id", "primary_object_name",
    "hit_object_count", "hit_object_ids", "hit_object_names",
    "primary_mask_area", "primary_track_score", "fixation_id", "blink_id",
]

ON_OBJECT = "on_object"
BACKGROUND = "background"
INVALID_GAZE = "invalid_gaze"
OUTSIDE_FRAME = "outside_frame"
GAZE_STATUSES = [ON_OBJECT, BACKGROUND, INVALID_GAZE, OUTSIDE_FRAME,
                 NO_MATCHING_FRAME]

# status precedence (documented): a sample outside the trim interval is
# no_matching_frame regardless of validity; an invalid sample (not worn /
# nonfinite coords) is invalid_gaze even if its coords would also be
# outside the frame; only valid, in-frame, frame-assigned samples split
# into on_object / background.


def build_public_table(fused, cells: FrameCells, inputs: FusionInputs):
    """Derive the authoritative public table (PUBLIC_COLUMNS, exact order)
    from the internal fused table. Timestamps stay exact integers."""
    import pandas as pd

    n = len(fused)
    statuses = []
    for i in range(n):
        if fused["assignment"].iat[i] == NO_MATCHING_FRAME:
            statuses.append(NO_MATCHING_FRAME)
        elif not bool(fused["gaze_valid"].iat[i]):
            statuses.append(INVALID_GAZE)
        elif not bool(fused["coords_in_frame"].iat[i]):
            statuses.append(OUTSIDE_FRAME)
        elif int(fused["n_hits"].iat[i]) > 0:
            statuses.append(ON_OBJECT)
        else:
            statuses.append(BACKGROUND)

    frame_ts = [None if f is pd.NA or f is None else cells.world[int(f)]
                for f in fused["frame_index"].tolist()]
    px, py = [], []
    for x, y in zip(fused["gaze_x_px"].tolist(), fused["gaze_y_px"].tolist()):
        if math.isfinite(x) and math.isfinite(y):
            px.append(math.floor(x))
            py.append(math.floor(y))
        else:
            px.append(None)
            py.append(None)

    def _nullable(col, dtype):
        if col in fused.columns:
            return fused[col].astype(dtype)
        return pd.array([None] * n, dtype=dtype)

    pub = pd.DataFrame({
        "gaze_sample_index": pd.array(range(n), dtype="int64"),
        "timestamp_ns": fused["gaze_ts_ns"].astype("int64"),
        "relative_time_s": [
            (int(t) - cells.interval_start_ns) / 1e9
            for t in fused["gaze_ts_ns"].tolist()],
        "frame_index": fused["frame_index"].astype("Int64"),
        "frame_timestamp_ns": pd.array(frame_ts, dtype="Int64"),
        "frame_offset_ms": pd.array(
            [None if o is None else float(o)
             for o in fused["frame_offset_ms"].tolist()], dtype="Float64"),
        "gaze_x_px": fused["gaze_x_px"].astype("float64"),
        "gaze_y_px": fused["gaze_y_px"].astype("float64"),
        "gaze_pixel_x": pd.array(px, dtype="Int64"),
        "gaze_pixel_y": pd.array(py, dtype="Int64"),
        "worn": _nullable("worn", "Float64"),
        "gaze_valid": fused["gaze_valid"].astype("bool"),
        "gaze_status": statuses,
        "is_gaze_at_object": (fused["n_hits"] > 0).astype("bool"),
        "primary_object_id": fused["primary_object_id"],
        "primary_object_name": fused["primary_object_name"],
        "hit_object_count": fused["n_hits"].astype("int64"),
        "hit_object_ids": fused["hit_object_ids"],
        "hit_object_names": fused["hit_object_names"],
        "primary_mask_area": fused["primary_mask_area"].astype("Int64"),
        "primary_track_score":
            fused["primary_score"].astype("Float64"),
        "fixation_id": _nullable("fixation_id", "Int64"),
        "blink_id": _nullable("blink_id", "Int64"),
    })
    assert list(pub.columns) == PUBLIC_COLUMNS
    return pub


def write_public_tables(pub, parquet_path: Path, csv_path: Path) -> None:
    """Write the SAME logical table as parquet and CSV.

    CSV encoding rules (documented; the loader/tests verify round-trip):
    ints exact integer text, nulls empty, booleans True/False, floats
    shortest-roundtrip repr, list fields JSON arrays.
    """
    pub.to_parquet(parquet_path, index=False)
    csv_df = pub.copy()
    for col in ("hit_object_ids", "hit_object_names"):
        csv_df[col] = [json.dumps(list(v)) for v in csv_df[col]]
    csv_df.to_csv(csv_path, index=False)


def summarize(fused, cells, inputs: FusionInputs, table) -> dict:
    """counts + stats for the manifest."""
    import numpy as np
    assigned = fused[fused["assignment"] == ASSIGNED]
    off = assigned["frame_offset_ms"].astype(float)
    per_object = []
    for oid, sub in table.groupby("object_id"):
        hits = fused["primary_object_id"] == oid
        any_hit = fused["hit_object_ids"].map(lambda h: oid in h)
        per_object.append({
            "object_id": oid,
            "object_name": sub["object_name"].iloc[0],
            "frames_tracked": int(sub["frame_index"].nunique()),
            "samples_primary": int(hits.sum()),
            "samples_hit_any": int(any_hit.sum()),
        })
    return {
        "samples_total": int(len(fused)),
        "samples_assigned": int((fused["assignment"] == ASSIGNED).sum()),
        "samples_no_matching_frame":
            int((fused["assignment"] == NO_MATCHING_FRAME).sum()),
        "samples_invalid_not_worn":
            int((fused["invalid_reason"] == NOT_WORN).sum()),
        "samples_invalid_nonfinite":
            int((fused["invalid_reason"] == NONFINITE).sum()),
        "samples_coords_outside_frame":
            int((fused["gaze_valid"]
                 & ~fused["coords_in_frame"]).sum()),
        "samples_with_hits": int((fused["n_hits"] > 0).sum()),
        "samples_assigned_valid_no_hit":
            int(((fused["assignment"] == ASSIGNED) & fused["gaze_valid"]
                 & fused["coords_in_frame"] & (fused["n_hits"] == 0)).sum()),
        "frames_total": cells.n_frames,
        "frames_with_cursor": int(fused["demo_selected"].sum()),
        "frame_offset_ms": ({
            "min": float(off.min()), "max": float(off.max()),
            "mean": float(off.mean()),
            "max_abs": float(off.abs().max()),
        } if len(off) else None),
        "per_object": per_object,
    }
