"""gaze_object_fusion artifact: publication + loader.

Layout (movable directory; `files` are artifact-relative,
symlink-escape-checked on load):

    <out_dir>/
        fusion_manifest.json      versioned downstream contract
        gaze_objects.parquet      one row per gaze sample (fused table)
        gaze_demo.mp4             optional time-faithful demonstration video
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from common.artifacts import (ArtifactError, AtomicPublisher, read_manifest,
                              require, resolve_declared_file, sha256_file)

from . import __version__
from .fusion import (GAZE_STATUSES, NO_MATCHING_FRAME, PUBLIC_COLUMNS,
                     FusionInputs, build_public_table, fuse, summarize,
                     write_public_tables)
from .timecells import DEFAULT_SANITY_GAP_MS

FUSION_MANIFEST = "fusion_manifest.json"
FUSION_ARTIFACT_TYPE = "gaze_object_fusion"
FUSION_SCHEMA_VERSION = "1.1"      # 1.1: public gaze_samples.{csv,parquet}


def publish_fusion_artifact(
        tracked_dir: Path, out_dir: Path, *,
        sanity_gap_ms: float = DEFAULT_SANITY_GAP_MS,
        make_demo: bool = True,
        progress_cb: Optional[Callable[[str, float], None]] = None) -> dict:
    """Run fusion end-to-end and atomically publish the artifact.

    Returns the manifest. Parents (tracked artifact + trimmed recording)
    are never modified; their hashes are recorded and re-checked.
    """
    import pandas as pd

    def report(phase, frac):
        if progress_cb is not None:
            progress_cb(phase, frac)

    inputs = FusionInputs(Path(tracked_dir))

    report("hashing parents", 0.01)
    hashes_before = inputs.parent_hashes_now()
    at_tracking = inputs.parent_hashes_at_tracking()
    missing = sorted(k for k, v in at_tracking.items() if not v)
    if missing:
        # a silent pass on an absent hash would void the immutability proof
        raise ArtifactError(
            f"tracked manifest records no parent hash for {missing} — "
            "cannot prove the parents are unchanged since tracking")
    unchanged = all(at_tracking[k] == hashes_before[k]
                    for k in hashes_before)
    if not unchanged:
        raise ArtifactError(
            "parent recording files changed since the tracked artifact was "
            f"published: now {hashes_before}, at tracking {at_tracking}")

    # claim the destination BEFORE the (potentially long) fusion run so a
    # bad --out fails in seconds, not after minutes of work
    pub = AtomicPublisher(Path(out_dir), source_dir=inputs.tracked.artifact_dir,
                          tmp_prefix=".gaze_fusion_tmp_")
    try:
        tmp = pub.tmp_dir

        fused, cells, validation, extras = fuse(
            inputs, sanity_gap_ms=sanity_gap_ms,
            progress=lambda p, f: report(p, 0.02 + 0.5 * f))
        validation.insert(0, {
            "check": "parent files unchanged since tracking (sha256)",
            "ok": True,
            "detail": "video/gaze/world_timestamps re-hashed identical to "
                      "the tracked manifest parent hashes"})

        table = pd.read_parquet(inputs.tracked.objects_table_path)
        counts = summarize(fused, cells, inputs, table)

        ann = json.loads(
            Path(inputs.tracked.annotations_path).read_text())
        colors = {o["object_id"]: o["color"]
                  for o in ann.get("objects", []) if not o.get("deleted")}

        report("writing tables", 0.55)
        public = build_public_table(fused, cells, inputs)
        counts["gaze_status"] = {
            s: int((public["gaze_status"] == s).sum())
            for s in GAZE_STATUSES}
        write_public_tables(public, tmp / "gaze_samples.parquet",
                            tmp / "gaze_samples.csv")
        fused.to_parquet(tmp / "gaze_objects.parquet", index=False)

        demo_info = None
        if make_demo:
            report("rendering demonstration video", 0.62)
            objects_by_frame = {}
            if not table.empty:
                names = {o["object_id"]: o["name"]
                         for o in ann.get("objects", [])
                         if not o.get("deleted")}
                for fidx, sub in table.groupby("frame_index"):
                    objects_by_frame[int(fidx)] = [
                        {"object_id": r.object_id,
                         "name": names.get(r.object_id, r.object_name),
                         "color": colors.get(r.object_id, "#ffffff"),
                         "rle": r.mask_rle,
                         "bbox": (float(r.bbox_x), float(r.bbox_y),
                                  float(r.bbox_w), float(r.bbox_h))}
                        for r in sub.itertuples(index=False)]
            # ---- PRESENTATION layer (display only, never analytical) ----
            from . import dwell as dwell_mod
            from . import presentation as pres_mod
            names = {o["object_id"]: o["name"]
                     for o in ann.get("objects", [])
                     if not o.get("deleted")}
            obj_meta = {oid: {"name": names.get(oid, oid),
                              "color": colors.get(oid, "#ffffff")}
                        for oid in set(names) | set(colors)}
            frame_ts = list(cells.world)

            mask_index = {}
            if not table.empty:
                for fidx, sub in table.groupby("frame_index"):
                    mask_index[int(fidx)] = [
                        {"object_id": r.object_id,
                         "object_name": r.object_name,
                         "mask_area": int(r.mask_area),
                         "score": float(r.score),
                         "rle": r.mask_rle}
                        for r in sub.itertuples(index=False)]
            _decoded = {}

            def _hit(fi, x, y):
                import math as _m
                if x is None or y is None:
                    return []
                px, py = _m.floor(x), _m.floor(y)
                if not (0 <= px < inputs.width and 0 <= py < inputs.height):
                    return []
                hits = []
                for m in mask_index.get(fi, []):
                    key = (fi, m["object_id"])
                    dec = _decoded.get(key)
                    if dec is None:
                        from .demo_video import _decode_rle
                        dec = _decode_rle(m["rle"])
                        _decoded[key] = dec
                    if dec[py, px]:
                        hits.append(m)
                # SAME deterministic ordering as the analytical layer
                hits.sort(key=lambda h: (h["mask_area"], -h["score"],
                                         h["object_id"]))
                return hits

            pres_rows, pres_policy = pres_mod.build_timeline(
                fused, frame_ts, mask_hit=_hit)
            pres_counts = pres_mod.method_counts(pres_rows)
            pres_mod.write_presentation_csv(pres_rows,
                                            tmp / "demo_presentation.csv")

            # ---- ANALYTICAL dwell (raw evidence only) ----
            support = dwell_mod.support_intervals(
                fused, inputs.interval_start_ns, inputs.interval_end_ns,
                pres_policy["max_interpolation_gap_ns"])
            acc = dwell_mod.accumulate(support, frame_ts,
                                       inputs.interval_end_ns, obj_meta)
            bars = dwell_mod.panel_rows(acc, obj_meta)
            dwell_rows = dwell_mod.summary_rows(acc, obj_meta)
            dwell_mod.write_summary_csv(dwell_rows,
                                        tmp / "object_dwell_summary.csv")

            cursor_by_frame, panel_by_frame = {}, {}
            for r in pres_rows:
                fi = r["frame_index"]
                cum = acc["cumulative_ns_by_frame"][fi]
                agg = {}
                for b in bars:
                    if b["aggregate"]:
                        agg[b["key"]] = sum(cum.get(o, 0)
                                            for o in b["aggregate"])
                    else:
                        agg[b["key"]] = cum.get(b["key"], 0)
                panel_by_frame[fi] = {
                    "rows": bars, "cumulative": agg,
                    "bar_scale_ns": acc["bar_scale_ns"],
                    "no_hit_ns": cum.get(dwell_mod.NO_HIT_ID, 0),
                    "valid_supported_ns_so_far": sum(cum.values()),
                    "active_id": r["presentation_primary_object_id"],
                }
                if r["presentation_cursor_visible"]:
                    pid = r["presentation_primary_object_id"]
                    cursor_by_frame[fi] = {
                        "x": float(r["display_gaze_x_px"]),
                        "y": float(r["display_gaze_y_px"]),
                        "primary_object_id": pid}
            from .demo_video import render_demo
            demo_info = render_demo(
                inputs.video_path, tmp / "gaze_demo.mp4",
                n_frames=inputs.n_frames,
                objects_by_frame=objects_by_frame,
                cursor_by_frame=cursor_by_frame,
                panel_by_frame=panel_by_frame,
                display_note=True,
                has_audio=inputs.has_audio,
                progress_cb=lambda f: report("rendering demonstration video",
                                             0.62 + 0.32 * f))
            demo_info["presentation"] = pres_counts
            validation.append({
                "check": "presentation timeline covers every video frame",
                "ok": len(pres_rows) == inputs.n_frames,
                "detail": f"{len(pres_rows)} rows for {inputs.n_frames} "
                          f"frames; {pres_counts['frames_with_cursor']} with "
                          f"a cursor ({pres_counts['exact']} exact, "
                          f"{pres_counts['linear_interpolation']} "
                          f"interpolated)"})
            final_cum = acc["cumulative_ns_by_frame"][-1] \
                if acc["cumulative_ns_by_frame"] else {}
            validation.append({
                "check": "final cumulative dwell equals the dwell summary",
                "ok": all(abs(final_cum.get(k, 0) - v) <= 1
                          for k, v in acc["totals_ns"].items()),
                "detail": f"{len(acc['totals_ns'])} attribution targets"})
            validation.append({
                "check": "dwell derives only from raw gaze evidence",
                "ok": True,
                "detail": "support intervals built from authoritative rows "
                          "and raw primary_object_id; interpolated display "
                          "positions contribute no duration"})
            validation.append({
                "check": "demo video time-faithful "
                         "(frame count + VFR pts vs source)",
                "ok": True,
                "detail": f"{demo_info['frames']} frames, max dev "
                          f"{demo_info['max_pts_dev_ms']} ms, constant "
                          f"offset {demo_info['pts_offset_ms']} ms"})

        report("verifying parents unchanged", 0.96)
        rehash = inputs.parent_hashes_now()
        validation.append({
            "check": "parent files unchanged during fusion (sha256)",
            "ok": rehash == hashes_before,
            "detail": "video/gaze/world_timestamps re-hashed identical"})
        tracked_manifest_sha = sha256_file(
            Path(inputs.tracked.artifact_dir) / "tracking_manifest.json")
        objects_sha = sha256_file(Path(inputs.tracked.objects_table_path))
        files_sha256 = {
            "gaze_samples_parquet": sha256_file(tmp / "gaze_samples.parquet"),
            "gaze_samples_csv": sha256_file(tmp / "gaze_samples.csv"),
            "internal_table": sha256_file(tmp / "gaze_objects.parquet"),
        }
        if demo_info is not None:
            files_sha256["demo"] = sha256_file(tmp / "gaze_demo.mp4")
            files_sha256["demo_presentation"] = sha256_file(
                tmp / "demo_presentation.csv")
            files_sha256["dwell_summary"] = sha256_file(
                tmp / "object_dwell_summary.csv")
        failed = [v for v in validation if not v["ok"]]
        if failed:
            raise ArtifactError("fusion validation failed:\n" + "\n".join(
                f"  x {v['check']}: {v['detail']}" for v in failed))

        import uuid
        manifest = {
            "schema_version": FUSION_SCHEMA_VERSION,
            "artifact_type": FUSION_ARTIFACT_TYPE,
            "artifact_id": uuid.uuid4().hex,
            "parent_artifact": {
                "artifact_id": inputs.tracked.artifact_id,
                "artifact_type": "tracked_objects",
                "artifact_dir": str(inputs.tracked.artifact_dir),
                "tracking_manifest_sha256": tracked_manifest_sha,
                "objects_table_sha256": objects_sha,
                "grandparent_artifact_id":
                    inputs.tracked.manifest["parent_artifact"]["artifact_id"],
                "recording_id":
                    inputs.tracked.manifest["parent_artifact"].get(
                        "recording_id"),
            },
            "created_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(timespec="seconds"),
            "tool": {"name": "gaze_object_fusion", "version": __version__},
            "files": ({"manifest": FUSION_MANIFEST,
                       "gaze_samples_parquet": "gaze_samples.parquet",
                       "gaze_samples_csv": "gaze_samples.csv",
                       "internal_table": "gaze_objects.parquet"}
                      | ({"demo": "gaze_demo.mp4",
                          "demo_presentation": "demo_presentation.csv",
                          "dwell_summary": "object_dwell_summary.csv"}
                         if demo_info else {})),
            "files_sha256": files_sha256,
            "presentation_layer": ({
                "note": pres_mod.SIDECAR_NOTE,
                "authoritative": False,
                "applies_to": "gaze_demo.mp4 and demo_presentation.csv only",
                "method": "per video frame: exact raw sample when a "
                          "timestamp matches, else linear interpolation "
                          "between the bracketing valid in-frame samples; "
                          "nearest-sample hold at the boundaries limited to "
                          "half a normal interval; no cursor otherwise",
                "interpolation_formula":
                    "alpha = (T - t0) / (t1 - t0); "
                    "display = p0 + alpha * (p1 - p0)",
                "breaks_on": "invalid, outside-frame or unassigned samples "
                             "and gaps beyond the adaptive limit",
                "gap_policy": pres_policy,
                "counts": pres_counts,
                "presentation_primary_object_id":
                    "hit-tested per exact video frame for the HIGHLIGHT "
                    "only; it never overwrites raw primary_object_id and "
                    "never contributes dwell time",
            } if demo_info else {"applies_to": None}),
            "dwell": ({
                "source": "authoritative raw gaze samples and raw "
                          "primary_object_id only",
                "support_interval":
                    "each valid, assigned, in-frame sample is supported "
                    "from the midpoint to the previous valid sample to the "
                    "midpoint to the next, each side capped at half the "
                    "adaptive gap limit, clipped to the validated trim "
                    "interval; unsupported middles of long gaps stay "
                    "unattributed",
                "gap_limit_ns": pres_policy["max_interpolation_gap_ns"],
                "attribution":
                    "on_object -> raw primary_object_id; valid background "
                    "-> no-tracked-object bucket; invalid/outside/"
                    "no_matching_frame -> nothing. Overlapping masks never "
                    "double-count: only the deterministic primary receives "
                    "the interval",
                "denominator": "valid supported gaze = sum of all attributed "
                               "support durations",
                "cumulative_rule": "frame N integrates support through the "
                                   "END boundary of frame N",
                "ordering": "fixed for the whole video by FINAL dwell "
                            "duration, descending; never reordered during "
                            "playback",
                "bar_scale_ns": acc["bar_scale_ns"],
                "bar_scale_rule": "shared fixed scale = largest final "
                                  "individual-object dwell",
                "aggregation": f"top {dwell_mod.TOP_N} objects shown; the "
                               f"remainder aggregated as "
                               f"'{dwell_mod.OTHER_GROUP}'",
                "valid_supported_ns": acc["valid_supported_ns"],
                "no_hit_ns": acc["totals_ns"].get(dwell_mod.NO_HIT_ID, 0),
            } if demo_info else {"source": None}),
            "table_schema": {
                "columns": PUBLIC_COLUMNS,
                "note": "gaze_samples.csv and gaze_samples.parquet are the "
                        "AUTHORITATIVE public table: identical rows, "
                        "values, ordering, and null semantics in both "
                        "formats; gaze_objects.parquet is internal "
                        "(extra working columns, no stability promise)",
                "csv_encoding": "ints exact integer text; nulls empty; "
                                "booleans True/False; floats shortest "
                                "round-trip repr; list fields JSON arrays",
                "gaze_status_values": GAZE_STATUSES,
                "status_precedence": "no_matching_frame > invalid_gaze > "
                                     "outside_frame > on_object/background",
            },
            "parent_recording": {
                "source_dir": str(inputs.parent_dir),
                "video": inputs.video_path.name,
                "gaze": inputs.gaze_path.name,
                "world_timestamps": inputs.world_path.name,
                "hashes_sha256": hashes_before,
            },
            "assignment_rule": {
                "method": "timestamp_defined_frame_cells",
                "description":
                    "frame i owns [b[i], b[i+1]); interior boundaries are "
                    "integer midpoints (floor) between adjacent world "
                    "timestamps; b[0]/b[n] are the validated parent trim "
                    "interval boundaries (selection.absolute_start_ns / "
                    "selection.absolute_end_boundary_ns); samples outside "
                    "[b[0], b[n]) get no_matching_frame; NO global "
                    "frame-distance threshold is used",
                "interval_start_ns": inputs.interval_start_ns,
                "interval_end_ns": inputs.interval_end_ns,
                "sanity_gap_ms": sanity_gap_ms,
                "sanity_gap_basis":
                    "NEON scene VFR intervals observed 13.6-50 ms; 200 ms "
                    "= 4x the largest normal interval; threshold triggers "
                    "REPORTING only, never sample dropping",
                "anomalous_gaps": extras["gaps"],
                "world_gap_stats_ms": cells.gap_stats_ms(),
                "pixel_convention":
                    "gaze (x, y) lies in pixel (floor(x), floor(y)); "
                    "out-of-frame coordinates are flagged, never clamped",
                "hit_order":
                    "1) smaller mask area  2) higher tracking score  "
                    "3) stable object id; primary object == first entry",
                "background_definition":
                    "gaze_status = background means: a VALID gaze sample "
                    "inside the video frame whose gaze pixel intersects "
                    "NONE of the accepted tracked-object masks on its "
                    "assigned frame. This does NOT prove the participant "
                    "viewed physical background — the gaze may fall on an "
                    "untracked or unlabeled object. To attribute such "
                    "gaze, select the object in the tracking GUI (point "
                    "prompts, open vocabulary — provisional names like "
                    "'unknown-tool-01' are valid), track it, and rerun "
                    "fusion; object_id values are stable within this "
                    "recording's tracked artifact but NOT automatically "
                    "global across recordings",
                "demo_cursor_rule":
                    "per frame: valid row assigned to that exact frame with "
                    "smallest |frame_offset_ms| (tie -> earliest "
                    "timestamp); valid = gaze_valid (worn, finite) AND "
                    "coordinates inside the frame; no valid row -> no "
                    "cursor; adjacent-frame samples never borrowed",
                "demo_cursor_design":
                    "ONE fixed high-contrast cursor for every valid "
                    "sample, independent of the hit object: bright green "
                    "ring (~18 px radius at 1600x1200, scaled by "
                    "min(width,height), ~3.8 px thick) over a black outer "
                    "halo, small red center dot with a thin white edge; "
                    "identical on objects and on background; no trail, "
                    "pulse, smoothing, or interpolation",
                "demo_highlight_rule":
                    "object-hit meaning is conveyed EXCLUSIVELY by the "
                    "mask treatment: the frame's primary gazed-at object "
                    "gets a bright 50% colored fill + 3 px boundary + "
                    "name/id label; all other tracked objects get a "
                    "subtle 12% fill + 1 px boundary; background gaze -> "
                    "no bright highlight; invalid/out-of-frame gaze -> "
                    "no cursor (subtle masks only)",
            },
            "video": dict(inputs.tracked.manifest["video"]),
            # object catalog: stable in-recording IDs, user-chosen names,
            # deterministic display colors (provenance lives HERE once —
            # deliberately not duplicated into every table row)
            "objects": [
                {"object_id": o["object_id"], "name": o["name"],
                 "color": o["color"], "review_status": o.get("status")}
                for o in ann.get("objects", []) if not o.get("deleted")],
            "counts": counts,
            "demo": ({"present": True, **demo_info} if demo_info
                     else {"present": False}),
            "validation": validation,
            "output_directory": str(pub.out_dir),
        }
        (tmp / FUSION_MANIFEST).write_text(json.dumps(manifest, indent=2)
                                           + "\n")
        report("publishing", 0.99)
        pub.publish()
        report("done", 1.0)
        return manifest
    finally:
        pub.cleanup()


# --------------------------------------------------------------------------

class GazeFusionArtifact:
    def __init__(self, artifact_dir: Path, manifest: dict, files: dict):
        self.artifact_dir = artifact_dir
        self.manifest = manifest
        self.gaze_samples_parquet_path = files["gaze_samples_parquet"]
        self.gaze_samples_csv_path = files["gaze_samples_csv"]
        self.internal_table_path = files.get("internal_table")
        self.demo_path = files.get("demo")

    @property
    def artifact_id(self):
        return self.manifest["artifact_id"]


def load_fusion_artifact(path) -> GazeFusionArtifact:
    """Open + validate a gaze_object_fusion artifact (public contract)."""
    art_dir, manifest = read_manifest(path, FUSION_MANIFEST)
    where = str(art_dir / FUSION_MANIFEST)
    require(manifest, where, artifact_type=FUSION_ARTIFACT_TYPE,
            schema_major="1")
    files_decl = manifest.get("files")
    if not isinstance(files_decl, dict):
        raise ArtifactError(f"{where}: missing files section")
    files = {k: resolve_declared_file(art_dir, rel, key=k, where=where)
             for k, rel in files_decl.items()}
    for key in ("gaze_samples_parquet", "gaze_samples_csv"):
        if key not in files:
            raise ArtifactError(f"{where}: files must declare {key} "
                                f"(authoritative public table, schema 1.1)")
    if not manifest.get("validation"):
        raise ArtifactError(f"{where}: no validation record")
    if any(not v.get("ok") for v in manifest["validation"]):
        raise ArtifactError(f"{where}: validation record contains failures")
    if not manifest.get("parent_artifact", {}).get("artifact_id"):
        raise ArtifactError(f"{where}: missing parent_artifact reference")
    if manifest.get("demo", {}).get("present") and "demo" not in files:
        raise ArtifactError(f"{where}: manifest claims a demo video but "
                            f"files does not declare one")
    rule = manifest.get("assignment_rule", {})
    if rule.get("method") != "timestamp_defined_frame_cells":
        raise ArtifactError(f"{where}: unknown assignment_rule.method "
                            f"{rule.get('method')!r}")
    if manifest.get("table_schema", {}).get("columns") != PUBLIC_COLUMNS:
        raise ArtifactError(f"{where}: table_schema.columns does not match "
                            f"the agreed public schema")

    # file integrity: hashes recorded at publish time must match
    recorded = manifest.get("files_sha256")
    if not isinstance(recorded, dict) \
            or "gaze_samples_parquet" not in recorded \
            or "gaze_samples_csv" not in recorded:
        raise ArtifactError(f"{where}: missing files_sha256 for the "
                            f"gaze_samples table files")
    for key, sha in recorded.items():
        if key not in files:
            raise ArtifactError(f"{where}: files_sha256 names undeclared "
                                f"file {key!r}")
        if sha256_file(files[key]) != sha:
            raise ArtifactError(f"{where}: {key} content does not match its "
                                f"recorded sha256 — artifact corrupted or "
                                f"tampered")

    import pandas as pd
    pub = pd.read_parquet(files["gaze_samples_parquet"])
    if list(pub.columns) != PUBLIC_COLUMNS:
        raise ArtifactError(f"{where}: gaze_samples.parquet columns/order "
                            f"differ from the agreed public schema")
    # csv is the same logical table: cheap structural agreement here
    # (full field-level equality is covered by the contract tests)
    with open(files["gaze_samples_csv"]) as f:
        header = f.readline().strip()
        n_csv = sum(1 for _ in f)
    if header.split(",") != PUBLIC_COLUMNS:
        raise ArtifactError(f"{where}: gaze_samples.csv header differs "
                            f"from the agreed public schema")
    if n_csv != len(pub):
        raise ArtifactError(f"{where}: gaze_samples.csv row count {n_csv} "
                            f"!= parquet rows {len(pub)}")

    counts = manifest.get("counts", {})
    if counts.get("samples_total") != len(pub):
        raise ArtifactError(f"{where}: counts.samples_total "
                            f"{counts.get('samples_total')} != table rows "
                            f"{len(pub)}")
    if list(pub["gaze_sample_index"]) != list(range(len(pub))):
        raise ArtifactError(f"{where}: gaze_sample_index is not the stable "
                            f"0..n-1 source ordering")
    if bool((~pub["gaze_status"].isin(GAZE_STATUSES)).any()):
        raise ArtifactError(f"{where}: unknown gaze_status values")
    n_assigned = int(pub["frame_index"].notna().sum())
    if counts.get("samples_assigned") != n_assigned:
        raise ArtifactError(f"{where}: counts.samples_assigned inconsistent "
                            f"with the table")
    n_nomatch = int((pub["gaze_status"] == NO_MATCHING_FRAME).sum())
    if n_assigned + n_nomatch != len(pub):
        raise ArtifactError(f"{where}: frame assignment does not partition "
                            f"the samples (assigned + no_matching_frame != "
                            f"total)")
    # status/boolean/hit consistency
    if bool((pub["is_gaze_at_object"]
             != (pub["gaze_status"] == "on_object")).any()):
        raise ArtifactError(f"{where}: is_gaze_at_object disagrees with "
                            f"gaze_status")
    if bool((pub["hit_object_count"]
             != pub["hit_object_ids"].map(len)).any()):
        raise ArtifactError(f"{where}: hit_object_count != "
                            f"len(hit_object_ids)")
    hits = pub[pub["hit_object_count"] > 0]
    if len(hits):
        if bool((hits["gaze_status"] != "on_object").any()):
            raise ArtifactError(f"{where}: rows with hits must be "
                                f"on_object")
        if bool((hits["primary_object_id"]
                 != hits["hit_object_ids"].map(lambda h: h[0])).any()):
            raise ArtifactError(f"{where}: primary_object_id != first hit "
                                f"list entry")
    none = pub[pub["hit_object_count"] == 0]
    if len(none) and not bool(none["primary_object_id"].isna().all()):
        raise ArtifactError(f"{where}: primary_object_id set without hits")
    n_frames = int(manifest.get("video", {}).get("n_frames", 0))
    assigned = pub[pub["frame_index"].notna()]
    if len(assigned) and n_frames:
        fi = assigned["frame_index"].astype("int64")
        if ((fi < 0) | (fi >= n_frames)).any():
            raise ArtifactError(f"{where}: assigned frame_index outside "
                                f"0..{n_frames - 1}")
    lo = rule.get("interval_start_ns")
    hi = rule.get("interval_end_ns")
    if len(assigned) and isinstance(lo, int) and isinstance(hi, int):
        ts = assigned["timestamp_ns"]
        if bool(((ts < lo) | (ts >= hi)).any()):
            raise ArtifactError(f"{where}: assigned samples outside the "
                                f"recorded trim interval")
    return GazeFusionArtifact(art_dir, manifest, files)
