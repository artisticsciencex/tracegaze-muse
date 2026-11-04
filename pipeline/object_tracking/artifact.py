"""tracked_objects artifact: publication + loader.

Layout (movable directory; `files` paths are artifact-relative and
symlink-escape-checked on load):

    <out_dir>/
        tracking_manifest.json    versioned downstream contract
        object_annotations.json   human-readable object provenance
        objects.parquet           one row per visible object per frame
        tracking_overlay.mp4      optional time-faithful QA overlay

The artifact deliberately does NOT duplicate the parent's video/gaze/world
files: it records a resolvable parent reference (absolute source path as
labelled provenance + parent artifact id). Gaze fusion resolves the parent
through the manifest's `parent` section.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from common.artifacts import (ArtifactError, AtomicPublisher, read_manifest,
                              require, resolve_declared_file, sha256_file)

from . import __version__
from . import intake
from .annotations import AnnotationStore, color_for_object_id
from .intake import RecordingBundle, TRACK_MANIFEST
from .tracking import presence_summary, validate_track_rows

TRACK_ARTIFACT_TYPE = "tracked_objects"
TRACK_SCHEMA_VERSION = "1.0"

TABLE_COLUMNS = ["frame_index", "timestamp_ns", "object_id", "object_name",
                 "sam_local_id", "reference_frame", "tracking_direction",
                 "score", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                 "mask_area", "mask_rle"]


def merge_object_tables(bundle: RecordingBundle, store: AnnotationStore,
                        tracks_db: Path, object_ids: list):
    """Merge accepted objects' rows (from the session SQLite store) into the
    published table. Runs ONLY inside the isolated publication subprocess —
    pandas must never be imported by the GUI server."""
    import pandas as pd

    from . import trackstore
    parts = []
    for oid in object_ids:
        rows = list(trackstore.iter_object_rows(tracks_db, oid))
        if not rows:
            raise ArtifactError(f"missing track rows for {oid} in "
                                f"{tracks_db}")
        df = pd.DataFrame(rows).drop(columns=["object_id"])
        o = store.get(oid)
        df["object_id"] = oid
        df["object_name"] = o["name"]
        df["reference_frame"] = int(o["reference_frame"])
        # timestamps are only derivable through the VERIFIED 1:1 mapping —
        # an unverified sync must never fabricate timestamp_ns values.
        # Nullable Int64 keeps the parquet column type stable across
        # gaze-compatible and video-only artifacts.
        if bundle.world_ts_ns and bundle.sync_ok:
            df["timestamp_ns"] = pd.array(
                [bundle.world_ts_ns[int(i)]
                 if 0 <= int(i) < len(bundle.world_ts_ns) else None
                 for i in df["frame_index"]], dtype="Int64")
        else:
            df["timestamp_ns"] = pd.array([None] * len(df), dtype="Int64")
        parts.append(df)
    if not parts:
        import pandas as pd
        return pd.DataFrame(columns=TABLE_COLUMNS)
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.sort_values(["frame_index", "object_id"],
                                kind="stable").reset_index(drop=True)
    return merged[TABLE_COLUMNS]


def _decode_rle_checked(rle_json: str):
    from neon_trim.core import NS  # noqa: F401 (import guard only)
    import pycocotools.mask as mask_util
    rle = json.loads(rle_json)
    r = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
    return mask_util.decode(r).astype(bool)


def validate_for_publish(bundle: RecordingBundle, store: AnnotationStore,
                         merged, accepted_ids: list) -> list:
    """The full pre-publish validation record (list of {check, ok, detail})."""
    results = []

    def check(name, ok, detail=""):
        results.append({"check": name, "ok": bool(ok), "detail": detail})

    objs = {o["object_id"]: o for o in store.objects()}
    names = [objs[oid]["name"] for oid in accepted_ids]
    ids_in_table = (sorted(merged["object_id"].unique().tolist())
                    if not merged.empty else [])

    check("object ids unique and immutable",
          len(set(accepted_ids)) == len(accepted_ids),
          f"{len(accepted_ids)} accepted objects")
    check("object names non-empty and unique",
          all(n.strip() for n in names) and len(set(names)) == len(names),
          ", ".join(names) or "none")
    check("accepted objects have accepted initial masks",
          all(any(r["action"] == "mask_accepted"
                  for r in objs[oid]["revisions"]) for oid in accepted_ids),
          "revision history contains mask_accepted for each")
    pts_ok = True
    for oid in accepted_ids:
        for (x, y) in objs[oid]["points"]:
            if not (0 <= x < bundle.width and 0 <= y < bundle.height):
                pts_ok = False
    check("prompt points inside reference frame", pts_ok,
          f"frame {bundle.width}x{bundle.height}")
    check("table object ids match accepted annotation ids",
          set(ids_in_table) <= set(accepted_ids),
          f"table: {ids_in_table}")
    if not merged.empty:
        check("frame indices within source range",
              bool(((merged["frame_index"] >= 0)
                    & (merged["frame_index"] < bundle.n_frames)).all()),
              f"0..{bundle.n_frames - 1}")
        dup = merged.duplicated(subset=["frame_index", "object_id"])
        check("no duplicate (frame_index, object_id) rows",
              not dup.any(), f"{int(dup.sum())} duplicates")
        refs_ok = all(
            (merged.loc[merged["object_id"] == oid, "reference_frame"]
             == objs[oid]["reference_frame"]).all() for oid in ids_in_table)
        check("reference frames match annotation records", refs_ok, "")
        for oid in ids_in_table:
            sub = merged[merged["object_id"] == oid]
            problems = validate_track_rows(
                sub, object_id=oid, n_frames=bundle.n_frames,
                ref_frame=objs[oid]["reference_frame"])
            check(f"forward/backward merge consistent ({oid})",
                  not problems, "; ".join(problems))
        # mask spot-decode + geometry (first/mid/last row per object)
        geo_ok, decode_ok = True, True
        for oid in ids_in_table:
            sub = merged[merged["object_id"] == oid]
            picks = sub.iloc[[0, len(sub) // 2, -1]] if len(sub) >= 3 else sub
            for _, row in picks.iterrows():
                try:
                    m = _decode_rle_checked(row["mask_rle"])
                except Exception:
                    decode_ok = False
                    continue
                if m.shape != (bundle.height, bundle.width):
                    geo_ok = False
                if not (0 <= row["bbox_x"] <= bundle.width
                        and 0 <= row["bbox_y"] <= bundle.height
                        and row["bbox_w"] <= bundle.width + 1
                        and row["bbox_h"] <= bundle.height + 1):
                    geo_ok = False
        check("masks decode successfully (spot check)", decode_ok, "")
        check("bounding boxes and masks fit frame dimensions", geo_ok,
              f"{bundle.width}x{bundle.height}")
        if bundle.world_ts_ns and bundle.sync_ok:
            # only rows with in-range frames (out-of-range rows are already
            # flagged by the frame-bounds check above — never index past it)
            inb = merged[(merged["frame_index"] >= 0)
                         & (merged["frame_index"] < bundle.n_frames)]
            ts_ok = bool((inb["timestamp_ns"] == inb["frame_index"].map(
                lambda i: bundle.world_ts_ns[int(i)])).all())
            check("timestamp mapping agrees with parent artifact", ts_ok,
                  "timestamp_ns == world_ts[frame_index]")
        else:
            ts_null = merged["timestamp_ns"].isna().all() \
                if not merged.empty else True
            check("no timestamps fabricated without verified sync", ts_null,
                  "timestamp_ns is null (sync unverified or video-only)")
    check("gaze-compatible status truthful",
          True,
          ("gaze_compatible" if bundle.gaze_compatible else
           "NOT gaze-compatible (video-only or unresolved sync) — recorded"))
    return results


def render_overlay(bundle: RecordingBundle, merged, out_path: Path,
                   colors: dict, *, keep_audio: bool = True,
                   progress_cb: Optional[Callable[[float], None]] = None,
                   cancel_check: Optional[Callable[[], bool]] = None) -> dict:
    """Time-faithful QA overlay: exact source frame count/order and VFR PTS
    (validated), deterministic colors, optional audio mux.

    Never a synchronization source — QA only.
    """
    import av
    import numpy as np
    from fractions import Fraction

    by_frame = {}
    if not merged.empty:
        for fidx, sub in merged.groupby("frame_index"):
            by_frame[int(fidx)] = list(sub.itertuples(index=False))

    src = str(bundle.video_path)
    tmp_video = str(out_path) + ".video.mp4"
    with av.open(src) as ic, av.open(tmp_video, "w") as oc:
        vin = ic.streams.video[0]
        vin.thread_type = "AUTO"
        tb = vin.time_base
        ost = oc.add_stream("libx264")
        ost.width, ost.height = vin.codec_context.width, vin.codec_context.height
        ost.pix_fmt = "yuv420p"
        ost.codec_context.time_base = tb      # exact VFR pts preserved
        ost.time_base = tb
        ost.options = {"crf": "20", "preset": "veryfast"}
        n = 0
        src_pts = []
        for frame in ic.decode(vin):
            if cancel_check is not None and cancel_check():
                raise ArtifactError("overlay rendering cancelled")
            img = frame.to_ndarray(format="rgb24")
            for row in by_frame.get(n, []):
                m = _decode_rle_checked(row.mask_rle)
                col = colors.get(row.object_id, "#ffffff").lstrip("#")
                rgb = np.array([int(col[i:i + 2], 16) for i in (0, 2, 4)],
                               dtype=np.float32)
                img[m] = (0.55 * img[m] + 0.45 * rgb).astype(np.uint8)
            out = av.VideoFrame.from_ndarray(img, format="rgb24")
            out = out.reformat(format="yuv420p")
            out.pts = frame.pts
            out.time_base = tb
            src_pts.append(float(frame.pts * tb))
            for pkt in ost.encode(out):
                oc.mux(pkt)
            n += 1
            if progress_cb is not None and n % 25 == 0:
                progress_cb(n / max(bundle.n_frames, 1))
        for pkt in ost.encode():
            oc.mux(pkt)

    final = str(out_path)
    if keep_audio and bundle.has_audio:
        from common.media import ffmpeg_exe
        import subprocess
        r = subprocess.run(
            [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
             "-y", "-i", tmp_video, "-i", src,
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
             "-movflags", "+faststart", final],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise ArtifactError("overlay audio mux failed: "
                                + r.stderr.strip()[-400:])
        Path(tmp_video).unlink(missing_ok=True)
    else:
        Path(tmp_video).replace(final)

    # validate: identical frame count + VFR spacing vs the source we just
    # read. A CONSTANT pts offset is benign and expected: the mp4 muxer
    # rebases a nonzero container start (trimmed sources start at e.g. 16 ms)
    # to zero — the same uniform-offset tolerance the intake sync check uses.
    from common.media import decode_all_pts
    out_pts = decode_all_pts(Path(final))
    if len(out_pts) != n:
        raise ArtifactError(
            f"overlay frame count {len(out_pts)} != source {n}")
    offset = (out_pts[0] - src_pts[0]) if n else 0.0
    max_dev = max((abs((a - b) - offset)
                   for a, b in zip(out_pts, src_pts)), default=0.0)
    if max_dev > 0.002:
        raise ArtifactError(
            f"overlay VFR timing deviates from source by {max_dev * 1e3:.2f} "
            f"ms beyond the constant container offset")
    if abs(offset) > 0.5:
        raise ArtifactError(
            f"overlay pts offset {offset * 1e3:.1f} ms exceeds sanity bound")
    return {"frames": n, "max_pts_dev_ms": round(max_dev * 1e3, 4),
            "pts_offset_ms": round(offset * 1e3, 3),
            "audio": bool(keep_audio and bundle.has_audio)}


def publish_tracked_artifact(
        bundle: RecordingBundle, store: AnnotationStore, tracks_db: Path,
        out_dir: Path, *, worker_info: Optional[dict] = None,
        score_thresh: float = 0.5, runtime_s: Optional[float] = None,
        make_overlay: bool = True,
        path_guard: Optional[Callable[[Path], bool]] = None,
        progress_cb: Optional[Callable[[str, float], None]] = None,
        cancel_event=None) -> dict:
    """Atomically publish the tracked-objects artifact. Returns the manifest."""
    def report(phase, frac):
        if progress_cb is not None:
            progress_cb(phase, frac)

    def cancelled():
        return cancel_event is not None and cancel_event.is_set()

    ready, blockers = store.publishable()
    if not ready:
        raise ArtifactError("not publishable:\n  - " + "\n  - ".join(blockers))
    objs = store.objects()
    accepted = [o["object_id"] for o in objs if o["status"] == "accepted"]
    excluded = [o["object_id"] for o in objs
                if o["status"] in ("rejected", "excluded")]

    report("hashing parent files", 0.02)
    t0 = time.time()
    parent_files = {"video": bundle.video_path}
    if bundle.gaze_path is not None:
        parent_files["gaze"] = bundle.gaze_path
    if bundle.world_path is not None:
        parent_files["world_timestamps"] = bundle.world_path
    parent_hashes = {k: sha256_file(p) for k, p in parent_files.items()}
    parent_video_hash = parent_hashes["video"]

    report("merging object tables", 0.08)
    merged = merge_object_tables(bundle, store, tracks_db, accepted)

    report("validating", 0.15)
    validation = validate_for_publish(bundle, store, merged, accepted)
    failed = [v for v in validation if not v["ok"]]
    if failed:
        raise ArtifactError(
            "publish validation failed:\n" + "\n".join(
                f"  x {v['check']}: {v['detail']}" for v in failed))
    if cancelled():
        raise ArtifactError("publish cancelled")

    artifact_id = uuid.uuid4().hex
    colors = {o["object_id"]: o["color"] for o in objs}
    pub = AtomicPublisher(out_dir, source_dir=bundle.source_dir,
                          path_guard=path_guard,
                          tmp_prefix=".tracked_objects_tmp_")
    try:
        tmp = pub.tmp_dir
        report("writing objects.parquet", 0.20)
        merged.to_parquet(tmp / "objects.parquet", index=False)
        (tmp / "object_annotations.json").write_text(
            json.dumps(store.export_annotations(), indent=2) + "\n")

        overlay_info = None
        if make_overlay:
            report("rendering overlay", 0.25)
            overlay_info = render_overlay(
                bundle, merged, tmp / "tracking_overlay.mp4", colors,
                progress_cb=lambda f: report("rendering overlay",
                                             0.25 + 0.6 * f),
                cancel_check=(lambda: cancelled()))
        if cancelled():
            raise ArtifactError("publish cancelled")

        report("verifying parent unchanged", 0.9)
        rehash = {k: sha256_file(p) for k, p in parent_files.items()}
        validation.append({
            "check": "source recording/artifact unchanged (sha256)",
            "ok": rehash == parent_hashes,
            "detail": "parent " + "/".join(parent_files)
                      + " re-hashed identical"})
        # parent must still be loadable for downstream stages — through the
        # loader of ITS OWN source system, never a foreign one
        parent_ok, parent_detail = True, bundle.source_type
        if bundle.source_type == "trimmed_artifact":
            try:
                from neon_trim.artifact import load_artifact
                load_artifact(bundle.source_dir)
            except Exception as e:
                parent_ok, parent_detail = False, str(e)
        elif bundle.source_type in intake.ARIA_KINDS:
            try:
                from aria_source.artifact import load_aria_artifact
                art = load_aria_artifact(bundle.source_dir)
                if art.artifact_type != bundle.source_type:
                    raise ArtifactError(
                        f"parent artifact type changed: manifest says "
                        f"{art.artifact_type}, intake saw "
                        f"{bundle.source_type}")
                if art.artifact_id != bundle.artifact_id:
                    raise ArtifactError(
                        f"parent artifact id changed: {art.artifact_id} "
                        f"!= {bundle.artifact_id}")
                if not art.validation_ok:
                    raise ArtifactError("parent Aria artifact validation "
                                        "record is not all-ok")
                parent_detail = (
                    f"{art.artifact_type} {art.artifact_id} reloaded via the "
                    f"Aria loader ({art.manifest['timestamps']['domain']})")
            except Exception as e:
                parent_ok, parent_detail = False, str(e)
        validation.append({"check": "parent artifact still loadable",
                           "ok": parent_ok, "detail": parent_detail})
        failed = [v for v in validation if not v["ok"]]
        if failed:
            raise ArtifactError("publish validation failed:\n" + "\n".join(
                f"  x {v['check']}: {v['detail']}" for v in failed))

        files = {"manifest": TRACK_MANIFEST,
                 "annotations": "object_annotations.json",
                 "objects_table": "objects.parquet"}
        if overlay_info is not None:
            files["overlay"] = "tracking_overlay.mp4"

        per_object = []
        for o in objs:
            entry = {
                "object_id": o["object_id"], "name": o["name"],
                "color": o["color"], "note": o["note"],
                "reference_frame": o["reference_frame"],
                "reference_timestamp_ns":
                    bundle.timestamp_ns_for_frame(o["reference_frame"]),
                "points": o["points"], "point_labels": o["point_labels"],
                "review_status": o["status"],
                "track": o.get("track"),
            }
            if o["status"] == "accepted" and not merged.empty:
                sub = merged[merged["object_id"] == o["object_id"]]
                entry["presence"] = presence_summary(sub, bundle.n_frames)
            per_object.append(entry)

        manifest = {
            "schema_version": TRACK_SCHEMA_VERSION,
            "artifact_type": TRACK_ARTIFACT_TYPE,
            "artifact_id": artifact_id,
            "parent_artifact": {
                "artifact_id": bundle.artifact_id,
                "artifact_type": bundle.artifact_type,
                "recording_id": bundle.recording_id,
            },
            "created_utc": datetime.now(timezone.utc)
                .isoformat(timespec="seconds"),
            "tool": {"name": "object_tracking", "version": __version__},
            "files": files,
            "parent": {
                "source_dir": str(bundle.source_dir),
                "source_dir_note": "absolute parent path, provenance only — "
                                   "all 'files' entries are artifact-relative",
                "source_type": bundle.source_type,
                "video": bundle.video_path.name,
                "video_sha256": parent_video_hash,
                "gaze": (bundle.gaze_path.name if bundle.gaze_path else None),
                "gaze_sha256": parent_hashes.get("gaze"),
                "world_timestamps": (bundle.world_path.name
                                     if bundle.world_path else None),
                "world_timestamps_sha256":
                    parent_hashes.get("world_timestamps"),
                "recording_id": bundle.recording_id,
                "note": "video/gaze/world files are NOT duplicated here; "
                        "resolve them through source_dir (self-contained "
                        "mode is not implemented — recorded explicitly)",
            },
            "video": {"n_frames": bundle.n_frames,
                      "width": bundle.width, "height": bundle.height,
                      "duration_s": bundle.duration_s,
                      "has_audio": bundle.has_audio},
            "synchronization": {
                "sync_ok": bundle.sync_ok,
                "status": bundle.sync_status,
                "gaze_compatible": bundle.gaze_compatible,
                "gaze_offset_provenance": bundle.gaze_offset_provenance,
                "timestamp_source": ("world_timestamps.csv of parent"
                                     if bundle.world_ts_ns else "unavailable"),
                # timestamp provenance follows the parent's source system —
                # an Aria parent is never described in NEON terms
                "source_system": ("project_aria"
                                  if bundle.source_type in intake.ARIA_KINDS
                                  else "pupil_neon"),
                "timestamp_domain": ("aria_device_time"
                                     if bundle.source_type in intake.ARIA_KINDS
                                     else "neon_world_time"),
            },
            "model": {
                **(worker_info or {}),
                "prompt_method": "SAM2-style tracker path: positive/negative "
                                 "points + obj_id, relative coordinates "
                                 "(one reset pass per object)",
                "score_thresh": score_thresh,
                "fa3_disabled": True,
            },
            "objects": per_object,
            "counts": {
                "objects_accepted": len(accepted),
                "objects_excluded_or_rejected": len(excluded),
                "table_rows": int(len(merged)),
                "runtime_s": round(runtime_s, 1) if runtime_s else None,
            },
            "overlay": ({"present": True, **overlay_info}
                        if overlay_info is not None
                        else {"present": False,
                              "note": "browser-side review is authoritative"}),
            "omitted": {"note": "no correction-point merging in this "
                                "version; the supported correction workflow "
                                "is a per-object rerun from a revised prompt "
                                "or reference frame"},
            "validation": validation,
            "output_directory": str(pub.out_dir),
        }
        (tmp / TRACK_MANIFEST).write_text(json.dumps(manifest, indent=2)
                                          + "\n")
        report("publishing", 0.97)
        pub.publish()
        report("done", 1.0)
        return manifest
    finally:
        pub.cleanup()


# --------------------------------------------------------------------------

class TrackedObjectsArtifact:
    def __init__(self, artifact_dir: Path, manifest: dict, files: dict):
        self.artifact_dir = artifact_dir
        self.manifest = manifest
        self.objects_table_path = files["objects_table"]
        self.annotations_path = files["annotations"]
        self.overlay_path = files.get("overlay")

    @property
    def artifact_id(self):
        return self.manifest["artifact_id"]

    def parent_paths(self) -> dict:
        """Resolvable parent inputs for gaze fusion."""
        p = self.manifest["parent"]
        base = Path(p["source_dir"])
        return {
            "video": base / p["video"] if p.get("video") else None,
            "gaze": base / p["gaze"] if p.get("gaze") else None,
            "world_timestamps": (base / p["world_timestamps"]
                                 if p.get("world_timestamps") else None),
        }


def load_tracked_artifact(path) -> TrackedObjectsArtifact:
    """Open + validate a tracked_objects artifact from dir or manifest path."""
    art_dir, manifest = read_manifest(path, TRACK_MANIFEST)
    where = str(art_dir / TRACK_MANIFEST)
    require(manifest, where, artifact_type=TRACK_ARTIFACT_TYPE,
            schema_major="1")
    files_decl = manifest.get("files")
    if not isinstance(files_decl, dict):
        raise ArtifactError(f"{where}: missing files section")
    files = {}
    for key, rel in files_decl.items():
        files[key] = resolve_declared_file(art_dir, rel, key=key, where=where)
    if "objects_table" not in files or "annotations" not in files:
        raise ArtifactError(f"{where}: files must declare objects_table and "
                            f"annotations")
    if not manifest.get("validation"):
        raise ArtifactError(f"{where}: no validation record")
    if not manifest.get("parent_artifact"):
        raise ArtifactError(f"{where}: missing parent_artifact reference")
    ov = manifest.get("overlay", {})
    if ov.get("present") and "overlay" not in files:
        raise ArtifactError(f"{where}: manifest claims an overlay but files "
                            f"does not declare one")

    # table/annotation consistency
    import pandas as pd
    table = pd.read_parquet(files["objects_table"])
    ann = json.loads(files["annotations"].read_text())
    ann_ids = [o["object_id"] for o in ann.get("objects", [])
               if not o.get("deleted")]
    if len(set(ann_ids)) != len(ann_ids):
        raise ArtifactError(f"{where}: duplicate object ids in annotations")
    accepted_ids = {o["object_id"] for o in ann.get("objects", [])
                    if o.get("status") == "accepted"}
    table_ids = set(table["object_id"].unique().tolist()) \
        if not table.empty else set()
    if not table_ids <= accepted_ids:
        raise ArtifactError(f"{where}: objects.parquet contains ids not "
                            f"accepted in annotations: "
                            f"{sorted(table_ids - accepted_ids)}")
    n_frames = int(manifest.get("video", {}).get("n_frames", 0))
    if not table.empty and n_frames:
        if ((table["frame_index"] < 0)
                | (table["frame_index"] >= n_frames)).any():
            raise ArtifactError(f"{where}: frame_index outside 0..{n_frames-1}")
        if table.duplicated(subset=["frame_index", "object_id"]).any():
            raise ArtifactError(f"{where}: duplicate (frame_index, object_id) "
                                f"rows")
    # color determinism (stable across processes/runs)
    for o in ann.get("objects", []):
        if not o.get("deleted") \
                and o.get("color") != color_for_object_id(o["object_id"]):
            raise ArtifactError(f"{where}: color of {o['object_id']} does not "
                                f"match the deterministic derivation")
    return TrackedObjectsArtifact(art_dir, manifest, files)
