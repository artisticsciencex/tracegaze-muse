"""attention_trace artifact: build, publish atomically, load + validate.

The parent fusion artifact is only ever READ; its identity and file
hashes are recorded and re-checked so a trace can always be tied back to
the exact evidence it was derived from.
"""

from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from common.artifacts import (ArtifactError, AtomicPublisher, read_manifest,
                              require, resolve_declared_file, sha256_file)

from . import (ATTENTION_ARTIFACT_TYPE, ATTENTION_MANIFEST,
               ATTENTION_SCHEMA_VERSION, NO_TRACKED_OBJECT,
               NO_TRACKED_OBJECT_NAME, __version__)
from . import policy as pol
from . import trace as tr

EVIDENCE_NOTE = (
    "Derived exclusively from the authoritative gaze_samples table of the "
    "parent fusion artifact. The demonstration video and its "
    "presentation-only interpolated cursor (demo_presentation.csv) are "
    "never used as evidence."
)
NO_HIT_NOTE = (
    "__no_tracked_object__ means: a valid, in-frame gaze sample that hit "
    "none of the accepted tracked-object masks. It does NOT assert that "
    "the participant looked at physical background, and it never implies "
    "an unlabeled object."
)


def _fusion_objects(manifest: dict) -> dict:
    """Object catalog (id -> name/color) from the fusion manifest."""
    out = {}
    for o in manifest.get("objects") or []:
        oid = o.get("object_id")
        if oid:
            out[oid] = {"name": o.get("name"), "color": o.get("color"),
                        "review_status": o.get("review_status")}
    return out


def build_trace(fusion_dir, *, progress: Optional[Callable] = None) -> dict:
    """Compute the whole trace from a validated fusion artifact (read-only)."""
    import pandas as pd
    from gaze_object_fusion.artifact import load_fusion_artifact

    def report(phase, frac):
        if progress:
            progress(phase, frac)

    report("loading fusion artifact", 0.05)
    art = load_fusion_artifact(fusion_dir)      # full contract validation
    fm = art.manifest
    fused = pd.read_parquet(art.gaze_samples_parquet_path)

    rule = fm.get("assignment_rule") or {}
    lo, hi = rule.get("interval_start_ns"), rule.get("interval_end_ns")
    if not isinstance(lo, int) or not isinstance(hi, int) or hi <= lo:
        raise ArtifactError(
            "parent fusion manifest lacks a usable validated interval "
            "(assignment_rule.interval_start_ns / interval_end_ns)")

    report("selecting eligible samples", 0.25)
    rows = pol.eligible_rows(fused)
    breaks = pol.breaking_rows(fused)
    policy = pol.gap_policy([r["ts_ns"] for r in rows])
    supported, dropped = pol.support_intervals(
        rows, lo, hi, policy["max_gap_ns"], breaks=breaks)

    report("building episodes and transitions", 0.55)
    episodes = tr.build_episodes(supported)
    transitions = tr.build_transitions(episodes)
    objects = _fusion_objects(fm)
    summary = tr.build_summary(episodes, objects)

    report("validating", 0.8)
    ok_tile, tile_problems = tr.check_no_double_counting(episodes)
    supported_ns = sum(s["support_end_ns"] - s["support_start_ns"]
                       for s in supported)
    episode_ns = sum(e["duration_ns"] for e in episodes)
    summary_ns = sum(r["total_attention_ns"] for r in summary)
    span_ns = hi - lo

    validation = [
        {"check": "parent fusion artifact loads and validates", "ok": True,
         "detail": f"{art.artifact_id} ({len(fused)} authoritative rows)"},
        {"check": "evidence is the authoritative gaze table only", "ok": True,
         "detail": EVIDENCE_NOTE},
        {"check": "episodes do not overlap (no double counting)",
         "ok": ok_tile, "detail": "; ".join(tile_problems) or
         f"{len(episodes)} episodes tile their supported time"},
        {"check": "episode duration equals supported sample duration",
         "ok": episode_ns == supported_ns,
         "detail": f"{episode_ns} ns vs {supported_ns} ns"},
        {"check": "summary duration equals episode duration",
         "ok": summary_ns == episode_ns,
         "detail": f"{summary_ns} ns vs {episode_ns} ns"},
        {"check": "supported time never exceeds the validated interval",
         "ok": episode_ns <= span_ns,
         "detail": f"{episode_ns} ns of {span_ns} ns interval"},
        {"check": "all timestamps are exact integers", "ok": all(
            isinstance(e["start_timestamp_ns"], int)
            and isinstance(e["end_timestamp_ns"], int)
            and isinstance(e["duration_ns"], int) for e in episodes),
         "detail": "integer nanoseconds end to end; seconds are derived"},
        {"check": "transitions link consecutive episodes",
         "ok": len(transitions) == max(0, len(episodes) - 1),
         "detail": f"{len(transitions)} transitions for {len(episodes)} "
                   f"episodes"},
        {"check": "no-hit target kept explicit",
         "ok": all(r["object_id"] != "background" for r in summary),
         "detail": NO_HIT_NOTE},
    ]
    return {"artifact": art, "fusion_manifest": fm, "fused": fused,
            "rows": rows, "supported": supported, "policy": policy,
            "episodes": episodes, "transitions": transitions,
            "summary": summary, "objects": objects,
            "interval": (lo, hi), "validation": validation,
            "supported_ns": supported_ns, "breaks": breaks,
            "dropped": dropped}


# ----------------------------------------------------------------- writers

def _write_csv(path: Path, columns: list, rows: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {}
            for c in columns:
                v = r.get(c)
                if isinstance(v, list):
                    v = json.dumps(v)
                elif isinstance(v, bool):
                    v = "true" if v else "false"
                elif v is None:
                    v = ""
                out[c] = v
            w.writerow(out)


def _write_parquet(path: Path, columns: list, rows: list) -> None:
    import pandas as pd
    df = pd.DataFrame([{c: r.get(c) for c in columns} for r in rows],
                      columns=columns)
    df.to_parquet(path, index=False)


def publish_attention_trace(fusion_dir, out_dir, *,
                            progress_cb: Optional[Callable] = None,
                            path_guard: Optional[Callable] = None) -> dict:
    """Build + atomically publish the trace. Refuses an existing output."""
    def report(phase, frac):
        if progress_cb:
            progress_cb(phase, frac)

    res = build_trace(fusion_dir, progress=report)
    art, fm = res["artifact"], res["fusion_manifest"]
    lo, hi = res["interval"]

    prov = provenance(fm)
    parent_hashes = {
        "fusion_manifest": sha256_file(
            Path(art.artifact_dir) / "fusion_manifest.json"),
        "gaze_samples_parquet": sha256_file(art.gaze_samples_parquet_path),
        "gaze_samples_csv": sha256_file(art.gaze_samples_csv_path),
    }

    pub = AtomicPublisher(Path(out_dir), source_dir=Path(art.artifact_dir),
                          path_guard=path_guard,
                          tmp_prefix=".attention_trace_tmp_")
    try:
        tmp = pub.tmp_dir
        report("writing tables", 0.88)
        _write_csv(tmp / "attention_episodes.csv", tr.EPISODE_COLUMNS,
                   res["episodes"])
        _write_parquet(tmp / "attention_episodes.parquet",
                       tr.EPISODE_COLUMNS, res["episodes"])
        _write_csv(tmp / "attention_transitions.csv", tr.TRANSITION_COLUMNS,
                   res["transitions"])
        _write_parquet(tmp / "attention_transitions.parquet",
                       tr.TRANSITION_COLUMNS, res["transitions"])
        _write_csv(tmp / "object_attention_summary.csv", tr.SUMMARY_COLUMNS,
                   res["summary"])

        object_dict = {}
        for r in res["summary"]:
            object_dict[r["object_id"]] = {
                "name": r["object_name"], "color": r["object_color"],
                "is_tracked_object": r["is_tracked_object"],
                "total_attention_ns": r["total_attention_ns"],
                "episode_count": r["episode_count"],
                "revisit_count": r["revisit_count"]}

        trace_json = {
            "schema_version": ATTENTION_SCHEMA_VERSION,
            "artifact_type": ATTENTION_ARTIFACT_TYPE,
            "parent_fusion_artifact_id": art.artifact_id,
            "timestamp_domain": prov["timestamp_domain"],
            "timestamp_units": "nanoseconds",
            "evidence": EVIDENCE_NOTE,
            "no_tracked_object": {"id": NO_TRACKED_OBJECT,
                                  "name": NO_TRACKED_OBJECT_NAME,
                                  "meaning": NO_HIT_NOTE},
            "policy": res["policy"],
            "objects": object_dict,
            "episodes": [
                {"episode_id": e["episode_id"],
                 "target_id": e["target_id"],
                 "target_name": e["target_name"],
                 "is_tracked_object": e["is_tracked_object"],
                 "start_timestamp_ns": e["start_timestamp_ns"],
                 "end_timestamp_ns": e["end_timestamp_ns"],
                 "duration_ns": e["duration_ns"],
                 "frames": [e["first_frame_index"], e["last_frame_index"]],
                 "gaze_sample_indices": e["gaze_sample_indices"],
                 "sample_count": e["sample_count"]}
                for e in res["episodes"]],
            "transitions": [
                {"transition_id": t["transition_id"],
                 "from_episode_id": t["from_episode_id"],
                 "to_episode_id": t["to_episode_id"],
                 "from_object_id": t["from_object_id"],
                 "to_object_id": t["to_object_id"],
                 "transition_timestamp_ns": t["transition_timestamp_ns"],
                 "unsupported_gap_ns": t["unsupported_gap_ns"],
                 "transition_type": t["transition_type"]}
                for t in res["transitions"]],
        }
        (tmp / "attention_trace.json").write_text(
            json.dumps(trace_json, indent=2) + "\n")

        report("verifying parent unchanged", 0.95)
        rehash = {
            "fusion_manifest": sha256_file(
                Path(art.artifact_dir) / "fusion_manifest.json"),
            "gaze_samples_parquet": sha256_file(
                art.gaze_samples_parquet_path),
            "gaze_samples_csv": sha256_file(art.gaze_samples_csv_path),
        }
        validation = list(res["validation"])
        validation.append({
            "check": "parent fusion artifact unchanged (sha256)",
            "ok": rehash == parent_hashes,
            "detail": "manifest and both gaze tables re-hashed identical"})
        failed = [v for v in validation if not v["ok"]]
        if failed:
            raise ArtifactError("attention-trace validation failed:\n"
                                + "\n".join(f"  x {v['check']}: {v['detail']}"
                                            for v in failed))

        files = {"manifest": ATTENTION_MANIFEST,
                 "episodes_csv": "attention_episodes.csv",
                 "episodes_parquet": "attention_episodes.parquet",
                 "transitions_csv": "attention_transitions.csv",
                 "transitions_parquet": "attention_transitions.parquet",
                 "summary_csv": "object_attention_summary.csv",
                 "trace_json": "attention_trace.json"}
        files_sha256 = {k: sha256_file(tmp / v) for k, v in files.items()
                        if k != "manifest"}

        counts_by_type = {}
        for t in res["transitions"]:
            counts_by_type[t["transition_type"]] = counts_by_type.get(
                t["transition_type"], 0) + 1

        manifest = {
            "schema_version": ATTENTION_SCHEMA_VERSION,
            "artifact_type": ATTENTION_ARTIFACT_TYPE,
            "artifact_id": uuid.uuid4().hex,
            "created_utc": datetime.now(timezone.utc)
                .isoformat(timespec="seconds"),
            "tool": {"name": "attention_trace", "version": __version__},
            "files": files,
            "files_sha256": files_sha256,
            "parent_artifact": {
                "artifact_id": art.artifact_id,
                "artifact_type": fm.get("artifact_type"),
                "artifact_dir": str(art.artifact_dir),
                "files_sha256": parent_hashes,
                "grandparent_tracked_artifact_id":
                    (fm.get("parent_artifact") or {}).get("artifact_id"),
                "recording_id": (fm.get("parent_artifact") or {}).get(
                    "recording_id"),
            },
            "source": {
                "system": prov["source_system"],
                "timestamp_domain": prov["timestamp_domain"],
                "provenance_resolved_from": prov["resolved_from"],
                "timestamp_units": "nanoseconds",
                "video_frames": (fm.get("video") or {}).get("n_frames"),
                "validated_interval_ns": [lo, hi],
                "validated_interval_duration_ns": hi - lo,
            },
            "evidence_policy": {
                "note": EVIDENCE_NOTE,
                "authoritative_table": "gaze_samples.parquet",
                "excluded": ["gaze_demo.mp4", "demo_presentation.csv"],
                "eligible_statuses": list(pol.SUPPORTING_STATUSES),
                "breaking_statuses": list(pol.BREAKING_STATUSES),
                "no_tracked_object": {"id": NO_TRACKED_OBJECT,
                                      "name": NO_TRACKED_OBJECT_NAME,
                                      "meaning": NO_HIT_NOTE},
                "overlap_resolution":
                    "primary_object_id as already decided by gaze fusion "
                    "(smaller mask area, then higher track score, then "
                    "stable object id) — never re-decided here",
            },
            "support_policy": {**res["policy"],
                               "support_interval":
                                   "midpoint boundaries to the neighbouring "
                                   "eligible samples, each side capped at "
                                   "max_gap_ns // 2, clipped to the "
                                   "validated fusion interval",
                               "episode_rule":
                                   "maximal run of one target whose support "
                                   "intervals touch or overlap; no semantic "
                                   "smoothing, no short-episode relabelling",
                               "revisit_rule":
                                   "returns after leaving = episode_count - 1"},
            "counts": {
                "authoritative_rows": int(len(res["fused"])),
                "eligible_samples": len(res["rows"]),
                "supported_samples": len(res["supported"]),
                "episodes": len(res["episodes"]),
                "transitions": len(res["transitions"]),
                "transition_types": counts_by_type,
                "objects_attended": sum(1 for r in res["summary"]
                                        if r["is_tracked_object"]),
                "supported_duration_ns": res["supported_ns"],
                "supported_duration_s": res["supported_ns"] / 1e9,
                "no_hit_duration_ns": next(
                    (r["total_attention_ns"] for r in res["summary"]
                     if r["object_id"] == NO_TRACKED_OBJECT), 0),
            },
            "validation": validation,
            "output_directory": str(pub.out_dir),
        }
        (tmp / ATTENTION_MANIFEST).write_text(json.dumps(manifest, indent=2)
                                              + "\n")
        report("publishing", 0.98)
        pub.publish()
        report("done", 1.0)
        return manifest
    finally:
        pub.cleanup()


# source system -> timestamp domain, as declared by the producing stages
_DOMAIN_BY_SOURCE_TYPE = {
    "trimmed_aria_recording": ("project_aria", "aria_device_time"),
    "prepared_aria_recording": ("project_aria", "aria_device_time"),
    "trimmed_artifact": ("pupil_neon", "neon_world_time"),
    "prepared_artifact": ("pupil_neon", "neon_world_time"),
    "csv_export": ("pupil_neon", "neon_world_time"),
}


def provenance(fm: dict) -> dict:
    """Authoritative source system + timestamp domain for a fusion artifact.

    Resolved from DECLARED provenance, never by sniffing strings: the
    tracked-object manifest states `synchronization.source_system` /
    `timestamp_domain` (newer artifacts) or `parent.source_type` (older
    ones). If neither is available the domain is reported as `unknown`
    rather than guessed — mislabelling Aria device time as NEON world
    time would be a provenance error.
    """
    out = {"source_system": "unknown", "timestamp_domain": "unknown",
           "resolved_from": "unavailable"}
    tracked_dir = (fm.get("parent_artifact") or {}).get("artifact_dir")
    if not tracked_dir:
        return out
    tm_path = Path(tracked_dir) / "tracking_manifest.json"
    try:
        tm = json.loads(tm_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {**out, "resolved_from": "tracked manifest unreadable"}
    declared = (fm.get("parent_artifact") or {}).get("artifact_id")
    if declared and tm.get("artifact_id") != declared:
        # the folder no longer holds the artifact fusion recorded: refuse
        # to inherit provenance from an unrelated manifest
        return {**out, "resolved_from":
                "tracked manifest identity mismatch — provenance not "
                "inherited"}
    sync = tm.get("synchronization") or {}
    if sync.get("timestamp_domain") and sync.get("source_system"):
        return {"source_system": sync["source_system"],
                "timestamp_domain": sync["timestamp_domain"],
                "resolved_from": "tracked manifest synchronization block"}
    src_type = (tm.get("parent") or {}).get("source_type")
    if src_type in _DOMAIN_BY_SOURCE_TYPE:
        system, domain = _DOMAIN_BY_SOURCE_TYPE[src_type]
        return {"source_system": system, "timestamp_domain": domain,
                "resolved_from": f"tracked manifest parent.source_type="
                                 f"{src_type}"}
    return out


# ------------------------------------------------------------------ loader

class AttentionTraceArtifact:
    def __init__(self, artifact_dir: Path, manifest: dict, files: dict):
        self.artifact_dir = artifact_dir
        self.manifest = manifest
        self.files = files

    @property
    def artifact_id(self):
        return self.manifest["artifact_id"]

    def path(self, key):
        return self.files[key]


def load_attention_trace(path) -> AttentionTraceArtifact:
    """Open + validate an attention_trace artifact."""
    art_dir, manifest = read_manifest(path, ATTENTION_MANIFEST)
    where = str(art_dir / ATTENTION_MANIFEST)
    require(manifest, where, artifact_type=ATTENTION_ARTIFACT_TYPE,
            schema_major="1")
    decl = manifest.get("files")
    if not isinstance(decl, dict):
        raise ArtifactError(f"{where}: missing files section")
    files = {k: resolve_declared_file(art_dir, rel, key=k, where=where)
             for k, rel in decl.items()}
    for key in ("episodes_csv", "episodes_parquet", "transitions_csv",
                "transitions_parquet", "summary_csv", "trace_json"):
        if key not in files:
            raise ArtifactError(f"{where}: files must declare {key}")
    if not manifest.get("validation"):
        raise ArtifactError(f"{where}: no validation record")
    if any(not v.get("ok") for v in manifest["validation"]):
        raise ArtifactError(f"{where}: validation record contains failures")
    if not (manifest.get("parent_artifact") or {}).get("artifact_id"):
        raise ArtifactError(f"{where}: missing parent fusion artifact id")

    recorded = manifest.get("files_sha256") or {}
    must_hash = {k for k in files if k != "manifest"}
    missing = sorted(must_hash - set(recorded))
    if missing:
        raise ArtifactError(f"{where}: files_sha256 is missing entries for "
                            f"{missing} — integrity cannot be self-declared")
    for key, sha in recorded.items():
        if key not in files:
            raise ArtifactError(f"{where}: files_sha256 names undeclared "
                                f"file {key!r}")
        if sha256_file(files[key]) != sha:
            raise ArtifactError(f"{where}: {key} does not match its "
                                f"recorded sha256")

    import pandas as pd
    eps = pd.read_parquet(files["episodes_parquet"])
    trs = pd.read_parquet(files["transitions_parquet"])
    counts = manifest.get("counts") or {}
    if counts.get("episodes") != len(eps):
        raise ArtifactError(f"{where}: counts.episodes != episode rows")
    if counts.get("transitions") != len(trs):
        raise ArtifactError(f"{where}: counts.transitions != transition rows")
    if len(eps) and list(eps["episode_index"]) != list(range(len(eps))):
        raise ArtifactError(f"{where}: episode_index is not 0..n-1 in order")
    if len(eps) > 1:
        starts = list(eps["start_timestamp_ns"])
        ends = list(eps["end_timestamp_ns"])
        if any(b < a for a, b in zip(ends, starts[1:])):
            raise ArtifactError(f"{where}: episodes overlap in time")
    total = int(eps["duration_ns"].sum()) if len(eps) else 0
    if counts.get("supported_duration_ns") != total:
        raise ArtifactError(f"{where}: counts.supported_duration_ns "
                            f"disagrees with the episode table")
    trace = json.loads(files["trace_json"].read_text())
    if len(trace.get("episodes", [])) != len(eps):
        raise ArtifactError(f"{where}: attention_trace.json disagrees with "
                            f"the episode table")
    return AttentionTraceArtifact(art_dir, manifest, files)
