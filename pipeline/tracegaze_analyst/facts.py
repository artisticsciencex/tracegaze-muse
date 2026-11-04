"""Deterministic facts derived from an attention-trace artifact."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_trace(trace_dir: str | Path) -> dict:
    root = Path(trace_dir).expanduser().resolve()
    required = [
        "attention_trace_manifest.json",
        "object_attention_summary.csv",
        "attention_episodes.csv",
        "attention_transitions.csv",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"not an attention-trace artifact; missing: {', '.join(missing)}")
    manifest = json.loads((root / required[0]).read_text(encoding="utf-8"))
    bad = [v for v in manifest.get("validation", []) if not v.get("ok")]
    if bad:
        raise ValueError("attention-trace artifact has failed validations")
    return {
        "root": root,
        "manifest": manifest,
        "summary": _read_csv(root / required[1]),
        "episodes": _read_csv(root / required[2]),
        "transitions": _read_csv(root / required[3]),
    }


def _fact(fid: str, metric: str, value, unit: str, refs: list[str], **context) -> dict:
    out = {"fact_id": fid, "metric": metric, "value": value,
           "unit": unit, "evidence_refs": refs}
    if context:
        out["context"] = context
    return out


def build_fact_catalog(trace: dict) -> list[dict]:
    """Create exact, citable facts. No model inference occurs here."""
    m = trace["manifest"]
    counts = m["counts"]
    facts = [
        _fact("session.authoritative_rows", "authoritative gaze rows",
              int(counts["authoritative_rows"]), "samples",
              ["attention_trace_manifest.json#/counts/authoritative_rows"]),
        _fact("session.supported_duration_s", "supported duration",
              float(counts["supported_duration_s"]), "seconds",
              ["attention_trace_manifest.json#/counts/supported_duration_s"]),
        _fact("session.no_hit_duration_s", "no accepted tracked-object hit duration",
              int(counts["no_hit_duration_ns"]) / 1e9, "seconds",
              ["attention_trace_manifest.json#/counts/no_hit_duration_ns"]),
        _fact("session.episode_count", "attention episodes",
              int(counts["episodes"]), "episodes",
              ["attention_trace_manifest.json#/counts/episodes"]),
        _fact("session.transition_count", "episode transitions",
              int(counts["transitions"]), "transitions",
              ["attention_trace_manifest.json#/counts/transitions"]),
    ]

    for row in trace["summary"]:
        oid = row["object_id"]
        base = f"object.{oid}"
        refs = [f"object_attention_summary.csv#object_id={oid}"]
        context = {"object_id": oid, "object_name": row["object_name"],
                   "is_tracked_object": row["is_tracked_object"].lower() == "true"}
        specs = [
            ("total_attention_s", "accumulated gaze-supported time", float(row["total_attention_s"]), "seconds"),
            ("share_of_supported_time", "share of supported time", float(row["share_of_supported_time"]), "proportion"),
            ("episode_count", "episode count", int(row["episode_count"]), "episodes"),
            ("revisit_count", "revisit count", int(row["revisit_count"]), "revisits"),
            ("sample_count", "supporting gaze samples", int(row["sample_count"]), "samples"),
            ("display_rank", "duration rank", int(row["display_rank"]), "rank"),
        ]
        for suffix, metric, value, unit in specs:
            facts.append(_fact(f"{base}.{suffix}", metric, value, unit, refs, **context))

    pairs = Counter((r["from_object_id"], r["to_object_id"],
                     r["from_object_name"], r["to_object_name"])
                    for r in trace["transitions"])
    ordered = sorted(pairs.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    for rank, ((from_id, to_id, from_name, to_name), count) in enumerate(ordered, 1):
        facts.append(_fact(
            f"transition_pair.{from_id}.to.{to_id}.count",
            "transition-pair count", count, "transitions",
            [f"attention_transitions.csv#from_object_id={from_id}&to_object_id={to_id}"],
            from_object_id=from_id, to_object_id=to_id,
            from_object_name=from_name, to_object_name=to_name,
            transition_pair_rank=rank))
    return facts

