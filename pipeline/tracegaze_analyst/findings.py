"""Muse findings: prompt, response schema and validation.

A finding is checked for citation integrity and numeric fidelity only.
"references_verified" means every cited ID exists, every copied number
matches the deterministic package, and evidence statements avoid
construct and personal-attribute language. It does NOT mean the
interpretation is true; that is what the manual review records.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .validate import UNSUPPORTED_ASSERTION_FRAGMENTS

EVIDENCE_TYPES = ("measured_pattern", "visible_observation",
                  "possible_interpretation")
SUPPORT_STATUSES = ("candidate_for_investigation", "not_supported_by_evidence",
                    "not_applicable")
NO_HIT = "__no_tracked_object__"

_STR = {"type": "string"}
_STRS = {"type": "array", "items": _STR}

FINDINGS_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["answer_summary", "findings", "remaining_questions"],
    "properties": {
        "answer_summary": _STR,
        "findings": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["finding_id", "title", "statement", "evidence_type",
                         "interval_ids", "object_ids", "episode_ids", "frame_ids",
                         "fact_ids", "quantities", "visible_observations",
                         "interpretation", "alternative_explanations",
                         "limitations", "uncertainty", "support_opportunity"],
            "properties": {
                "finding_id": _STR, "title": _STR, "statement": _STR,
                "evidence_type": {"type": "string", "enum": list(EVIDENCE_TYPES)},
                "interval_ids": _STRS, "object_ids": _STRS,
                "episode_ids": _STRS, "frame_ids": _STRS, "fact_ids": _STRS,
                "quantities": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["fact_id", "value", "unit"],
                    "properties": {"fact_id": _STR, "value": {"type": "number"},
                                   "unit": _STR}}},
                "visible_observations": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["frame_id", "observation"],
                    "properties": {"frame_id": _STR, "observation": _STR}}},
                "interpretation": _STR,
                "alternative_explanations": _STRS,
                "limitations": _STRS,
                "uncertainty": {"type": "string", "enum": ["low", "medium", "high"]},
                "support_opportunity": {
                    "type": "object", "additionalProperties": False,
                    "required": ["status", "description", "evidence_needed"],
                    "properties": {
                        "status": {"type": "string", "enum": list(SUPPORT_STATUSES)},
                        "description": _STR, "evidence_needed": _STR}},
            }}},
        "remaining_questions": _STRS,
    },
}

SYSTEM_PROMPT = """\
You assist researchers who study everyday activities and design assistance \
for older adults. You analyse ONE retrospective first-person recording \
through a TraceGaze evidence package: deterministic gaze-to-object \
measurements plus a few timestamped scene frames. You do not see the video.

Return three findings (four at most) that together answer every part of \
the question: how gaze and visible activity unfolded; which candidate \
intervals deserve closer examination and why; and what those moments might \
imply for designing assistance, including when the evidence supports none. \
Prefer findings that rest on different intervals.

Evidence rules
1. Every number you write must be copied exactly from a supplied value: a \
fact `value`, or an interval, episode or frame time. Never compute new \
totals, averages, percentages or differences. If a number you want is not \
supplied, describe it in words.
2. Put each number you rely on in `quantities` (fact_id, value, unit exactly \
as supplied) and list that fact_id in `fact_ids`.
3. Cite only IDs that exist in the package (interval_ids, object_ids, \
episode_ids, frame_ids, fact_ids).
4. Gaze-to-object assignments are fixed. Never reassign a gaze sample to a \
different object because it seems more plausible; if a frame makes an \
assignment look doubtful, say so as a limitation.
5. Runs of single-sample episodes and boundary-near samples may reflect gaze \
near mask edges rather than a change in what was looked at. Do not treat \
rapid switching as behaviour unless frames give independent support. When a \
finding relies on a target's time or transitions, check that target's \
boundary_near_sample_count against its sample_count.
6. A frame shows one instant. Describe only what is visible (hands, objects, \
posture, who is doing what). The green ring marks the gaze sample; its \
radius equals the 25 px proximity convention and is not an accuracy bound.

Keep evidence and interpretation apart
7. evidence_type is "measured_pattern" (stated from facts), \
"visible_observation" (stated from frames) or "possible_interpretation" (a \
reading that goes beyond both).
8. `statement` and `visible_observations` contain only what the evidence \
shows. Any reading about purpose, state of mind, difficulty, intention or \
need goes in `interpretation` (empty string if none), together with at \
least one material alternative explanation and medium or high uncertainty.
9. Repeated looking, returns or switching do not by themselves establish \
confusion, memory failure, searching, difficulty or a need for help. \
Ordinary successful activity and ambiguous cases are valid findings.
10. Age is not evidence of difficulty. The participant's age, health and \
familiarity with the task are unknown. Do not infer age, gender, health, \
identity or any other personal attribute from appearance; write "the \
wearer" and "the partner", and avoid gendered pronouns.
11. Write "gaze", "looked toward" or "gaze-supported time"; do not use the \
words fixation or saccade, and do not claim that attention was on something.

Support opportunities
12. Use support_opportunity.status "candidate_for_investigation" only when \
the evidence points to a moment where assistance might plausibly be worth \
studying; then say what to investigate and what additional evidence would \
distinguish the explanations. Otherwise use "not_supported_by_evidence" or \
"not_applicable". Concluding that no opportunity is evidenced is acceptable.

In remaining_questions, list what this package cannot answer and which \
additional evidence would help. Return only JSON matching the schema."""


# --------------------------------------------------------------------------
# request construction
# --------------------------------------------------------------------------

def model_view(package: dict) -> dict:
    """The package as sent to the model: no local paths, hashes or ns."""
    drop_iv = {"start_timestamp_ns", "end_timestamp_ns"}
    view = {k: v for k, v in package.items() if k != "source_refs"}
    view["recording"] = {k: v for k, v in package["recording"].items()
                         if k not in ("interval_timestamp_ns", "artifact_ids")}
    view["intervals"] = [{k: v for k, v in iv.items() if k not in drop_iv}
                         for iv in package["intervals"]]
    view["frames"] = [{k: v for k, v in f.items()
                       if k not in ("file", "sha256", "width", "height",
                                    "frame_time_s")}
                      for f in package["frames"]]
    view["facts"] = [{k: v for k, v in f.items() if k != "source_ref"}
                     for f in package["facts"]]
    return view


def frame_caption(f: dict) -> str:
    alts = ", ".join(f["alternatives_within_px"]) or "none"
    return (f"frame_id {f['frame_id']} | interval {f['interval_id']} | "
            f"t = {f['time_s']} s | gaze sample {f['gaze_sample_index']} "
            f"assigned to {f['assigned_target_id']} ({f['assigned_target_name']}) "
            f"| other targets within 25 px: {alts}")


def build_messages(package: dict, package_dir: Path) -> tuple[list, str]:
    """Chat messages with interleaved frame images, plus a text-only record."""
    from .muse import image_part
    head = ("RESEARCHER QUESTION\n" + package["question"] + "\n\n"
            "EVIDENCE PACKAGE (JSON)\n"
            + json.dumps(model_view(package), ensure_ascii=False, indent=1)
            + "\n\nSCENE FRAMES (each caption precedes its image)\n")
    content = [{"type": "text", "text": head}]
    record = [head]
    for f in package["frames"]:
        cap = frame_caption(f)
        content.append({"type": "text", "text": cap})
        content.append(image_part((package_dir / f["file"]).read_bytes()))
        record.append(cap + f"\n[image {f['file']} sha256={f['sha256']}]\n")
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}]
    prompt_record = ("# System\n\n" + SYSTEM_PROMPT + "\n\n# User\n\n"
                     + "\n".join(record))
    return messages, prompt_record


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------

CONSTRUCT_PATTERN = re.compile(
    r"\b(confus\w*|forg[eo]t\w*|memory|difficult\w*|struggl\w*|impair\w*|"
    r"declin\w*|dementia|cognitiv\w*|frail\w*|disorient\w*|anxi\w*|"
    r"frustrat\w*|hesitat\w*|unsure|uncertain about|search\w*|looking for|"
    r"intend\w*|intention\w*|wants? to|wanted to|trying to|tried to|"
    r"needs? help|needed help|in need)\b", re.I)
PERSONAL_PATTERN = re.compile(
    r"\b(woman|women|man|men|female|male|girl|boy|lady|gentleman|elderly|"
    r"senior|aged|young|older adults?|older person|years old|"
    r"he|she|him|his|her|hers)\b", re.I)
_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(\s*%)?(?![\w])")


def _number_ok(token: str, pct: bool, allowed: set[float]) -> bool:
    d = len(token.split(".")[1]) if "." in token else 0
    x = float(token)
    if pct:
        x, d = x / 100.0, d + 2
    tol = 0.5 * 10 ** (-d) + 1e-9
    return any(abs(x - v) <= tol for v in allowed)


def _allowed_numbers(package, finding, fact_map) -> set[float]:
    vals: set[float] = {float(package["recording"]["duration_s"]), 25.0,
                        float(package["data_quality"]["gaze_median_interval_ms"])}
    for fid in finding.get("fact_ids", []):
        v = fact_map.get(fid, {}).get("value")
        if isinstance(v, (int, float)):
            vals.add(float(v))
    iv_map = {iv["interval_id"]: iv for iv in package["intervals"]}
    for iid in finding.get("interval_ids", []):
        iv = iv_map.get(iid)
        if iv:
            vals |= {float(iv["start_s"]), float(iv["end_s"])}
    fr_map = {f["frame_id"]: f for f in package["frames"]}
    for frid in finding.get("frame_ids", []):
        if frid in fr_map:
            vals.add(float(fr_map[frid]["time_s"]))
    ep_map = {e["episode_id"]: e for e in package["episodes"]}
    for eid in finding.get("episode_ids", []):
        e = ep_map.get(eid)
        if e:
            vals |= {float(e["start_s"]), float(e["duration_s"]),
                     float(e["sample_count"])}
    return vals


def _texts(finding) -> list[tuple[str, str]]:
    out = [("title", finding.get("title", "")),
           ("statement", finding.get("statement", "")),
           ("interpretation", finding.get("interpretation", ""))]
    for i, v in enumerate(finding.get("visible_observations", []) or []):
        if isinstance(v, dict):
            out.append((f"visible_observations[{i}]", v.get("observation", "")))
    return out


def _value_equal(claimed, actual) -> bool:
    if isinstance(claimed, bool) or not isinstance(claimed, (int, float)):
        return False
    return abs(float(claimed) - float(actual)) <= 1e-9


# --------------------------------------------------------------------------
# independent re-derivation of package facts from the trace files
# --------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rederive_facts(package: dict, trace_dir: Path,
                   proximity_path: Path) -> dict:
    """Recompute every package fact from the attention-trace CSVs.

    Written independently of support_package's construction so that a
    stale or edited package is caught, not merely re-echoed.
    """
    manifest = json.loads((trace_dir / "attention_trace_manifest.json").read_text())
    summary = {r["object_id"]: r for r in _read_csv(trace_dir / "object_attention_summary.csv")}
    eps = _read_csv(trace_dir / "attention_episodes.csv")
    trans = _read_csv(trace_dir / "attention_transitions.csv")
    near = {int(k) for k, v in json.loads(proximity_path.read_text())["samples"].items()
            if v["boundary_near"]}
    for e in eps:
        e["_s"], e["_e"] = int(e["start_timestamp_ns"]), int(e["end_timestamp_ns"])
        e["_samples"] = json.loads(e["gaze_sample_indices"])
    t0, t1 = (int(x) for x in manifest["source"]["validated_interval_ns"])
    counts = manifest["counts"]
    iv_bounds = {iv["interval_id"]: (iv["start_timestamp_ns"], iv["end_timestamp_ns"])
                 for iv in package["intervals"]}

    def window(s, e, what, oid=None):
        inside = [x for x in eps if x["_e"] > s and x["_s"] < e]
        if what == "time":
            return sum(max(0, min(x["_e"], e) - max(x["_s"], s))
                       for x in inside if x["target_id"] == oid) / 1e9
        if what == "transition_count":
            return sum(1 for t in trans if s < int(t["transition_timestamp_ns"]) < e)
        if what == "episode_count":
            return len(inside)
        if what == "single_sample_episode_count":
            return sum(1 for x in inside if x["sample_count"] == "1")
        if what == "sample_count":
            return sum(len(x["_samples"]) for x in inside)
        if what == "boundary_near_sample_count":
            return sum(1 for x in inside for i in x["_samples"] if i in near)
        if what == "duration_s":
            return (e - s) / 1e9
        raise KeyError(what)

    def expected(fid: str):
        parts = fid.split(".")
        if parts[0] == "session":
            key = parts[1]
            return {
                "authoritative_rows": counts["authoritative_rows"],
                "supported_duration_s": counts["supported_duration_s"],
                "no_hit_duration_s": counts["no_hit_duration_ns"] / 1e9,
                "episode_count": counts["episodes"],
                "transition_count": counts["transitions"],
                "single_sample_episode_count": sum(1 for x in eps if x["sample_count"] == "1"),
                "boundary_near_sample_count": sum(1 for x in eps for i in x["_samples"] if i in near),
            }[key]
        if parts[0] == "object":
            if parts[2] == "boundary_near_sample_count":
                return sum(1 for x in eps if x["target_id"] == parts[1]
                           for i in x["_samples"] if i in near)
            row = summary[parts[1]]
            v = row[parts[2]]
            return int(v) if parts[2].endswith("count") else float(v)
        if parts[0] == "transition_pair":
            a, b = parts[1], parts[3]
            return sum(1 for t in trans if t["from_object_id"] == a and t["to_object_id"] == b)
        if parts[0] == "interval":
            s, e = iv_bounds[parts[1]]
            if parts[2] == "target":
                return window(s, e, "time", parts[3])
            return window(s, e, parts[2])
        if parts[0] == "phase":
            k = int(parts[1][1:])
            s = t0 + (k - 1) * 10_000_000_000
            e = min(s + 10_000_000_000, t1)
            if parts[2] == "target":
                return window(s, e, "time", parts[3])
            return window(s, e, parts[2])
        raise KeyError(fid)

    mismatches, checked = [], 0
    for f in package["facts"]:
        try:
            exp = expected(f["fact_id"])
        except (KeyError, IndexError, ValueError) as exc:
            mismatches.append({"fact_id": f["fact_id"], "problem": f"cannot re-derive: {exc}"})
            continue
        checked += 1
        ok = (round(float(exp), 3) == float(f["value"]) if isinstance(exp, float)
              else exp == f["value"])
        if not ok:
            mismatches.append({"fact_id": f["fact_id"], "package_value": f["value"],
                               "rederived_value": exp})
    return {"checked": checked, "mismatches": mismatches}


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def validate_findings(package: dict, response: dict,
                      rederivation: dict | None = None) -> dict:
    fact_map = {f["fact_id"]: f for f in package["facts"]}
    iv_map = {iv["interval_id"]: iv for iv in package["intervals"]}
    ep_ids = {e["episode_id"] for e in package["episodes"]}
    fr_map = {f["frame_id"]: f for f in package["frames"]}
    obj_ids = {o["object_id"] for o in package["objects"]}
    global_problems = []

    if not isinstance(response.get("answer_summary"), str) or not response["answer_summary"].strip():
        global_problems.append("answer_summary must be a non-empty string")
    for frag in UNSUPPORTED_ASSERTION_FRAGMENTS:
        if frag in str(response.get("answer_summary", "")).casefold():
            global_problems.append(f"answer_summary uses unsupported language {frag!r}")
    for m in PERSONAL_PATTERN.finditer(str(response.get("answer_summary", ""))):
        global_problems.append(f"answer_summary uses personal-attribute word {m.group(0)!r}")
    findings = response.get("findings")
    if not isinstance(findings, list) or not findings:
        global_problems.append("findings must be a non-empty list")
        findings = []
    if not isinstance(response.get("remaining_questions"), list):
        global_problems.append("remaining_questions must be a list")

    results, seen = [], set()
    for i, f in enumerate(findings):
        errors, warnings = [], []
        fid = f.get("finding_id") or f"findings[{i}]"
        if fid in seen:
            errors.append(f"duplicate finding_id {fid}")
        seen.add(fid)
        kind = f.get("evidence_type")
        if kind not in EVIDENCE_TYPES:
            errors.append(f"invalid evidence_type {kind!r}")
        for key in ("title", "statement"):
            if not isinstance(f.get(key), str) or not f[key].strip():
                errors.append(f"{key} must be non-empty")

        # references
        for key, valid in (("interval_ids", set(iv_map)), ("episode_ids", ep_ids),
                           ("frame_ids", set(fr_map)), ("fact_ids", set(fact_map)),
                           ("object_ids", obj_ids)):
            vals = f.get(key, [])
            if not isinstance(vals, list):
                errors.append(f"{key} must be a list")
                continue
            unknown = [v for v in vals if v not in valid]
            if unknown:
                errors.append(f"unknown {key}: {unknown}")
        cited_iv = [iv_map[x] for x in f.get("interval_ids", []) if x in iv_map]
        iv_eps = {e for iv in cited_iv for e in iv["episode_ids"]}
        iv_frames = {x for iv in cited_iv for x in iv["frame_ids"]}
        if cited_iv:
            off = [e for e in f.get("episode_ids", []) if e in ep_ids and e not in iv_eps]
            if off:
                warnings.append(f"episodes outside the cited intervals: {off}")
            off = [x for x in f.get("frame_ids", []) if x in fr_map and x not in iv_frames]
            if off:
                errors.append(f"frames outside the cited intervals: {off}")

        # quantities: copied numbers must equal the deterministic package
        qchecks = []
        for j, q in enumerate(f.get("quantities", []) or []):
            qid = q.get("fact_id")
            if qid not in fact_map:
                errors.append(f"quantities[{j}]: unknown fact_id {qid!r}")
                continue
            if qid not in f.get("fact_ids", []):
                errors.append(f"quantities[{j}]: {qid} not listed in fact_ids")
            fact = fact_map[qid]
            match = _value_equal(q.get("value"), fact["value"]) and q.get("unit") == fact["unit"]
            qchecks.append({"fact_id": qid, "claimed": q.get("value"),
                            "claimed_unit": q.get("unit"),
                            "deterministic": fact["value"], "unit": fact["unit"],
                            "match": match})
            if not match:
                errors.append(f"quantities[{j}]: {q.get('value')!r} {q.get('unit')!r} "
                              f"does not match {qid} = {fact['value']!r} {fact['unit']!r}")

        # numbers written in prose must be traceable to cited evidence
        allowed = _allowed_numbers(package, f, fact_map)
        prose = []
        for field, text in _texts(f):
            for m in _NUM.finditer(text or ""):
                ok = _number_ok(m.group(1), bool(m.group(2)), allowed)
                prose.append({"field": field, "token": m.group(0).strip(), "traceable": ok})
                if not ok:
                    errors.append(f"{field}: number {m.group(0).strip()!r} is not "
                                  f"traceable to the finding's cited evidence")

        # language gates
        for field, text in _texts(f):
            low = (text or "").casefold()
            for frag in UNSUPPORTED_ASSERTION_FRAGMENTS:
                if frag in low:
                    errors.append(f"{field}: unsupported language {frag!r}")
            for m in PERSONAL_PATTERN.finditer(text or ""):
                errors.append(f"{field}: personal-attribute word {m.group(0)!r}")
            if field != "interpretation":
                for m in CONSTRUCT_PATTERN.finditer(text or ""):
                    errors.append(f"{field}: construct word {m.group(0)!r} belongs "
                                  f"in interpretation, not evidence")

        # evidence-type obligations
        interp = (f.get("interpretation") or "").strip()
        alts = [a for a in f.get("alternative_explanations", []) or [] if str(a).strip()]
        if kind == "measured_pattern" and not qchecks:
            errors.append("measured_pattern needs at least one checked quantity")
        if kind == "visible_observation" and not f.get("visible_observations"):
            errors.append("visible_observation needs at least one frame observation")
        for j, v in enumerate(f.get("visible_observations", []) or []):
            if v.get("frame_id") not in f.get("frame_ids", []):
                errors.append(f"visible_observations[{j}]: frame {v.get('frame_id')!r} "
                              f"not listed in frame_ids")
        if kind == "possible_interpretation" and not interp:
            errors.append("possible_interpretation needs a non-empty interpretation")
        if interp:
            if not alts:
                errors.append("an interpretation needs at least one alternative explanation")
            if f.get("uncertainty") not in ("medium", "high"):
                errors.append("an interpretation needs medium or high uncertainty")
            if not (f.get("fact_ids") or f.get("frame_ids")):
                errors.append("an interpretation needs cited facts or frames")
        so = f.get("support_opportunity") or {}
        if so.get("status") not in SUPPORT_STATUSES:
            errors.append(f"invalid support_opportunity.status {so.get('status')!r}")
        elif so["status"] == "candidate_for_investigation":
            if not str(so.get("evidence_needed", "")).strip():
                errors.append("a candidate support opportunity must state the evidence needed")
            if not alts:
                warnings.append("candidate support opportunity without an alternative explanation")

        results.append({"finding_id": fid,
                        "status": "references_verified" if not errors else "needs_review",
                        "errors": errors, "warnings": warnings,
                        "quantity_checks": qchecks, "prose_numbers": prose})

    if not 2 <= len(findings) <= 4:
        global_problems.append(f"expected 2-4 findings, got {len(findings)}")
    if rederivation and rederivation.get("mismatches"):
        global_problems.append(f"{len(rederivation['mismatches'])} package facts do not "
                               f"re-derive from the trace files")
    n_ok = sum(1 for r in results if r["status"] == "references_verified")
    return {
        "ok": not global_problems and n_ok == len(results),
        "meaning": ("references_verified = cited IDs exist, copied numbers match the "
                    "deterministic package, evidence text avoids construct and "
                    "personal-attribute language. It does not establish that an "
                    "interpretation is correct."),
        "summary": {"findings": len(results), "references_verified": n_ok,
                    "needs_review": len(results) - n_ok},
        "global_problems": global_problems,
        "findings": results,
        "rederivation": rederivation,
    }
