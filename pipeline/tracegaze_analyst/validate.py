"""Validate that a structured analyst response is grounded in the packet."""

from __future__ import annotations

from . import CLAIM_KINDS


UNSUPPORTED_ASSERTION_FRAGMENTS = (
    "fixation", "saccade", "most attended", "attended entity",
    "untracked visual space", "attention was", "head movement",
    "outside the camera's field of view", "outside the camera field of view",
)


def validate_response(packet: dict, response: dict) -> dict:
    problems = []
    fact_map = {f["fact_id"]: f for f in packet.get("facts", [])}
    fact_ids = set(fact_map)
    episode_map = {e["episode_id"]: e for e in packet.get("episodes", [])}
    transition_map = {t["transition_id"]: t for t in packet.get("transitions", [])}
    episode_ids = set(episode_map)
    transition_ids = set(transition_map)
    seen = set()
    answer_text = str(response.get("answer", "")).casefold()
    for fragment in UNSUPPORTED_ASSERTION_FRAGMENTS:
        if fragment in answer_text:
            problems.append(f"answer uses unsupported assertion language: {fragment!r}")
    claims = response.get("claims")
    if not isinstance(response.get("answer"), str) or not response["answer"].strip():
        problems.append("answer must be a non-empty string")
    if not isinstance(claims, list):
        problems.append("claims must be a list")
        claims = []
    for i, claim in enumerate(claims):
        where = f"claims[{i}]"
        cid = claim.get("claim_id")
        if not isinstance(cid, str) or not cid:
            problems.append(f"{where}: missing claim_id")
        elif cid in seen:
            problems.append(f"{where}: duplicate claim_id {cid}")
        seen.add(cid)
        kind = claim.get("claim_kind")
        if kind not in CLAIM_KINDS:
            problems.append(f"{where}: invalid claim_kind {kind!r}")
        if not isinstance(claim.get("statement"), str) or not claim["statement"].strip():
            problems.append(f"{where}: statement must be non-empty")
        else:
            statement_text = claim["statement"].casefold()
            for fragment in UNSUPPORTED_ASSERTION_FRAGMENTS:
                if fragment in statement_text:
                    problems.append(f"{where}: unsupported assertion language: {fragment!r}")
        refs = []
        for key, valid in (("fact_ids", fact_ids), ("episode_ids", episode_ids),
                           ("transition_ids", transition_ids)):
            values = claim.get(key, [])
            if not isinstance(values, list):
                problems.append(f"{where}: {key} must be a list")
                continue
            refs.extend(values)
            unknown = [v for v in values if v not in valid]
            if unknown:
                problems.append(f"{where}: unknown {key}: {unknown}")
        if kind in {"measured_fact", "observed_pattern"} and not refs:
            problems.append(f"{where}: {kind} requires evidence references")
        quantities = claim.get("quantities", [])
        if not isinstance(quantities, list):
            problems.append(f"{where}: quantities must be a list")
            quantities = []
        if kind == "measured_fact" and not quantities:
            problems.append(f"{where}: measured_fact requires a checked quantity")
        quantity_fact_ids = {q.get("fact_id") for q in quantities if isinstance(q, dict)}
        if kind == "measured_fact":
            missing_quantities = set(claim.get("fact_ids", [])) - quantity_fact_ids
            if missing_quantities:
                problems.append(f"{where}: unchecked measured fact IDs: {sorted(missing_quantities)}")
        for j, quantity in enumerate(quantities):
            qwhere = f"{where}.quantities[{j}]"
            fid = quantity.get("fact_id")
            if fid not in claim.get("fact_ids", []):
                problems.append(f"{qwhere}: fact_id is not listed in claim fact_ids")
            if fid not in fact_map:
                problems.append(f"{qwhere}: unknown fact_id {fid!r}")
                continue
            expected = fact_map[fid]
            if quantity.get("unit") != expected["unit"]:
                problems.append(f"{qwhere}: unit does not match {fid}")
            supplied = quantity.get("value")
            actual = expected["value"]
            if isinstance(actual, float):
                equal = (isinstance(supplied, (int, float))
                         and abs(float(supplied) - actual) <= 1e-9)
            else:
                equal = supplied == actual
            if not equal:
                problems.append(f"{qwhere}: value does not match {fid}")
        pair_contexts = set()
        object_contexts = set()
        for fid in claim.get("fact_ids", []):
            context = fact_map.get(fid, {}).get("context", {})
            if context.get("object_id"):
                object_contexts.add(context["object_id"])
            if context.get("from_object_id") and context.get("to_object_id"):
                pair_contexts.add((context["from_object_id"], context["to_object_id"]))
        if pair_contexts:
            for tid in claim.get("transition_ids", []):
                row = transition_map.get(tid)
                if row and (row["from_object_id"], row["to_object_id"]) not in pair_contexts:
                    problems.append(f"{where}: {tid} does not match a cited transition pair")
        if object_contexts:
            for eid in claim.get("episode_ids", []):
                row = episode_map.get(eid)
                if row and row["target_id"] not in object_contexts:
                    problems.append(f"{where}: {eid} does not match a cited object")
        if kind in {"interpretation", "hypothesis"}:
            if not refs:
                problems.append(f"{where}: {kind} requires supporting evidence")
            if claim.get("uncertainty") not in {"low", "medium", "high"}:
                problems.append(f"{where}: {kind} requires low/medium/high uncertainty")
    return {"ok": not problems, "claim_count": len(claims), "problems": problems}

