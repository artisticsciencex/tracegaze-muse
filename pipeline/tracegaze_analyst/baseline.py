"""A zero-API deterministic response used to test the full analysis path."""

from __future__ import annotations


def make_baseline_response(packet: dict) -> dict:
    facts = {f["fact_id"]: f for f in packet["facts"]}
    tracked = [o for o in packet["objects"] if o["is_tracked_object"].lower() == "true"]
    tracked.sort(key=lambda o: int(o["display_rank"]))
    claims = []
    answer_parts = []
    if tracked:
        top = tracked[0]
        fid = f"object.{top['object_id']}.total_attention_s"
        value = facts[fid]["value"]
        claims.append({
            "claim_id": "c1", "claim_kind": "measured_fact",
            "statement": f"{top['object_name']} had the most accumulated gaze-supported time ({value:.3f} s).",
            "fact_ids": [fid, f"object.{top['object_id']}.display_rank"],
            "episode_ids": [], "transition_ids": [], "uncertainty": "none",
            "quantities": [
                {"fact_id": fid, "value": value, "unit": "seconds"},
                {"fact_id": f"object.{top['object_id']}.display_rank",
                 "value": int(top["display_rank"]), "unit": "rank"},
            ],
        })
        answer_parts.append(claims[-1]["statement"])
    nohit = facts.get("session.no_hit_duration_s")
    if nohit:
        claims.append({
            "claim_id": f"c{len(claims)+1}", "claim_kind": "measured_fact",
            "statement": f"{nohit['value']:.3f} s had no accepted tracked-object hit.",
            "fact_ids": [nohit["fact_id"]], "episode_ids": [],
            "transition_ids": [], "uncertainty": "none",
            "quantities": [{"fact_id": nohit["fact_id"],
                            "value": nohit["value"], "unit": "seconds"}],
        })
        answer_parts.append(claims[-1]["statement"])
    pair_facts = [f for f in packet["facts"] if f["fact_id"].startswith("transition_pair.")]
    if pair_facts:
        pair = min(pair_facts, key=lambda f: f["context"]["transition_pair_rank"])
        ctx = pair["context"]
        claims.append({
            "claim_id": f"c{len(claims)+1}", "claim_kind": "observed_pattern",
            "statement": (f"The most frequent supplied transition pair was "
                          f"{ctx['from_object_name']} → {ctx['to_object_name']} "
                          f"({pair['value']} transitions)."),
            "fact_ids": [pair["fact_id"]], "episode_ids": [],
            "transition_ids": [], "uncertainty": "none",
            "quantities": [{"fact_id": pair["fact_id"],
                            "value": pair["value"], "unit": "transitions"}],
        })
        answer_parts.append(claims[-1]["statement"])
    return {
        "answer": " ".join(answer_parts),
        "claims": claims,
        "limitations": [
            "Object hits establish geometric mask membership, not intention, expertise, task success, or fixation.",
            "No accepted tracked-object hit may include gaze on an unlabeled physical object.",
        ],
    }

