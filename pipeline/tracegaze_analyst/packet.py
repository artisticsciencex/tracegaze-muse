"""Build a bounded, question-driven evidence packet for an LLM."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .facts import build_fact_catalog, load_trace


CONTRACT = {
    "claim_kinds": {
        "measured_fact": "Direct restatement of a supplied deterministic fact.",
        "observed_pattern": "A pattern assembled only from supplied facts or episode/transition rows.",
        "interpretation": "A plausible semantic reading, explicitly marked as interpretation.",
        "hypothesis": "A testable explanation requiring additional evidence.",
        "limitation": "A boundary on what the evidence can support.",
    },
    "rules": [
        "Never infer attention, intention, expertise, causality, task success, or fixation from an object hit alone.",
        "Every measured fact and observed pattern must cite supplied fact IDs and/or row IDs.",
        "Interpretations and hypotheses must cite supporting evidence and state uncertainty.",
        "Use 'gaze-supported time' or 'gaze-supported pattern'; do not call geometric hits attention or fixation.",
        "Use the exact phrase 'no accepted tracked-object hit'; it may include an unlabeled or untracked physical object and does not mean empty space or outside the camera view.",
        "Do not invent numbers, object identities, episodes, transitions, or timestamps.",
        "For measured facts, return a checked quantity for every cited fact ID.",
        "Cite episode or transition IDs only when those rows directly instantiate the claim; aggregate facts need no decorative row citations.",
    ],
}


def _mentioned_object_ids(question: str, summary: list[dict]) -> set[str]:
    q = question.casefold()
    found = set()
    for row in summary:
        oid, name = row["object_id"], row["object_name"]
        if oid.casefold() in q or (len(name) >= 3 and name.casefold() in q):
            found.add(oid)
    return found


def build_packet(trace_dir: str | Path, question: str,
                 max_objects: int = 10, max_episodes: int = 30,
                 max_transitions: int = 30,
                 max_transition_pair_facts: int = 20) -> dict:
    trace = load_trace(trace_dir)
    facts = build_fact_catalog(trace)
    ranked = sorted(trace["summary"], key=lambda r: int(r["display_rank"]))
    mentioned = _mentioned_object_ids(question, ranked)
    selected_ids = list(mentioned)
    for row in ranked:
        if len(selected_ids) >= max_objects:
            break
        if row["object_id"] not in selected_ids:
            selected_ids.append(row["object_id"])
    selected = set(selected_ids)

    selected_facts = [f for f in facts
                      if f["fact_id"].startswith("session.")
                      or f.get("context", {}).get("object_id") in selected]
    pair_facts = [f for f in facts if f["fact_id"].startswith("transition_pair.")
                  and (f.get("context", {}).get("from_object_id") in selected
                       or f.get("context", {}).get("to_object_id") in selected)]
    pair_facts.sort(key=lambda f: f["context"]["transition_pair_rank"])
    selected_facts.extend(pair_facts[:max_transition_pair_facts])
    object_rows = [r for r in ranked if r["object_id"] in selected]

    relevant_eps = [e for e in trace["episodes"] if e["target_id"] in selected]
    relevant_eps.sort(key=lambda e: (-float(e["duration_s"]), int(e["episode_index"])))
    episode_fields = ("episode_id", "target_id", "target_name",
                      "start_timestamp_ns", "end_timestamp_ns", "duration_s",
                      "first_gaze_sample_index", "last_gaze_sample_index",
                      "first_frame_index", "last_frame_index", "sample_count")
    episodes = [{k: e[k] for k in episode_fields} for e in relevant_eps[:max_episodes]]

    relevant_trans = [t for t in trace["transitions"]
                      if t["from_object_id"] in selected or t["to_object_id"] in selected]
    transition_fields = ("transition_id", "from_episode_id", "to_episode_id",
                         "from_object_id", "to_object_id", "from_object_name",
                         "to_object_name", "transition_timestamp_ns",
                         "unsupported_gap_s", "transition_type")
    transitions = [{k: t[k] for k in transition_fields}
                   for t in relevant_trans[:max_transitions]]

    m = trace["manifest"]
    return {
        "schema_version": "tracegaze-evidence-packet-0.1",
        "question": question.strip(),
        "source": {
            "attention_trace_artifact_id": m["artifact_id"],
            "source_system": m["source"]["system"],
            "timestamp_domain": m["source"]["timestamp_domain"],
        },
        "selection": {
            "method": "question-mentioned objects plus highest accumulated gaze-supported time",
            "limits": {"objects": max_objects, "episodes": max_episodes,
                       "transitions": max_transitions,
                       "transition_pair_facts": max_transition_pair_facts},
            "selected_object_ids": selected_ids,
        },
        "analysis_contract": CONTRACT,
        "facts": selected_facts,
        "objects": object_rows,
        "episodes": episodes,
        "transitions": transitions,
        "response_schema": {
            "answer": "concise direct answer",
            "claims": [{
                "claim_id": "c1", "claim_kind": "observed_pattern",
                "statement": "...", "fact_ids": ["..."],
                "episode_ids": ["..."], "transition_ids": ["..."],
                "quantities": [{"fact_id": "...", "value": 1.23,
                                "unit": "seconds"}],
                "uncertainty": "none|low|medium|high",
            }],
            "limitations": ["..."],
        },
    }


def make_prompt(packet: dict) -> str:
    payload = json.dumps(packet, ensure_ascii=False, indent=2)
    return (
        "You are analyzing first-person gaze evidence. Answer only from the supplied "
        "evidence packet and obey its analysis_contract. Separate deterministic "
        "observations from interpretation and hypotheses. Return JSON matching "
        "response_schema; do not wrap it in Markdown.\n\n"
        f"EVIDENCE PACKET\n{payload}\n")


def write_packet(trace_dir: str | Path, question: str, out_dir: str | Path,
                 **limits) -> dict:
    out = Path(out_dir).expanduser().resolve()
    if out.exists():
        raise ValueError(f"output directory already exists: {out}")
    out.mkdir(parents=True)
    packet = build_packet(trace_dir, question, **limits)
    (out / "evidence_packet.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "analyst_prompt.md").write_text(make_prompt(packet), encoding="utf-8")
    template = {"answer": "", "claims": [], "limitations": []}
    (out / "response_template.json").write_text(
        json.dumps(template, indent=2) + "\n", encoding="utf-8")
    return packet

