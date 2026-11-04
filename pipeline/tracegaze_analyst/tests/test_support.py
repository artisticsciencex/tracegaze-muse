import base64
import copy
import json
import os

import pytest

from tracegaze_analyst import muse
from tracegaze_analyst.findings import (FINDINGS_SCHEMA, model_view,
                                        rederive_facts, validate_findings)


def _trace(tmp_path):
    root = tmp_path / "trace"
    root.mkdir()
    (root / "attention_trace_manifest.json").write_text(json.dumps({
        "artifact_id": "trace-1",
        "source": {"system": "test", "timestamp_domain": "ns",
                   "validated_interval_ns": [0, 400_000_000]},
        "counts": {"authoritative_rows": 4, "supported_duration_s": 0.4,
                   "no_hit_duration_ns": 100_000_000, "episodes": 3, "transitions": 2},
        "validation": [{"ok": True}]}))
    (root / "object_attention_summary.csv").write_text(
        "object_id,object_name,object_color,is_tracked_object,total_attention_s,"
        "share_of_supported_time,episode_count,revisit_count,sample_count,display_rank\n"
        "obj_1,cup,#fff,true,0.3,0.75,2,1,3,1\n"
        "__no_tracked_object__,No accepted tracked-mask hit,,false,0.1,0.25,1,0,1,2\n")
    (root / "attention_episodes.csv").write_text(
        "episode_id,target_id,start_timestamp_ns,end_timestamp_ns,sample_count,gaze_sample_indices\n"
        "ep_00001,obj_1,0,200000000,2,\"[0, 1]\"\n"
        "ep_00002,__no_tracked_object__,200000000,300000000,1,[2]\n"
        "ep_00003,obj_1,300000000,400000000,1,[3]\n")
    (root / "attention_transitions.csv").write_text(
        "transition_id,from_object_id,to_object_id,transition_timestamp_ns\n"
        "tr_00001,obj_1,__no_tracked_object__,200000000\n"
        "tr_00002,__no_tracked_object__,obj_1,300000000\n")
    prox = tmp_path / "boundary_proximity.json"
    prox.write_text(json.dumps({"samples": {
        "0": {"boundary_near": False}, "1": {"boundary_near": True},
        "2": {"boundary_near": True}, "3": {"boundary_near": False}}}))
    return root, prox


def _package():
    fact = lambda fid, v, unit, **ctx: {"fact_id": fid, "metric": fid, "value": v,  # noqa: E731
                                        "unit": unit, "source_ref": "x", "context": ctx}
    return {
        "question": "q",
        "recording": {"duration_s": 0.4},
        "data_quality": {"gaze_median_interval_ms": 100.0},
        "objects": [{"object_id": "obj_1", "name": "cup"},
                    {"object_id": "__no_tracked_object__", "name": "No hit"}],
        "facts": [
            fact("session.episode_count", 3, "episodes"),
            fact("session.boundary_near_sample_count", 2, "samples"),
            fact("object.obj_1.total_attention_s", 0.3, "seconds"),
            fact("object.obj_1.revisit_count", 1, "revisits"),
            fact("object.obj_1.boundary_near_sample_count", 1, "samples"),
            fact("transition_pair.obj_1.to.__no_tracked_object__.count", 1, "transitions"),
            fact("interval.iv_01.duration_s", 0.3, "seconds", interval_id="iv_01"),
            fact("interval.iv_01.target.obj_1.time_s", 0.2, "seconds", interval_id="iv_01"),
            fact("interval.iv_01.single_sample_episode_count", 1, "episodes", interval_id="iv_01"),
            fact("phase.p1.target.obj_1.time_s", 0.3, "seconds"),
        ],
        "intervals": [{"interval_id": "iv_01", "start_s": 0.0, "end_s": 0.3,
                       "start_timestamp_ns": 0, "end_timestamp_ns": 300_000_000,
                       "episode_ids": ["ep_00001", "ep_00002"], "frame_ids": ["fr_0002"]}],
        "episodes": [{"episode_id": "ep_00001", "target_id": "obj_1", "start_s": 0.0,
                      "duration_s": 0.2, "sample_count": 2},
                     {"episode_id": "ep_00002", "target_id": "__no_tracked_object__",
                      "start_s": 0.2, "duration_s": 0.1, "sample_count": 1}],
        "frames": [{"frame_id": "fr_0002", "interval_id": "iv_01", "time_s": 0.1,
                    "file": "frames/fr_0002.jpg", "sha256": "0" * 64,
                    "alternatives_within_px": []}],
        "source_refs": {"scene_video": "/local/path.mp4"},
    }


def _response():
    return {
        "answer_summary": "The cup dominated gaze-supported time.",
        "findings": [{
            "finding_id": "F1", "title": "Cup returns",
            "statement": "Gaze returned to the cup once; it held 0.3 s of gaze-supported time.",
            "evidence_type": "measured_pattern", "interval_ids": ["iv_01"],
            "object_ids": ["obj_1"], "episode_ids": ["ep_00001"], "frame_ids": ["fr_0002"],
            "fact_ids": ["object.obj_1.total_attention_s", "object.obj_1.revisit_count"],
            "quantities": [{"fact_id": "object.obj_1.total_attention_s", "value": 0.3, "unit": "seconds"},
                           {"fact_id": "object.obj_1.revisit_count", "value": 1, "unit": "revisits"}],
            "visible_observations": [{"frame_id": "fr_0002", "observation": "A cup is on the table."}],
            "interpretation": "", "alternative_explanations": [], "limitations": ["Short clip."],
            "uncertainty": "low",
            "support_opportunity": {"status": "not_supported_by_evidence",
                                    "description": "", "evidence_needed": ""},
        }, {
            "finding_id": "F2", "title": "Brief no-hit",
            "statement": "One single-sample episode had no accepted tracked-mask hit.",
            "evidence_type": "possible_interpretation", "interval_ids": ["iv_01"],
            "object_ids": ["__no_tracked_object__"], "episode_ids": ["ep_00002"], "frame_ids": [],
            "fact_ids": ["interval.iv_01.single_sample_episode_count"],
            "quantities": [{"fact_id": "interval.iv_01.single_sample_episode_count", "value": 1,
                            "unit": "episodes"}],
            "visible_observations": [],
            "interpretation": "The wearer may have been checking something off the table.",
            "alternative_explanations": ["Gaze near the cup's mask edge."],
            "limitations": [], "uncertainty": "high",
            "support_opportunity": {"status": "candidate_for_investigation",
                                    "description": "Study brief off-object glances.",
                                    "evidence_needed": "Hand tracking and audio."},
        }],
        "remaining_questions": ["What was said?"],
    }


def test_clean_response_verifies():
    v = validate_findings(_package(), _response())
    assert v["ok"], v
    assert all(q["match"] for r in v["findings"] for q in r["quantity_checks"])


def test_invalid_references_are_rejected():
    r = _response()
    r["findings"][0]["frame_ids"] = ["fr_9999"]
    r["findings"][0]["visible_observations"][0]["frame_id"] = "fr_9999"
    r["findings"][1]["interval_ids"].append("iv_99")
    v = validate_findings(_package(), r)
    errs = [e for f in v["findings"] for e in f["errors"]]
    assert any("fr_9999" in e for e in errs) and any("iv_99" in e for e in errs)
    assert v["summary"]["needs_review"] == 2


def test_altered_quantity_is_rejected():
    r = _response()
    r["findings"][0]["quantities"][0]["value"] = 0.31
    v = validate_findings(_package(), r)
    assert any("does not match" in e for e in v["findings"][0]["errors"])


def test_untraceable_prose_number_is_rejected_and_rounding_allowed():
    r = _response()
    r["findings"][0]["statement"] = "Gaze held the cup for 0.30 s, about 75% of the time, then 9.5 s more."
    r["findings"][0]["fact_ids"].append("object.obj_1.share_of_supported_time")
    pkg = _package()
    pkg["facts"].append({"fact_id": "object.obj_1.share_of_supported_time", "metric": "share",
                         "value": 0.75, "unit": "proportion", "source_ref": "x"})
    v = validate_findings(pkg, r)
    errs = v["findings"][0]["errors"]
    assert errs == ["statement: number '9.5' is not traceable to the finding's cited evidence"]


def test_construct_and_personal_language_gated():
    r = _response()
    r["findings"][0]["statement"] = "She looked confused by the cup."
    v = validate_findings(_package(), r)
    errs = " ".join(v["findings"][0]["errors"])
    assert "'She'" in errs and "'confused'" in errs
    # construct words are allowed inside a hedged interpretation
    r = _response()
    r["findings"][1]["interpretation"] = "This could reflect searching or memory demands."
    assert validate_findings(_package(), r)["findings"][1]["status"] == "references_verified"


def test_interpretation_needs_alternative_and_uncertainty():
    r = _response()
    r["findings"][1]["alternative_explanations"] = []
    r["findings"][1]["uncertainty"] = "low"
    errs = validate_findings(_package(), r)["findings"][1]["errors"]
    assert any("alternative" in e for e in errs) and any("uncertainty" in e for e in errs)


def test_rederivation_matches_and_catches_edited_package(tmp_path):
    trace, prox = _trace(tmp_path)
    pkg = _package()
    red = rederive_facts(pkg, trace, prox)
    assert red["mismatches"] == [] and red["checked"] == len(pkg["facts"])
    bad = copy.deepcopy(pkg)
    bad["facts"][2]["value"] = 2.3
    red = rederive_facts(bad, trace, prox)
    assert [m["fact_id"] for m in red["mismatches"]] == ["object.obj_1.total_attention_s"]
    assert not validate_findings(bad, _response(), red)["ok"]


def test_model_view_strips_local_paths_without_mutating():
    pkg = _package()
    view = model_view(pkg)
    assert "source_refs" not in view and "file" not in view["frames"][0]
    assert "source_ref" not in view["facts"][0]
    assert pkg["facts"][0]["source_ref"] == "x" and "file" in pkg["frames"][0]


def test_schema_is_strict_compatible():
    def walk(node):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                walk(child)
        if node.get("type") == "array":
            walk(node["items"])
    walk(FINDINGS_SCHEMA)


def test_request_redaction_and_key_handling(tmp_path, monkeypatch):
    jpeg = b"\xff\xd8fake-jpeg"
    body = muse.chat_request([{"role": "user", "content": [
        {"type": "text", "text": "hi"}, muse.image_part(jpeg)]}], FINDINGS_SCHEMA)
    red = muse.redact_images(body)
    part = red["messages"][0]["content"][1]["image_url"]
    assert part["bytes"] == len(jpeg) and "base64" not in json.dumps(red)
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(body["messages"][0]["content"][1]["image_url"]["url"].split(",")[1]) == jpeg

    monkeypatch.delenv(muse.KEY_ENV, raising=False)
    with pytest.raises(muse.MuseError, match="not set"):
        muse._key()
    env = tmp_path / "x.env"
    env.write_text("OTHER=1\nexport MODEL_API_KEY='sk-test-123'\n")
    assert muse.load_env_file(env) == ["MODEL_API_KEY"]
    assert os.environ["MODEL_API_KEY"] == "sk-test-123" and "OTHER" not in os.environ
    assert muse._scrub("error sk-test-123 here") == "error [REDACTED] here"
