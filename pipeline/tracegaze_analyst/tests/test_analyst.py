import json

from tracegaze_analyst.baseline import make_baseline_response
from tracegaze_analyst.packet import build_packet, make_prompt
from tracegaze_analyst.validate import validate_response


def _trace(tmp_path):
    root = tmp_path / "trace"
    root.mkdir()
    manifest = {
        "artifact_id": "trace-1", "source": {"system": "test", "timestamp_domain": "ns"},
        "counts": {"authoritative_rows": 4, "supported_duration_s": 0.4,
                   "no_hit_duration_ns": 100_000_000, "episodes": 3, "transitions": 2},
        "validation": [{"ok": True}],
    }
    (root / "attention_trace_manifest.json").write_text(json.dumps(manifest))
    (root / "object_attention_summary.csv").write_text(
        "object_id,object_name,object_color,is_tracked_object,total_attention_ns,total_attention_s,share_of_supported_time,episode_count,revisit_count,sample_count,first_attention_timestamp_ns,last_attention_timestamp_ns,longest_episode_ns,mean_episode_ns,display_rank\n"
        "obj_1,cup,#fff,true,300000000,0.3,0.75,2,1,3,0,400000000,200000000,150000000,1\n"
        "__no_tracked_object__,No accepted tracked-mask hit,,false,100000000,0.1,0.25,1,0,1,200000000,300000000,100000000,100000000,2\n")
    (root / "attention_episodes.csv").write_text(
        "episode_id,episode_index,target_id,target_name,is_tracked_object,start_timestamp_ns,end_timestamp_ns,duration_ns,duration_s,first_gaze_sample_index,last_gaze_sample_index,first_frame_index,last_frame_index,sample_count,mean_frame_offset_ms,median_frame_offset_ms,mean_track_score,gaze_sample_indices,frame_indices\n"
        "ep_00001,0,obj_1,cup,true,0,200000000,200000000,0.2,0,1,0,1,2,0,0,1.0,[],[]\n"
        "ep_00002,1,__no_tracked_object__,No accepted tracked-mask hit,false,200000000,300000000,100000000,0.1,2,2,2,2,1,0,0,,[],[]\n"
        "ep_00003,2,obj_1,cup,true,300000000,400000000,100000000,0.1,3,3,3,3,1,0,0,1.0,[],[]\n")
    (root / "attention_transitions.csv").write_text(
        "transition_id,transition_index,from_episode_id,to_episode_id,from_object_id,to_object_id,from_object_name,to_object_name,transition_timestamp_ns,unsupported_gap_ns,unsupported_gap_s,transition_type,from_episode_duration_ns,to_episode_duration_ns\n"
        "tr_00001,0,ep_00001,ep_00002,obj_1,__no_tracked_object__,cup,No accepted tracked-mask hit,200000000,0,0,object_to_no_hit,200000000,100000000\n"
        "tr_00002,1,ep_00002,ep_00003,__no_tracked_object__,obj_1,No accepted tracked-mask hit,cup,300000000,0,0,no_hit_to_object,100000000,100000000\n")
    return root


def test_packet_and_baseline_validate(tmp_path):
    packet = build_packet(_trace(tmp_path), "How was the cup revisited?")
    assert packet["selection"]["selected_object_ids"][0] == "obj_1"
    assert "Never infer attention" in make_prompt(packet)
    response = make_baseline_response(packet)
    assert validate_response(packet, response)["ok"]


def test_unknown_reference_is_rejected(tmp_path):
    packet = build_packet(_trace(tmp_path), "What happened?")
    response = {"answer": "x", "limitations": [], "claims": [{
        "claim_id": "c1", "claim_kind": "observed_pattern", "statement": "x",
        "fact_ids": ["invented.fact"], "episode_ids": [], "transition_ids": [],
        "quantities": [], "uncertainty": "none"}]}
    result = validate_response(packet, response)
    assert not result["ok"]
    assert "invented.fact" in result["problems"][0]


def test_interpretation_requires_uncertainty_and_evidence(tmp_path):
    packet = build_packet(_trace(tmp_path), "What happened?")
    response = {"answer": "x", "limitations": [], "claims": [{
        "claim_id": "c1", "claim_kind": "interpretation", "statement": "x",
        "fact_ids": [], "episode_ids": [], "transition_ids": [],
        "quantities": [], "uncertainty": "none"}]}
    result = validate_response(packet, response)
    assert not result["ok"]
    assert len(result["problems"]) == 2


def test_wrong_numeric_value_is_rejected(tmp_path):
    packet = build_packet(_trace(tmp_path), "What happened?")
    response = {"answer": "The duration was 99 seconds.", "limitations": [], "claims": [{
        "claim_id": "c1", "claim_kind": "measured_fact", "statement": "99 seconds",
        "fact_ids": ["session.supported_duration_s"], "episode_ids": [],
        "transition_ids": [], "uncertainty": "none",
        "quantities": [{"fact_id": "session.supported_duration_s",
                        "value": 99, "unit": "seconds"}]}]}
    result = validate_response(packet, response)
    assert not result["ok"]
    assert "value does not match" in result["problems"][0]


def test_unsupported_fixation_language_is_rejected(tmp_path):
    packet = build_packet(_trace(tmp_path), "What happened?")
    response = {"answer": "The subject's attention was anchored.", "limitations": [], "claims": [{
        "claim_id": "c1", "claim_kind": "interpretation",
        "statement": "Frequent fixation likely reflects saccades.",
        "fact_ids": ["session.episode_count"], "episode_ids": [],
        "transition_ids": [], "quantities": [], "uncertainty": "medium"}]}
    result = validate_response(packet, response)
    assert not result["ok"]
    assert any("unsupported assertion language" in p for p in result["problems"])
