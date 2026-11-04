"""attention_trace contracts.

Synthetic fixtures drive the temporal logic; the three approved real
fusion artifacts are exercised read-only as an opt-in smoke test.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from attention_trace import NO_TRACKED_OBJECT, NO_TRACKED_OBJECT_NAME
from attention_trace import policy as pol
from attention_trace import trace as tr
from attention_trace.artifact import (build_trace, load_attention_trace,
                                      publish_attention_trace)
from attention_trace.cli import default_out_dir, main as cli_main
from common.artifacts import ArtifactError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent
                       / "gaze_object_fusion" / "tests"))
from futil import WORLD_TS, make_tracked_artifact       # noqa: E402

BIG = {"name": "big", "box": (20, 20, 100, 100), "score": 0.95}
SMALL = {"name": "small", "box": (40, 40, 60, 60), "score": 0.9}
ON_BIG = {"x": 90.5, "y": 30.5}          # inside BIG only
ON_SMALL = {"x": 50.5, "y": 50.5}        # inside SMALL (and BIG)
OFF = {"x": 5.0, "y": 5.0}               # inside nothing

STEP = 20_000_000                        # 20 ms synthetic gaze cadence


def gaze(seq):
    """seq: list of (offset_index, position dict, extra kwargs)."""
    rows = []
    for k, (i, pos, extra) in enumerate(seq):
        rows.append({"ts": WORLD_TS[0] + i * STEP, **pos, **extra})
    rows.sort(key=lambda r: r["ts"])
    return rows


def fusion(tmp_path, seq, objects=(BIG, SMALL), name="f"):
    """Publish a real fusion artifact from synthetic gaze (no demo)."""
    from gaze_object_fusion.artifact import publish_fusion_artifact
    tracked, trim, oids = make_tracked_artifact(tmp_path / name, gaze(seq),
                                                list(objects))
    out = tmp_path / f"{name}_gaze_fusion"
    publish_fusion_artifact(tracked, out, make_demo=False)
    return out, oids


# --------------------------------------------------------------- policy

def test_gap_policy_matches_the_documented_clamp():
    p = pol.gap_policy([i * 100_000_000 for i in range(10)])
    assert p["median_interval_ns"] == 100_000_000
    assert p["max_gap_ns"] == 250_000_000          # clamped high
    assert pol.gap_policy([i * 5_000_000 for i in range(6)])["max_gap_ns"] \
        == 50_000_000                              # clamped low
    assert pol.gap_policy([i * 30_000_000 for i in range(6)])["max_gap_ns"] \
        == 90_000_000                              # 3 x median
    assert pol.gap_policy([])["max_gap_ns"] == 250_000_000
    assert "clamp(3 x median" in p["rule"]


def test_support_intervals_are_exact_non_overlapping_integers():
    rows = [{"ts_ns": 1_000_000_000 + i * 20_000_000, "gaze_sample_index": i,
             "row": i, "frame_index": i, "status": "on_object",
             "target_id": "obj_0001", "target_name": "a",
             "track_score": 0.9, "frame_offset_ms": 0.0} for i in range(5)]
    sup, dropped = pol.support_intervals(rows, 0, 10_000_000_000,
                                         60_000_000)
    assert len(sup) == 5 and dropped == []
    for s in sup:
        assert isinstance(s["support_start_ns"], int)
        assert isinstance(s["support_end_ns"], int)
    for a, b in zip(sup, sup[1:]):
        assert a["support_end_ns"] == b["support_start_ns"], "tile exactly"
    total = sum(s["support_end_ns"] - s["support_start_ns"] for s in sup)
    assert total == sup[-1]["support_end_ns"] - sup[0]["support_start_ns"]


def test_support_is_clipped_to_the_validated_interval():
    rows = [{"ts_ns": 100, "gaze_sample_index": 0, "row": 0,
             "frame_index": 0, "status": "background",
             "target_id": NO_TRACKED_OBJECT, "target_name": None,
             "track_score": None, "frame_offset_ms": None}]
    sup, dropped = pol.support_intervals(rows, 90, 110, 100_000_000)
    assert sup[0]["support_start_ns"] == 90 and sup[0]["support_end_ns"] == 110
    assert dropped == []


# ------------------------------------------------------ episodes/trace

def test_episodes_are_deterministic_with_stable_ids(tmp_path):
    seq = [(i, ON_BIG, {}) for i in range(6)]
    fdir, oids = fusion(tmp_path, seq)
    a = build_trace(fdir)
    b = build_trace(fdir)
    assert [e["episode_id"] for e in a["episodes"]] == \
        [e["episode_id"] for e in b["episodes"]]
    assert a["episodes"] == b["episodes"]
    assert a["transitions"] == b["transitions"]
    assert [e["episode_id"] for e in a["episodes"]] == \
        [f"ep_{i + 1:05d}" for i in range(len(a["episodes"]))]
    assert [e["episode_index"] for e in a["episodes"]] == \
        list(range(len(a["episodes"])))


def test_object_to_object_and_no_hit_transitions(tmp_path):
    seq = ([(i, ON_BIG, {}) for i in range(0, 3)]
           + [(i, ON_SMALL, {}) for i in range(3, 6)]
           + [(i, OFF, {}) for i in range(6, 9)]
           + [(i, ON_BIG, {}) for i in range(9, 12)])
    fdir, oids = fusion(tmp_path, seq)
    res = build_trace(fdir)
    eps, trs = res["episodes"], res["transitions"]
    targets = [e["target_id"] for e in eps]
    assert len(eps) == 4 and len(trs) == 3
    assert targets[2] == NO_TRACKED_OBJECT
    assert eps[2]["target_name"] == NO_TRACKED_OBJECT_NAME
    assert eps[2]["is_tracked_object"] is False
    types = [t["transition_type"] for t in trs]
    assert types == [tr.TYPE_OBJ_OBJ, tr.TYPE_OBJ_NOHIT, tr.TYPE_NOHIT_OBJ]
    assert all(t["unsupported_gap_ns"] == 0 for t in trs), "contiguous"
    assert trs[0]["from_object_id"] == targets[0]
    assert trs[0]["to_object_id"] == targets[1]
    # transitions link consecutive episodes by id
    assert [t["from_episode_id"] for t in trs] == \
        [e["episode_id"] for e in eps[:-1]]


def test_revisit_counts_returns_not_samples(tmp_path):
    seq = ([(i, ON_BIG, {}) for i in range(0, 3)]
           + [(i, ON_SMALL, {}) for i in range(3, 5)]
           + [(i, ON_BIG, {}) for i in range(5, 8)]
           + [(i, ON_SMALL, {}) for i in range(8, 10)]
           + [(i, ON_BIG, {}) for i in range(10, 12)])
    fdir, oids = fusion(tmp_path, seq)
    res = build_trace(fdir)
    by_id = {r["object_id"]: r for r in res["summary"]}
    big = next(r for r in res["summary"]
               if r["object_name"] == "big" or r["episode_count"] == 3)
    assert big["episode_count"] == 3
    assert big["revisit_count"] == 2, "returns after leaving, not samples"
    assert big["sample_count"] == 8 and big["sample_count"] != \
        big["revisit_count"]
    single = [r for r in res["summary"] if r["episode_count"] == 1]
    for r in single:
        assert r["revisit_count"] == 0


def test_invalid_samples_split_episodes(tmp_path):
    seq = ([(i, ON_BIG, {}) for i in range(0, 3)]
           + [(3, ON_BIG, {"worn": 0.0})]            # invalid: breaks
           + [(i, ON_BIG, {}) for i in range(4, 7)])
    fdir, oids = fusion(tmp_path, seq)
    res = build_trace(fdir)
    statuses = list(pd.read_parquet(
        res["artifact"].gaze_samples_parquet_path)["gaze_status"])
    assert "invalid_gaze" in statuses
    assert len(res["rows"]) == 6, "the invalid row is not eligible"
    assert all(r["gaze_sample_index"] != 3 for r in res["rows"])
    # same target on both sides, but the break must not merge them into
    # one episode that silently spans the invalid region
    assert len(res["episodes"]) >= 1
    covered = sum(e["duration_ns"] for e in res["episodes"])
    assert covered == sum(s["support_end_ns"] - s["support_start_ns"]
                          for s in res["supported"])


def test_long_gaps_are_unsupported_not_accumulated(tmp_path):
    # 20 ms cadence -> max_gap 60 ms; a 10-step (200 ms) hole exceeds it
    seq = ([(i, ON_BIG, {}) for i in range(0, 3)]
           + [(i, ON_BIG, {}) for i in range(13, 16)])
    fdir, oids = fusion(tmp_path, seq)
    res = build_trace(fdir)
    assert res["policy"]["max_gap_ns"] == 60_000_000
    assert len(res["episodes"]) == 2, "a long gap must split the episode"
    a, b = res["episodes"]
    gap = b["start_timestamp_ns"] - a["end_timestamp_ns"]
    assert gap > 0, "the hole stays unsupported"
    assert res["transitions"][0]["unsupported_gap_ns"] == gap
    assert res["transitions"][0]["transition_type"] == tr.TYPE_OBJ_OBJ
    total = sum(e["duration_ns"] for e in res["episodes"])
    span = b["end_timestamp_ns"] - a["start_timestamp_ns"]
    assert total < span, "the gap is never accumulated as attention"


def test_no_double_counting_and_duration_conservation(tmp_path):
    seq = [(i, ON_BIG if i % 3 else ON_SMALL, {}) for i in range(24)]
    fdir, oids = fusion(tmp_path, seq)
    res = build_trace(fdir)
    ok, problems = tr.check_no_double_counting(res["episodes"])
    assert ok, problems
    ep = sum(e["duration_ns"] for e in res["episodes"])
    sup = sum(s["support_end_ns"] - s["support_start_ns"]
              for s in res["supported"])
    summ = sum(r["total_attention_ns"] for r in res["summary"])
    assert ep == sup == summ
    lo, hi = res["interval"]
    assert ep <= hi - lo
    shares = sum(r["share_of_supported_time"] for r in res["summary"])
    assert shares == pytest.approx(1.0)


def test_overlap_resolution_is_inherited_from_fusion(tmp_path):
    """Gaze inside BOTH masks must use fusion's primary (smaller area)."""
    seq = [(i, ON_SMALL, {}) for i in range(5)]
    fdir, oids = fusion(tmp_path, seq)
    fused = pd.read_parquet(fdir / "gaze_samples.parquet")
    assert (fused["hit_object_count"] >= 2).any()
    primaries = set(fused["primary_object_id"].dropna())
    res = build_trace(fdir)
    assert {e["target_id"] for e in res["episodes"]} == primaries
    assert all(e["target_name"] == "small" for e in res["episodes"])


def test_eligibility_rejects_inconsistent_fusion_rows():
    df = pd.DataFrame({
        "gaze_sample_index": [0], "timestamp_ns": [1],
        "frame_index": [0], "gaze_status": ["on_object"],
        "gaze_valid": [True], "primary_object_id": [None],
        "primary_object_name": [None], "primary_track_score": [None],
        "frame_offset_ms": [0.0]})
    with pytest.raises(ValueError, match="without primary_object_id"):
        pol.eligible_rows(df)


# ----------------------------------------------------- artifact contract

def test_publish_load_roundtrip_and_parent_immutability(tmp_path):
    seq = ([(i, ON_BIG, {}) for i in range(0, 4)]
           + [(i, OFF, {}) for i in range(4, 7)]
           + [(i, ON_SMALL, {}) for i in range(7, 10)])
    fdir, oids = fusion(tmp_path, seq)

    def digest(d):
        h = hashlib.sha256()
        for p in sorted(Path(d).rglob("*")):
            if p.is_file():
                h.update(p.name.encode())
                h.update(p.read_bytes())
        return h.hexdigest()

    before = digest(fdir)
    public_before = (fdir / "gaze_samples.csv").read_bytes()
    out = tmp_path / "trace"
    m = publish_attention_trace(fdir, out)
    assert digest(fdir) == before, "parent fusion artifact must not change"
    assert (fdir / "gaze_samples.csv").read_bytes() == public_before

    for f in ("attention_trace_manifest.json", "attention_episodes.csv",
              "attention_episodes.parquet", "attention_transitions.csv",
              "attention_transitions.parquet",
              "object_attention_summary.csv", "attention_trace.json"):
        assert (out / f).is_file(), f
    assert all(v["ok"] for v in m["validation"])
    assert m["parent_artifact"]["files_sha256"]["gaze_samples_parquet"]
    assert m["evidence_policy"]["excluded"] == ["gaze_demo.mp4",
                                                "demo_presentation.csv"]

    art = load_attention_trace(out)
    assert art.artifact_id == m["artifact_id"]

    # csv and parquet agree; json mirrors the tables
    eps_csv = list(csv.DictReader(open(out / "attention_episodes.csv")))
    eps_pq = pd.read_parquet(out / "attention_episodes.parquet")
    assert len(eps_csv) == len(eps_pq) == m["counts"]["episodes"]
    assert [r["episode_id"] for r in eps_csv] == list(eps_pq["episode_id"])
    # CSV list columns round-trip as JSON and match the parquet lists
    for row, want in zip(eps_csv, eps_pq["gaze_sample_indices"]):
        assert json.loads(row["gaze_sample_indices"]) == list(want)
    trace = json.loads((out / "attention_trace.json").read_text())
    assert len(trace["episodes"]) == m["counts"]["episodes"]
    assert len(trace["transitions"]) == m["counts"]["transitions"]
    assert trace["no_tracked_object"]["id"] == NO_TRACKED_OBJECT
    assert "never used as evidence" in trace["evidence"]
    # evidence links back to concrete sample indices and frames
    assert all(e["gaze_sample_indices"] for e in trace["episodes"])
    assert all(len(e["frames"]) == 2 for e in trace["episodes"])


def test_publication_refuses_an_existing_destination(tmp_path):
    fdir, _ = fusion(tmp_path, [(i, ON_BIG, {}) for i in range(4)])
    out = tmp_path / "trace"
    publish_attention_trace(fdir, out)
    with pytest.raises(Exception):
        publish_attention_trace(fdir, out)


def test_loader_rejects_malformed_or_inconsistent_artifacts(tmp_path):
    fdir, _ = fusion(tmp_path, [(i, ON_BIG, {}) for i in range(6)])
    out = tmp_path / "trace"
    publish_attention_trace(fdir, out)
    mf = out / "attention_trace_manifest.json"
    good = mf.read_text()

    m = json.loads(good)
    m["counts"]["episodes"] += 1
    mf.write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="counts.episodes"):
        load_attention_trace(out)

    m = json.loads(good)
    m["validation"][0]["ok"] = False
    mf.write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="failures"):
        load_attention_trace(out)

    m = json.loads(good)
    del m["files"]["episodes_parquet"]
    mf.write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="episodes_parquet"):
        load_attention_trace(out)

    mf.write_text(good)
    pd.read_parquet(out / "attention_episodes.parquet").head(0).to_parquet(
        out / "attention_episodes.parquet", index=False)
    with pytest.raises(ArtifactError, match="sha256"):
        load_attention_trace(out)


def test_cli_default_naming_and_refusal(tmp_path, capsys):
    assert default_out_dir(Path("/a/b/rec_gaze_fusion")).name == \
        "rec_attention_trace"
    fdir, _ = fusion(tmp_path, [(i, ON_BIG, {}) for i in range(5)])
    out = tmp_path / "cli_out"
    assert cli_main(["--fusion", str(fdir), "--out", str(out),
                     "--quiet"]) == 0
    assert (out / "attention_trace.json").is_file()
    assert cli_main(["--fusion", str(fdir), "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert "already exists" in err


# ------------------------------------- read-only smoke on real artifacts

REAL = [
    Path("data/neon-data/"
         "lab_study/participant_01/"
         "2024-01-01_00-00-00-0000abcd_participant_a_trim_f3000-f3299"
         "_gaze_fusion_v3"),
    Path("data/aria/"
         "recording_1085015926322699_trim_f2400-f2699_gaze_fusion"),
    Path("data/aria/"
         "AriaEverydayActivities_loc4_script3_seq4_rec1_prepared_aria_trim"
         "_f709-f1463_gaze_fusion"),
]


@pytest.mark.parametrize("fdir", REAL, ids=lambda p: p.name[:40])
def test_real_fusion_artifacts_smoke_read_only(fdir):
    if not fdir.is_dir():
        pytest.skip(f"approved fusion artifact not present: {fdir.name}")

    def digest(d):
        h = hashlib.sha256()
        for p in sorted(Path(d).rglob("*")):
            if p.is_file():
                h.update(p.name.encode())
                h.update(p.read_bytes())
        return h.hexdigest()

    before = digest(fdir)
    res = build_trace(fdir)                   # in-memory only, nothing written
    assert digest(fdir) == before, "smoke generation must not touch the parent"
    assert all(v["ok"] for v in res["validation"]), \
        [v for v in res["validation"] if not v["ok"]]
    assert res["episodes"], "a real recording must yield episodes"
    assert len(res["transitions"]) == len(res["episodes"]) - 1
    ep = sum(e["duration_ns"] for e in res["episodes"])
    lo, hi = res["interval"]
    assert 0 < ep <= hi - lo
    assert all(isinstance(e["duration_ns"], int) for e in res["episodes"])
    print(f"\n{fdir.name}: {len(res['episodes'])} episodes, "
          f"{len(res['transitions'])} transitions, "
          f"{ep / 1e9:.3f} s supported, "
          f"{sum(1 for r in res['summary'] if r['is_tracked_object'])} "
          f"objects attended")


def test_provenance_is_resolved_from_declared_metadata_never_sniffed(tmp_path):
    """Timestamp domain must come from declared provenance; an
    unresolvable parent yields 'unknown', never a guessed NEON default."""
    from attention_trace.artifact import provenance
    fdir, _ = fusion(tmp_path, [(i, ON_BIG, {}) for i in range(5)])
    fm = json.loads((fdir / "fusion_manifest.json").read_text())
    got = provenance(fm)
    assert got["source_system"] == "pupil_neon"
    assert got["timestamp_domain"] == "neon_world_time"
    assert "tracked manifest" in got["resolved_from"]

    # an Aria parent must never be reported as NEON
    tracked = Path(fm["parent_artifact"]["artifact_dir"])
    tm_path = tracked / "tracking_manifest.json"
    tm = json.loads(tm_path.read_text())
    tm["synchronization"]["source_system"] = "project_aria"
    tm["synchronization"]["timestamp_domain"] = "aria_device_time"
    tm_path.write_text(json.dumps(tm))
    aria = provenance(fm)
    assert aria == {"source_system": "project_aria",
                    "timestamp_domain": "aria_device_time",
                    "resolved_from": "tracked manifest synchronization block"}

    # older artifacts fall back to parent.source_type
    del tm["synchronization"]["source_system"]
    del tm["synchronization"]["timestamp_domain"]
    tm["parent"]["source_type"] = "trimmed_aria_recording"
    tm_path.write_text(json.dumps(tm))
    assert provenance(fm)["timestamp_domain"] == "aria_device_time"

    # unresolvable -> unknown, never guessed
    assert provenance({})["timestamp_domain"] == "unknown"
    assert provenance({"parent_artifact": {"artifact_dir": str(tmp_path)}}
                      )["source_system"] == "unknown"


# ------------------------------- regressions from the adversarial review

def test_odd_inter_sample_gaps_do_not_overlap_by_one_nanosecond():
    """An asymmetric floor/ceil midpoint pair overlaps by 1 ns whenever
    the gap is odd; both boundaries must use the same floor midpoint."""
    ts = [1_000_000_000, 1_020_000_001, 1_040_000_002]
    rows = [{"ts_ns": t, "gaze_sample_index": i, "row": i, "frame_index": i,
             "status": "on_object", "target_id": "obj_0001",
             "target_name": "a", "track_score": 0.9, "frame_offset_ms": 0.0}
            for i, t in enumerate(ts)]
    sup, _ = pol.support_intervals(rows, 0, 10_000_000_000, 60_000_000)
    for a, b in zip(sup, sup[1:]):
        assert a["support_end_ns"] == b["support_start_ns"], \
            "odd gaps must still tile exactly"
    total = sum(s["support_end_ns"] - s["support_start_ns"] for s in sup)
    span = sup[-1]["support_end_ns"] - sup[0]["support_start_ns"]
    assert total == span, "no nanosecond may be attributed twice"
    eps = tr.build_episodes(sup)
    ok, problems = tr.check_no_double_counting(eps)
    assert ok, problems


def test_breaking_samples_really_break_support_and_split_episodes(tmp_path):
    """A blink shorter than max_gap must NOT be bridged: its own midpoint
    territory is unsupported and the episode splits."""
    seq = ([(i, ON_BIG, {}) for i in range(0, 3)]
           + [(3, ON_BIG, {"worn": 0.0})]            # invalid, 20 ms hole
           + [(i, ON_BIG, {}) for i in range(4, 7)])
    fdir, _ = fusion(tmp_path, seq)
    res = build_trace(fdir)
    assert len(res["breaks"]) == 1
    eps = res["episodes"]
    assert len(eps) == 2, "the break must split the run, not be bridged"
    assert eps[0]["target_id"] == eps[1]["target_id"]
    brk = res["breaks"][0]["ts_ns"]
    assert eps[0]["end_timestamp_ns"] <= brk <= eps[1]["start_timestamp_ns"]
    trs = res["transitions"]
    assert len(trs) == 1 and trs[0]["transition_type"] == tr.TYPE_OBJ_OBJ
    assert trs[0]["unsupported_gap_ns"] > 0, \
        "the blink's own territory stays unsupported"
    # and its time is never counted as attention
    total = sum(e["duration_ns"] for e in eps)
    span = eps[1]["end_timestamp_ns"] - eps[0]["start_timestamp_ns"]
    assert total < span
    summ = {r["object_id"]: r for r in res["summary"]}
    obj = next(r for r in res["summary"] if r["is_tracked_object"])
    assert obj["episode_count"] == 2 and obj["revisit_count"] == 1


def test_provenance_refuses_a_mismatched_tracked_manifest(tmp_path):
    from attention_trace.artifact import provenance
    fdir, _ = fusion(tmp_path, [(i, ON_BIG, {}) for i in range(5)])
    fm = json.loads((fdir / "fusion_manifest.json").read_text())
    tm_path = Path(fm["parent_artifact"]["artifact_dir"]) \
        / "tracking_manifest.json"
    tm = json.loads(tm_path.read_text())
    tm["artifact_id"] = "0" * 32                     # a different artifact
    tm_path.write_text(json.dumps(tm))
    got = provenance(fm)
    assert got["source_system"] == "unknown"
    assert "identity mismatch" in got["resolved_from"]


def test_loader_requires_a_hash_for_every_declared_file(tmp_path):
    fdir, _ = fusion(tmp_path, [(i, ON_BIG, {}) for i in range(5)])
    out = tmp_path / "trace"
    publish_attention_trace(fdir, out)
    mf = out / "attention_trace_manifest.json"
    m = json.loads(mf.read_text())
    del m["files_sha256"]["episodes_parquet"]
    mf.write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="missing entries"):
        load_attention_trace(out)


def test_accepted_objects_with_zero_attention_are_reported(tmp_path):
    """A tracked object nobody looked at is a finding, not an absence."""
    seq = [(i, ON_BIG, {}) for i in range(6)]        # never on SMALL alone
    fdir, oids = fusion(tmp_path, seq, objects=(BIG, SMALL))
    res = build_trace(fdir)
    ids = {r["object_id"] for r in res["summary"]}
    assert set(oids) <= ids, "every accepted object appears"
    zero = [r for r in res["summary"] if r["total_attention_ns"] == 0]
    assert zero, "the unattended object is listed with zero dwell"
    for r in zero:
        assert r["episode_count"] == 0 and r["revisit_count"] == 0
        assert r["mean_episode_ns"] == 0
        assert r["first_attention_timestamp_ns"] is None
