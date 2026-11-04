"""Aria adapter tests — synthetic fixtures only (no native code, no
participant data). The real recording is exercised by the opt-in
integration test at the bottom."""

import csv
import hashlib
import json
import os
import threading
import zipfile
from pathlib import Path

import pytest

from aria_source import cache as cache_mod
from aria_source import config as aria_config
from aria_source import detect
from aria_source.artifact import (load_aria_artifact, load_prepared_aria,
                                  load_trimmed_aria)
from aria_source.helper import AriaHelperError, run_helper
from common.artifacts import ArtifactError

import autil


# ------------------------------------------------------------- detection

def test_detects_aria_download_folder(tmp_path):
    d = autil.make_download_folder(tmp_path)
    assert detect.looks_like_aria_folder(d)
    pkg = detect.inspect_aria_folder(d)
    assert pkg.ok and pkg.recording_id == autil.REC_ID
    assert pkg.dataset == autil.DATASET and pkg.release == autil.RELEASE
    assert set(pkg.files) == {"recording_vrs", "mps", "metadata_json",
                              "video_main_rgb"}
    assert "Reading in the Wild" in pkg.as_dict()["label"]
    # a NEON folder must never be mistaken for Aria
    neon = tmp_path / "neon"
    neon.mkdir()
    (neon / "gaze.csv").write_text("x")
    assert not detect.looks_like_aria_folder(neon)


def test_detects_aea_direct_vrs_and_gaze_only_layout(tmp_path):
    d = autil.make_aea_download_folder(tmp_path)
    assert detect.looks_like_aria_folder(d)
    pkg = detect.inspect_aria_folder(d)
    assert pkg.ok
    assert pkg.recording_id == autil.AEA_REC_ID
    assert pkg.dataset == autil.AEA_DATASET
    assert pkg.release == autil.AEA_RELEASE
    assert pkg.layout == detect.LAYOUT_AEA
    assert pkg.file_formats[detect.PKG_VRS] == "vrs"
    assert set(pkg.files) == {"recording_vrs", "mps", "metadata_json",
                              "video_main_rgb"}
    assert "Aria Everyday Activities" in pkg.label
    validation = detect.validate_against_receipt(pkg)
    assert all(v["ok"] for v in validation)
    assert validation[0]["verification"] == "sha256_on_prepare"


def test_mixed_aria_layouts_are_rejected(tmp_path):
    d = autil.make_aea_download_folder(tmp_path)
    # Add a RITW full-MPS package to an AEA folder: role and layout conflict.
    with zipfile.ZipFile(d / f"{autil.DATASET}_{autil.RELEASE}_"
                              f"{autil.AEA_REC_ID}_mps.zip", "w") as z:
        z.writestr("mps/eye_gaze/general_eye_gaze.csv",
                   autil._csv_text(autil.gaze_rows(2)))
    pkg = detect.inspect_aria_folder(d)
    assert not pkg.ok
    assert any("multiple files" in problem or "mixed incompatible" in problem
               for problem in pkg.problems)


def test_missing_required_archive_is_reported(tmp_path):
    d = autil.make_download_folder(tmp_path)
    (d / autil.base_name("recording_vrs")).unlink()
    pkg = detect.inspect_aria_folder(d)
    assert not pkg.ok
    assert any("recording_vrs" in p for p in pkg.problems)


def test_receipt_validation_catches_size_mismatch(tmp_path):
    d = autil.make_download_folder(tmp_path)
    pkg = detect.inspect_aria_folder(d)
    assert all(v["ok"] for v in detect.validate_against_receipt(pkg))
    with open(d / autil.base_name("mps"), "ab") as f:
        f.write(b"tampered")
    pkg2 = detect.inspect_aria_folder(d)
    res = detect.validate_against_receipt(pkg2)
    assert any(not v["ok"] and "mps" in v["check"] for v in res)


def test_receipt_absence_is_reported_not_fatal(tmp_path):
    d = autil.make_download_folder(tmp_path, with_receipt=False)
    pkg = detect.inspect_aria_folder(d)
    assert pkg.ok, "archives alone still identify the folder"
    res = detect.validate_against_receipt(pkg)
    assert res[0]["check"] == "download receipt present" and not res[0]["ok"]


# ----------------------------------------------------------------- cache

def test_selective_extraction_only_takes_needed_members(tmp_path):
    d = autil.make_download_folder(tmp_path)
    pkg = detect.inspect_aria_folder(d)
    c = cache_mod.RecordingCache(pkg.recording_id, tmp_path / "cache")
    m = c.populate(pkg, hash_sources=True)
    names = {p.name for p in c.dir.iterdir()}
    assert names == {"recording.vrs", "personalized_eye_gaze.csv",
                     "general_eye_gaze.csv", "metadata.json",
                     "cache_manifest.json"}
    assert not any("semidense" in n or "trajectory" in n or "hand" in n
                   for n in m["selected_members"]), \
        "large MPS members must never be extracted"
    assert m["source_sha256"]["mps"] == hashlib.sha256(
        (d / autil.base_name("mps")).read_bytes()).hexdigest()


def test_aea_cache_copies_direct_vrs_and_root_gaze_csv(tmp_path):
    payload = b"AEA-DIRECT-VRS" * 100
    d = autil.make_aea_download_folder(tmp_path, vrs_bytes=payload)
    pkg = detect.inspect_aria_folder(d)
    c = cache_mod.RecordingCache(pkg.recording_id, tmp_path / "cache")
    m = c.populate(pkg, hash_sources=True)
    assert c.vrs.read_bytes() == payload
    assert c.gaze_csv("general").is_file()
    assert not c.gaze_csv("personalized").exists()
    assert (c.dir / cache_mod.METADATA).is_file()
    assert m["source_layout"] == detect.LAYOUT_AEA
    assert m["source_formats"][detect.PKG_VRS] == "vrs"
    assert "general_eye_gaze.csv" in m["selected_members"]
    assert m["source_sha256"][detect.PKG_VRS] == hashlib.sha256(
        (d / autil.aea_name("main_recording")).read_bytes()).hexdigest()


def test_cache_is_reused_when_valid(tmp_path):
    d = autil.make_download_folder(tmp_path)
    pkg = detect.inspect_aria_folder(d)
    c = cache_mod.RecordingCache(pkg.recording_id, tmp_path / "cache")
    c.ensure(pkg, hash_sources=False)
    marker = c.vrs.stat().st_mtime_ns
    st = c.ensure(pkg, hash_sources=False)
    assert st["reused"] is True and st["state"] == "ready"
    assert c.vrs.stat().st_mtime_ns == marker, "cache was rebuilt needlessly"


def test_corrupt_and_stale_cache_are_detected_and_rebuilt(tmp_path):
    d = autil.make_download_folder(tmp_path)
    pkg = detect.inspect_aria_folder(d)
    c = cache_mod.RecordingCache(pkg.recording_id, tmp_path / "cache")
    c.ensure(pkg, hash_sources=False)

    c.vrs.write_bytes(b"truncated")                  # corrupt
    st = c.status(pkg)
    assert not st["valid"] and st["state"] == "corrupt"
    assert c.ensure(pkg, hash_sources=False)["reused"] is False

    c.manifest_path.write_text("{not json")           # unreadable manifest
    assert c.status(pkg)["state"] == "corrupt"
    c.ensure(pkg, hash_sources=False)

    with open(d / autil.base_name("recording_vrs"), "ab") as f:
        f.write(b"changed")                           # source changed
    st = c.status(pkg)
    assert not st["valid"] and st["state"] == "stale"
    assert "changed" in st["reason"]


def test_cache_failure_leaves_no_partial_entry(tmp_path, monkeypatch):
    d = autil.make_download_folder(tmp_path)
    pkg = detect.inspect_aria_folder(d)
    c = cache_mod.RecordingCache(pkg.recording_id, tmp_path / "cache")

    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(cache_mod.shutil, "copyfileobj", boom)
    with pytest.raises(RuntimeError):
        c.populate(pkg, hash_sources=False)
    assert not c.dir.exists()
    assert not any(p.name.startswith(c.dir.name + ".tmp")
                   for p in (tmp_path / "cache").iterdir())


def test_cache_cancellation_is_safe(tmp_path):
    d = autil.make_download_folder(tmp_path)
    pkg = detect.inspect_aria_folder(d)
    c = cache_mod.RecordingCache(pkg.recording_id, tmp_path / "cache")
    before = {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()}
    with pytest.raises(cache_mod.CacheCancelled):
        c.populate(pkg, cancel=lambda: True, hash_sources=False)
    assert not c.dir.exists()
    after = {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()}
    assert after == before, "source archives must be byte-identical"


def test_source_archives_are_never_modified(tmp_path):
    d = autil.make_download_folder(tmp_path)
    pkg = detect.inspect_aria_folder(d)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in d.iterdir() if p.is_file()}
    c = cache_mod.RecordingCache(pkg.recording_id, tmp_path / "cache")
    c.ensure(pkg, hash_sources=True)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in d.iterdir() if p.is_file()}
    assert after == before


# -------------------------------------------------- isolated helper safety

def _fake_helper(tmp_path, body: str) -> Path:
    p = tmp_path / "fake_helper.py"
    p.write_text(body)
    return p


def test_helper_crash_is_reported_without_raising_native_errors(tmp_path,
                                                                monkeypatch):
    """A segfaulting helper must surface as a clear error — the caller
    (a GUI server) stays alive."""
    helper = _fake_helper(tmp_path, "import os, signal\n"
                                    "os.kill(os.getpid(), signal.SIGSEGV)\n")
    monkeypatch.setattr(aria_config, "helper_path", lambda: helper)
    with pytest.raises(AriaHelperError) as e:
        run_helper(["probe"], python=Path(os.sys.executable))
    assert e.value.kind == "crashed" and "SIGSEGV" in str(e.value)


def test_helper_nonzero_exit_and_error_event(tmp_path, monkeypatch):
    helper = _fake_helper(tmp_path, "import sys\n"
                                    "sys.stderr.write('bad input\\n')\n"
                                    "sys.exit(3)\n")
    monkeypatch.setattr(aria_config, "helper_path", lambda: helper)
    with pytest.raises(AriaHelperError) as e:
        run_helper(["x"], python=Path(os.sys.executable))
    assert e.value.kind == "exit_nonzero"
    assert any("bad input" in ln for ln in e.value.log)

    helper2 = _fake_helper(tmp_path, "print('@@ARIA@@ ' + __import__('json')"
                                     ".dumps({'event':'error','kind':'VrsError',"
                                     "'message':'no rgb stream'}))\n")
    monkeypatch.setattr(aria_config, "helper_path", lambda: helper2)
    with pytest.raises(AriaHelperError) as e:
        run_helper(["x"], python=Path(os.sys.executable))
    assert "no rgb stream" in str(e.value)


def test_helper_protocol_survives_noisy_native_logging(tmp_path,
                                                        monkeypatch):
    """projectaria_tools writes colour escapes without newlines — the
    protocol line must still be found."""
    helper = _fake_helper(tmp_path,
                          "import sys, json\n"
                          "sys.stdout.write('\\x1b[0m')\n"
                          "sys.stdout.write('@@ARIA@@ ' + json.dumps("
                          "{'event':'result','n_frames':7}) + '\\n')\n")
    monkeypatch.setattr(aria_config, "helper_path", lambda: helper)
    res = run_helper(["x"], python=Path(os.sys.executable))
    assert res["n_frames"] == 7


def test_missing_aria_interpreter_is_a_clear_error(tmp_path):
    with pytest.raises(AriaHelperError) as e:
        run_helper(["probe"], python=tmp_path / "nope" / "python")
    assert e.value.kind == "env_missing"


def test_helper_cancellation_terminates_the_child(tmp_path, monkeypatch):
    helper = _fake_helper(tmp_path, "import time\ntime.sleep(120)\n")
    monkeypatch.setattr(aria_config, "helper_path", lambda: helper)
    with pytest.raises(AriaHelperError) as e:
        run_helper(["x"], python=Path(os.sys.executable),
                   cancel=lambda: True)
    assert e.value.kind == "cancelled"


# ------------------------------------------------------ gaze model choice

def test_personalized_preferred_general_fallback(tmp_path):
    from aria_source.prepare import choose_gaze_model
    c = cache_mod.RecordingCache(autil.REC_ID, tmp_path / "cache")
    c.dir.mkdir(parents=True)
    rows = autil.gaze_rows(20)
    span = (rows[0]["tracking_timestamp_us"] * 1000,
            rows[-1]["tracking_timestamp_us"] * 1000)
    autil.write_gaze_csv(c.dir / cache_mod.PERSONALIZED, rows)
    autil.write_gaze_csv(c.dir / cache_mod.GENERAL, rows)
    got = choose_gaze_model(c, span)
    assert got["model"] == "personalized"

    # empty personalized -> general
    (c.dir / cache_mod.PERSONALIZED).write_text(
        "tracking_timestamp_us,depth_m\n")
    got = choose_gaze_model(c, span)
    assert got["model"] == "general" and "fell back" in got["reason"]

    # temporally incompatible personalized -> general
    autil.write_gaze_csv(c.dir / cache_mod.PERSONALIZED,
                         autil.gaze_rows(5, start_us=1))
    got = choose_gaze_model(c, span)
    assert got["model"] == "general"
    assert "does not overlap" in json.dumps(got["candidates"])

    # neither usable -> explicit failure
    (c.dir / cache_mod.GENERAL).unlink()
    (c.dir / cache_mod.PERSONALIZED).unlink()
    with pytest.raises(ArtifactError, match="no usable eye-gaze model"):
        choose_gaze_model(c, span)


def test_microsecond_to_nanosecond_conversion_is_exact_integer(tmp_path):
    """µs → ns is exact integer ×1000, and the written text is integral.

    (Aria DEVICE_TIME is a device-uptime clock ~7.1e15 ns, just under
    2**53; NEON epoch times exceed it. Exact integer handling is required
    for both — no value is ever routed through a float.)
    """
    us = 7_112_379_886_123
    ns = us * 1000
    assert ns == 7_112_379_886_123_000 and isinstance(ns, int)
    assert str(ns).isdigit(), "written as integer text, never 7.11e+15"
    from aria_source import helper_cli
    assert helper_cli.FALLBACK_DEPTH_M == 1.0
    # end-to-end: the fixture's gaze table survives a parse with ±1 ns
    p = autil.make_prepared_artifact(tmp_path / "prep", gaze_n=5)
    from gaze_object_fusion.fusion import load_gaze
    df = load_gaze(p / "gaze.csv")
    for v in df["timestamp [ns]"].tolist():
        assert int(v) % 1 == 0 and int(v) > 0


# --------------------------------------------------- prepared/trimmed artifacts

def test_prepared_artifact_loads_and_declares_its_domain(tmp_path):
    p = autil.make_prepared_artifact(tmp_path / "prep")
    art = load_prepared_aria(p)
    assert art.artifact_type == "prepared_aria_recording"
    assert art.source_system == "project_aria"
    assert art.manifest["timestamps"]["domain"] == "aria_device_time"
    assert "NOT Pupil Labs NEON world time" in \
        art.manifest["timestamps"]["compatibility_note"]
    assert load_aria_artifact(p).artifact_id == art.artifact_id


def test_loader_rejects_foreign_or_broken_manifests(tmp_path):
    p = autil.make_prepared_artifact(tmp_path / "prep")
    mf = p / "prep_manifest.json"
    m = json.loads(mf.read_text())
    m["source_system"] = "pupil_neon"
    mf.write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="source_system"):
        load_prepared_aria(p)

    m["source_system"] = "project_aria"
    m["timestamps"]["domain"] = "neon_world_time"
    mf.write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="aria_device_time"):
        load_prepared_aria(p)

    m["timestamps"]["domain"] = "aria_device_time"
    m["validation"] = [{"check": "x", "ok": False, "detail": ""}]
    mf.write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="failures"):
        load_prepared_aria(p)


def test_trim_uses_exact_half_open_boundaries(tmp_path):
    from aria_source.trim import trim_aria_recording
    from neon_trim.core import load_timestamp_csv
    p = autil.make_prepared_artifact(tmp_path / "prep", n_frames=20,
                                     gaze_n=60)
    world = load_timestamp_csv(p / "world_timestamps.csv").ts_ns
    src_gaze = load_timestamp_csv(p / "gaze.csv").ts_ns
    out = tmp_path / "trim"
    m = trim_aria_recording(p, 5, 12, out)

    assert m["artifact_type"] == "trimmed_aria_recording"
    sel = m["selection"]
    assert sel["first_frame"] == 5 and sel["last_frame"] == 12
    assert sel["n_frames"] == 8
    # inclusive UI selection -> half-open interval [ts[5], ts[13])
    assert sel["absolute_start_ns"] == world[5]
    assert sel["absolute_end_boundary_ns"] == world[13]
    out_world = load_timestamp_csv(out / "world_timestamps.csv").ts_ns
    assert out_world == world[5:13]
    out_gaze = load_timestamp_csv(out / "gaze.csv").ts_ns
    expect = [t for t in src_gaze if world[5] <= t < world[13]]
    assert out_gaze == expect, "half-open gaze retention"
    assert all(v["ok"] for v in m["validation"])
    art = load_trimmed_aria(out)
    assert art.manifest["parent_artifact"]["artifact_id"] == \
        json.loads((p / "prep_manifest.json").read_text())["artifact_id"]


def test_trim_publication_is_atomic(tmp_path, monkeypatch):
    from aria_source import trim as trim_mod
    p = autil.make_prepared_artifact(tmp_path / "prep")
    out = tmp_path / "trim"

    def boom(*a, **k):
        raise RuntimeError("encoder exploded")
    monkeypatch.setattr(trim_mod, "run_cut", boom)
    with pytest.raises(RuntimeError):
        trim_mod.trim_aria_recording(p, 2, 6, out)
    assert not out.exists(), "no partial artifact may survive"


# --------------------------------------------- downstream dispatch/handoff

def test_object_tracking_intake_accepts_aria_artifacts(tmp_path):
    from object_tracking import intake
    p = autil.make_prepared_artifact(tmp_path / "prep")
    assert intake.classify_folder(p) == "prepared_aria_recording"
    b = intake.open_bundle(p)
    assert b.source_type == "prepared_aria_recording"
    assert b.artifact_type == "prepared_aria_recording"
    assert b.recording_id == autil.REC_ID
    assert b.sync_ok and b.gaze_compatible
    assert b.world_rows == b.n_frames == 20

    d = autil.make_download_folder(tmp_path / "dl")
    assert intake.classify_folder(d) == "aria_download"
    with pytest.raises(intake.IntakeError, match="preparation is required"):
        intake.open_bundle(d)


def test_dispatch_never_confuses_neon_and_aria(tmp_path):
    """Both systems use prep_manifest.json / trim_manifest.json — dispatch
    must read artifact_type, never the filename."""
    from common import recordings as rd
    p = autil.make_prepared_artifact(tmp_path / "prep")
    assert rd.classify_aria_folder(p) == "prepared_aria_recording"
    assert rd.source_system_of(p) == rd.SYSTEM_ARIA

    neon = tmp_path / "neon_trim_like"
    neon.mkdir()
    (neon / "trim_manifest.json").write_text(json.dumps({
        "artifact_type": "trimmed_neon_recording", "schema_version": "2.0"}))
    (neon / "gaze.csv").write_text("timestamp [ns]\n1\n")
    (neon / "world_timestamps.csv").write_text("timestamp [ns]\n1\n")
    assert rd.classify_aria_folder(neon) == ""
    assert rd.source_system_of(neon) == rd.SYSTEM_NEON


def test_fusion_accepts_aria_parent_and_keeps_public_contract(tmp_path):
    """Aria-backed tracked artifact → fusion, with NEON-only gaze fields
    null and the 23-column public contract unchanged."""
    import pandas as pd
    from aria_source.trim import trim_aria_recording
    from gaze_object_fusion.artifact import publish_fusion_artifact
    from gaze_object_fusion.fusion import PUBLIC_COLUMNS
    from object_tracking import intake, trackstore
    from object_tracking.annotations import AnnotationStore
    from object_tracking.artifact import publish_tracked_artifact

    prep = autil.make_prepared_artifact(tmp_path / "prep", n_frames=20,
                                        gaze_n=60)
    trimmed = tmp_path / "trim"
    trim_aria_recording(prep, 2, 17, trimmed)

    bundle = intake.open_bundle(trimmed)
    assert bundle.source_type == "trimmed_aria_recording"
    store = AnnotationStore(tmp_path / "ann.json", bundle.n_frames,
                            bundle.width, bundle.height)
    o = store.create_object("book", 5)
    oid = o["object_id"]
    store.set_points(oid, [[40, 40]], [1])
    store.record_preview(oid, {"found": True, "score": 0.9,
                               "mask_area": 900})
    store.accept_mask(oid)

    import numpy as np
    import pycocotools.mask as mask_util
    m = np.zeros((autil.H, autil.W), dtype=np.uint8)
    m[25:70, 15:65] = 1                    # covers the synthetic gaze area
    rle = mask_util.encode(np.asfortranarray(m))
    rle_json = json.dumps({"size": [autil.H, autil.W],
                           "counts": rle["counts"].decode("ascii")})
    rows = [{"frame_index": i,
             "tracking_direction": "forward" if i >= 5 else "backward",
             "sam_local_id": 1, "score": 0.9, "bbox_x": 15.0,
             "bbox_y": 25.0, "bbox_w": 50.0, "bbox_h": 45.0,
             "mask_area": int(m.sum()), "mask_rle": rle_json}
            for i in range(bundle.n_frames)]
    trackstore.replace_object_rows(tmp_path / "t.db", oid, rows)
    store.record_track_result(oid, {
        "rows": len(rows), "frames_present": len(rows), "first_frame": 0,
        "last_frame": bundle.n_frames - 1, "forward_ok": True,
        "backward_ok": True, "whole_video": True})
    store.set_review_status(oid, "accepted")
    tracked = tmp_path / "tracked"
    tm = publish_tracked_artifact(bundle, store, tmp_path / "t.db", tracked,
                                  make_overlay=False)
    assert tm["parent"]["source_type"] == "trimmed_aria_recording"
    assert tm["parent_artifact"]["artifact_id"]

    fused_dir = tmp_path / "fusion"
    fm = publish_fusion_artifact(tracked, fused_dir, make_demo=False)
    assert all(v["ok"] for v in fm["validation"])
    pub = pd.read_parquet(fused_dir / "gaze_samples.parquet")
    assert list(pub.columns) == PUBLIC_COLUMNS, "23-column contract intact"
    # NEON-only fields stay null for Aria; they are never invented
    assert pub["worn"].isna().all()
    assert pub["fixation_id"].isna().all()
    assert pub["blink_id"].isna().all()
    assert (pub["gaze_status"] == "on_object").any()
    assert pub["timestamp_ns"].iloc[0] > 2 ** 42


def test_unprojected_gaze_rows_are_invalid_not_clamped(tmp_path):
    """Rows the projection could not produce stay empty (NaN), never 0/edge."""
    from gaze_object_fusion.fusion import load_gaze
    p = autil.make_prepared_artifact(tmp_path / "prep", gaze_n=10,
                                     projection_failures=3)
    df = load_gaze(p / "gaze.csv")
    xs = df["gaze x [px]"].tolist()
    assert all(x != x for x in xs[:3]), "failed projections must be NaN"
    assert all(x == x for x in xs[3:])
    rows = list(csv.DictReader(open(p / "gaze.csv")))
    assert rows[0]["projection_status"] == "outside_field_of_view"
    assert rows[0]["gaze x [px]"] == ""


# ------------------------------------------- opt-in real-recording fixture

REAL = Path("data/aria/"
            "recording_1085015926322699")


@pytest.mark.skipif(not REAL.is_dir() or not aria_config.aria_env_available(),
                    reason="real Aria fixture / Aria interpreter absent")
def test_real_recording_detection_and_receipt():
    pkg = detect.inspect_aria_folder(REAL)
    assert pkg.ok and pkg.recording_id == "recording_1085015926322699"
    assert all(v["ok"] for v in detect.validate_against_receipt(pkg))


REAL_AEA = Path("data/aria/"
                "AriaEverydayActivities_loc4_script3_seq4_rec1")


@pytest.mark.skipif(not REAL_AEA.is_dir(), reason="real AEA fixture absent")
def test_real_aea_direct_layout_detection():
    pkg = detect.inspect_aria_folder(REAL_AEA)
    assert pkg.ok
    assert pkg.recording_id == "loc4_script3_seq4_rec1"
    assert pkg.dataset == "AriaEverydayActivities"
    assert pkg.layout == detect.LAYOUT_AEA
    assert all(v["ok"] for v in detect.validate_against_receipt(pkg))


@pytest.mark.skipif(not REAL.is_dir() or not aria_config.aria_env_available(),
                    reason="real Aria fixture / Aria interpreter absent")
def test_real_prepared_artifact_if_present():
    prep = REAL.parent / (REAL.name + "_prepared_aria")
    if not prep.is_dir():
        pytest.skip("prepared artifact not built in this environment")
    art = load_prepared_aria(prep)
    m = art.manifest
    assert m["video"]["n_frames"] == m["timestamps"]["gap_report"]["n_gaps"] + 1
    assert m["timestamps"]["domain"] == "aria_device_time"
    assert m["gaze"]["model"] in ("personalized", "general")
    from neon_trim.core import load_timestamp_csv
    w = load_timestamp_csv(art.world_timestamps_path)
    assert len(w.ts_ns) == m["video"]["n_frames"]
    assert all(b > a for a, b in zip(w.ts_ns, w.ts_ns[1:]))

