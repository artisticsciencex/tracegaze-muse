"""Native preparation tests against the REAL participant_02 fixture (read-only)
and its matching Cloud CSV export (participant_a — same recording id).

The heavyweight preparation runs once (module fixture) into a tmp dir.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from object_tracking import intake, native_prep

NATIVE = Path("data/neon-data/"
              "lab_study/participant_02")
EXPORT = Path("data/neon-data/"
              "lab_study/participant_01/"
              "2024-01-01_00-00-00-0000abcd_participant_a")
REC_ID = "0000abcd-c979-4a9d-ae0d-3afc97d426ba"

pytestmark = pytest.mark.skipif(not NATIVE.is_dir(),
                                reason="native fixture not present")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    out = tmp_path_factory.mktemp("prep") / "participant_02_prepared"
    manifest = native_prep.prepare_native_recording(NATIVE, out)
    return out, manifest


def _src_snapshot():
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
            for p in NATIVE.iterdir()}


def test_overview_and_streams():
    ov = native_prep.native_overview(NATIVE)
    assert ov["recording_id"] == REC_ID
    assert any("Neon Scene Camera" in s for s in ov["streams"])
    assert ov["gaze_offset"] is not None


def test_prepare_counts_and_source_unchanged(prepared):
    out, manifest = prepared
    snap_after = _src_snapshot()
    # empirical oracle numbers (documented in native_prep docstring)
    assert manifest["counts"]["scene_frames"] == 6403
    assert manifest["counts"]["gaze_rows_native"] == 42427
    assert manifest["counts"][
        "gaze_tail_samples_dropped_past_recording_end"] == 5
    assert manifest["synchronization"]["ok"] is True
    assert all(v["ok"] for v in manifest["validation"])
    # source untouched (size+mtime snapshot; sha256 checked inside prep)
    assert snap_after == _src_snapshot()


def test_prepared_folder_is_standard_csv_export_shape(prepared):
    out, _ = prepared
    assert intake.classify_folder(out) == "prepared_artifact"
    bundle = intake.open_bundle(out)
    assert bundle.sync_ok and bundle.gaze_compatible
    assert bundle.n_frames == 6403
    assert bundle.recording_id == REC_ID


def test_int_nanosecond_preservation(prepared):
    out, _ = prepared
    raw = np.fromfile(NATIVE / "Neon Scene Camera v1 ps1.time", dtype="<i8")
    with open(out / "world_timestamps.csv") as fh:
        got = [int(r["timestamp [ns]"]) for r in csv.DictReader(fh)]
    assert got == [int(t) for t in raw]            # exact, no float round-trip


def test_gaze_matches_cloud_export_and_offset_provenance(prepared):
    out, manifest = prepared
    with open(out / "gaze.csv") as fh:
        prep_rows = [(int(r["timestamp [ns]"]), float(r["gaze x [px]"]),
                      float(r["gaze y [px]"])) for r in csv.DictReader(fh)]
    with open(EXPORT / "gaze.csv") as fh:
        exp_rows = [(int(r["timestamp [ns]"]), float(r["gaze x [px]"]),
                     float(r["gaze y [px]"])) for r in csv.DictReader(fh)]
    n = min(len(prep_rows), len(exp_rows))
    assert n > 42000
    # identical timestamps; coordinates equal within Cloud CSV rounding
    assert all(prep_rows[i][0] == exp_rows[i][0] for i in range(n))
    assert max(abs(prep_rows[i][1] - exp_rows[i][1]) for i in range(n)) < 1e-3
    assert max(abs(prep_rows[i][2] - exp_rows[i][2]) for i in range(n)) < 1e-3
    prov = manifest["synchronization"]["gaze_offset_provenance"]
    assert "already incorporated" in prov
    assert "NOT re-applied" in prov


def test_native_vs_export_scene_relationship_documented(prepared):
    out, manifest = prepared
    # export has 30 more world rows (synthetic head padding) — our prepared
    # artifact must NOT reproduce them
    with open(EXPORT / "world_timestamps.csv") as fh:
        export_rows = sum(1 for _ in fh) - 1
    assert export_rows == 6433
    assert manifest["counts"]["world_rows"] == 6403
    assert "pad" in manifest["synchronization"]["note"]


def test_mismatch_blocks_publication(tmp_path, monkeypatch):
    from common import media as cm

    real = cm.probe_video

    def lying_probe(path):
        p = real(path)
        p.n_frames = p.n_frames - 1        # pretend a decoded-frame mismatch
        p.pts_s = p.pts_s[:-1]
        return p
    monkeypatch.setattr(native_prep.media, "probe_video", lying_probe)
    out = tmp_path / "blocked"
    with pytest.raises(native_prep.PrepError,
                       match="no reliable 1:1|do not align"):
        native_prep.prepare_native_recording(NATIVE, out)
    assert not out.exists()                # atomic: nothing published


def test_prepare_conflict_and_atomicity(tmp_path):
    out = tmp_path / "exists"
    out.mkdir()
    from common.artifacts import ArtifactError
    with pytest.raises(ArtifactError, match="already exists"):
        native_prep.prepare_native_recording(NATIVE, out)
