"""Tests for the versioned artifact contract: manifest v2 fields, relative
portability, loader validation, and re-trim parent references.

Reuses the synthetic-recording machinery from test_export (real encoded VFR
video, so exports run end-to-end).
"""

import json
import shutil

import pytest

from neon_trim import artifact, export

from test_export import (GAZE_TS, OFFSETS_MS, WORLD_TS, _encode_vfr_video,
                         _write_csvs)


@pytest.fixture(scope="module")
def rec_dir(tmp_path_factory):
    rec = tmp_path_factory.mktemp("art_rec")
    _encode_vfr_video(rec / "scene.mp4", OFFSETS_MS)
    _write_csvs(rec, WORLD_TS, GAZE_TS)
    return rec


@pytest.fixture(scope="module")
def exported(rec_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("art_out") / "trim1"
    manifest = export.trim_recording(rec_dir, 2, 6, out)
    return out, manifest


def test_manifest_v2_contract_fields(exported):
    out, m = exported
    assert m["schema_version"] == artifact.SCHEMA_VERSION
    assert m["artifact_type"] == artifact.ARTIFACT_TYPE
    assert len(m["artifact_id"]) == 32
    assert m["parent_artifact"] is None            # source was a raw recording
    assert m["created_utc"]
    assert m["tool"]["version"]
    # files are RELATIVE, no separators
    for key in ("video", "gaze", "world_timestamps", "manifest"):
        assert "/" not in m["files"][key]
    # absolute path only under source provenance, clearly labelled
    assert m["source"]["directory"].startswith("/")
    assert "provenance" in m["source"]["directory_note"]
    # legacy/provenance fields retained
    for key in ("selection", "counts", "video_encoding", "synchronization",
                "omitted_files", "validation"):
        assert key in m


def test_load_exported_artifact(exported):
    out, m = exported
    art = artifact.load_artifact(out)              # by directory
    assert art.artifact_id == m["artifact_id"]
    assert art.source_recording_id == "rec-1"
    assert art.video_path.is_file()
    assert art.gaze_path.name == "gaze.csv"
    assert art.world_timestamps_path.is_file()
    assert art.validation_ok
    assert art.selection["first_frame"] == 2
    # by manifest file path too
    art2 = artifact.load_artifact(out / "trim_manifest.json")
    assert art2.artifact_id == art.artifact_id


def test_artifact_portable_after_move(exported, tmp_path):
    out, m = exported
    moved = tmp_path / "relocated_artifact"
    shutil.copytree(out, moved)
    art = artifact.load_artifact(moved)            # no stale absolute paths used
    assert art.video_path == moved / m["files"]["video"]
    assert art.video_path.is_file()
    assert art.artifact_id == m["artifact_id"]


def test_loader_rejects_bad_artifacts(exported, tmp_path):
    out, m = exported
    # missing manifest
    empty = tmp_path / "empty"; empty.mkdir()
    with pytest.raises(artifact.ArtifactError, match="no trim_manifest"):
        artifact.load_artifact(empty)
    # wrong type / version / missing file
    for mutate, err in [
        (lambda d: d.__setitem__("artifact_type", "other"), "artifact_type"),
        (lambda d: d.__setitem__("schema_version", "1.0"), "schema_version"),
        (lambda d: d.pop("artifact_id"), "artifact_id"),
        (lambda d: d["files"].__setitem__("video", "missing.mp4"), "not found"),
        (lambda d: d["files"].__setitem__("gaze", "../evil.csv"), "relative"),
        (lambda d: d["files"].__setitem__("gaze", "/abs/evil.csv"), "relative"),
        (lambda d: d.__setitem__("validation", []), "validation"),
    ]:
        broken = tmp_path / f"broken_{err.replace(' ', '_')[:12]}"
        if broken.exists():
            shutil.rmtree(broken)
        shutil.copytree(out, broken)
        data = json.loads((broken / "trim_manifest.json").read_text())
        mutate(data)
        (broken / "trim_manifest.json").write_text(json.dumps(data))
        with pytest.raises(artifact.ArtifactError, match=err):
            artifact.load_artifact(broken)


def test_retrim_records_parent_artifact(exported, tmp_path):
    out, m = exported
    # trim the trimmed artifact again -> parent_artifact must reference it
    out2 = tmp_path / "retrim"
    m2 = export.trim_recording(out, 1, 3, out2)
    assert m2["parent_artifact"] is not None
    assert m2["parent_artifact"]["artifact_id"] == m["artifact_id"]
    art2 = artifact.load_artifact(out2)
    assert art2.validation_ok


def test_loader_rejects_symlink_escape(exported, tmp_path):
    """A manifest-declared file that is a symlink pointing outside the
    artifact directory must be rejected (no arbitrary-file reads)."""
    out, m = exported
    victim = tmp_path / "victim.mp4"
    victim.write_bytes(b"outside data")
    evil = tmp_path / "evil_artifact"
    shutil.copytree(out, evil)
    vid = evil / m["files"]["video"]
    vid.unlink()
    vid.symlink_to(victim)
    with pytest.raises(artifact.ArtifactError, match="symlink escape"):
        artifact.load_artifact(evil)


def test_parent_ref_flags_corrupt_parent_manifest(exported, tmp_path):
    out, m = exported
    corrupt = tmp_path / "corrupt_parent"
    shutil.copytree(out, corrupt)
    data = json.loads((corrupt / "trim_manifest.json").read_text())
    data["artifact_id"] = 12345                     # not a string
    (corrupt / "trim_manifest.json").write_text(json.dumps(data))
    ref = artifact.parent_artifact_ref(corrupt)
    assert ref is not None and ref["artifact_id"] is None
    assert "unresolvable" in ref["note"]
