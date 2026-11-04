"""End-to-end export tests on a tiny synthetic VFR recording.

A real (PyAV-encoded) H.264 video with IRREGULAR frame timestamps is built on
the fly, together with matching world/gaze CSVs, so the full pipeline —
probe → sync-check → plan → ffmpeg cut → CSV trim → validation → atomic
publish — runs for real in a few hundred ms.
"""

import json
from fractions import Fraction
from pathlib import Path

import pytest

from neon_trim import core, export, media

NS = core.NS
T0 = 1_733_200_839_984_000_000

# irregular VFR timing (ms) — includes 50 ms head + a 14 ms short gap
OFFSETS_MS = [0, 50, 100, 133, 166, 180, 220, 253, 286, 320]
WORLD_TS = [T0 + int(ms * 1e6) for ms in OFFSETS_MS]
DUR_MS = OFFSETS_MS[-1] + 34
REC_END = T0 + int(DUR_MS * 1e6)
# gaze @ ~200 Hz but starting late (missing at the beginning, like the fixture)
GAZE_TS = [T0 + int(70e6) + i * 5_000_000 for i in range(40)]  # 70 ms .. 265 ms


def _encode_vfr_video(path: Path, offsets_ms) -> None:
    import av
    import numpy as np
    tb = Fraction(1, 90000)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264")
        s.width, s.height, s.pix_fmt = 160, 120, "yuv420p"
        s.codec_context.time_base = tb   # keeps exact 90 kHz VFR pts (no fps grid)
        s.time_base = tb
        s.options = {"crf": "20", "preset": "ultrafast", "g": "3"}
        for i, ms in enumerate(offsets_ms):
            img = np.full((120, 160, 3), (i * 23) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame = frame.reformat(format="yuv420p")
            frame.pts = int(round(ms * 90))          # exact 90 kHz ticks
            frame.time_base = tb
            for pkt in s.encode(frame):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def _write_csvs(rec: Path, world_ts, gaze_ts) -> None:
    with open(rec / "world_timestamps.csv", "w") as f:
        f.write("section id,recording id,timestamp [ns]\n")
        for t in world_ts:
            f.write(f"sec-1,rec-1,{t}\n")
    with open(rec / "gaze.csv", "w") as f:
        f.write("section id,recording id,timestamp [ns],gaze x [px],gaze y [px],worn\n")
        for t in gaze_ts:
            f.write(f"sec-1,rec-1,{t},100.500,200.250,1.0\n")
    (rec / "info.json").write_text(json.dumps({
        "recording_id": "rec-1", "start_time": T0,
        "duration": REC_END - T0}))
    # decoy auxiliary file that must NOT be copied into the trim output
    (rec / "fixations.csv").write_text("section id,recording id,fixation id\n")


@pytest.fixture(scope="module")
def rec_dir(tmp_path_factory):
    rec = tmp_path_factory.mktemp("neon_rec")
    _encode_vfr_video(rec / "scene.mp4", OFFSETS_MS)
    _write_csvs(rec, WORLD_TS, GAZE_TS)
    return rec


@pytest.fixture(scope="module")
def loaded(rec_dir):
    return export.load_recording(rec_dir)


# ---------------------------------------------------------------------------

def test_probe_and_sync_on_synthetic_vfr(loaded):
    assert loaded.video.n_frames == len(WORLD_TS)
    assert loaded.sync.ok, loaded.sync.message
    # PTS must match world offsets to within 90 kHz quantization
    for pts, ms in zip(loaded.video.pts_s, OFFSETS_MS):
        assert abs(pts - ms / 1000) < 1e-4


def test_full_export_roundtrip(loaded, tmp_path):
    out = tmp_path / "trim_out"
    src_bytes = {p.name: p.read_bytes() for p in loaded.rec_dir.iterdir()}

    manifest = export.run_export(loaded, 2, 6, out)

    # source untouched, byte for byte
    for p in loaded.rec_dir.iterdir():
        assert p.read_bytes() == src_bytes[p.name], f"{p.name} was modified!"

    # output contents: exactly the 4 expected files
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted(["scene.mp4", "gaze.csv", "world_timestamps.csv",
                            "trim_manifest.json"])

    # frame-accurate video
    out_pts = media.decode_all_pts(out / "scene.mp4")
    assert len(out_pts) == 5                       # frames 2..6 inclusive

    # world rows verbatim
    w = core.load_timestamp_csv(out / "world_timestamps.csv")
    assert w.ts_ns == WORLD_TS[2:7]

    # gaze: interval is [world[2], world[7])
    g = core.load_timestamp_csv(out / "gaze.csv")
    lo, hi = WORLD_TS[2], WORLD_TS[7]
    assert all(lo <= t < hi for t in g.ts_ns)
    assert g.n_rows == sum(1 for t in GAZE_TS if lo <= t < hi)
    # verbatim precision: quirky trailing zeros survive
    assert ",100.500,200.250," in g.lines[0]

    # manifest sanity
    assert manifest["counts"]["output_frames"] == 5
    assert manifest["counts"]["retained_gaze_rows"] == g.n_rows
    assert manifest["selection"]["absolute_start_ns"] == WORLD_TS[2]
    assert manifest["selection"]["absolute_end_boundary_ns"] == WORLD_TS[7]
    assert manifest["synchronization"]["method"] == core.SYNC_METHOD
    assert "fixations.csv" in manifest["omitted_files"]["files"]
    assert all(v["ok"] for v in manifest["validation"])
    # no aux files copied
    assert not (out / "fixations.csv").exists()


def test_export_last_frame_selection_uses_recording_end(loaded, tmp_path):
    out = tmp_path / "tail"
    manifest = export.run_export(loaded, 7, len(WORLD_TS) - 1, out)
    assert manifest["selection"]["absolute_end_boundary_ns"] == REC_END
    out_pts = media.decode_all_pts(out / "scene.mp4")
    assert len(out_pts) == 3


def test_export_single_frame(loaded, tmp_path):
    out = tmp_path / "single"
    manifest = export.run_export(loaded, 4, 4, out)
    assert manifest["counts"]["output_frames"] == 1
    assert len(media.decode_all_pts(out / "scene.mp4")) == 1


def test_export_empty_gaze_interval_reported(loaded, tmp_path):
    out = tmp_path / "empty_gaze"
    manifest = export.run_export(loaded, 0, 0, out)   # frame 0: before first gaze
    assert manifest["counts"]["retained_gaze_rows"] == 0
    g = core.load_timestamp_csv(out / "gaze.csv")
    assert g.n_rows == 0
    warn = [v for v in manifest["validation"] if "empty gaze" in v["check"]]
    assert warn and "no gaze" in warn[0]["detail"]


def test_export_refuses_existing_output(loaded, tmp_path):
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(export.ExportError, match="already exists"):
        export.run_export(loaded, 0, 3, out)


def test_export_refuses_output_inside_source(loaded):
    with pytest.raises(export.ExportError, match="outside the source"):
        export.run_export(loaded, 0, 3, loaded.rec_dir / "sub")


def test_export_refuses_reversed_selection(loaded, tmp_path):
    with pytest.raises((export.ExportError, ValueError)):
        export.run_export(loaded, 5, 2, tmp_path / "rev")


def test_sync_mismatch_blocks_export(rec_dir, tmp_path):
    """The observed failure class: world rows != decoded frames must block."""
    broken = tmp_path / "broken_rec"
    broken.mkdir()
    (broken / "scene.mp4").write_bytes((rec_dir / "scene.mp4").read_bytes())
    _write_csvs(broken, WORLD_TS[:-1], GAZE_TS)     # one world row missing
    rec = export.load_recording(broken)
    assert not rec.sync.ok
    assert "mismatch" in rec.sync.message
    with pytest.raises(export.ExportError, match="blocked"):
        export.run_export(rec, 0, 3, tmp_path / "nope")
    assert not (tmp_path / "nope").exists()


def test_failed_export_leaves_no_partial_output(loaded, tmp_path, monkeypatch):
    out = tmp_path / "boom"

    def exploding_cut(*a, **k):
        raise media.CutError("synthetic encoder failure")
    monkeypatch.setattr(media, "run_cut", exploding_cut)
    with pytest.raises(media.CutError):
        export.run_export(loaded, 0, 3, out)
    assert not out.exists()
    leftovers = [p for p in out.parent.iterdir() if p.name.startswith(".neon_trim_tmp")]
    assert leftovers == []


def test_cancelled_export_leaves_no_partial_output(loaded, tmp_path):
    import threading
    ev = threading.Event()
    ev.set()                                        # cancelled before it starts
    out = tmp_path / "cancelled"
    with pytest.raises(export.ExportError):
        export.run_export(loaded, 0, 3, out, cancel_event=ev)
    assert not out.exists()
    leftovers = [p for p in out.parent.iterdir() if p.name.startswith(".neon_trim_tmp")]
    assert leftovers == []


def test_validation_failure_blocks_publish(loaded, tmp_path, monkeypatch):
    """If output decode disagrees with the plan, nothing must be published."""
    real_decode = media.decode_all_pts

    def lying_decode(path):
        return real_decode(path)[:-1]               # pretend a frame is missing
    monkeypatch.setattr(media, "decode_all_pts", lying_decode)
    out = tmp_path / "badval"
    with pytest.raises(export.ExportError, match="validation failed"):
        export.run_export(loaded, 2, 6, out)
    assert not out.exists()


# ---------------------------------------------------------------------------
# regressions from the adversarial review
# ---------------------------------------------------------------------------

def test_probe_has_no_spurious_pts_warning(rec_dir):
    """PyAV's EOF flush packet must not be reported as a missing-PTS packet."""
    probe = media.probe_video(rec_dir / "scene.mp4")
    assert not any("no PTS" in w for w in probe.warnings), probe.warnings


def test_export_start_after_large_vfr_gap(tmp_path):
    """A dropped-frames gap (>120 ms) before the selected start frame must not
    abort the export: the first-output-frame offset equals half that gap by
    cut geometry and the validation must expect exactly that."""
    offs = [0, 33, 66, 100, 240, 273, 306, 340, 373, 406]   # 140 ms gap 100->240
    rec = tmp_path / "gap_rec"
    rec.mkdir()
    _encode_vfr_video(rec / "scene.mp4", offs)
    world = [T0 + int(ms * 1e6) for ms in offs]
    with open(rec / "world_timestamps.csv", "w") as f:
        f.write("section id,recording id,timestamp [ns]\n")
        for t in world:
            f.write(f"sec-1,rec-1,{t}\n")
    with open(rec / "gaze.csv", "w") as f:
        f.write("section id,recording id,timestamp [ns],gaze x [px],gaze y [px],worn\n")
        for i in range(60):
            f.write(f"sec-1,rec-1,{T0 + i * 7_000_000},1.0,2.0,1.0\n")
    (rec / "info.json").write_text(json.dumps(
        {"recording_id": "rec-1", "start_time": T0,
         "duration": int((offs[-1] + 34) * 1e6)}))
    loaded = export.load_recording(rec)
    assert loaded.sync.ok, loaded.sync.message
    out = tmp_path / "gap_out"
    manifest = export.run_export(loaded, 4, 9, out)         # start right after the gap
    assert manifest["counts"]["output_frames"] == 6
    assert all(v["ok"] for v in manifest["validation"])
    assert len(media.decode_all_pts(out / "scene.mp4")) == 6


def test_sync_tolerates_half_gap_offset_on_reload():
    """A trimmed output cut mid-way in a 140 ms gap carries a ~70 ms constant
    offset; check_sync must accept it (spread is the real invariant)."""
    world = [T0 + int(ms * 1e6) for ms in [240, 273, 306, 340]]
    pts = [(w - world[0]) / core.NS + 0.070 for w in world]  # +70 ms constant
    s = core.check_sync(pts, world)
    assert s.ok, s.message


def test_manifest_command_is_replayable(loaded, tmp_path):
    """Provenance command must not contain the -progress pair and must end
    with the output basename (replaying it must not truncate the output)."""
    out = tmp_path / "cmdcheck"
    manifest = export.run_export(loaded, 1, 3, out)
    cmd = manifest["video_encoding"]["command"]
    assert "-progress" not in cmd
    assert cmd.endswith("scene.mp4")
    assert cmd.count("scene.mp4") == 2          # input path + output basename


def test_failed_export_removes_created_parent_dirs(loaded, tmp_path, monkeypatch):
    def exploding_cut(*a, **k):
        raise media.CutError("boom")
    monkeypatch.setattr(media, "run_cut", exploding_cut)
    out = tmp_path / "a" / "b" / "boom_out"
    with pytest.raises(media.CutError):
        export.run_export(loaded, 0, 3, out)
    assert not out.exists()
    assert not (tmp_path / "a").exists()        # empty parent chain undone


def test_cancel_after_validation_prevents_publish(loaded, tmp_path, monkeypatch):
    import threading
    ev = threading.Event()
    orig = export._validate

    def validating_then_cancel(*a, **k):
        orig(*a, **k)
        ev.set()                                # cancel arrives mid-flight
    monkeypatch.setattr(export, "_validate", validating_then_cancel)
    out = tmp_path / "late_cancel"
    with pytest.raises(export.ExportCancelled):
        export.run_export(loaded, 0, 3, out, cancel_event=ev)
    assert not out.exists()
