"""Synthetic Aria fixtures — no projectaria_tools, no real recording.

Builds download folders (zips + receipt) and prepared/trimmed Aria
artifacts whose payload is a real small VFR video plus exact int-ns
tables, so the contracts, cache, trim math and downstream dispatch can be
tested without native code or participant data.
"""

import csv
import hashlib
import json
import zipfile
from fractions import Fraction
from pathlib import Path

REC_ID = "recording_9999999999999"
DATASET = "ReadingInTheWild"
RELEASE = "v1.0"
AEA_REC_ID = "loc4_script3_seq4_rec1"
AEA_DATASET = "AriaEverydayActivities"
AEA_RELEASE = "1.0.0"
# Aria DEVICE_TIME is a device-uptime clock (seconds-scale), not epoch
T0_NS = 7_112_379_792_600
FRAME_GAP_NS = 33_312_425
W = H = 96


def base_name(pkg: str) -> str:
    ext = "mp4" if pkg == "preview_rgb" else "zip"
    return f"{DATASET}_{RELEASE}_{REC_ID}_{pkg}.{ext}"


def gaze_rows(n=40, start_us=None, step_us=17_000, depth=0.9):
    """MPS-shaped eye-gaze rows (µs timestamps, CPF angles)."""
    start_us = start_us if start_us is not None else T0_NS // 1000
    out = []
    for i in range(n):
        out.append({
            "tracking_timestamp_us": start_us + i * step_us,
            "left_yaw_rads_cpf": 0.12 + i * 1e-4,
            "right_yaw_rads_cpf": 0.19 + i * 1e-4,
            "pitch_rads_cpf": 0.07,
            "depth_m": "" if depth is None else depth,
            "left_yaw_low_rads_cpf": 0.09, "right_yaw_low_rads_cpf": 0.16,
            "pitch_low_rads_cpf": 0.04, "left_yaw_high_rads_cpf": 0.15,
            "right_yaw_high_rads_cpf": 0.22, "pitch_high_rads_cpf": 0.11,
            "tx_left_eye_cpf": 0.0315, "ty_left_eye_cpf": 0.0,
            "tz_left_eye_cpf": 0.0, "tx_right_eye_cpf": -0.0315,
            "ty_right_eye_cpf": 0.0, "tz_right_eye_cpf": 0.0,
            "session_uid": "e52ab24d-b492-4a61-90e8-e418df4c420f",
        })
    return out


def write_gaze_csv(path: Path, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def make_download_folder(root: Path, *, with_receipt=True,
                         personalized=True, general=True,
                         vrs_bytes=b"FAKE-VRS-CONTENT" * 64,
                         extra_mps=True) -> Path:
    """A structurally-valid Aria download folder (small fake archives)."""
    d = root / REC_ID
    d.mkdir(parents=True)

    vrs_zip = d / base_name("recording_vrs")
    with zipfile.ZipFile(vrs_zip, "w") as z:
        z.writestr("recording.vrs", vrs_bytes)

    mps_zip = d / base_name("mps")
    with zipfile.ZipFile(mps_zip, "w") as z:
        if personalized:
            z.writestr("mps/eye_gaze/personalized_eye_gaze.csv",
                       _csv_text(gaze_rows(40)))
        if general:
            z.writestr("mps/eye_gaze/general_eye_gaze.csv",
                       _csv_text(gaze_rows(40, depth=None)))
        if extra_mps:      # large members that must NEVER be extracted
            z.writestr("mps/slam/semidense_points.csv.gz", b"x" * 4096)
            z.writestr("mps/slam/closed_loop_trajectory.csv", b"y" * 4096)
            z.writestr("mps/hand_tracking/hand_tracking_frames.jsonl",
                       b"z" * 2048)

    meta_zip = d / base_name("metadata_json")
    with zipfile.ZipFile(meta_zip, "w") as z:
        z.writestr("metadata.json", json.dumps({
            "recording_id": int(REC_ID.split("_")[1]),
            "campaign_task_name": "Task 19: Skimming (reading quickly)",
            "duration": 1.5, "notes": "synthetic fixture"}))

    preview = d / base_name("preview_rgb")
    preview.write_bytes(b"not-a-real-mp4")

    if with_receipt:
        files = []
        for p in (meta_zip, preview, mps_zip, vrs_zip):
            files.append({"filename": p.name, "bytes": p.stat().st_size,
                          "sha1": hashlib.sha1(p.read_bytes()).hexdigest(),
                          "verified": True, "package": "x"})
        (d / "download_receipt.json").write_text(json.dumps({
            "dataset": DATASET, "release": RELEASE, "sequence_id": REC_ID,
            "files": files}))
    return d


def aea_name(role: str) -> str:
    ext = {"main_recording": "vrs", "preview_rgb": "mp4"}.get(role, "zip")
    return f"{AEA_DATASET}_{AEA_RELEASE}_{AEA_REC_ID}_{role}.{ext}"


def make_aea_download_folder(root: Path,
                             vrs_bytes=b"FAKE-AEA-VRS" * 64) -> Path:
    """AEA layout: direct VRS, gaze-only ZIP, annotations, no receipt."""
    d = root / AEA_REC_ID
    d.mkdir(parents=True)
    (d / aea_name("main_recording")).write_bytes(vrs_bytes)
    with zipfile.ZipFile(d / aea_name("mps_eye_gaze"), "w") as z:
        z.writestr("general_eye_gaze.csv",
                   _csv_text(gaze_rows(30, depth=None)))
        z.writestr("summary.json", json.dumps({"status": "success"}))
    with zipfile.ZipFile(d / aea_name("annotations"), "w") as z:
        z.writestr("metadata.json", json.dumps({
            "location": 4, "script": 3, "sequence": 4, "recording": 1,
            "dataset_version": "1.0", "dataset_name": "AEA_2024"}))
        z.writestr("speech.csv", "timestamp,text\n")
    (d / aea_name("preview_rgb")).write_bytes(b"not-a-real-mp4")
    return d


def _csv_text(rows) -> str:
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def encode_video(path: Path, n_frames: int, ts_ns: list) -> None:
    """Small video whose PTS mirror the given device timestamps."""
    import av
    import numpy as np
    tb = Fraction(1, 90000)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264")
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.codec_context.time_base = tb
        s.time_base = tb
        s.options = {"crf": "20", "preset": "ultrafast", "g": "5"}
        for i in range(n_frames):
            img = np.full((H, W, 3), (i * 7) % 255, dtype=np.uint8)
            f = av.VideoFrame.from_ndarray(img, format="rgb24")
            f = f.reformat(format="yuv420p")
            f.pts = int(round((ts_ns[i] - ts_ns[0]) * 90000 / 1e9))
            f.time_base = tb
            for pkt in s.encode(f):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def make_prepared_artifact(out: Path, *, n_frames=20, gaze_n=60,
                           gaze_model="personalized",
                           projection_failures=0) -> Path:
    """A valid `prepared_aria_recording` artifact (synthetic payload)."""
    from aria_source import (ARIA_PREP_MANIFEST, PREPARED_ARIA_TYPE,
                             SOURCE_SYSTEM)
    from aria_source.artifact import (ARIA_SCHEMA_VERSION,
                                      TIMESTAMP_DOMAIN_NOTE)
    out = Path(out)
    out.mkdir(parents=True)
    ts = [T0_NS + i * FRAME_GAP_NS for i in range(n_frames)]
    encode_video(out / "scene.mp4", n_frames, ts)

    with open(out / "world_timestamps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section id", "recording id", "timestamp [ns]"])
        for t in ts:
            w.writerow([REC_ID, REC_ID, t])

    span = ts[-1] - ts[0]
    cols = ["timestamp [ns]", "gaze x [px]", "gaze y [px]",
            "tracking_timestamp_us", "left_yaw_rads_cpf",
            "right_yaw_rads_cpf", "combined_yaw_rads_cpf",
            "pitch_rads_cpf", "depth_m", "depth_source", "session_uid",
            "gaze_model", "projection_valid", "projection_status"]
    with open(out / "gaze.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(gaze_n):
            t = ts[0] + (span * i) // max(gaze_n - 1, 1)
            failed = i < projection_failures
            w.writerow({
                "timestamp [ns]": t,
                "gaze x [px]": "" if failed else f"{20 + (i % 40)}.5",
                "gaze y [px]": "" if failed else f"{30 + (i % 30)}.25",
                "tracking_timestamp_us": t // 1000,
                "left_yaw_rads_cpf": 0.12, "right_yaw_rads_cpf": 0.19,
                "combined_yaw_rads_cpf": 0.155, "pitch_rads_cpf": 0.07,
                "depth_m": 0.9, "depth_source": "measured",
                "session_uid": "sess-1", "gaze_model": gaze_model,
                "projection_valid": "false" if failed else "true",
                "projection_status": ("outside_field_of_view" if failed
                                      else "ok")})

    manifest = {
        "schema_version": ARIA_SCHEMA_VERSION,
        "artifact_type": PREPARED_ARIA_TYPE,
        "artifact_id": "aaaa1111bbbb2222",
        "source_system": SOURCE_SYSTEM,
        "dataset": DATASET, "dataset_release": RELEASE,
        "created_utc": "2026-08-08T00:00:00+00:00",
        "tool": {"name": "aria_source", "version": "test"},
        "files": {"manifest": ARIA_PREP_MANIFEST, "video": "scene.mp4",
                  "gaze": "gaze.csv",
                  "world_timestamps": "world_timestamps.csv"},
        "recording": {"recording_id": REC_ID, "source_dir": str(out),
                      "task": "Task 19: Skimming (reading quickly)"},
        "video": {"n_frames": n_frames, "width": W, "height": H,
                  "duration_s": span / 1e9, "has_audio": False},
        "rgb_stream": {"stream_id": "214-1", "label": "camera-rgb"},
        "timestamps": {"domain": "aria_device_time", "units": "nanoseconds",
                       "first_ns": ts[0], "last_ns": ts[-1],
                       "compatibility_note": TIMESTAMP_DOMAIN_NOTE,
                       "gap_report": {"irregular_count": 0}},
        "orientation": {"transform": "rot90_cw_upright", "detail": "test"},
        "gaze": {"model": gaze_model, "model_reason": "test fixture",
                 "rows_total": gaze_n,
                 "projection_ok": gaze_n - projection_failures,
                 "projection_outside_image": 0,
                 "projection_failed": projection_failures,
                 "fallback_depth_m": 1.0},
        "audio": {"present": False, "reason": "synthetic fixture"},
        "environment": {"projectaria_tools": "test"},
        "validation": [{"check": "synthetic fixture", "ok": True,
                        "detail": ""}],
        "output_directory": str(out),
    }
    (out / ARIA_PREP_MANIFEST).write_text(json.dumps(manifest, indent=2))
    return out
