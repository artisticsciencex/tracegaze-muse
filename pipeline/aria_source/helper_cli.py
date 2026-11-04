"""Isolated Project Aria helper — the ONLY place projectaria_tools runs.

Executed as a subprocess by the dedicated Aria interpreter (see
`aria_source.config.aria_python()`), never imported by a server process:
VRS/calibration are native code, and a segfault here must not be able to
take a long-lived GUI server with it.

Protocol: every protocol line is written to stdout prefixed with
`@@ARIA@@ ` followed by one JSON object. projectaria_tools logs freely to
stdout/stderr; the parent ignores anything without the prefix.

    {"event": "progress", "phase": "...", "frac": 0.42}
    {"event": "result", ...}          exactly one, on success
    {"event": "error", "message": "...", "kind": "..."}

Subcommands:
    probe    --vrs FILE
    prepare  --vrs FILE --eye-gaze CSV --gaze-model NAME --out DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from pathlib import Path

PREFIX = "@@ARIA@@ "
RGB_STREAM_ID = "214-1"
TIME_BASE = 90000                 # video PTS ticks per second (as NEON)

# canonical + provenance columns of the Aria gaze table
GAZE_COLUMNS = [
    "timestamp [ns]", "gaze x [px]", "gaze y [px]",
    "tracking_timestamp_us", "left_yaw_rads_cpf", "right_yaw_rads_cpf",
    "combined_yaw_rads_cpf", "pitch_rads_cpf", "depth_m", "depth_source",
    "session_uid", "gaze_model", "projection_valid", "projection_status",
    "left_yaw_low_rads_cpf", "right_yaw_low_rads_cpf", "pitch_low_rads_cpf",
    "left_yaw_high_rads_cpf", "right_yaw_high_rads_cpf",
    "pitch_high_rads_cpf", "tx_left_eye_cpf", "ty_left_eye_cpf",
    "tz_left_eye_cpf", "tx_right_eye_cpf", "ty_right_eye_cpf",
    "tz_right_eye_cpf",
]

STATUS_OK = "ok"
STATUS_OUTSIDE = "outside_image"
STATUS_NO_PROJECTION = "outside_field_of_view"
DEPTH_MEASURED = "measured"
DEPTH_FALLBACK = "fallback_1m"
FALLBACK_DEPTH_M = 1.0            # documented Project Aria example default


def emit(obj: dict) -> None:
    # leading newline: projectaria_tools' logger writes colour escapes
    # without a trailing newline, so a protocol line must start its own
    sys.stdout.write("\n" + PREFIX + json.dumps(obj) + "\n")
    sys.stdout.flush()


def progress(phase: str, frac: float) -> None:
    emit({"event": "progress", "phase": phase, "frac": round(frac, 4)})


def _versions() -> dict:
    import numpy
    out = {"python": sys.version.split()[0], "numpy": numpy.__version__}
    try:
        from importlib.metadata import version
        out["projectaria_tools"] = version("projectaria-tools")
    except Exception:                                    # noqa: BLE001
        out["projectaria_tools"] = "unknown"
    try:
        import av
        out["av"] = av.__version__
    except Exception:                                    # noqa: BLE001
        out["av"] = None
    return out


def _open_rgb(vrs_path: str):
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.stream_id import StreamId
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise RuntimeError(f"cannot open VRS file: {vrs_path}")
    sid = StreamId(RGB_STREAM_ID)
    label = provider.get_label_from_stream_id(sid)
    if not label:
        raise RuntimeError(f"VRS has no RGB stream {RGB_STREAM_ID}")
    return provider, sid, label


def _rgb_timestamps(provider, sid) -> list:
    """Exact int-ns DEVICE_TIME capture timestamp per RGB frame.

    Uses the stream's timestamp table (no image decoding); verified
    identical to each frame record's capture_timestamp_ns.
    """
    from projectaria_tools.core.sensor_data import TimeDomain
    ts = [int(t) for t in
          provider.get_timestamps_ns(sid, TimeDomain.DEVICE_TIME)]
    if ts:                      # spot-check the table against the records
        _, rec0 = provider.get_image_data_by_index(sid, 0)
        _, recN = provider.get_image_data_by_index(sid, len(ts) - 1)
        if (int(rec0.capture_timestamp_ns) != ts[0]
                or int(recN.capture_timestamp_ns) != ts[-1]):
            raise RuntimeError("RGB timestamp table disagrees with the "
                               "frame records — refusing to continue")
    return ts


def _gap_report(ts: list) -> dict:
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    if not gaps:
        return {"n_gaps": 0, "irregular": []}
    gaps_sorted = sorted(gaps)
    median = gaps_sorted[len(gaps_sorted) // 2]
    # a frame interval deviating >50% from the median is reported
    irregular = [{"between_frames": [i, i + 1], "gap_ms": g / 1e6}
                 for i, g in enumerate(gaps)
                 if abs(g - median) > 0.5 * median]
    return {"n_gaps": len(gaps), "median_gap_ms": median / 1e6,
            "min_gap_ms": min(gaps) / 1e6, "max_gap_ms": max(gaps) / 1e6,
            "nominal_fps": round(1e9 / median, 3) if median else None,
            "irregular_count": len(irregular),
            "irregular": irregular[:50],
            "strictly_increasing": all(g > 0 for g in gaps)}


def cmd_probe(args) -> int:
    provider, sid, label = _open_rgb(args.vrs)
    ts = _rgb_timestamps(provider, sid)
    img, _ = provider.get_image_data_by_index(sid, 0)
    arr = img.to_numpy_array()
    dev = provider.get_device_calibration()
    cam = dev.get_camera_calib(label)
    emit({"event": "result",
          "rgb_stream_id": RGB_STREAM_ID, "rgb_stream_label": label,
          "n_frames": len(ts),
          "native_size": [int(arr.shape[1]), int(arr.shape[0])],
          "calib_size": [int(v) for v in cam.get_image_size()],
          "first_device_time_ns": ts[0] if ts else None,
          "last_device_time_ns": ts[-1] if ts else None,
          "duration_s": ((ts[-1] - ts[0]) / 1e9) if len(ts) > 1 else 0.0,
          "timestamps_ns": ts,
          "gap_report": _gap_report(ts),
          "streams": [f"{s}={provider.get_label_from_stream_id(s)}"
                      for s in provider.get_all_streams()],
          "versions": _versions()})
    return 0


def _read_gaze_rows(path: Path) -> list:
    """Exact CSV rows (text preserved) — the timestamp never touches a
    float: tracking_timestamp_us is parsed as int and scaled by 1000."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"eye-gaze CSV has no rows: {path}")
    if "tracking_timestamp_us" not in rows[0]:
        raise RuntimeError(f"{path}: no tracking_timestamp_us column")
    return rows


def _fnum(row: dict, key: str):
    v = (row.get(key) or "").strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def cmd_prepare(args) -> int:
    import numpy as np
    from projectaria_tools.core import calibration, mps
    from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    progress("opening VRS", 0.01)
    provider, sid, label = _open_rgb(args.vrs)
    dev_calib = provider.get_device_calibration()
    cam_calib = dev_calib.get_camera_calib(label)
    # ONE documented orientation transform, applied to image AND
    # calibration: Aria RGB is stored rotated; upright = 90 deg clockwise.
    upright_calib = calibration.rotate_camera_calib_cw90deg(cam_calib)

    progress("reading RGB timestamps", 0.03)
    ts = _rgb_timestamps(provider, sid)
    n_all = len(ts)
    limit = min(n_all, args.frame_limit) if args.frame_limit else n_all
    ts = ts[:limit]
    if not ts:
        raise RuntimeError("RGB stream has no frames")
    if any(b <= a for a, b in zip(ts, ts[1:])):
        raise RuntimeError("RGB device timestamps are not strictly "
                           "increasing — refusing to build an artifact")
    gaps = _gap_report(ts)

    # ---------------------------------------------------------- video
    progress("encoding scene.mp4", 0.05)
    import av
    from fractions import Fraction
    scene = out / "scene.mp4"
    t0 = ts[0]
    width = height = None
    with av.open(str(scene), "w") as oc:
        stream = oc.add_stream("libx264")
        stream.pix_fmt = "yuv420p"
        tb = Fraction(1, TIME_BASE)
        stream.codec_context.time_base = tb
        stream.time_base = tb
        stream.options = {"crf": "20", "preset": "veryfast"}
        for i in range(limit):
            img, _ = provider.get_image_data_by_index(sid, i)
            arr = img.to_numpy_array()
            arr = np.rot90(arr, k=-1)            # CW90 -> upright
            arr = np.ascontiguousarray(arr)
            if width is None:
                height, width = int(arr.shape[0]), int(arr.shape[1])
                stream.width, stream.height = width, height
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            frame = frame.reformat(format="yuv420p")
            frame.pts = int(round((ts[i] - t0) * TIME_BASE / 1e9))
            frame.time_base = tb
            for pkt in stream.encode(frame):
                oc.mux(pkt)
            if i % 100 == 0:
                progress("encoding scene.mp4", 0.05 + 0.55 * i / limit)
        for pkt in stream.encode():
            oc.mux(pkt)

    # ------------------------------------------------ world timestamps
    progress("writing world_timestamps.csv", 0.62)
    rec_id = args.recording_id or ""
    with open(out / "world_timestamps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section id", "recording id", "timestamp [ns]"])
        for t in ts:
            w.writerow([rec_id, rec_id, t])     # exact ints, never floats

    # ------------------------------------------------------------ gaze
    progress("projecting gaze", 0.66)
    rows = _read_gaze_rows(Path(args.eye_gaze))
    eyegazes = mps.read_eyegaze(str(args.eye_gaze))
    if len(eyegazes) != len(rows):
        raise RuntimeError(
            f"eye-gaze row mismatch: csv has {len(rows)} rows, "
            f"projectaria_tools parsed {len(eyegazes)}")

    n_ok = n_outside = n_failed = 0
    n_measured = n_fallback = 0
    with open(out / "gaze.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GAZE_COLUMNS)
        w.writeheader()
        for i, (row, eg) in enumerate(zip(rows, eyegazes)):
            us = int(str(row["tracking_timestamp_us"]).strip())
            ts_ns = us * 1000                    # exact integer scaling
            depth = _fnum(row, "depth_m")
            if depth is not None and math.isfinite(depth) and depth > 0:
                depth_used, depth_source = depth, DEPTH_MEASURED
                n_measured += 1
            else:
                depth_used, depth_source = FALLBACK_DEPTH_M, DEPTH_FALLBACK
                n_fallback += 1
            try:
                xy = get_gaze_vector_reprojection(
                    eg, label, dev_calib, upright_calib, depth_used,
                    make_upright=True)
            except Exception:                        # noqa: BLE001
                xy = None
            if xy is None:
                status, valid = STATUS_NO_PROJECTION, False
                gx = gy = ""
                n_failed += 1
            else:
                gx_f, gy_f = float(xy[0]), float(xy[1])
                inside = (0.0 <= gx_f < width) and (0.0 <= gy_f < height)
                status = STATUS_OK if inside else STATUS_OUTSIDE
                valid = True
                # NEVER clamped: the true projection is written as-is
                gx, gy = f"{gx_f:.6f}", f"{gy_f:.6f}"
                if inside:
                    n_ok += 1
                else:
                    n_outside += 1
            w.writerow({
                "timestamp [ns]": ts_ns, "gaze x [px]": gx,
                "gaze y [px]": gy, "tracking_timestamp_us": us,
                "left_yaw_rads_cpf": row.get("left_yaw_rads_cpf", ""),
                "right_yaw_rads_cpf": row.get("right_yaw_rads_cpf", ""),
                "combined_yaw_rads_cpf": f"{float(eg.yaw):.12g}",
                "pitch_rads_cpf": row.get("pitch_rads_cpf", ""),
                "depth_m": row.get("depth_m", ""),
                "depth_source": depth_source,
                "session_uid": row.get("session_uid", ""),
                "gaze_model": args.gaze_model,
                "projection_valid": "true" if valid else "false",
                "projection_status": status,
                **{k: row.get(k, "") for k in (
                    "left_yaw_low_rads_cpf", "right_yaw_low_rads_cpf",
                    "pitch_low_rads_cpf", "left_yaw_high_rads_cpf",
                    "right_yaw_high_rads_cpf", "pitch_high_rads_cpf",
                    "tx_left_eye_cpf", "ty_left_eye_cpf", "tz_left_eye_cpf",
                    "tx_right_eye_cpf", "ty_right_eye_cpf",
                    "tz_right_eye_cpf")},
            })
            if i % 2000 == 0:
                progress("projecting gaze", 0.66 + 0.3 * i / len(rows))

    progress("done", 1.0)
    emit({"event": "result",
          "rgb_stream_id": RGB_STREAM_ID, "rgb_stream_label": label,
          "n_frames": limit, "n_frames_available": n_all,
          "width": width, "height": height,
          "orientation": "rot90_cw_upright",
          "orientation_detail":
              "numpy.rot90(k=-1) on every RGB frame with "
              "calibration.rotate_camera_calib_cw90deg + "
              "get_gaze_vector_reprojection(make_upright=True) — one "
              "transform applied to video, calibration and gaze alike",
          "timestamp_domain": "aria_device_time",
          "timestamp_units": "nanoseconds",
          "first_device_time_ns": ts[0], "last_device_time_ns": ts[-1],
          "duration_s": (ts[-1] - ts[0]) / 1e9,
          "video_time_base": [1, TIME_BASE],
          "gap_report": gaps,
          "gaze_model": args.gaze_model,
          "gaze_rows_total": len(rows),
          "gaze_projection_ok": n_ok,
          "gaze_projection_outside_image": n_outside,
          "gaze_projection_failed": n_failed,
          "gaze_depth_measured": n_measured,
          "gaze_depth_fallback": n_fallback,
          "fallback_depth_m": FALLBACK_DEPTH_M,
          "audio_present": False,
          "audio_note":
              "the VRS microphone stream (231-1) is not muxed in v1: a "
              "frame-accurate audio/RGB mapping across VRS time domains "
              "is not implemented, so audio is reported absent rather "
              "than silently claimed",
          "versions": _versions()})
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aria_helper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--vrs", required=True)
    p.set_defaults(fn=cmd_probe)
    q = sub.add_parser("prepare")
    q.add_argument("--vrs", required=True)
    q.add_argument("--eye-gaze", required=True)
    q.add_argument("--gaze-model", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--recording-id", default="")
    q.add_argument("--frame-limit", type=int, default=0)
    q.set_defaults(fn=cmd_prepare)
    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except Exception as e:                               # noqa: BLE001
        emit({"event": "error", "kind": type(e).__name__, "message": str(e),
              "traceback": traceback.format_exc()[-2000:]})
        return 2


if __name__ == "__main__":
    sys.exit(main())
