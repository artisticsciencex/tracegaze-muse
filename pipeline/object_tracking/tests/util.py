"""Shared synthetic-recording builders for object_tracking tests."""

import json
from fractions import Fraction
from pathlib import Path

T0 = 1_733_200_839_984_000_000
OFFSETS_MS = [0, 50, 100, 133, 166, 180, 220, 253, 286, 320]
WORLD_TS = [T0 + int(ms * 1e6) for ms in OFFSETS_MS]
REC_END = T0 + int((OFFSETS_MS[-1] + 34) * 1e6)
GAZE_TS = [T0 + int(70e6) + i * 5_000_000 for i in range(40)]

W, H = 160, 120


def encode_vfr_video(path: Path, offsets_ms=OFFSETS_MS) -> None:
    import av
    import numpy as np
    tb = Fraction(1, 90000)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264")
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.codec_context.time_base = tb
        s.time_base = tb
        s.options = {"crf": "20", "preset": "ultrafast", "g": "3"}
        for i, ms in enumerate(offsets_ms):
            img = np.full((H, W, 3), (i * 23) % 255, dtype=np.uint8)
            f = av.VideoFrame.from_ndarray(img, format="rgb24")
            f = f.reformat(format="yuv420p")
            f.pts = int(round(ms * 90))
            f.time_base = tb
            for pkt in s.encode(f):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def write_csvs(rec: Path, world_ts=WORLD_TS, gaze_ts=GAZE_TS) -> None:
    with open(rec / "world_timestamps.csv", "w") as f:
        f.write("section id,recording id,timestamp [ns]\n")
        for t in world_ts:
            f.write(f"sec-1,rec-1,{t}\n")
    with open(rec / "gaze.csv", "w") as f:
        f.write("section id,recording id,timestamp [ns],"
                "gaze x [px],gaze y [px]\n")
        for t in gaze_ts:
            f.write(f"sec-1,rec-1,{t},80.5,60.25\n")
    (rec / "info.json").write_text(json.dumps({
        "recording_id": "rec-1", "start_time": T0,
        "duration": REC_END - T0}))


def make_csv_export(root: Path, name: str = "rec") -> Path:
    d = root / name
    d.mkdir(parents=True)
    encode_vfr_video(d / "scene.mp4")
    write_csvs(d)
    return d


def make_trimmed_artifact(root: Path, name: str = "trim") -> Path:
    """A REAL trimmed artifact via the neon_trim pipeline (full contract)."""
    from neon_trim.export import trim_recording
    src = make_csv_export(root, name + "_src")
    out = root / name
    trim_recording(src, 0, len(OFFSETS_MS) - 1, out)
    return out


def rle_for_box(x0, y0, x1, y1, w=W, h=H) -> str:
    """COCO RLE json string for a filled rectangle (test masks)."""
    import numpy as np
    import pycocotools.mask as mask_util
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    rle = mask_util.encode(np.asfortranarray(m))
    return json.dumps({"size": [h, w],
                       "counts": rle["counts"].decode("ascii")})


def track_rows(frames_dirs, *, box=(30, 30, 70, 70)):
    rows = []
    for fidx, direction in frames_dirs:
        rows.append({
            "frame_index": fidx, "tracking_direction": direction,
            "sam_local_id": 1, "score": 0.9,
            "bbox_x": float(box[0]), "bbox_y": float(box[1]),
            "bbox_w": float(box[2] - box[0]), "bbox_h": float(box[3] - box[1]),
            "mask_area": (box[2] - box[0]) * (box[3] - box[1]),
            "mask_rle": rle_for_box(*box),
        })
    rows.sort(key=lambda r: r["frame_index"])
    return rows


def write_track_rows(db_path, object_id: str, frames_dirs, *,
                     box=(30, 30, 70, 70)) -> int:
    """Insert an object's rows into the session SQLite store (server-safe)."""
    from object_tracking import trackstore
    return trackstore.replace_object_rows(db_path, object_id,
                                          track_rows(frames_dirs, box=box))


def write_track_parquet(path: Path, frames_dirs, *, box=(30, 30, 70, 70)):
    """Legacy per-object parquet — used ONLY to simulate orphaned files for
    the isolated-recovery tests (tests may import pandas; the server not)."""
    import pandas as pd
    pd.DataFrame(track_rows(frames_dirs, box=box)).to_parquet(path,
                                                              index=False)
