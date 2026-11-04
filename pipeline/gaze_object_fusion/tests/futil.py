"""Synthetic builders for gaze_object_fusion tests.

Self-contained analogue of object_tracking/tests/util.py (deliberately NOT
imported from there: distinct module name avoids the pytest 'util'
sys.modules collision across suites, and fusion tests need parametrizable
gaze rows / object boxes / scores).

Everything goes through the REAL pipeline contracts: neon_trim trims the
synthetic source, object_tracking publishes the tracked artifact.
"""

import json
from fractions import Fraction
from pathlib import Path

T0 = 1_733_200_839_984_000_000
OFFSETS_MS = [0, 50, 100, 133, 166, 180, 220, 253, 286, 320]
WORLD_TS = [T0 + int(ms * 1e6) for ms in OFFSETS_MS]
REC_END = T0 + int((OFFSETS_MS[-1] + 34) * 1e6)

W, H = 160, 120
N = len(OFFSETS_MS)


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


def _gaze_line(r, with_worn, with_ids):
    line = f"sec-1,rec-1,{r['ts']},{r['x']},{r['y']}"
    if with_worn:
        line += f",{r.get('worn', 1.0)}"
    if with_ids:
        fx = r.get("fixation")
        bl = r.get("blink")
        line += f",{'' if fx is None else fx},{'' if bl is None else bl}"
    return line


def write_csvs(rec: Path, gaze_rows, world_ts=WORLD_TS,
               with_worn: bool = True, with_ids: bool = False) -> None:
    """gaze_rows: dicts {ts, x, y, worn?, fixation?, blink?}."""
    with open(rec / "world_timestamps.csv", "w") as f:
        f.write("section id,recording id,timestamp [ns]\n")
        for t in world_ts:
            f.write(f"sec-1,rec-1,{t}\n")
    with open(rec / "gaze.csv", "w") as f:
        cols = "section id,recording id,timestamp [ns],gaze x [px],gaze y [px]"
        if with_worn:
            cols += ",worn"
        if with_ids:
            cols += ",fixation id,blink id"
        f.write(cols + "\n")
        for r in gaze_rows:
            f.write(_gaze_line(r, with_worn, with_ids) + "\n")
    (rec / "info.json").write_text(json.dumps({
        "recording_id": "rec-1", "start_time": T0,
        "duration": REC_END - T0}))


def rle_for_box(x0, y0, x1, y1, w=W, h=H) -> str:
    import numpy as np
    import pycocotools.mask as mask_util
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    rle = mask_util.encode(np.asfortranarray(m))
    return json.dumps({"size": [h, w],
                       "counts": rle["counts"].decode("ascii")})


def make_tracked_artifact(tmp_path: Path, gaze_rows, objects,
                          with_worn: bool = True, post_trim_gaze=None):
    """Full REAL pipeline: synthetic source -> neon_trim -> object_tracking.

    objects: list of {"name", "box": (x0,y0,x1,y1), "score", "frames"?}
             frames defaults to every frame.
    post_trim_gaze: rows appended to the TRIMMED gaze.csv before tracking
             publishes (trim drops out-of-interval rows, so exercising the
             no_matching_frame path needs post-trim injection; hashes stay
             consistent because tracking hashes the final file).
    Returns (tracked_dir, trim_dir, [object_id...]).
    """
    from neon_trim.export import trim_recording
    from object_tracking import intake, trackstore
    from object_tracking.annotations import AnnotationStore
    from object_tracking.artifact import publish_tracked_artifact

    all_rows = list(gaze_rows) + list(post_trim_gaze or [])
    with_ids = any("fixation" in r or "blink" in r for r in all_rows)
    src = tmp_path / "src"
    src.mkdir(parents=True)
    encode_vfr_video(src / "scene.mp4")
    write_csvs(src, gaze_rows, with_worn=with_worn, with_ids=with_ids)
    trim_dir = tmp_path / "trim"
    trim_recording(src, 0, N - 1, trim_dir)
    if post_trim_gaze:
        with open(trim_dir / "gaze.csv", "a") as f:
            for r in post_trim_gaze:
                f.write(_gaze_line(r, with_worn, with_ids) + "\n")

    bundle = intake.open_bundle(trim_dir)
    store = AnnotationStore(tmp_path / "ann.json", bundle.n_frames,
                            bundle.width, bundle.height)
    tracks = tmp_path / "tracks.db"
    ref = 5
    oids = []
    for spec in objects:
        o = store.create_object(spec["name"], ref)
        oid = o["object_id"]
        oids.append(oid)
        x0, y0, x1, y1 = spec["box"]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        store.set_points(oid, [[cx, cy]], [1])
        store.record_preview(oid, {"found": True, "score": spec["score"],
                                   "mask_area": (x1 - x0) * (y1 - y0)})
        store.accept_mask(oid)
        frames = spec.get("frames", list(range(bundle.n_frames)))
        rows = []
        for fidx in sorted(frames):
            rows.append({
                "frame_index": fidx,
                "tracking_direction": "forward" if fidx >= ref
                                      else "backward",
                "sam_local_id": 1, "score": spec["score"],
                "bbox_x": float(x0), "bbox_y": float(y0),
                "bbox_w": float(x1 - x0), "bbox_h": float(y1 - y0),
                "mask_area": (x1 - x0) * (y1 - y0),
                "mask_rle": rle_for_box(x0, y0, x1, y1),
            })
        trackstore.replace_object_rows(tracks, oid, rows)
        store.record_track_result(oid, {
            "rows": len(rows), "frames_present": len(rows),
            "first_frame": min(frames), "last_frame": max(frames),
            "forward_ok": True, "backward_ok": True,
            "whole_video": len(frames) == bundle.n_frames})
        store.set_review_status(oid, "accepted")

    out = tmp_path / "tracked"
    publish_tracked_artifact(bundle, store, tracks, out, make_overlay=False)
    return out, trim_dir, oids


def trim_selection(trim_dir: Path) -> dict:
    m = json.loads((Path(trim_dir) / "trim_manifest.json").read_text())
    return m["selection"]
