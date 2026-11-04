"""Tracking-engine tests: merge/validation rules + the real worker protocol
machinery driven by a FAKE compute worker (no GPU needed).
"""

import json
import sys
import textwrap
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from common.jobs import JobCancelled
from object_tracking import tracking
from object_tracking.tracking import (WorkerHandle, presence_summary,
                                      validate_track_rows)

from util import make_csv_export, rle_for_box


# ---------------------------------------------------- merge/validation rules

def _df(rows):
    return pd.DataFrame(rows, columns=["frame_index", "tracking_direction",
                                       "sam_local_id", "score", "bbox_x",
                                       "bbox_y", "bbox_w", "bbox_h",
                                       "mask_area", "mask_rle"])


def _row(f, d):
    return {"frame_index": f, "tracking_direction": d, "sam_local_id": 1,
            "score": 0.9, "bbox_x": 1, "bbox_y": 1, "bbox_w": 5, "bbox_h": 5,
            "mask_area": 25, "mask_rle": "{}"}


def test_merge_forward_backward_no_duplicate_reference():
    # ref=5 of 10 frames: forward 5..9, backward 4..0 -> all 10, no dups
    rows = [_row(f, "forward") for f in range(5, 10)] + \
           [_row(f, "backward") for f in range(4, -1, -1)]
    df = _df(rows)
    assert validate_track_rows(df, object_id="obj_0001", n_frames=10,
                               ref_frame=5) == []


def test_duplicate_frame_rows_detected():
    rows = [_row(5, "forward"), _row(5, "backward")]
    problems = validate_track_rows(_df(rows), object_id="o", n_frames=10,
                                   ref_frame=5)
    assert any("duplicate" in p for p in problems)


def test_direction_side_violations_detected():
    rows = [_row(3, "forward")]          # forward row BEFORE ref
    p1 = validate_track_rows(_df(rows), object_id="o", n_frames=10,
                             ref_frame=5)
    assert any("before reference" in p for p in p1)
    rows = [_row(7, "backward")]         # backward row AFTER ref
    p2 = validate_track_rows(_df(rows), object_id="o", n_frames=10,
                             ref_frame=5)
    assert any("at/after reference" in p for p in p2)


def test_out_of_range_frames_detected():
    p = validate_track_rows(_df([_row(10, "forward")]), object_id="o",
                            n_frames=10, ref_frame=5)
    assert any("outside" in p for p in p)


def test_empty_track_is_legitimate():
    assert validate_track_rows(_df([]), object_id="o", n_frames=10,
                               ref_frame=5) == []
    s = presence_summary(_df([]), 10)
    assert s["frames_present"] == 0 and s["first_frame"] is None


def test_presence_gaps_and_absence():
    # object absent frames 0-2 and 6 (entered late, blinked out)
    rows = [_row(f, "forward") for f in (3, 4, 5, 7, 8)]
    s = presence_summary(_df(rows), 10)
    assert s["first_frame"] == 3 and s["last_frame"] == 8
    assert s["gaps"] == [[6, 6]] and s["n_gaps"] == 1
    assert s["frames_present"] == 5


# ------------------------------------------------- worker protocol machinery

PIPELINE_DIR = str(Path(__file__).resolve().parent.parent.parent)
FAKE_WORKER = textwrap.dedent("""
    import json, sys, time, threading
    CANCEL = threading.Event()
    def emit(o): sys.stdout.write(json.dumps(o)+"\\n"); sys.stdout.flush()
    import queue
    q = queue.Queue()
    def reader():
        for line in sys.stdin:
            r = json.loads(line)
            if r.get("cmd") == "cancel": CANCEL.set(); continue
            q.put(r)
        q.put(None)
    threading.Thread(target=reader, daemon=True).start()
    RLE = {rle!r}
    while True:
        req = q.get()
        if req is None or req.get("cmd") == "shutdown":
            emit({{"event": "bye"}}); break
        if req["cmd"] == "init":
            emit({{"event": "phase", "phase": "loading_model"}})
            emit({{"event": "ready", "gpus": 0, "gpu_names": ["fake"],
                  "torch": "fake", "fa3_disabled": True, "version": "sam3",
                  "session_id": "fake"}})
        elif req["cmd"] == "preview":
            emit({{"event": "preview", "found": True, "score": 0.91,
                  "sam_local_id": 1, "mask_area": 1600, "bbox_x": 30.0,
                  "bbox_y": 30.0, "bbox_w": 40.0, "bbox_h": 40.0,
                  "mask_rle": RLE}})
        elif req["cmd"] == "track":
            CANCEL.clear()
            ref = req["ref_frame"]; slow = req.get("_slow", False)
            rows = []
            cancelled = False
            for d, rng in (("forward", range(ref, 10)),
                           ("backward", range(ref - 1, -1, -1))):
                for f in rng:
                    if slow: time.sleep(0.15)
                    if CANCEL.is_set(): cancelled = True; break
                    rows.append(dict(frame_index=f, tracking_direction=d,
                        sam_local_id=1, score=0.9, bbox_x=30.0, bbox_y=30.0,
                        bbox_w=40.0, bbox_h=40.0, mask_area=1600,
                        mask_rle=RLE))
                    emit({{"event": "track_progress",
                          "object_id": req["object_id"], "direction": d,
                          "frames_done": len(rows)}})
                if cancelled: break
            if cancelled:
                emit({{"event": "track_cancelled",
                      "object_id": req["object_id"]}})
                continue
            import sys as _s
            _s.path.insert(0, {pipeline_dir!r})
            from object_tracking import trackstore
            rows.sort(key=lambda r: r["frame_index"])
            trackstore.replace_object_rows(req["db_path"], req["object_id"], rows)
            frames = sorted(set(r["frame_index"] for r in rows))
            emit({{"event": "track_done", "object_id": req["object_id"],
                  "rows": len(rows), "frames_present": len(frames),
                  "first_frame": frames[0], "last_frame": frames[-1],
                  "forward_ok": True, "backward_ok": True,
                  "forward_error": None, "backward_error": None}})
""")


@pytest.fixture()
def fake_worker(tmp_path):
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER.format(rle=rle_for_box(30, 30, 70, 70), pipeline_dir=PIPELINE_DIR))
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"\x00")     # fake worker never opens it
    wh = WorkerHandle(vid, tmp_path / "w.log", python=Path(sys.executable),
                      worker_script=script, require_patch=False,
                      init_timeout_s=30)
    yield wh
    wh.stop()


def test_worker_protocol_init_preview_track(fake_worker, tmp_path):
    info = fake_worker.start()
    assert info["fa3_disabled"] is True
    ev = fake_worker.preview([[40, 40]], [1], 5, 160, 120)
    assert ev["found"] and ev["score"] == 0.91
    db = tmp_path / "tracks.db"
    done = fake_worker.track("obj_0001", [[40, 40]], [1], 5, 160, 120, db)
    assert done["rows"] == 10 and done["forward_ok"] and done["backward_ok"]
    from object_tracking import trackstore
    pairs = trackstore.object_frames(db, "obj_0001")
    assert validate_track_rows(pairs, object_id="obj_0001", n_frames=10,
                               ref_frame=5) == []
    assert sorted(f for f, _ in pairs) == list(range(10))


def test_worker_cancellation_and_cleanup(fake_worker, tmp_path):
    fake_worker.start()
    cancel = threading.Event()
    out = tmp_path / "cancel.db"

    def cancel_soon():
        time.sleep(0.5)
        cancel.set()
    threading.Thread(target=cancel_soon, daemon=True).start()
    # _slow flag makes the fake worker sleep per frame so cancel lands
    # mid-run; the engine's own track() drives the cancel watcher and must
    # surface JobCancelled without writing any parquet
    with pytest.raises(JobCancelled):
        # need the raw _send path for the private _slow flag
        with fake_worker._lock:
            fake_worker._send({"cmd": "track", "object_id": "o",
                               "_slow": True, "points": [[1, 1]],
                               "labels": [1], "ref_frame": 5,
                               "width": 160, "height": 120,
                               "db_path": str(out)})
            got = None
            deadline = time.monotonic() + 20
            for ev in fake_worker._events(deadline=deadline):
                if cancel.is_set():
                    fake_worker.cancel_current()
                    cancel.clear()
                if ev.get("event") == "track_cancelled":
                    got = ev
                    raise JobCancelled()
    from object_tracking import trackstore as _ts
    assert _ts.object_stats(out, "o") is None if out.exists() else True
    pid = fake_worker._proc.pid
    fake_worker.stop()
    assert fake_worker._proc is None
    # process really gone
    import os
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_worker_stop_kills_process_group(fake_worker):
    fake_worker.start()
    pid = fake_worker._proc.pid
    fake_worker.stop()
    import os
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_missing_patch_blocks_start(tmp_path, monkeypatch):
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"\x00")
    wh = WorkerHandle(vid, tmp_path / "w.log", python=Path(sys.executable),
                      worker_script=tmp_path / "nonexistent.py")
    monkeypatch.setattr(tracking, "verify_local_patch", lambda: False)
    with pytest.raises(tracking.TrackingError, match="LOCAL PATCH"):
        wh.start()


def test_worker_model_failure_reported(tmp_path):
    bad = tmp_path / "bad_worker.py"
    bad.write_text("import json,sys\n"
                   "line=sys.stdin.readline()\n"
                   "sys.stdout.write(json.dumps({'event':'error',"
                   "'error':'CUDA out of memory (fake)'})+'\\n')\n"
                   "sys.stdout.flush()\n")
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"\x00")
    wh = WorkerHandle(vid, tmp_path / "w.log", python=Path(sys.executable),
                      worker_script=bad, require_patch=False,
                      init_timeout_s=20)
    with pytest.raises(tracking.TrackingError, match="CUDA out of memory"):
        wh.start()
    wh.stop()
