"""Arrow/Parquet isolation regressions.

Root cause being guarded: pyarrow segfaulted the FastAPI server process
twice (kernel log `segfault ... in libarrow.so.2500`), the second time from
``pandas.read_parquet`` in the tracking-job thread. The server must never
import pandas/pyarrow; all Parquet work is subprocess-isolated; track rows
live in a SQLite store.
"""

import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from object_tracking import table_io, trackstore

from util import make_csv_export, track_rows, write_track_parquet, \
    write_track_rows

PIPELINE_DIR = Path(__file__).resolve().parent.parent.parent


# ----------------------------------------------------------- store basics

def test_atomic_per_object_replacement(tmp_path):
    db = tmp_path / "t.db"
    write_track_rows(db, "obj_0001", [(f, "forward") for f in range(5)])
    assert trackstore.object_stats(db, "obj_0001")["rows"] == 5
    # replacement with duplicate frames must roll back COMPLETELY
    with pytest.raises(sqlite3.IntegrityError):
        trackstore.replace_object_rows(
            db, "obj_0001", track_rows([(1, "forward"), (1, "backward")]))
    st = trackstore.object_stats(db, "obj_0001")
    assert st["rows"] == 5 and st["first_frame"] == 0   # old rows intact
    # successful replacement swaps rows in one step
    write_track_rows(db, "obj_0001", [(f, "forward") for f in range(3, 9)])
    st = trackstore.object_stats(db, "obj_0001")
    assert st["rows"] == 6 and st["first_frame"] == 3


def test_bounded_range_retrieval_and_isolation_between_objects(tmp_path):
    db = tmp_path / "t.db"
    write_track_rows(db, "obj_0001", [(f, "forward") for f in range(10)])
    write_track_rows(db, "obj_0002", [(f, "forward") for f in range(4, 8)])
    got = trackstore.rows_for_range(db, 4, 6, ["obj_0001", "obj_0002"])
    assert sorted(got.keys()) == [4, 5, 6]
    assert all(len(v) == 2 for v in got.values())
    # deleting one object never touches another (accepted tracks preserved)
    trackstore.delete_object_rows(db, "obj_0002")
    assert trackstore.object_stats(db, "obj_0002") is None
    assert trackstore.object_stats(db, "obj_0001")["rows"] == 10


# ------------------------------------------------------ helper subprocess

def test_import_parquet_recovery_roundtrip(tmp_path):
    """An orphaned completed parquet is validated + imported atomically by
    the ISOLATED helper (the test process may use pandas; the server not)."""
    pq = tmp_path / "obj_0002.parquet"
    frames = [(f, "forward") for f in range(150, 300)] + \
             [(f, "backward") for f in range(0, 150)]
    write_track_parquet(pq, frames)
    db = tmp_path / "t.db"
    res = table_io.run_helper(
        ["import-parquet", "--db", db, "--object", "obj_0002",
         "--parquet", pq, "--n-frames", 300, "--ref-frame", 150])
    assert res["ok"] and res["rows"] == 300
    assert res["forward_rows"] == 150 and res["backward_rows"] == 150
    st = trackstore.object_stats(db, "obj_0002")
    assert st["rows"] == 300 and st["first_frame"] == 0 \
        and st["last_frame"] == 299


def test_import_parquet_rejects_invalid_and_preserves_others(tmp_path):
    db = tmp_path / "t.db"
    write_track_rows(db, "obj_0001", [(f, "forward") for f in range(5)])
    bad = tmp_path / "obj_0002.parquet"
    # invalid: forward rows before the claimed reference frame
    write_track_parquet(bad, [(f, "forward") for f in range(0, 5)])
    with pytest.raises(table_io.HelperError, match="failed validation"):
        table_io.run_helper(
            ["import-parquet", "--db", db, "--object", "obj_0002",
             "--parquet", bad, "--n-frames", 300, "--ref-frame", 150])
    assert trackstore.object_stats(db, "obj_0002") is None
    assert trackstore.object_stats(db, "obj_0001")["rows"] == 5  # untouched


def test_helper_segfault_is_clear_error(tmp_path, monkeypatch):
    """Exit 139 in the helper must surface as a precise message, not kill
    anything."""
    crasher = tmp_path / "crasher.py"
    crasher.write_text("import os, signal\n"
                       "os.kill(os.getpid(), signal.SIGSEGV)\n")
    monkeypatch.setattr(table_io, "HELPER_PATH", crasher)
    with pytest.raises(table_io.HelperError,
                       match="segmentation fault \\(exit 139\\)"):
        table_io.run_helper(["import-parquet", "--db", "x", "--object", "o",
                             "--parquet", "p", "--n-frames", 1,
                             "--ref-frame", 0])


# --------------------------------------------------------- server process

def _server_flow_script(root, state_dir):
    return textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(PIPELINE_DIR)!r})
        import hashlib, json
        from pathlib import Path

        # pre-seed a session: annotations + SQLite rows (NO pandas anywhere)
        src = Path({str(root)!r}) / "rec1"
        key = hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:16]
        wd = Path({str(state_dir)!r}) / "sessions" / key
        wd.mkdir(parents=True, exist_ok=True)

        from object_tracking.annotations import AnnotationStore
        from object_tracking import trackstore
        store = AnnotationStore(wd / "annotations.json", 10, 160, 120)
        o = store.create_object("thing", 5)
        store.set_points(o["object_id"], [[40, 40]], [1])
        store.record_preview(o["object_id"], {{"found": True}})
        store.accept_mask(o["object_id"])
        rows = [dict(frame_index=f, tracking_direction="forward" if f >= 5
                     else "backward", sam_local_id=1, score=0.9, bbox_x=1,
                     bbox_y=1, bbox_w=5, bbox_h=5, mask_area=25,
                     mask_rle='{{"size": [120, 160], "counts": "PP1"}}')
                for f in range(10)]
        trackstore.replace_object_rows(wd / "tracks.db", o["object_id"], rows)
        store.record_track_result(o["object_id"],
                                  {{"rows": 10, "forward_ok": True,
                                    "backward_ok": True}})

        from fastapi.testclient import TestClient
        from common import config as cc
        from object_tracking import server as srv
        cfg = cc.StageConfig(stage="object_tracking", port=8002,
                             browse_roots=[Path({str(root)!r})],
                             state_dir=Path({str(state_dir)!r}))
        c = TestClient(srv.create_app(cfg), base_url="http://localhost")
        r = c.post("/api/load", json={{"dir": str(src)}})
        assert r.status_code == 200, r.text
        r = c.get("/api/review/summary", params={{"dir": str(src)}})
        assert r.status_code == 200 and r.json()["objects"], r.text
        r = c.get("/api/review/masks",
                  params={{"dir": str(src), "start": 0, "end": 9}})
        assert r.status_code == 200 and r.json()["frames"], r.text
        r = c.get("/api/publish/check", params={{"dir": str(src)}})
        assert r.status_code == 200, r.text

        forbidden = [m for m in ("pandas", "pyarrow")
                     if m in sys.modules]
        assert not forbidden, f"SERVER IMPORTED {{forbidden}}"
        print("SERVER_PANDAS_FREE_OK")
    """)


def test_server_flow_never_imports_pandas(tmp_path):
    """Full load -> track-registered -> review-summary -> masks -> publish-
    check flow in a pristine interpreter: pandas/pyarrow must never load."""
    root = tmp_path / "root"
    root.mkdir()
    make_csv_export(root, "rec1")
    script = tmp_path / "flow.py"
    script.write_text(_server_flow_script(root, tmp_path / "state"))
    r = subprocess.run([sys.executable, str(script)], capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SERVER_PANDAS_FREE_OK" in r.stdout


def test_server_survives_helper_crash(tmp_path, monkeypatch):
    """/api/artifact via a crashing helper -> clear 4xx, server keeps
    serving."""
    from fastapi.testclient import TestClient
    from common import config as cc
    from object_tracking import server as srv
    root = tmp_path / "root"
    root.mkdir()
    make_csv_export(root, "rec1")
    (root / "fake_art").mkdir()
    crasher = tmp_path / "crasher.py"
    crasher.write_text("import os, signal\n"
                       "os.kill(os.getpid(), signal.SIGSEGV)\n")
    monkeypatch.setattr(table_io, "HELPER_PATH", crasher)
    cfg = cc.StageConfig(stage="object_tracking", port=8002,
                         browse_roots=[root], state_dir=tmp_path / "state")
    c = TestClient(srv.create_app(cfg), base_url="http://localhost")
    r = c.get("/api/artifact", params={"path": str(root / "fake_art")})
    assert r.status_code == 422
    assert "segmentation fault (exit 139)" in r.json()["detail"]
    assert "server is unaffected" in r.json()["detail"]
    # the server is alive and fully functional afterwards
    assert c.get("/api/health").status_code == 200
    assert c.post("/api/load",
                  json={"dir": str(root / "rec1")}).status_code == 200


def test_orphan_recovery_through_api_load(tmp_path):
    """A completed-but-unregistered parquet (server crashed before
    record_track_result) is recovered on load via the isolated helper."""
    from fastapi.testclient import TestClient
    from common import config as cc
    from object_tracking import server as srv
    from object_tracking.annotations import AnnotationStore
    root = tmp_path / "root"
    root.mkdir()
    src = make_csv_export(root, "rec1")
    key = hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:16]
    wd = tmp_path / "state" / "sessions" / key
    (wd / "tracks").mkdir(parents=True)
    store = AnnotationStore(wd / "annotations.json", 10, 160, 120)
    o = store.create_object("orphaned", 5)
    store.set_points(o["object_id"], [[40, 40]], [1])
    store.record_preview(o["object_id"], {"found": True})
    store.accept_mask(o["object_id"])          # mask_accepted, track=None
    write_track_parquet(wd / "tracks" / f"{o['object_id']}.parquet",
                        [(f, "forward") for f in range(5, 10)]
                        + [(f, "backward") for f in range(0, 5)])
    cfg = cc.StageConfig(stage="object_tracking", port=8002,
                         browse_roots=[root], state_dir=tmp_path / "state")
    c = TestClient(srv.create_app(cfg), base_url="http://localhost")
    r = c.post("/api/load", json={"dir": str(src)})
    assert r.status_code == 200
    d = r.json()
    assert any("recovered completed track" in w for w in d["warnings"])
    obj = [x for x in d["objects"] if x["object_id"] == o["object_id"]][0]
    assert obj["status"] == "tracked"
    assert obj["track"]["rows"] == 10 and obj["track"]["recovered"] is True
    # rows are queryable through the pandas-free review path
    r = c.get("/api/review/masks",
              params={"dir": str(src), "start": 0, "end": 9})
    assert r.status_code == 200 and len(r.json()["frames"]) == 10
    # the parquet was retired so it cannot be double-imported
    assert not (wd / "tracks" / f"{o['object_id']}.parquet").exists()


def test_corrupt_orphan_keeps_annotation_and_requires_rerun(tmp_path):
    from fastapi.testclient import TestClient
    from common import config as cc
    from object_tracking import server as srv
    from object_tracking.annotations import AnnotationStore
    root = tmp_path / "root"
    root.mkdir()
    src = make_csv_export(root, "rec1")
    key = hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:16]
    wd = tmp_path / "state" / "sessions" / key
    (wd / "tracks").mkdir(parents=True)
    store = AnnotationStore(wd / "annotations.json", 10, 160, 120)
    o = store.create_object("broken", 5)
    store.set_points(o["object_id"], [[40, 40]], [1])
    store.record_preview(o["object_id"], {"found": True})
    store.accept_mask(o["object_id"])
    (wd / "tracks" / f"{o['object_id']}.parquet").write_bytes(b"not parquet")
    cfg = cc.StageConfig(stage="object_tracking", port=8002,
                         browse_roots=[root], state_dir=tmp_path / "state")
    c = TestClient(srv.create_app(cfg), base_url="http://localhost")
    r = c.post("/api/load", json={"dir": str(src)})
    assert r.status_code == 200
    d = r.json()
    assert any("could not be recovered" in w and "rerun" in w
               for w in d["warnings"])
    obj = [x for x in d["objects"] if x["object_id"] == o["object_id"]][0]
    assert obj["status"] == "mask_accepted" and obj["track"] is None


# ------------------------------------------------------------ worker pipe

def test_worker_emit_survives_broken_pipe(monkeypatch):
    """The shutdown BrokenPipeError is secondary — emit must swallow it,
    flag the parent as gone, and never raise a traceback of its own."""
    from object_tracking import sam3_worker

    class DeadPipe:
        def write(self, *_):
            raise BrokenPipeError()

        def flush(self):
            raise BrokenPipeError()

    monkeypatch.setattr(sam3_worker.sys, "stdout", DeadPipe())
    sam3_worker.PARENT_GONE.clear()
    sam3_worker.CANCEL.clear()
    sam3_worker.emit({"event": "track_progress"})   # must not raise
    assert sam3_worker.PARENT_GONE.is_set()
    assert sam3_worker.CANCEL.is_set()              # aborts in-flight work
    sam3_worker.emit({"event": "again"})            # still silent
    sam3_worker.PARENT_GONE.clear()
    sam3_worker.CANCEL.clear()
