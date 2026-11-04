"""SQLite session store for track rows — the server-safe data plane.

WHY THIS EXISTS: pyarrow segfaulted the GUI server twice (kernel log:
``segfault ... in libarrow.so.2500``), the second time from
``pandas.read_parquet`` inside the FastAPI tracking-job thread. The server
process must therefore NEVER touch Pandas/PyArrow/Parquet. This module is
stdlib-``sqlite3`` only and provides everything the server needs:

* one row per visible object/frame (PRIMARY KEY (object_id, frame_index) —
  duplicate rows are impossible at the storage level)
* scalar tracking fields + RLE mask text, indexed by object id + frame index
* ATOMIC per-object replacement (single transaction: a cancelled or crashed
  rerun leaves the previous rows untouched)
* bounded frame-range mask retrieval for review (never loads all masks)

Writers: the SAM3 worker (GPU env) and the isolated table_io helper — both
import this same module; sqlite WAL mode gives one-writer/many-reader safety
across processes. Parquet remains ONLY the published artifact format,
produced by the isolated publication subprocess.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

ROW_FIELDS = ("object_id", "frame_index", "tracking_direction",
              "sam_local_id", "score", "bbox_x", "bbox_y", "bbox_w",
              "bbox_h", "mask_area", "mask_rle")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS track_rows(
    object_id           TEXT    NOT NULL,
    frame_index         INTEGER NOT NULL,
    tracking_direction  TEXT    NOT NULL,
    sam_local_id        INTEGER,
    score               REAL,
    bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
    mask_area           INTEGER,
    mask_rle            TEXT    NOT NULL,
    PRIMARY KEY (object_id, frame_index)
);
CREATE INDEX IF NOT EXISTS idx_track_rows_frame
    ON track_rows(frame_index);
"""


def connect(db_path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    return conn


# ---------------------------------------------------------------- writers

def replace_object_rows(db_path, object_id: str,
                        rows: Iterable[dict]) -> int:
    """Atomically replace ALL rows of one object (delete + insert in one
    transaction). Returns the number of rows written. Any failure —
    including a duplicate frame_index, which the PK forbids — rolls the
    transaction back and leaves the object's previous rows intact."""
    conn = connect(db_path)
    try:
        with conn:  # one transaction
            conn.execute("DELETE FROM track_rows WHERE object_id = ?",
                         (object_id,))
            n = 0
            for r in rows:
                conn.execute(
                    "INSERT INTO track_rows VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (object_id, int(r["frame_index"]),
                     str(r["tracking_direction"]),
                     int(r.get("sam_local_id") or 0),
                     float(r.get("score") or 0.0),
                     float(r.get("bbox_x") or 0.0),
                     float(r.get("bbox_y") or 0.0),
                     float(r.get("bbox_w") or 0.0),
                     float(r.get("bbox_h") or 0.0),
                     int(r.get("mask_area") or 0),
                     str(r["mask_rle"])))
                n += 1
        return n
    finally:
        conn.close()


def delete_object_rows(db_path, object_id: str) -> None:
    conn = connect(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM track_rows WHERE object_id = ?",
                         (object_id,))
    finally:
        conn.close()


# ---------------------------------------------------------------- readers

def objects_in_db(db_path) -> list:
    conn = connect(db_path)
    try:
        return [r[0] for r in conn.execute(
            "SELECT DISTINCT object_id FROM track_rows ORDER BY object_id")]
    finally:
        conn.close()


def object_frames(db_path, object_id: str) -> list:
    """[(frame_index, tracking_direction)] — the validation working set
    (no masks materialized)."""
    conn = connect(db_path)
    try:
        return list(conn.execute(
            "SELECT frame_index, tracking_direction FROM track_rows "
            "WHERE object_id = ? ORDER BY frame_index", (object_id,)))
    finally:
        conn.close()


def object_stats(db_path, object_id: str) -> Optional[dict]:
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*), MIN(frame_index), MAX(frame_index) "
            "FROM track_rows WHERE object_id = ?", (object_id,)).fetchone()
        if not row or row[0] == 0:
            return None
        dirs = dict(conn.execute(
            "SELECT tracking_direction, COUNT(*) FROM track_rows "
            "WHERE object_id = ? GROUP BY tracking_direction", (object_id,)))
        return {"rows": row[0], "first_frame": row[1], "last_frame": row[2],
                "frames_present": row[0], "directions": dirs}
    finally:
        conn.close()


def timelines(db_path, object_id: str, max_points: int = 2000) -> dict:
    """Downsampled per-frame area/score series for the review charts."""
    conn = connect(db_path)
    try:
        rows = list(conn.execute(
            "SELECT frame_index, mask_area, score FROM track_rows "
            "WHERE object_id = ? ORDER BY frame_index", (object_id,)))
    finally:
        conn.close()
    stride = max(1, len(rows) // max_points)
    sampled = rows[::stride]
    return {"area": [[int(f), int(a)] for f, a, _ in sampled],
            "score": [[int(f), float(s)] for f, _, s in sampled]}


def rows_for_range(db_path, start: int, end: int,
                   object_ids: Optional[list] = None) -> dict:
    """frame_index -> [{object_id, score, mask_rle, bbox}] for a BOUNDED
    range (the caller enforces the request cap)."""
    conn = connect(db_path)
    try:
        q = ("SELECT frame_index, object_id, score, mask_rle, "
             "bbox_x, bbox_y, bbox_w, bbox_h FROM track_rows "
             "WHERE frame_index BETWEEN ? AND ?")
        args = [int(start), int(end)]
        if object_ids is not None:
            q += (" AND object_id IN ("
                  + ",".join("?" * len(object_ids)) + ")")
            args += list(object_ids)
        out: dict = {}
        for f, oid, score, rle, bx, by, bw, bh in conn.execute(q, args):
            out.setdefault(int(f), []).append(
                {"object_id": oid, "score": float(score), "mask_rle": rle,
                 "bbox": [bx, by, bw, bh]})
        return out
    finally:
        conn.close()


def iter_object_rows(db_path, object_id: str):
    """Full row dicts for one object, streamed in frame order (used by the
    isolated publication subprocess — NOT by the server)."""
    conn = connect(db_path)
    try:
        cols = ",".join(ROW_FIELDS)
        for row in conn.execute(
                f"SELECT {cols} FROM track_rows WHERE object_id = ? "
                f"ORDER BY frame_index", (object_id,)):
            yield dict(zip(ROW_FIELDS, row))
    finally:
        conn.close()


def export_rows_json(db_path, object_id: str) -> str:
    return json.dumps(list(iter_object_rows(db_path, object_id)))
