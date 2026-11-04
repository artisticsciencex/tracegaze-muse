"""Open-vocabulary object annotations for tracking sessions.

Model rules (enforced here, not in HTTP routes or JS):

* immutable pipeline object IDs ``obj_0001``, ``obj_0002``, ... — never
  reused (a deleted object's ID stays retired) and never changed by rename
* manually entered names: arbitrary non-empty strings, unique per recording;
  duplicate names are rejected with a suggested suffix
* stable deterministic color derived from the immutable ID (golden-angle hue)
  — identical across processes/runs, unaffected by rename
* per-object reference frame + positive/negative prompt points in ORIGINAL
  video pixel coordinates
* full revision history (prompt edits, previews, reruns) and review status

The store persists as JSON under the stage state dir keyed by the recording
folder, so drafts survive server restarts. The same file becomes the
``object_annotations.json`` published inside the tracked artifact.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

VALID_STATUSES = ("draft", "mask_accepted", "tracked", "accepted",
                  "rejected", "excluded")


class AnnotationError(ValueError):
    pass


def color_for_object_id(object_id: str) -> str:
    """Deterministic hex color from the immutable ID (golden-angle hue)."""
    m = re.fullmatch(r"obj_(\d+)", object_id)
    if m:
        idx = int(m.group(1))
    else:  # fallback for any non-standard id: stable hash
        idx = int(hashlib.sha256(object_id.encode()).hexdigest()[:6], 16)
    hue = (idx * 137.508) % 360.0
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.55, 0.72)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255),
                                        int(b * 255))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AnnotationStore:
    """All object annotations for one recording (thread-safe, persisted)."""

    def __init__(self, path: Path, frame_count: int,
                 frame_w: int, frame_h: int):
        self.path = Path(path)
        self.frame_count = int(frame_count)
        self.frame_w, self.frame_h = int(frame_w), int(frame_h)
        self._lock = threading.RLock()
        self._data = {"next_index": 1, "objects": {}}
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text())
                if isinstance(loaded, dict) and "objects" in loaded:
                    self._data = loaded
            except (OSError, ValueError):
                pass  # corrupted draft: start fresh rather than crash

    # ---------------------------------------------------------------- io
    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        tmp.replace(self.path)

    # ------------------------------------------------------------- helpers
    def _obj(self, object_id: str) -> dict:
        o = self._data["objects"].get(object_id)
        if o is None:
            raise AnnotationError(f"unknown object: {object_id}")
        return o

    def _names(self, exclude: Optional[str] = None) -> set:
        return {o["name"] for oid, o in self._data["objects"].items()
                if oid != exclude and not o.get("deleted")}

    def _check_name(self, name: str, exclude: Optional[str] = None) -> str:
        name = (name or "").strip()
        if not name:
            raise AnnotationError("object name must be non-empty")
        if name in self._names(exclude):
            base, n = name, 2
            m = re.fullmatch(r"(.*?)(?:_(\d+))?", name)
            if m and m.group(2):
                base, n = m.group(1), int(m.group(2)) + 1
            suggestion = f"{base}_{n}"
            while suggestion in self._names(exclude):
                n += 1
                suggestion = f"{base}_{n}"
            raise AnnotationError(
                f"name {name!r} is already used — names must be unique "
                f"(suggestion: {suggestion!r})")
        return name

    def _check_points(self, points: list, labels: list, ref_frame: int):
        if not (0 <= int(ref_frame) < self.frame_count):
            raise AnnotationError(
                f"reference frame {ref_frame} outside video "
                f"(0..{self.frame_count - 1})")
        if len(points) != len(labels):
            raise AnnotationError("points/labels length mismatch")
        for (x, y) in points:
            if not (0 <= float(x) < self.frame_w
                    and 0 <= float(y) < self.frame_h):
                raise AnnotationError(
                    f"prompt point ({x:.1f}, {y:.1f}) lies outside the "
                    f"{self.frame_w}x{self.frame_h} frame")
        for lb in labels:
            if int(lb) not in (0, 1):
                raise AnnotationError(f"point label must be 0 or 1, got {lb}")

    # ------------------------------------------------------------ commands
    def create_object(self, name: str, ref_frame: int,
                      note: str = "") -> dict:
        with self._lock:
            name = self._check_name(name)
            if not (0 <= int(ref_frame) < self.frame_count):
                raise AnnotationError(
                    f"reference frame {ref_frame} outside video")
            idx = self._data["next_index"]
            self._data["next_index"] = idx + 1
            oid = f"obj_{idx:04d}"
            obj = {
                "object_id": oid,                     # immutable
                "name": name,
                "note": note or "",
                "color": color_for_object_id(oid),    # immutable
                "reference_frame": int(ref_frame),
                "points": [], "point_labels": [],
                "status": "draft",
                "preview": None,          # last accepted-preview metadata
                "track": None,            # last tracking result summary
                "deleted": False,
                "created_utc": _now(),
                "revisions": [{"utc": _now(), "action": "created",
                               "reference_frame": int(ref_frame)}],
            }
            self._data["objects"][oid] = obj
            self._save()
            return dict(obj)

    def set_points(self, object_id: str, points: list, labels: list,
                   ref_frame: Optional[int] = None) -> dict:
        with self._lock:
            o = self._obj(object_id)
            rf = int(ref_frame) if ref_frame is not None \
                else o["reference_frame"]
            self._check_points(points, labels, rf)
            o["points"] = [[float(x), float(y)] for x, y in points]
            o["point_labels"] = [int(v) for v in labels]
            if rf != o["reference_frame"]:
                o["reference_frame"] = rf
            # any prompt change invalidates a previously accepted mask AND a
            # previously computed track: the old rows no longer correspond to
            # these prompts, so they must never be reviewable or publishable
            if o["status"] in ("mask_accepted", "tracked", "accepted",
                               "rejected"):
                o["status"] = "draft"
            o["preview"] = None
            if o.get("track") is not None:
                o["track"] = None
                o["revisions"].append({"utc": _now(),
                                       "action": "track_invalidated",
                                       "reason": "prompt edited"})
            o["revisions"].append({"utc": _now(), "action": "points_set",
                                   "reference_frame": rf,
                                   "points": o["points"],
                                   "point_labels": o["point_labels"]})
            self._save()
            return dict(o)

    def move_reference_frame(self, object_id: str, new_frame: int) -> dict:
        """Explicitly re-anchor an object to a different reference frame.

        Anchor coordinates belong to the IMAGE they were placed on, so they
        can never be carried into another frame: this clears the anchors,
        the preview mask, any computed track and any review decision, and
        records an auditable revision. The immutable object id, its name
        and its deterministic color are preserved; the object returns to
        `draft` and must be re-prompted, previewed, accepted and retracked.
        """
        with self._lock:
            o = self._obj(object_id)
            rf = int(new_frame)
            if not (0 <= rf < self.frame_count):
                raise AnnotationError(
                    f"reference frame {rf} outside 0..{self.frame_count - 1}")
            old = o["reference_frame"]
            if rf == old:
                return dict(o)
            cleared = {
                "points": len(o.get("points") or []),
                "had_preview": o.get("preview") is not None,
                "had_track": o.get("track") is not None,
                "previous_status": o["status"],
            }
            o["reference_frame"] = rf
            o["points"] = []
            o["point_labels"] = []
            o["preview"] = None
            o["track"] = None
            o["status"] = "draft"
            o["revisions"].append({
                "utc": _now(), "action": "reference_frame_moved",
                "from_frame": old, "to_frame": rf, "cleared": cleared,
                "reason": "anchors belong to the previous reference image; "
                          "re-prompt, preview, use the selection and "
                          "retrack on the new frame"})
            self._save()
            return dict(o)

    def record_preview(self, object_id: str, preview_meta: dict) -> dict:
        with self._lock:
            o = self._obj(object_id)
            o["preview"] = dict(preview_meta)
            o["revisions"].append({"utc": _now(), "action": "previewed",
                                   "reference_frame": o["reference_frame"],
                                   "meta": {k: preview_meta.get(k) for k in
                                            ("score", "mask_area", "found")}})
            self._save()
            return dict(o)

    def accept_mask(self, object_id: str) -> dict:
        with self._lock:
            o = self._obj(object_id)
            if not o["points"] or 1 not in o["point_labels"]:
                raise AnnotationError(
                    "cannot accept: at least one positive point is required")
            if not o.get("preview") or not o["preview"].get("found"):
                raise AnnotationError(
                    "cannot accept: run the initial-mask preview first")
            o["status"] = "mask_accepted"
            o["revisions"].append({"utc": _now(), "action": "mask_accepted",
                                   "reference_frame": o["reference_frame"]})
            self._save()
            return dict(o)

    def record_track_result(self, object_id: str, track_meta: dict,
                            expected_revision_count: Optional[int] = None
                            ) -> dict:
        """Record a finished track. When expected_revision_count is given the
        write is compare-and-set: if the object was edited/deleted while the
        pass ran, the stale result is refused instead of overwriting."""
        with self._lock:
            o = self._obj(object_id)
            if o.get("deleted"):
                raise AnnotationError(
                    f"{object_id} was deleted while tracking ran — result "
                    f"discarded")
            if expected_revision_count is not None \
                    and len(o["revisions"]) != expected_revision_count:
                raise AnnotationError(
                    f"{object_id} was edited while tracking ran — stale "
                    f"result discarded; re-track from the current prompt")
            o["track"] = dict(track_meta)
            o["status"] = "tracked"
            o["revisions"].append({"utc": _now(), "action": "tracked",
                                   "meta": {k: track_meta.get(k) for k in
                                            ("rows", "forward_ok",
                                             "backward_ok")}})
            self._save()
            return dict(o)

    def set_review_status(self, object_id: str, status: str) -> dict:
        if status not in ("accepted", "rejected", "excluded"):
            raise AnnotationError(f"invalid review status {status!r}")
        with self._lock:
            o = self._obj(object_id)
            if status == "accepted" and not o.get("track"):
                raise AnnotationError("cannot accept an untracked object")
            if status == "accepted" and o["status"] == "draft":
                raise AnnotationError(
                    "prompt was edited after tracking — re-track before "
                    "accepting")
            o["status"] = status
            o["revisions"].append({"utc": _now(), "action": status})
            self._save()
            return dict(o)

    def rename(self, object_id: str, new_name: str) -> dict:
        with self._lock:
            o = self._obj(object_id)
            new_name = self._check_name(new_name, exclude=object_id)
            old = o["name"]
            o["name"] = new_name
            # explicitly NOT touching object_id or color
            o["revisions"].append({"utc": _now(), "action": "renamed",
                                   "from": old, "to": new_name})
            self._save()
            return dict(o)

    def set_note(self, object_id: str, note: str) -> dict:
        with self._lock:
            o = self._obj(object_id)
            o["note"] = str(note or "")
            self._save()
            return dict(o)

    def delete(self, object_id: str) -> None:
        with self._lock:
            o = self._obj(object_id)
            o["deleted"] = True
            o["revisions"].append({"utc": _now(), "action": "deleted"})
            self._save()

    # -------------------------------------------------------------- queries
    def objects(self, include_deleted: bool = False) -> list:
        with self._lock:
            out = [dict(o) for o in self._data["objects"].values()
                   if include_deleted or not o.get("deleted")]
            return sorted(out, key=lambda o: o["object_id"])

    def get(self, object_id: str) -> dict:
        with self._lock:
            return dict(self._obj(object_id))

    def publishable(self) -> tuple:
        """(ready: bool, blockers: [str]) — publish gate per spec: every
        non-deleted object must be accepted or explicitly excluded/rejected."""
        blockers = []
        objs = self.objects()
        if not any(o["status"] == "accepted" for o in objs):
            blockers.append("no accepted object tracks to publish")
        for o in objs:
            if o["status"] not in ("accepted", "rejected", "excluded"):
                blockers.append(
                    f"{o['object_id']} ({o['name']}) is {o['status']} — "
                    f"accept, reject, or exclude it before publishing")
        return (not blockers, blockers)

    def export_annotations(self) -> dict:
        """The human-readable object_annotations.json payload."""
        return {
            "format": "object_tracking.annotations/1",
            "frame_count": self.frame_count,
            "frame_size": [self.frame_w, self.frame_h],
            "objects": self.objects(include_deleted=True),
        }
