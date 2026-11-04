"""SAM3 worker subprocess for object_tracking.

Runs inside the GPU (sam3) environment, isolated from the GUI server so that
SAM3's multi-GPU multiprocessing, CUDA state, and crashes never take the
server down. Protocol: JSON lines on stdin (commands) and stdout (events);
all SAM3/torch logging goes to stderr (captured to a log file by the parent).

Commands
    {"cmd":"init","video":str,"version":"sam3","score_thresh":float}
    {"cmd":"preview","points":[[x,y]..],"labels":[0|1..],"ref_frame":int,
     "width":int,"height":int}
    {"cmd":"track","object_id":str,"points":..,"labels":..,"ref_frame":int,
     "width":int,"height":int,"out_path":str}
    {"cmd":"cancel"}          (async: honoured mid-propagation)
    {"cmd":"shutdown"}

Design notes validated against the vendored SAM3 source:
* instance selection uses the SAM2-style tracker path: positive/negative
  POINTS + obj_id with rel_coordinates=True (the reliable path; the raw
  box-exemplar path is not used)
* the add_prompt RESPONSE already contains the reference-frame mask -> the
  initial-mask preview needs no propagation
* propagate_in_video accepts propagation_direction "forward"/"backward" and
  "start_frame_index"; forward yields ref..end, backward yields ref-1..0, so
  the reference frame appears exactly once (in the forward pass)
* requires the LOCAL PATCH in sam3_video_inference.py (points branch marks
  previous_stages_out) — verified by the parent before spawning
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CANCEL = threading.Event()
PARENT_GONE = threading.Event()


def emit(obj):
    """Write one protocol line. If the parent server died, the pipe is
    broken — that is SECONDARY to whatever killed the parent, so it must
    never raise a noisy traceback of its own: log once and shut down."""
    if PARENT_GONE.is_set():
        return
    try:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()
    except (BrokenPipeError, OSError):
        PARENT_GONE.set()
        CANCEL.set()
        log("parent pipe closed — shutting down quietly")


def log(msg):
    print(f"[sam3_worker] {msg}", file=sys.stderr, flush=True)


def encode_rle(mask):
    import numpy as np
    import pycocotools.mask as mask_util
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(rle["size"][0]), int(rle["size"][1])],
            "counts": rle["counts"].decode("ascii")}


class Worker:
    def __init__(self):
        self.predictor = None
        self.session_id = None
        self.score_thresh = 0.5

    # ------------------------------------------------------------- init
    def init(self, req):
        import torch
        emit({"event": "phase", "phase": "loading_model"})
        # Blackwell (sm_120) + cuDNN 9.19: heuristic engine selection fails
        # for SAM3's neck convs ("GET was unable to find an engine ...");
        # benchmark mode's exhaustive search finds working engines (verified
        # on RTX PRO 6000). Because these flags CANNOT propagate into SAM3's
        # spawned multi-GPU rank workers, the default is a single in-process
        # GPU; opt into multi-GPU with OBJTRACK_GPUS=0,1 on architectures
        # that don't need the flag.
        torch.backends.cudnn.benchmark = True
        self.score_thresh = float(req.get("score_thresh", 0.5))
        version = req.get("version", "sam3")
        if version != "sam3":
            # project policy: FA3 must stay disabled; base sam3 has no FA3
            # path at all, sam3.1 is only reachable with use_fa3=False.
            log(f"requested version={version}; forcing use_fa3=False")
        if version == "sam3":
            from sam3.model_builder import build_sam3_video_predictor
            gpus_env = os.environ.get("OBJTRACK_GPUS", "").strip()
            if gpus_env:
                gpus = [int(g) for g in gpus_env.split(",") if g.strip()]
            else:
                gpus = [0] if torch.cuda.device_count() else None
            self.predictor = build_sam3_video_predictor(gpus_to_use=gpus)
        else:
            from sam3.model_builder import build_sam3_predictor
            self.predictor = build_sam3_predictor(
                version=version, use_fa3=False)
        emit({"event": "phase", "phase": "loading_video"})
        resp = self.predictor.handle_request(
            {"type": "start_session", "resource_path": req["video"]})
        self.session_id = resp["session_id"]
        emit({"event": "ready",
              "gpus": torch.cuda.device_count(),
              "gpu_names": [torch.cuda.get_device_name(i)
                            for i in range(torch.cuda.device_count())],
              "torch": torch.__version__,
              "fa3_disabled": True,
              "version": version,
              "session_id": self.session_id})

    # ------------------------------------------------------- prompt utils
    def _add_point_prompt(self, req):
        import torch
        pts = torch.tensor(req["points"], dtype=torch.float32)
        pts[:, 0] /= float(req["width"])
        pts[:, 1] /= float(req["height"])
        lbl = torch.tensor(req["labels"], dtype=torch.int32)
        return self.predictor.handle_request({
            "type": "add_prompt", "session_id": self.session_id,
            "frame_index": int(req["ref_frame"]),
            "points": pts, "point_labels": lbl,
            "obj_id": 1, "rel_coordinates": True,
        })

    @staticmethod
    def _extract(outputs, score_thresh):
        """outputs dict -> list of per-object dicts (px bbox, RLE mask)."""
        import numpy as np
        obj_ids = np.asarray(outputs["out_obj_ids"]).reshape(-1)
        probs = np.asarray(outputs.get("out_probs")
                           if outputs.get("out_probs") is not None
                           else []).reshape(-1)
        boxes = np.asarray(outputs["out_boxes_xywh"]).reshape(-1, 4)
        masks = outputs["out_binary_masks"]
        out = []
        for i, oid in enumerate(obj_ids.tolist()):
            score = float(probs[i]) if i < len(probs) else 1.0
            if score < score_thresh:
                continue
            mask = np.asarray(masks[i]).astype(bool)
            if not mask.any():
                continue
            H, W = mask.shape
            bx, by, bw, bh = (boxes[i] * np.array([W, H, W, H])).tolist()
            out.append({
                "sam_local_id": int(oid), "score": round(score, 4),
                "bbox_x": round(bx, 2), "bbox_y": round(by, 2),
                "bbox_w": round(bw, 2), "bbox_h": round(bh, 2),
                "mask_area": int(mask.sum()),
                "mask_rle": json.dumps(encode_rle(mask)),
            })
        return out

    # ----------------------------------------------------------- preview
    def preview(self, req):
        self.predictor.handle_request(
            {"type": "reset_session", "session_id": self.session_id})
        resp = self._add_point_prompt(req)
        # SAME threshold as tracking: a mask the user can accept here must
        # also survive the tracking filter (no silent empty-track surprises)
        objs = self._extract(resp["outputs"], score_thresh=self.score_thresh)
        if not objs:
            below = self._extract(resp["outputs"], score_thresh=0.0)
            best_score = max((o["score"] for o in below), default=None)
            emit({"event": "preview", "found": False,
                  "best_score": best_score,
                  "score_thresh": self.score_thresh,
                  "note": (f"a mask exists but its score {best_score} is "
                           f"below the tracking threshold "
                           f"{self.score_thresh} — refine the points"
                           if best_score is not None else
                           "no mask found at these points")})
            return
        best = max(objs, key=lambda o: o["score"])
        emit({"event": "preview", "found": True, **best})

    # ------------------------------------------------------------- track
    def track(self, req):
        CANCEL.clear()
        ref = int(req["ref_frame"])
        self.predictor.handle_request(
            {"type": "reset_session", "session_id": self.session_id})
        self._add_point_prompt(req)

        rows = []
        status = {"forward": {"ok": False, "error": None, "frames": 0},
                  "backward": {"ok": False, "error": None, "frames": 0}}
        cancelled = False
        for direction in ("forward", "backward"):
            if cancelled:
                break
            try:
                stream = self.predictor.handle_stream_request({
                    "type": "propagate_in_video",
                    "session_id": self.session_id,
                    "propagation_direction": direction,
                    "start_frame_index": ref,
                })
                n = 0
                for fr in stream:
                    if CANCEL.is_set():
                        cancelled = True
                        try:
                            self.predictor.handle_request(
                                {"type": "cancel_propagation",
                                 "session_id": self.session_id})
                        except Exception:
                            pass
                        # drain remaining frames quickly without storing
                        continue
                    fidx = int(fr["frame_index"])
                    for o in self._extract(fr["outputs"], self.score_thresh):
                        rows.append({"frame_index": fidx,
                                     "tracking_direction": direction, **o})
                    n += 1
                    if n % 25 == 0:
                        emit({"event": "track_progress",
                              "object_id": req["object_id"],
                              "direction": direction, "frames_done": n})
                status[direction]["frames"] = n
                status[direction]["ok"] = not cancelled
            except Exception as e:  # per-direction failure, keep other rows
                status[direction]["error"] = f"{type(e).__name__}: {e}"
                log(f"{direction} propagation failed:\n"
                    + traceback.format_exc())

        if cancelled:
            emit({"event": "track_cancelled", "object_id": req["object_id"]})
            return

        # persist rows ATOMICALLY into the session SQLite store (single
        # DELETE+INSERT transaction): a rerun never destroys the previous
        # good rows unless the replacement commits, and the GUI server reads
        # this store WITHOUT pandas/pyarrow (which segfaulted it before).
        from object_tracking import trackstore
        rows.sort(key=lambda r: r["frame_index"])
        n = trackstore.replace_object_rows(req["db_path"],
                                           req["object_id"], rows)
        frames = sorted({r["frame_index"] for r in rows})
        emit({"event": "track_done", "object_id": req["object_id"],
              "rows": n, "frames_present": len(frames),
              "first_frame": frames[0] if frames else None,
              "last_frame": frames[-1] if frames else None,
              "forward_ok": status["forward"]["ok"],
              "backward_ok": status["backward"]["ok"],
              "forward_error": status["forward"]["error"],
              "backward_error": status["backward"]["error"]})

    def shutdown(self):
        try:
            if self.predictor is not None:
                if self.session_id is not None:
                    self.predictor.handle_request(
                        {"type": "close_session",
                         "session_id": self.session_id})
                self.predictor.shutdown()
        except Exception:
            pass


def stdin_reader(queue_put):
    """Reader thread: cancel is handled immediately, others are queued."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            emit({"event": "error", "error": f"bad json: {line[:120]}"})
            continue
        if req.get("cmd") == "cancel":
            CANCEL.set()
            continue
        queue_put(req)
    queue_put(None)  # stdin closed -> shut down


def main():
    import queue
    q = queue.Queue()
    t = threading.Thread(target=stdin_reader, args=(q.put,), daemon=True)
    t.start()
    worker = Worker()
    while True:
        req = q.get()
        if req is None or req.get("cmd") == "shutdown" \
                or PARENT_GONE.is_set():
            worker.shutdown()
            emit({"event": "bye"})
            return
        cmd = req.get("cmd")
        try:
            if cmd == "init":
                worker.init(req)
            elif cmd == "preview":
                worker.preview(req)
            elif cmd == "track":
                worker.track(req)
            else:
                emit({"event": "error", "error": f"unknown cmd {cmd!r}"})
        except Exception as e:  # command-level failure, worker stays alive
            log(traceback.format_exc())
            emit({"event": "error", "cmd": cmd,
                  "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
