"""Demonstration video: gaze-driven object highlighting + gaze cursor.

Semantics (fixed by the stage specification):
- NON-gazed tracked objects: subtle treatment — very low-opacity fill
  (12%) plus a thin (1 px) boundary in the object's color.
- The PRIMARY gazed-at object of the frame: bright 50% colored fill,
  thicker (3 px) boundary, and a name+id label near the object.
- ONE fixed high-contrast cursor (see CURSOR_* below) for every valid
  gaze sample — identical on objects and on background; object-hit
  meaning is conveyed exclusively by the mask highlight + label.
- Invalid / out-of-frame gaze (or no assigned sample): NO cursor drawn —
  the neutral treatment is simply the subtle-mask frame.
- The highlighted object is EXACTLY the frame's demo row
  `primary_object_id` (tested pixel-level).

Timing is unchanged from the verified renderer: identical frame count and
order, source VFR PTS copied per frame, deviation validated (<= 2 ms
beyond the constant container offset), audio retained. Never a
synchronization source.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Callable, Optional

from common.artifacts import ArtifactError

FILL_PRIMARY = 0.50          # spec: 45-55% colored fill for the gazed object
FILL_SUBTLE = 0.12           # spec: very low-opacity mask for the others


def _decode_rle(rle_json: str):
    import pycocotools.mask as mask_util
    r = json.loads(rle_json)
    return mask_util.decode({"size": r["size"],
                             "counts": r["counts"].encode("ascii")}
                            ).astype(bool)


def _hex_rgb(color: str):
    import numpy as np
    c = color.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)],
                    dtype=np.float32)


def _erode(m):
    import numpy as np
    e = m.copy()
    t = np.zeros_like(m)
    t[1:, :] = m[:-1, :]
    e &= t
    t = np.zeros_like(m)
    t[:-1, :] = m[1:, :]
    e &= t
    t = np.zeros_like(m)
    t[:, 1:] = m[:, :-1]
    e &= t
    t = np.zeros_like(m)
    t[:, :-1] = m[:, 1:]
    e &= t
    return e


def _boundary(m, thickness: int):
    e = m
    for _ in range(max(1, thickness)):
        e = _erode(e)
    return m & ~e


# ------------------------------------------------------ tiny bitmap font
# 5x7 glyphs (5-bit rows, MSB left); enough for object names/ids:
# letters (case-insensitive), digits, '-', '_', space.
_F = {
    "a": (0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "b": (0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E),
    "c": (0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E),
    "d": (0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E),
    "e": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F),
    "f": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10),
    "g": (0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F),
    "h": (0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "i": (0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "j": (0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C),
    "k": (0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11),
    "l": (0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F),
    "m": (0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11),
    "n": (0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11),
    "o": (0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "p": (0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10),
    "q": (0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D),
    "r": (0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11),
    "s": (0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E),
    "t": (0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "u": (0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "v": (0x11, 0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04),
    "w": (0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11),
    "x": (0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11),
    "y": (0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04),
    "z": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F),
    "0": (0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E),
    "1": (0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "2": (0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F),
    "3": (0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E),
    "4": (0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02),
    "5": (0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E),
    "6": (0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E),
    "7": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08),
    "8": (0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E),
    "9": (0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C),
    "-": (0x00, 0x00, 0x00, 0x0E, 0x00, 0x00, 0x00),
    "_": (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F),
    " ": (0, 0, 0, 0, 0, 0, 0),
    ".": (0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C),
    ",": (0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x08),
    ":": (0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00),
    "%": (0x19, 0x1A, 0x02, 0x04, 0x08, 0x0B, 0x13),
    "/": (0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10),
    "(": (0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02),
    ")": (0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08),
    "+": (0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00),
    "…": (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x15),
}
_SCALE = 2                                   # glyph cell: 12x14 px
_CW, _CH = 6 * _SCALE, 7 * _SCALE


def draw_label(img, x: int, y: int, text: str, rgb) -> None:
    """name+id label with a dark backing strip at top-left corner (x, y)."""
    import numpy as np
    h, w, _ = img.shape
    text = text.lower()
    tw, th = len(text) * _CW + 8, _CH + 6
    x = max(0, min(w - tw, x))
    y = max(0, min(h - th, y))
    if tw > w or th > h:
        return
    strip = img[y:y + th, x:x + tw]
    strip[:] = (0.30 * strip).astype(strip.dtype)          # dark backing
    color = np.clip(rgb * 0.4 + 255 * 0.6, 0, 255)         # light tinted text
    for ci, ch in enumerate(text):
        g = _F.get(ch)
        if g is None:
            continue
        gx = x + 4 + ci * _CW
        for ry in range(7):
            row = g[ry]
            for rx in range(5):
                if row & (0x10 >> rx):
                    yy = y + 3 + ry * _SCALE
                    xx = gx + rx * _SCALE
                    img[yy:yy + _SCALE, xx:xx + _SCALE] = color


# fixed high-contrast cursor design (identical for gaze-on-object and
# gaze-on-background; object-hit meaning is conveyed EXCLUSIVELY by the
# bright mask fill + thick boundary + name/id label):
#   bright green ring, ~18 px radius at 1600x1200 (scaled by min(w,h)),
#   ~3.8 px ring thickness, solid black outer halo, small red center dot
#   with a thin white edge. No trail, pulse, smoothing, or interpolation.
CURSOR_GREEN = (0, 255, 64)
CURSOR_RED = (235, 30, 30)
CURSOR_WHITE = (255, 255, 255)


def cursor_radius(w: int, h: int) -> int:
    return max(8, round(min(w, h) * 18 / 1200))


def draw_cursor(img, x: float, y: float) -> None:
    """Fixed gaze cursor drawn in-place with numpy at subpixel (x, y)."""
    import numpy as np
    h, w, _ = img.shape
    R = cursor_radius(w, h)
    pad = R + 7
    x0, x1 = max(0, math.floor(x) - pad), min(w, math.floor(x) + pad + 1)
    y0, y1 = max(0, math.floor(y) - pad), min(h, math.floor(y) + pad + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    region = img[y0:y1, x0:x1]
    region[np.abs(d - R) <= 4.6] = (0, 0, 0)            # black outer halo
    region[np.abs(d - R) <= 1.9] = CURSOR_GREEN         # bright green ring
    region[d <= 4.4] = CURSOR_WHITE                     # thin white edge
    region[d <= 2.8] = CURSOR_RED                       # red center dot


def _paint_objects(img, objs: list, primary_id) -> None:
    """objs: [{object_id, name, color, rle, bbox}] — paint per semantics."""
    import numpy as np
    primary = None
    for o in objs:
        m = _decode_rle(o["rle"])
        rgb = _hex_rgb(o["color"])
        if o["object_id"] == primary_id:
            primary = (o, m, rgb)
            continue                          # painted last, on top
        img[m] = ((1 - FILL_SUBTLE) * img[m]
                  + FILL_SUBTLE * rgb).astype(np.uint8)
        edge = _boundary(m, 1)
        img[edge] = (0.4 * img[edge] + 0.6 * rgb).astype(np.uint8)
    if primary is not None:
        o, m, rgb = primary
        img[m] = ((1 - FILL_PRIMARY) * img[m]
                  + FILL_PRIMARY * rgb).astype(np.uint8)
        img[_boundary(m, 3)] = rgb.astype(np.uint8)
        bx, by = int(o["bbox"][0]), int(o["bbox"][1])
        draw_label(img, bx, by - _CH - 8,
                   f"{o['name']} {o['object_id']}", rgb)


PANEL_MIN_SIDE = 360        # below this the panel would dominate the frame
PANEL_TITLE = "ACCUMULATED GAZE ON TRACKED OBJECTS"
DISPLAY_NOTE = "GAZE DISPLAY: LINEARLY INTERPOLATED FOR VISUALIZATION"
DURATION_NOTE = "DURATIONS: RAW GAZE EVIDENCE"


def draw_panel(img, panel: dict) -> None:
    """Upper-right accumulated-gaze panel (numpy only).

    Bars come from ANALYTICAL raw-evidence durations; only the active-row
    marker follows the presentation highlight. Order and scale are fixed
    for the whole video — never rescaled frame to frame.
    """
    import numpy as np
    h, w, _ = img.shape
    # a legible panel needs real estate; on tiny frames it would cover the
    # scene (and the cursor), so it is skipped and recorded as such
    if min(w, h) < PANEL_MIN_SIDE:
        return
    rows = panel["rows"]
    scale = max(1, panel["bar_scale_ns"])
    cum = panel["cumulative"]
    active = panel.get("active_id")
    s = max(1, round(min(w, h) / 700))              # size unit
    pad, line = 8 * s, 15 * s
    pw = min(int(w * 0.46), 350 * s)
    ph = pad * 2 + line * (len(rows) + 4)
    x0, y0 = w - pw - 10 * s, 10 * s
    x1, y1 = x0 + pw, min(h - 1, y0 + ph)
    if x0 < 0 or y1 <= y0:
        return
    region = img[y0:y1, x0:x1]
    region[:] = (0.32 * region).astype(region.dtype)      # semi-transparent
    _text(img, x0 + pad, y0 + pad, PANEL_TITLE, (235, 240, 246), s)
    y = y0 + pad + line
    total_valid = max(1, panel["valid_supported_ns_so_far"])
    for r in rows:
        ns = cum.get(r["key"], 0)
        rgb = _hex_rgb(r["color"])
        sw = 6 * s
        img[y + 2 * s:y + 2 * s + sw, x0 + pad:x0 + pad + sw] = \
            rgb.astype(img.dtype)
        label = f"{r['name']}"[:16]
        _text(img, x0 + pad + sw + 4 * s, y, label, (222, 229, 236), s)
        _text(img, x0 + pad + sw + 4 * s, y + 7 * s, r["object_id"][:16],
              (138, 148, 163), max(1, s - 1))
        secs = ns / 1e9
        pct = 100.0 * ns / total_valid
        val = f"{secs:.1f}s {pct:.0f}%"
        _text(img, x1 - pad - len(val) * 6 * s, y, val, (222, 229, 236), s)
        # bar (fixed shared scale)
        by = y + line - 5 * s
        bw = pw - 2 * pad
        img[by:by + 3 * s, x0 + pad:x0 + pad + bw] = (58, 68, 80)
        fill = int(bw * min(1.0, ns / scale))
        if fill > 0:
            img[by:by + 3 * s, x0 + pad:x0 + pad + fill] = rgb.astype(
                img.dtype)
        if active is not None and r["key"] == active:
            img[by - s:by + 4 * s, x0 + pad - 2 * s:x0 + pad - s] = \
                (255, 255, 255)
        y += line
    _text(img, x0 + pad, y,
          f"No tracked-object hit: {panel['no_hit_ns'] / 1e9:.1f} s",
          (150, 160, 172), s)
    y += line
    _text(img, x0 + pad, y,
          f"Valid supported gaze: {panel['valid_supported_ns_so_far'] / 1e9:.1f} s",
          (150, 160, 172), s)
    y += line
    _text(img, x0 + pad, y, DURATION_NOTE, (138, 148, 163),
          max(1, s - 1))


def _text(img, x, y, text, rgb, scale):
    """Bitmap text via the label font (uppercase 5x7 glyphs)."""
    import numpy as np
    h, w, _ = img.shape
    col = np.array(rgb, dtype=np.float32)
    cw = 6 * scale
    for ci, ch in enumerate(str(text).lower()):
        g = _F.get(ch)
        if g is None:
            continue
        gx = x + ci * cw
        if gx + 5 * scale >= w:
            return
        for ry in range(7):
            row = g[ry]
            for rx in range(5):
                if row & (0x10 >> rx):
                    yy, xx = y + ry * scale, gx + rx * scale
                    if 0 <= yy < h - scale and 0 <= xx < w - scale:
                        img[yy:yy + scale, xx:xx + scale] = col


def render_demo(video_path: Path, out_path: Path, *, n_frames: int,
                objects_by_frame: dict, cursor_by_frame: dict,
                has_audio: bool, keep_audio: bool = True,
                panel_by_frame: Optional[dict] = None,
                display_note: bool = False,
                progress_cb: Optional[Callable[[float], None]] = None
                ) -> dict:
    """objects_by_frame: {frame: [{object_id, name, color, rle, bbox}]};
    cursor_by_frame: {frame: {"x","y","color","primary_object_id"}} —
    the selected demo row per frame (absent => no cursor, subtle masks
    only). Returns {"frames", "max_pts_dev_ms", "pts_offset_ms", "audio"}.
    """
    import av

    tmp_video = str(out_path) + ".video.mp4"
    with av.open(str(video_path)) as ic, av.open(tmp_video, "w") as oc:
        vin = ic.streams.video[0]
        vin.thread_type = "AUTO"
        tb = vin.time_base
        ost = oc.add_stream("libx264")
        ost.width = vin.codec_context.width
        ost.height = vin.codec_context.height
        ost.pix_fmt = "yuv420p"
        ost.codec_context.time_base = tb        # exact VFR pts preserved
        ost.time_base = tb
        ost.options = {"crf": "20", "preset": "veryfast"}
        n = 0
        src_pts = []
        for frame in ic.decode(vin):
            img = frame.to_ndarray(format="rgb24")
            cur = cursor_by_frame.get(n)
            primary_id = cur.get("primary_object_id") if cur else None
            _paint_objects(img, objects_by_frame.get(n, []), primary_id)
            if panel_by_frame is not None and n in panel_by_frame:
                draw_panel(img, panel_by_frame[n])
            if cur is not None:
                draw_cursor(img, cur["x"], cur["y"])   # cursor on top
            if display_note and min(img.shape[1], img.shape[0]) \
                    >= PANEL_MIN_SIDE:
                s = max(1, round(img.shape[0] / 700))
                _text(img, 12 * s, img.shape[0] - 22 * s, DISPLAY_NOTE,
                      (222, 229, 236), s)
            out = av.VideoFrame.from_ndarray(img, format="rgb24")
            out = out.reformat(format="yuv420p")
            out.pts = frame.pts
            out.time_base = tb
            src_pts.append(float(frame.pts * tb))
            for pkt in ost.encode(out):
                oc.mux(pkt)
            n += 1
            if progress_cb is not None and n % 25 == 0:
                progress_cb(n / max(n_frames, 1))
        for pkt in ost.encode():
            oc.mux(pkt)

    if n != n_frames:
        Path(tmp_video).unlink(missing_ok=True)
        raise ArtifactError(f"demo decoded {n} frames, tracked artifact "
                            f"says {n_frames}")

    final = str(out_path)
    if keep_audio and has_audio:
        from common.media import ffmpeg_exe
        r = subprocess.run(
            [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
             "-y", "-i", tmp_video, "-i", str(video_path),
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
             "-movflags", "+faststart", final],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise ArtifactError("demo audio mux failed: "
                                + r.stderr.strip()[-400:])
        Path(tmp_video).unlink(missing_ok=True)
    else:
        Path(tmp_video).replace(final)

    # validate timing exactly like the tracked overlay: constant container
    # offset tolerated, per-frame deviation must stay within 2 ms
    from common.media import decode_all_pts
    out_pts = decode_all_pts(Path(final))
    if len(out_pts) != n:
        raise ArtifactError(f"demo frame count {len(out_pts)} != source {n}")
    offset = (out_pts[0] - src_pts[0]) if n else 0.0
    max_dev = max((abs((a - b) - offset)
                   for a, b in zip(out_pts, src_pts)), default=0.0)
    if max_dev > 0.002:
        raise ArtifactError(f"demo VFR timing deviates {max_dev * 1e3:.2f} "
                            f"ms beyond the constant container offset")
    if abs(offset) > 0.5:
        raise ArtifactError(f"demo pts offset {offset * 1e3:.1f} ms exceeds "
                            f"sanity bound")
    return {"frames": n, "max_pts_dev_ms": round(max_dev * 1e3, 4),
            "pts_offset_ms": round(offset * 1e3, 3),
            "audio": bool(keep_audio and has_audio)}
