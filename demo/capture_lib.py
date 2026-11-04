"""Capture helpers for the demo: local report server, headless Chrome,
visible synthetic cursor, and a CDP screencast recorder (frames + timestamps).

Run with ~/.venvs/demo-capture/bin/python (Playwright); the report server is
the project's own range-capable handler, started in-process.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import threading
import time
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from tracegaze_analyst.support_report import _RangeHandler  # noqa: E402

EXAMPLE = ROOT / "out" / "aea_meal_v1"
W, H = 1920, 1080

CURSOR_JS = r"""
(() => {
  if (window.__cursor) return;
  const c = document.createElement('div');
  c.id = '__cursor';
  c.innerHTML = `<svg width="30" height="30" viewBox="0 0 24 24"><path d="M4 2 L4 20 L9 15.5 L12.5 22 L15.5 20.6 L12 14.2 L19 14.2 Z"
     fill="#fff" stroke="#111" stroke-width="1.4" stroke-linejoin="round"/></svg>`;
  Object.assign(c.style, {position:'fixed', left:'0px', top:'0px', zIndex:2147483647,
    pointerEvents:'none', transform:'translate(-3px,-2px)', filter:'drop-shadow(0 1px 2px rgba(0,0,0,.45))'});
  const ring = document.createElement('div');
  Object.assign(ring.style, {position:'fixed', width:'44px', height:'44px', borderRadius:'50%',
    border:'3px solid #2f6fd1', zIndex:2147483646, pointerEvents:'none', opacity:0,
    transform:'translate(-50%,-50%) scale(.4)', transition:'opacity .45s, transform .45s'});
  const add = () => { document.body.appendChild(ring); document.body.appendChild(c); };
  if (document.body) add(); else document.addEventListener('DOMContentLoaded', add);
  window.__cursor = c;
  document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
  document.addEventListener('mousedown', e => {
    ring.style.transition = 'none'; ring.style.left = e.clientX + 'px'; ring.style.top = e.clientY + 'px';
    ring.style.opacity = 1; ring.style.transform = 'translate(-50%,-50%) scale(.4)';
    requestAnimationFrame(() => { ring.style.transition = 'opacity .5s, transform .5s';
      ring.style.opacity = 0; ring.style.transform = 'translate(-50%,-50%) scale(1.3)'; });
  }, true);
})();
"""


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass            # browsers drop video range requests when sources switch


def start_server(port: int = 8770) -> ThreadingHTTPServer:
    handler = partial(_RangeHandler, directory=str(EXAMPLE))
    httpd = _QuietServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


async def open_page(pw, url: str, headless: bool = True):
    browser = await pw.chromium.launch(
        channel="chrome", headless=headless,
        args=["--autoplay-policy=no-user-gesture-required", "--hide-scrollbars",
              "--force-color-profile=srgb", "--disable-features=Translate"])
    ctx = await browser.new_context(viewport={"width": W, "height": H},
                                    device_scale_factor=1, color_scheme="light")
    await ctx.add_init_script(CURSOR_JS)
    page = await ctx.new_page()
    await page.goto(url, wait_until="networkidle")
    await page.wait_for_function("document.getElementById('video').readyState >= 2")
    return browser, page


class Cursor:
    """Deliberate, eased cursor motion and clicks."""

    def __init__(self, page, x=W * 0.55, y=H * 0.6):
        self.page, self.x, self.y = page, x, y

    async def park(self):
        await self.page.mouse.move(self.x, self.y)

    async def move(self, x, y, dur=0.7):
        n = max(8, int(dur * 60))
        x0, y0 = self.x, self.y
        for i in range(1, n + 1):
            t = i / n
            e = 0.5 - 0.5 * math.cos(math.pi * t)          # ease in-out
            await self.page.mouse.move(x0 + (x - x0) * e, y0 + (y - y0) * e)
            await asyncio.sleep(dur / n)
        self.x, self.y = x, y

    async def to(self, locator, dur=0.7, dx=0.5, dy=0.5):
        await locator.scroll_into_view_if_needed()
        box = await locator.bounding_box()
        await self.move(box["x"] + box["width"] * dx, box["y"] + box["height"] * dy, dur)

    async def click(self, locator, dur=0.7, pause=0.25, dx=0.5, dy=0.5):
        await self.to(locator, dur, dx, dy)
        await asyncio.sleep(pause)
        await self.page.mouse.down()
        await asyncio.sleep(0.08)
        await self.page.mouse.up()


class Recorder:
    """CDP screencast → JPEG frames on disk with wall-clock timestamps."""

    def __init__(self, page, out_dir: Path):
        self.page, self.dir = page, out_dir
        self.frames: list[tuple[float, str]] = []
        self.marks: list[tuple[str, float]] = []
        self.t0 = None
        out_dir.mkdir(parents=True, exist_ok=True)

    async def start(self):
        import base64
        self.cdp = await self.page.context.new_cdp_session(self.page)

        async def on_frame(ev):
            ts = ev["metadata"].get("timestamp") or time.time()
            if self.t0 is None:
                self.t0 = ts
            name = f"f{len(self.frames):06d}.jpg"
            (self.dir / name).write_bytes(base64.b64decode(ev["data"]))
            self.frames.append((ts - self.t0, name))
            try:
                await self.cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})
            except Exception:                                  # noqa: BLE001
                pass

        self.cdp.on("Page.screencastFrame", lambda ev: asyncio.ensure_future(on_frame(ev)))
        self.wall0 = time.time()
        await self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 93,
                                                     "maxWidth": W, "maxHeight": H,
                                                     "everyNthFrame": 1})

    def mark(self, name: str):
        self.marks.append((name, time.time() - self.wall0))
        print(f"  [{time.time() - self.wall0:7.2f}s] {name}", flush=True)

    async def stop(self):
        await asyncio.sleep(0.5)
        await self.cdp.send("Page.stopScreencast")
        await asyncio.sleep(0.3)
        # screencast timestamps and wall clock share an epoch; align marks to frames
        offset = (self.t0 or self.wall0) - self.wall0
        (self.dir / "frames.json").write_text(json.dumps({
            "frames": self.frames,
            "marks": [(n, round(t - offset, 3)) for n, t in self.marks]}, indent=0))


def encode(frame_dir: Path, out_mp4: Path, fps: int = 30) -> None:
    """Variable-timestamp screencast frames → constant-fps H.264 (yuv420p)."""
    import subprocess
    meta = json.loads((frame_dir / "frames.json").read_text())
    frames = meta["frames"]
    end = max([t for n, t in meta.get("marks", []) if n == "end"] + [frames[-1][0] + 1 / fps])
    lines = []
    for (t, name), nxt in zip(frames, frames[1:] + [[end, None]]):
        lines += [f"file '{name}'", f"duration {max(nxt[0] - t, 0.001):.4f}"]
    lines.append(f"file '{frames[-1][1]}'")
    (frame_dir / "concat.txt").write_text("\n".join(lines) + "\n")
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(frame_dir / "concat.txt"), "-vf", f"fps={fps},scale={W}:{H}",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_mp4)],
                   check=True)


def ffmpeg() -> str:
    """imageio-ffmpeg's bundled binary if importable, else ffmpeg on PATH,
    else the copy bundled in any ~/.venvs environment."""
    import glob
    import shutil
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    found = shutil.which("ffmpeg") or next(iter(sorted(glob.glob(str(
        Path.home() / ".venvs/*/lib/python3*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*")))), None)
    if not found:
        raise RuntimeError("ffmpeg not found: pip install imageio-ffmpeg")
    return found
