"""Record one continuous walkthrough of the real dashboard (1920x1080).

Shows the saved result of the real Muse run (muse_run/, muse-spark-1.3);
no request is sent. Marks for clips A-E are written next to the frames.
"""

import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

from capture_lib import ROOT, Cursor, Recorder, encode, open_page, start_server

WORK = ROOT / "demo_work" / "walkthrough_frames"
OUT = ROOT / "demo_exports" / "00_walkthrough_continuous.mp4"
MAIN = "f3_switching_with_edge_effects"


async def wait(s):
    await asyncio.sleep(s)


async def scroll_to(page, selector, column, block="start"):
    await page.evaluate(
        """([sel, col, block]) => { const el = document.querySelector(sel);
             const c = document.getElementById(col);
             const top = el.getBoundingClientRect().top - c.getBoundingClientRect().top + c.scrollTop;
             c.scrollTo({top: block === 'start' ? top - 12 : top - c.clientHeight / 2, behavior: 'smooth'}); }""",
        [selector, column, block])


async def main():
    srv = start_server(8773)
    async with async_playwright() as pw:
        browser, page = await open_page(pw, "http://127.0.0.1:8773/report/index.html?demo=1#none")
        v = "document.getElementById('video')"
        cur = Cursor(page, 1600, 640)
        await cur.park()
        await page.evaluate(f"seek(0); {v}.pause()")
        await wait(1.0)
        rec = Recorder(page, WORK)
        await rec.start()
        await wait(0.6)

        # overview
        rec.mark("overview")
        await wait(3.5)

        # A — first-person experience (scene video, enlarged)
        rec.mark("A_start")
        await cur.click(page.locator("[data-src=scene]"), 0.8)
        await wait(0.5)
        await cur.click(page.locator("#theater"), 0.6)
        await wait(0.6)
        await cur.move(1720, 560, 0.6)
        await page.evaluate(f"{v}.play()")
        await wait(12.5)
        rec.mark("A_end")

        # B — SAM 3 objects, then gaze on objects (same moment)
        rec.mark("B_start")
        await page.evaluate(f"{v}.pause()")
        await wait(0.6)
        await cur.click(page.locator("[data-src=sam]"), 0.9)
        await wait(2.5)
        await page.evaluate(f"{v}.play()")
        await cur.move(1720, 560, 0.6)
        await wait(4.5)
        await cur.click(page.locator("[data-src=vis]"), 0.8)
        await cur.move(1720, 560, 0.6)
        await wait(9.0)
        await page.evaluate(f"{v}.pause()")
        await wait(1.2)
        await cur.click(page.locator("#theater"), 0.7)
        await wait(0.8)
        rec.mark("B_end")

        # C — temporal pattern: one object across the whole meal
        rec.mark("C_start")
        await cur.move(560 + 260, 800, 0.9)
        await cur.move(1250, 800, 1.6)
        await wait(0.6)
        await cur.click(page.locator("[data-obj=obj_0027]"), 0.9, dx=0.25)
        await wait(1.0)
        await cur.move(640, 800, 1.0)
        await cur.move(1270, 800, 2.6)
        await wait(4.0)
        await cur.click(page.locator("[data-obj=obj_0027]"), 0.9, dx=0.25)
        await wait(1.5)
        rec.mark("C_end")

        # D — the question and Muse's reading of it
        rec.mark("D_start")
        await cur.to(page.locator("#question"), 1.0, dx=0.7)
        await wait(4.0)
        await cur.to(page.locator("#summary"), 0.8, dx=0.85, dy=0.3)
        await wait(9.0)
        await scroll_to(page, f".card[data-f={MAIN}]", "col-left")
        await wait(1.2)
        await cur.to(page.locator(f".card[data-f={MAIN}]"), 0.8, dx=0.85, dy=0.25)
        await wait(8.0)
        rec.mark("D_end")

        # E — open the moment behind the insight
        rec.mark("E_start")
        await cur.click(page.locator(f".card[data-f={MAIN}] h3"), 0.6)
        await wait(3.5)
        await cur.click(page.locator("[data-iv=iv_03]"), 1.0)
        await cur.move(1100, 470, 0.7)
        await wait(3.6)
        await cur.click(page.locator("[data-fr=fr_0332]"), 1.0)
        await wait(5.0)
        await cur.click(page.locator("#lbimg"), 0.6)
        await wait(0.8)
        await scroll_to(page, "#evidence table", "col-right")
        await cur.move(1640, 520, 0.8)
        await wait(4.5)
        await page.evaluate("""(() => { const h = [...document.querySelectorAll('#evidence h4')]
                               .find(x => x.textContent.startsWith('Possible')); h.id = 'h_interp'; })()""")
        await scroll_to(page, "#h_interp", "col-right")
        await wait(1.0)
        await cur.to(page.locator("#evidence .interp"), 0.8, dx=0.8)
        await wait(6.0)
        await page.evaluate("""(() => { const h = [...document.querySelectorAll('#evidence h4')]
                               .find(x => x.textContent.startsWith('Support')); h.id = 'h_support'; })()""")
        await scroll_to(page, "#h_support", "col-right")
        await wait(6.5)
        rec.mark("E_end")
        await wait(1.0)
        rec.mark("end")
        await rec.stop()
        await browser.close()
    srv.shutdown()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encode(WORK, OUT)
    print(f"walkthrough -> {OUT}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
