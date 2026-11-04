"""Static screenshots of key dashboard states (visual QA and README images)."""
import asyncio, sys
from pathlib import Path
from playwright.async_api import async_playwright
from capture_lib import start_server, open_page

OUT = Path(sys.argv[1])

async def main():
    srv = start_server(8771)
    async with async_playwright() as pw:
        b, page = await open_page(pw, "http://127.0.0.1:8771/report/index.html?demo=1#none")
        await page.screenshot(path=OUT / "1_overview.png")
        await page.click("#theater"); await page.wait_for_timeout(400)
        await page.click("[data-src=sam]"); await page.wait_for_timeout(300)
        await page.evaluate("seek(16.6)"); await page.wait_for_timeout(900)
        await page.screenshot(path=OUT / "2_theater_sam.png")
        await page.click("[data-src=vis]"); await page.wait_for_timeout(900)
        await page.screenshot(path=OUT / "3_theater_gaze.png")
        await page.keyboard.press("Escape"); await page.wait_for_timeout(300)
        await page.click("[data-obj=obj_0027]"); await page.wait_for_timeout(300)
        await page.screenshot(path=OUT / "4_bars_focus.png")
        await page.click("[data-obj=obj_0027]")
        await page.click(".card[data-f=f3_switching_with_edge_effects]"); await page.wait_for_timeout(1200)
        await page.screenshot(path=OUT / "5_finding3.png")
        await page.click("[data-fr=fr_0332]"); await page.wait_for_timeout(900)
        await page.wait_for_timeout(400); await page.screenshot(path=OUT / "6_frame.png")
        await b.close()
    srv.shutdown()

asyncio.run(main())
