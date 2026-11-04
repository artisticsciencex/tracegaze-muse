"""README screenshots with dataset imagery masked (AEA footage is not redistributed)."""
import asyncio, sys
from pathlib import Path
from playwright.async_api import async_playwright
from capture_lib import start_server, open_page

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
MASK_CSS = """
video{visibility:hidden}
.frame img{visibility:hidden;height:110px}
.frame{background:#1a2027}
.frame::before{content:'dataset frame (not redistributed)';display:block;color:#9aa5b1;font-size:12px;padding:44px 8px 0;text-align:center;height:0}
"""
MASK_JS = """(() => { const v = document.getElementById('video'); const r = v.getBoundingClientRect();
  const d = document.createElement('div'); d.id = 'vmask';
  Object.assign(d.style, {position:'absolute', left:(r.left+scrollX)+'px', top:(r.top+scrollY)+'px', width:r.width+'px', height:r.height+'px',
    background:'#1a2027', color:'#c9d1da', display:'flex', flexDirection:'column', gap:'10px', alignItems:'center', justifyContent:'center', textAlign:'center',
    borderRadius:'8px', font:'18px system-ui', zIndex:3, padding:'24px'});
  d.innerHTML = 'Scene / SAM 3 / gaze video<br><span style="font-size:14px;color:#8b96a3">Aria Everyday Activities footage is not redistributed in this repository;<br>see the demo video</span>';
  v.parentElement.style.position = 'relative'; v.parentElement.appendChild(d);
  d.style.left = (v.offsetLeft)+'px'; d.style.top = (v.offsetTop)+'px'; })()"""

async def main():
    srv = start_server(8774)
    async with async_playwright() as pw:
        b, page = await open_page(pw, "http://127.0.0.1:8774/report/index.html?demo=1#f3_switching_with_edge_effects")
        await page.add_style_tag(content=MASK_CSS)
        await page.evaluate("document.querySelector('[data-obj=obj_0027]').click()")
        await page.evaluate(MASK_JS)
        await page.wait_for_timeout(500)
        await page.mouse.move(1900, 1070)
        await page.screenshot(path=OUT / "dashboard.png")
        await page.evaluate("""(() => { const h = [...document.querySelectorAll('#evidence h4')].find(x => x.textContent.startsWith('Measurements'));
            document.getElementById('col-right').scrollTop = h.offsetTop - 20; })()""")
        await page.wait_for_timeout(300)
        await page.locator('#col-right').screenshot(path=OUT / "evidence_panel.png")
        await page.locator('#col-left').screenshot(path=OUT / "findings.png")
        await b.close()
    srv.shutdown()

asyncio.run(main())
