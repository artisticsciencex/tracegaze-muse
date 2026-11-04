"""Cut clips A-E from the continuous walkthrough and assemble the rough cut.

    ~/.venvs/demo-capture/bin/python make_cut.py [--repo-url URL]

Inputs : demo_exports/00_walkthrough_continuous.mp4 + demo_work/walkthrough_frames/frames.json
Outputs: demo_exports/clips/*.mp4 (captioned), demo_exports/clips_clean/*.mp4,
         demo_exports/tracegaze_muse_rough_cut.mp4, cards and diagram PNG/SVG.
"""

from __future__ import annotations

import argparse
import html
import asyncio
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from capture_lib import ROOT, ffmpeg
from diagram import architecture_svg

EXP = ROOT / "demo_exports"
WORK = ROOT / "demo_work"
SRC = EXP / "00_walkthrough_continuous.mp4"
W, H = 1920, 1080
VW, VH = 1728, 972                     # recording inside the captioned frame
BAND = H - VH
BG = "#0b0e12"
BOLD = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
REG = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"

MARKS = dict(json.loads((WORK / "walkthrough_frames" / "frames.json").read_text())["marks"])

# zoom crops (16:9, source pixels): reading-heavy moments
LEFT = (0, 74, 1120, 630)
CENTER_LOW = (470, 450, 1120, 630)
RIGHT = (800, 74, 1120, 630)

CLIPS = [
    {"name": "A_first_person_experience", "step": "1 · What the wearer experienced",
     "segments": [(MARKS["A_start"], MARKS["A_start"] + 15.0, None)],
     "captions": [(0, 15.0, "First-person video with eye tracking from Project Aria glasses: "
                            "a shared breakfast at home.")]},
    {"name": "B_gaze_to_object_context", "step": "2 · Turning gaze into object context",
     "segments": [(MARKS["B_start"], MARKS["B_end"] - 0.5, None)],
     "captions": [(0, 9.4, "SAM 3 tracks each object a researcher selected: plates, table, "
                           "partner, phones."),
                  (9.4, 99, "TraceGaze matches every gaze sample to the object it lands on.")]},
    {"name": "C_temporal_patterns", "step": "3 · Making patterns over time visible",
     "segments": [(MARKS["C_start"], MARKS["C_end"], CENTER_LOW)],
     "captions": [(0, 5.4, "Each coloured block is one gaze episode on one object, across "
                           "the 37.75-second clip."),
                  (5.4, 99, "The wearer's own plate: 14.9 s, 39.5% of gaze time, in 15 "
                            "separate episodes across the meal.")]},
    {"name": "D_muse_insight", "step": "4 · Muse interprets the evidence",
     "segments": [(MARKS["D_start"], MARKS["D_end"], LEFT)],
     "captions": [(0, 5.2, "A researcher asks how the activity unfolded and what it "
                           "implies for designing assistance."),
                  (5.2, 15.8, "Muse Spark 1.3 reads TraceGaze measurements and 12 key frames "
                              "(saved result of a real request)."),
                  (15.8, 99, "Finding 3: the busiest switching moments are flagged for "
                             "caution, not read as confusion.")]},
    {"name": "E_explore_the_moment", "step": "5 · Inspect the moment behind the insight",
     "segments": [(96.5, 117.5, None), (119.0, 121.5, RIGHT), (125.5, 133.0, RIGHT)],
     "captions": [(0, 12.5, "One click opens the moment: the video jumps to the interval "
                            "and replays it."),
                  (12.5, 21.0, "The exact frame Muse saw, with the gaze sample and Muse's "
                               "own description."),
                  (21.0, 23.5, "Every number Muse quotes is checked against TraceGaze."),
                  (23.5, 99, "Interpretation stays separate from evidence, with "
                             "alternatives and a review verdict.")]},
]


def run(args):
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", *args], check=True)


def duration(path: Path) -> float:
    out = subprocess.run([ffmpeg(), "-i", str(path)], capture_output=True, text=True).stderr
    hms = out.split("Duration: ")[1].split(",")[0].split(":")
    return int(hms[0]) * 3600 + int(hms[1]) * 60 + float(hms[2])


def caption_png(step: str, text: str, path: Path) -> Path:
    img = Image.new("RGB", (W, BAND), BG)
    d = ImageDraw.Draw(img)
    d.text((96, 14), step.upper(), font=ImageFont.truetype(BOLD, 22), fill="#7aa7f0")
    d.text((96, 46), text, font=ImageFont.truetype(REG, 36), fill="#f2f4f7")
    img.save(path)
    return path


def cut_clip(spec: dict) -> tuple[Path, Path]:
    clean = EXP / "clips_clean" / f"{spec['name']}_clean.mp4"
    out = EXP / "clips" / f"{spec['name']}.mp4"
    clean.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, (a, b, crop) in enumerate(spec["segments"]):
        p = WORK / f"{spec['name']}_{i}.mp4"
        vf = (f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]},scale={W}:{H}:flags=lanczos"
              if crop else f"scale={W}:{H}")
        run(["-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", str(SRC), "-vf", vf + ",fps=30",
             "-c:v", "libx264", "-crf", "15", "-preset", "medium", "-pix_fmt", "yuv420p", str(p)])
        parts.append(p)
    lst = WORK / f"{spec['name']}.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    run(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(clean)])

    dur = duration(clean)
    inputs, chain, last = ["-i", str(clean)], [], "[base]"
    chain.append(f"[0:v]scale={VW}:{VH}:flags=lanczos,pad={W}:{H}:{(W - VW) // 2}:0:color={BG}[base]")
    for k, (t0, t1, text) in enumerate(spec["captions"]):
        png = caption_png(spec["step"], text, WORK / f"{spec['name']}_cap{k}.png")
        inputs += ["-i", str(png)]
        nxt = f"[v{k}]"
        chain.append(f"{last}[{k + 1}:v]overlay=0:{VH}:enable='between(t,{t0},{min(t1, dur)})'{nxt}")
        last = nxt
    chain.append(f"{last}fade=t=in:st=0:d=0.25,fade=t=out:st={dur - 0.25:.3f}:d=0.25[out]")
    run([*inputs, "-filter_complex", ";".join(chain), "-map", "[out]", "-r", "30",
         "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", str(out)])
    return clean, out


CARD_CSS = """
body{margin:0;width:1920px;height:1080px;background:#0b0e12;color:#f2f4f7;
 font-family:'Noto Sans',system-ui,sans-serif;display:flex;align-items:center;justify-content:center}
.wrap{width:1720px} .kick{color:#7aa7f0;font-weight:700;letter-spacing:.12em;text-transform:uppercase;font-size:26px;margin-bottom:28px}
h1{font-size:70px;line-height:1.12;margin:0 0 28px;font-weight:700} p{font-size:38px;line-height:1.45;color:#c9d1da;margin:0 0 18px}
.small{font-size:26px;color:#8b96a3} .accent{color:#7aa7f0} ul{font-size:34px;line-height:1.5;color:#c9d1da;margin:18px 0 0;padding-left:34px}
"""


def cards(repo_url: str) -> dict[str, str]:
    return {
        "card_open1": """<div class=wrap><div class=kick>The ambition</div>
          <h1>AI that helps older adults in everyday life,<br>even when asking for help is hard.</h1>
          <p>Many moments where support matters are hard to recognise, or to put into words.</p></div>""",
        "card_open2": """<div class=wrap><div class=kick>TraceGaze + Muse</div>
          <h1>An understanding layer:<br><span class=accent>what someone saw, did, and might need.</span></h1>
          <p>First-person video and gaze become object-grounded evidence.<br>Muse turns that evidence into interpretations and candidate support opportunities.</p>
          <p class=small>Meta models: Muse Spark 1.3 (Meta Model API) · SAM 3 · recorded with Project Aria glasses</p></div>""",
        "card_close1": """<div class=wrap><div class=kick>Today</div>
          <h1>Retrospective understanding,<br>traceable to the moment behind it.</h1>
          <p>Every Muse finding links to its video interval, frames and measurements, and is checked automatically.</p></div>""",
        "card_close2": """<div class=wrap><div class=kick>Next</div>
          <h1>Assistance that notices the moment,<br><span class=accent>before anyone has to ask.</span></h1>
          <p>Proactive support for older adults is the future this understanding layer enables, and the next thing to test with older adults themselves.</p></div>""",
        "card_end": f"""<div class=wrap><h1>TraceGaze + Muse</h1>
          <p class=accent>{html.escape(repo_url)}</p>
          <p class=small>Demo data: Aria Everyday Activities Dataset, Meta Reality Labs-R. Engineering example: participant age unknown,
          not an older-adult recording. Findings shown are the saved output of a real Muse Spark 1.3 request.</p></div>""",
    }


async def render_cards(repo_url: str) -> dict[str, Path]:
    from playwright.async_api import async_playwright
    out = {}
    svg = architecture_svg()
    (EXP / "architecture.svg").write_text(svg, encoding="utf-8")
    pages = cards(repo_url)
    pages["card_built"] = f"""<div style="width:1760px"><div class=kick>Built during the hackathon</div>
      <h1 style="font-size:54px">Muse on top of our existing TraceGaze pipeline</h1>{svg}
      <ul><li>Evidence package: deterministic facts, rule-selected moments and frames; Muse never computes numbers</li>
      <li>Muse Spark 1.3 integration with typed findings that cite intervals, frames and facts</li>
      <li>Automatic checks of every reference and quoted number, and this inspectable dashboard</li></ul></div>"""
    async with async_playwright() as pw:
        b = await pw.chromium.launch(channel="chrome")
        page = await b.new_page(viewport={"width": W, "height": H})
        for name, body in pages.items():
            await page.set_content(f"<html><head><style>{CARD_CSS}</style></head><body>{body}</body></html>")
            await page.wait_for_timeout(150)
            out[name] = WORK / f"{name}.png"
            await page.screenshot(path=out[name])
        await page.set_viewport_size({"width": 1800, "height": 560})
        await page.set_content(f"<html><body style='margin:0;background:#fff'>{architecture_svg(dark=False)}</body></html>")
        await page.screenshot(path=EXP / "architecture.png")
        (EXP / "architecture_light.svg").write_text(architecture_svg(dark=False), encoding="utf-8")
        await b.close()
    return out


def card_video(png: Path, seconds: float, path: Path) -> Path:
    run(["-loop", "1", "-t", f"{seconds}", "-i", str(png), "-vf",
         f"fps=30,format=yuv420p,fade=t=in:st=0:d=0.5,fade=t=out:st={seconds - 0.5}:d=0.5",
         "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", str(path)])
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="github.com/<account>/tracegaze-muse")
    args = ap.parse_args()
    WORK.mkdir(exist_ok=True)
    clips = {}
    for spec in CLIPS:
        clean, out = cut_clip(spec)
        clips[spec["name"]] = out
        print(f"{out.name}: {duration(out):.1f} s")
    pngs = asyncio.run(render_cards(args.repo_url))
    seq = [card_video(pngs["card_open1"], 7.0, WORK / "s01.mp4"),
           card_video(pngs["card_open2"], 8.0, WORK / "s02.mp4"),
           *[clips[s["name"]] for s in CLIPS],
           card_video(pngs["card_built"], 15.0, WORK / "s08.mp4"),
           card_video(pngs["card_close1"], 8.0, WORK / "s09.mp4"),
           card_video(pngs["card_close2"], 9.0, WORK / "s10.mp4"),
           card_video(pngs["card_end"], 6.0, WORK / "s11.mp4")]
    lst = WORK / "roughcut.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in seq))
    rough = EXP / "tracegaze_muse_rough_cut.mp4"
    run(["-f", "concat", "-safe", "0", "-i", str(lst), "-f", "lavfi", "-i",
         "anullsrc=channel_layout=stereo:sample_rate=48000", "-shortest",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-r", "30",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(rough)])
    timeline, t = [], 0.0
    for p in seq:
        d = duration(p)
        timeline.append({"segment": p.stem, "start": round(t, 2), "end": round(t + d, 2)})
        t += d
    (EXP / "rough_cut_timeline.json").write_text(json.dumps(timeline, indent=1))
    print(f"rough cut: {duration(rough):.1f} s -> {rough}")


if __name__ == "__main__":
    main()
