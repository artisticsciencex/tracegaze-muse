"""Narration script + SRT for the rough cut, aligned to rough_cut_timeline.json.

Cues are (segment, start, end) relative to the segment, so re-timing the cut
keeps the subtitles aligned.
"""

import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1] / "demo_exports"

CUES = [
    ("s01", 0.5, 6.8, "We want AI that helps older adults in everyday life, even when asking for help is hard."),
    ("s02", 0.3, 7.7, "TraceGaze and Muse are an understanding layer: they turn what someone sees and does into evidence AI can reason about."),
    ("A_first_person_experience", 0.3, 7.5, "This is a first-person recording from Project Aria glasses, with eye tracking: a shared breakfast at home."),
    ("A_first_person_experience", 7.6, 14.7, "It comes from a public Meta dataset. It is an engineering example, not yet an older adult."),
    ("B_gaze_to_object_context", 0.3, 9.3, "SAM 3 tracks each object a researcher selected: the plates, the table, the partner, the phones."),
    ("B_gaze_to_object_context", 9.6, 16.8, "TraceGaze then matches every gaze sample to the object it lands on,"),
    ("B_gaze_to_object_context", 16.9, 24.6, "so we know not just where the eyes pointed, but what they were on."),
    ("C_temporal_patterns", 0.3, 6.4, "That becomes a timeline of gaze episodes, object by object."),
    ("C_temporal_patterns", 6.6, 14.4, "Select the wearer's own plate: fourteen point nine seconds, almost forty percent of gaze time,"),
    ("C_temporal_patterns", 14.5, 21.5, "spread over fifteen separate episodes across the meal."),
    ("D_muse_insight", 0.3, 7.8, "Now a researcher asks: how did this unfold, which moments deserve a closer look, and what could they mean for assistance?"),
    ("D_muse_insight", 8.0, 15.6, "Muse Spark reads the TraceGaze measurements and twelve key frames. This is the saved result of a real request."),
    ("D_muse_insight", 15.8, 26.7, "Its findings keep measurement, observation and interpretation apart. Here it flags the busiest switching moments, and cautions that they are not evidence of confusion."),
    ("E_explore_the_moment", 0.3, 7.3, "One click opens the moment behind that insight. The video jumps to the interval and replays it,"),
    ("E_explore_the_moment", 7.5, 12.4, "and here the gaze really does move to the partner."),
    ("E_explore_the_moment", 12.6, 21.2, "We see the exact frame Muse described, and every number it quotes, checked against TraceGaze."),
    ("E_explore_the_moment", 21.4, 30.7, "The interpretation comes with alternatives, and the review marks it only partially supported: two frames show genuine glances."),
    ("s08", 0.4, 7.4, "During the hackathon, we built this Muse layer on top of our existing TraceGaze pipeline:"),
    ("s08", 7.5, 14.7, "the evidence package, the Muse Spark integration, automatic checks on every reference and number, and this dashboard."),
    ("s09", 0.4, 7.7, "Today, that means retrospective understanding, with every insight traceable to the moment behind it."),
    ("s10", 0.4, 8.7, "Next, the same layer could let assistance notice the moment as it happens, so help arrives before anyone has to ask."),
    ("s11", 0.4, 5.6, "TraceGaze and Muse. Thank you."),
]

SECTIONS = {
    "s01": "Opening: the ambition", "s02": "Opening: the understanding layer",
    "A_first_person_experience": "A · The first-person experience",
    "B_gaze_to_object_context": "B · Turning gaze into object context",
    "C_temporal_patterns": "C · Making temporal patterns visible",
    "D_muse_insight": "D · Muse produces an insight",
    "E_explore_the_moment": "E · Explore the moment behind the insight",
    "s08": "What we built during the hackathon", "s09": "Close: today", "s10": "Close: next",
    "s11": "End card",
}


def ts(t: float, sep=","):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int(round((s % 1) * 1000)):03d}"


def main():
    timeline = {x["segment"]: x for x in json.loads((EXP / "rough_cut_timeline.json").read_text())}
    srt, md, last = [], ["# TraceGaze + Muse: narration script (rough cut)", "",
                         f"Running time {timeline['s11']['end']:.1f} s. Read at a calm pace "
                         "(about 130 words per minute); timecodes match the rough cut and the SRT.", ""], None
    for i, (seg, a, b, text) in enumerate(CUES, 1):
        base, end = timeline[seg]["start"], timeline[seg]["end"]
        t0, t1 = base + a, min(base + b, end - 0.05)
        srt += [str(i), f"{ts(t0)} --> {ts(t1)}", text, ""]
        if seg != last:
            md += ["", f"## {SECTIONS[seg]}  ({ts(timeline[seg]['start'], '.')[3:8]}–{ts(end, '.')[3:8]})", ""]
            last = seg
        md.append(f"- `{ts(t0, '.')[3:10]}` {text}")
    words = sum(len(c[3].split()) for c in CUES)
    md += ["", f"_{words} words._", "",
           "## Accuracy notes for the speaker", "",
           "- The recording is an engineering example from the Aria Everyday Activities dataset; "
           "participant age is unknown. Do not describe the wearer as an older adult.",
           "- The Muse result shown is the saved output of a real `muse-spark-1.3` request "
           "(response chatcmpl-01a1a7ba…), not a live call.",
           "- \"The review\" refers to `muse_run/manual_review.json`, written by the coding "
           "assistant and still to be confirmed by a researcher.",
           "- Proactive assistance is the future direction; the prototype is retrospective."]
    (EXP / "tracegaze_muse_narration.srt").write_text("\n".join(srt), encoding="utf-8")
    (EXP / "narration_script.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{len(CUES)} cues, {words} words")


if __name__ == "__main__":
    main()
