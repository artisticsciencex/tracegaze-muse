"""Architecture diagram (SVG) for the demo cards and the README."""

BOXES = [
    ("Project Aria", ["glasses: first-person", "video + eye tracking"], "input"),
    ("SAM 3 (Meta)", ["Tracks objects a", "researcher selected", "and reviewed"], "foundation"),
    ("TraceGaze", ["Gaze → object", "per sample; episodes,", "time per object"], "foundation"),
    ("Evidence package", ["Facts, key moments,", "12 frames, edge flags,", "source links"], "new"),
    ("Muse Spark 1.3", ["Meta Model API", "Typed findings:", "measured, visible,", "interpretation"], "new"),
    ("Checks + dashboard", ["Every reference and", "number verified;", "click to the moment"], "new"),
]


def architecture_svg(dark: bool = True) -> str:
    ink, sub, bg = ("#f2f4f7", "#c9d1da", "#0b0e12") if dark else ("#17202a", "#4a5562", "#ffffff")
    fills = {"input": ("none", "#8b96a3"), "foundation": ("none", "#8b96a3"),
             "new": ("#1f3a66" if dark else "#e6eefb", "#7aa7f0" if dark else "#2f6fd1")}
    bw, bh, gap, x0, y0 = 255, 230, 40, 15, 60
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1760" height="420" viewBox="0 0 1760 420" '
             f'font-family="Noto Sans, Helvetica, Arial, sans-serif">',
             f'<rect width="1760" height="420" fill="{bg}"/>']
    for i, (title, lines, kind) in enumerate(BOXES):
        x = x0 + i * (bw + gap)
        fill, stroke = fills[kind]
        dash = ' stroke-dasharray="6 5"' if kind == "input" else ""
        parts.append(f'<rect x="{x}" y="{y0}" width="{bw}" height="{bh}" rx="16" fill="{fill}" '
                     f'stroke="{stroke}" stroke-width="3"{dash}/>')
        parts.append(f'<text x="{x + bw / 2}" y="{y0 + 52}" text-anchor="middle" font-size="22" '
                     f'font-weight="700" fill="{ink}">{title}</text>')
        for k, line in enumerate(lines):
            parts.append(f'<text x="{x + bw / 2}" y="{y0 + 96 + k * 31}" text-anchor="middle" '
                         f'font-size="19" fill="{sub}">{line.replace("→", "&#8594;")}</text>')
        if i < len(BOXES) - 1:
            ax = x + bw + 6
            parts.append(f'<path d="M{ax} {y0 + bh / 2} h{gap - 16}" stroke="{sub}" stroke-width="3"/>'
                         f'<path d="M{ax + gap - 12} {y0 + bh / 2} l-12 -8 v16 z" fill="{sub}"/>')
    ly = y0 + bh + 70
    parts += [f'<rect x="{x0}" y="{ly - 20}" width="26" height="26" rx="5" fill="none" stroke="#8b96a3" stroke-width="3"/>',
              f'<text x="{x0 + 40}" y="{ly + 2}" font-size="22" fill="{sub}">Existing TraceGaze research pipeline (before the hackathon)</text>',
              f'<rect x="{x0 + 760}" y="{ly - 20}" width="26" height="26" rx="5" fill="{fills["new"][0]}" stroke="{fills["new"][1]}" stroke-width="3"/>',
              f'<text x="{x0 + 800}" y="{ly + 2}" font-size="22" fill="{sub}">Built during the hackathon</text>',
              "</svg>"]
    return "".join(parts)
