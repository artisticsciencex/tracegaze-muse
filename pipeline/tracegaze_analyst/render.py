"""Render a compact report for human verification."""

from __future__ import annotations

import html
from pathlib import Path


def render_report(packet: dict, response: dict, validation: dict,
                  out_path: str | Path) -> None:
    esc = html.escape
    claims = []
    for c in response.get("claims", []):
        refs = c.get("fact_ids", []) + c.get("episode_ids", []) + c.get("transition_ids", [])
        quantities = ", ".join(
            f"{q.get('value')} {q.get('unit')} [{q.get('fact_id')}]"
            for q in c.get("quantities", []))
        claims.append(
            f"<article><span class='kind'>{esc(c.get('claim_kind',''))}</span> "
            f"<strong>{esc(c.get('statement',''))}</strong>"
            f"<div class='refs'>Evidence: {esc(', '.join(refs) or 'none')}"
            f"{'<br>Checked quantities: ' + esc(quantities) if quantities else ''}</div></article>")
    problems = "".join(f"<li>{esc(p)}</li>" for p in validation.get("problems", []))
    limitations = "".join(f"<li>{esc(x)}</li>" for x in response.get("limitations", []))
    status = "Verified" if validation.get("ok") else "Needs review"
    css_class = "verified" if validation.get("ok") else "needs-review"
    doc = f"""<!doctype html><meta charset='utf-8'><title>TraceGaze analyst report</title>
<style>body{{font:16px system-ui;max-width:980px;margin:40px auto;padding:0 20px;color:#17212b}}h1,h2{{line-height:1.15}}.status{{padding:12px;border-radius:8px}}.verified{{background:#def7e5}}.needs-review{{background:#ffe8e8}}article{{border:1px solid #ccd5df;border-radius:8px;padding:14px;margin:10px 0}}.kind{{font:12px ui-monospace;background:#e7f2fa;padding:3px 6px;border-radius:4px}}.refs{{margin-top:8px;color:#52606d;font:13px ui-monospace}}</style>
<h1>TraceGaze evidence-grounded analysis</h1>
<p><b>Question:</b> {esc(packet['question'])}</p>
<div class='status {css_class}'><b>{status}</b> — {validation.get('claim_count',0)} claims checked.</div>
<h2>Answer</h2><p>{esc(response.get('answer',''))}</p>
<h2>Claims</h2>{''.join(claims) or '<p>No claims.</p>'}
<h2>Limitations</h2><ul>{limitations}</ul>
<h2>Validation problems</h2><ul>{problems or '<li>None</li>'}</ul>
"""
    Path(out_path).write_text(doc, encoding="utf-8")

