"""Local report: Muse findings beside the TraceGaze evidence they cite.

One static page with its data embedded. Videos are symlinked from the
original artifacts (never copied or modified); frames are copied from the
package. Serve the run directory with `serve-report` (HTTP range support,
so video seeking works) or open the page directly from disk.
"""

from __future__ import annotations

import json
import shutil
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .support_package import first_frame_pts_s


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def build_report(example_dir: Path) -> Path:
    """example_dir holds package/ and muse_run/; writes report/index.html."""
    pkg_dir, run_dir = example_dir / "package", example_dir / "muse_run"
    out = example_dir / "report"
    package = _load(pkg_dir / "support_package.json")
    findings = _load(run_dir / "findings.json")
    validation = _load(run_dir / "validation.json")
    if package is None or findings is None or validation is None:
        raise ValueError("package/support_package.json, muse_run/findings.json and "
                         "muse_run/validation.json are required")
    out.mkdir(exist_ok=True)
    frames = out / "frames"
    frames.mkdir(exist_ok=True)
    for f in package["frames"]:
        shutil.copy2(pkg_dir / f["file"], frames / Path(f["file"]).name)
    media = out / "media"
    media.mkdir(exist_ok=True)
    refs = package["source_refs"]
    sources = [("scene", "scene.mp4", refs["scene_video"]),
               ("sam", "sam_objects.mp4",
                str(Path(refs["tracked_objects_dir"]) / "tracking_overlay.mp4")),
               ("vis", "tracegaze_visualization.mp4", refs["tracegaze_visualization_video"])]
    media_meta = {}
    for key, name, target in sources:
        link = media / name
        if link.is_symlink() or link.exists():
            link.unlink()
        if not Path(target).is_file():
            continue
        link.symlink_to(target)
        media_meta[key] = {"src": f"media/{name}",
                           "first_frame_pts_s": first_frame_pts_s(Path(target))}

    data = {
        "media": media_meta,
        "package": package,
        "findings": findings,
        "validation": validation,
        "run": _load(run_dir / "run_metadata.json"),
        "review": _load(run_dir / "manual_review.json"),
        "tamper": _load(run_dir / "tamper_check.json"),
    }
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    (out / "index.html").write_text(PAGE.replace("__DATA__", blob), encoding="utf-8")
    return out / "index.html"


class _RangeHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler plus single-range requests for video seeking."""

    def send_head(self):
        rng = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not rng or not rng.startswith("bytes=") or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        first, _, last = rng[6:].partition("-")
        try:
            start = int(first) if first else max(0, size - int(last))
            end = int(last) if first and last else size - 1
        except ValueError:
            return super().send_head()
        end = min(end, size - 1)
        if start > end:
            self.send_error(416)
            return None
        f = path.open("rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining > 0:
            chunk = source.read(min(1 << 16, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)
        self._remaining = None

    def log_message(self, fmt, *args):
        pass


def serve(example_dir: Path, port: int) -> None:
    handler = partial(_RangeHandler, directory=str(example_dir))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"report: http://127.0.0.1:{port}/report/index.html  (Ctrl-C to stop)",
              flush=True)
        httpd.serve_forever()


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TraceGaze Muse Review</title>
<style>
:root{
  --bg:#f6f7f9;--panel:#ffffff;--ink:#17202a;--muted:#5b6672;--line:#d9dee4;
  --accent:#2f6fd1;--accent-soft:#e6eefb;
  --ok:#1f7a45;--ok-soft:#e3f4ea;--warn:#9a5b00;--warn-soft:#fbf0dc;
  --bad:#b3261e;--bad-soft:#fbe5e3;
  --measured:#2f6fd1;--visible:#7a4fc9;--interp:#b45f06;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#12161b;--panel:#1a2027;--ink:#e6eaef;--muted:#9aa5b1;--line:#2c343d;
  --accent:#7aa7f0;--accent-soft:#1f2c40;
  --ok:#6fcf97;--ok-soft:#173326;--warn:#f0b35a;--warn-soft:#3a2c14;
  --bad:#f28b82;--bad-soft:#3b1d1b;
  --measured:#7aa7f0;--visible:#b79cf0;--interp:#f0a35a;}}
:root[data-theme="dark"]{
  --bg:#12161b;--panel:#1a2027;--ink:#e6eaef;--muted:#9aa5b1;--line:#2c343d;
  --accent:#7aa7f0;--accent-soft:#1f2c40;
  --ok:#6fcf97;--ok-soft:#173326;--warn:#f0b35a;--warn-soft:#3a2c14;
  --bad:#f28b82;--bad-soft:#3b1d1b;
  --measured:#7aa7f0;--visible:#b79cf0;--interp:#f0a35a;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:20px 24px 8px;max-width:1500px;margin:0 auto}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:15px;margin:0 0 10px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
h3{font-size:16px;margin:0 0 6px}
.meta{color:var(--muted);font-size:13px}
.badge{display:inline-block;font-size:12px;padding:2px 8px;border-radius:999px;border:1px solid var(--line);margin:2px 4px 2px 0;white-space:nowrap}
.badge.wrap{white-space:normal}
.b-eng{background:var(--warn-soft);color:var(--warn);border-color:transparent}
.b-ok{background:var(--ok-soft);color:var(--ok);border-color:transparent}
.b-bad{background:var(--bad-soft);color:var(--bad);border-color:transparent}
.b-warn{background:var(--warn-soft);color:var(--warn);border-color:transparent}
.t-measured_pattern{color:var(--measured);border-color:var(--measured)}
.t-visible_observation{color:var(--visible);border-color:var(--visible)}
.t-possible_interpretation{color:var(--interp);border-color:var(--interp)}
main{max-width:1500px;margin:0 auto;padding:8px 24px 40px;display:grid;grid-template-columns:minmax(0,5fr) minmax(0,7fr);gap:20px}
main>section{min-width:0}
.scroll{overflow-x:auto}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
.question{font-size:17px;font-weight:600}
.card{border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0;cursor:pointer;background:var(--panel)}
.card:hover{border-color:var(--accent)}
.card.sel{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent-soft)}
.card p{margin:6px 0 0;font-size:14px}
video{width:100%;max-height:52vh;background:#000;border-radius:8px;display:block}
.vbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0}
button{font:inherit;font-size:13px;padding:4px 10px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--accent)}
button.on{background:var(--accent-soft);border-color:var(--accent)}
#timeline{width:100%;height:96px;display:block;cursor:pointer}
.legend{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:12px;color:var(--muted)}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{font-size:13px;padding:3px 9px;border-radius:6px;border:1px solid var(--line);cursor:pointer;background:var(--panel);color:var(--ink)}
.chip:hover{border-color:var(--accent)}
.frames{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.frame{border:1px solid var(--line);border-radius:8px;overflow:hidden;cursor:pointer;background:var(--panel)}
.frame img{width:100%;display:block}
.frame div{font-size:12px;padding:6px 8px;color:var(--muted)}
.frame.sel{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent-soft)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:5px 6px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;overflow-wrap:anywhere}
.sec{margin-top:14px}
.sec h4{margin:0 0 6px;font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.interp{border-left:3px solid var(--interp);padding:6px 10px;background:var(--warn-soft);border-radius:0 6px 6px 0}
.shows{border-left:3px solid var(--measured);padding:6px 10px;background:var(--accent-soft);border-radius:0 6px 6px 0}
ul{margin:4px 0;padding-left:20px}
.err{color:var(--bad)}
.small{font-size:12px;color:var(--muted)}
#lightbox{position:fixed;inset:0;background:#0b0e12;display:none;align-items:center;justify-content:center;z-index:10;padding:16px}
#lightbox.on{display:flex}
#lightbox figure{margin:0;max-width:min(94vw,1100px);color:#eee;display:flex;flex-direction:column;align-items:center}
#lightbox img{max-height:76vh;max-width:100%;width:auto;border-radius:8px}
#lightbox figcaption{font-size:15px;line-height:1.45;margin-top:12px;max-width:min(94vw,980px)}
main{grid-template-areas:"left center" "left right"}
#col-left{grid-area:left}#col-center{grid-area:center}#col-right{grid-area:right}
.bars{display:grid;gap:6px;margin-top:10px}
.bar{display:grid;grid-template-columns:180px 1fr 92px;gap:10px;align-items:center;font-size:13px;cursor:pointer;padding:2px 4px;border-radius:6px;border:1px solid transparent;background:none;color:var(--ink);text-align:left}
.bar:hover{border-color:var(--line)}
.bar.on{border-color:var(--accent);background:var(--accent-soft)}
.bar .track{height:14px;background:var(--bg);border-radius:4px;overflow:hidden}
.bar .fill{height:100%;border-radius:4px}
.bar .val{font-variant-numeric:tabular-nums;color:var(--muted);text-align:right}
.fid{font-family:ui-monospace,monospace;font-size:11px;color:var(--muted);font-weight:400}
.card.sel{border-left:5px solid var(--accent)}
#vpanel.theater{position:fixed;inset:0;z-index:5;margin:0;border-radius:0;background:#0b0e12;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:18px}
#vpanel.theater video{max-height:84vh;width:auto;max-width:96vw}
#vpanel.theater .vbar{color:#e6eaef}
#vpanel.theater #timeline,#vpanel.theater .bars,#vpanel.theater #barsTitle{display:none}
#vpanel.theater .legend{max-width:1100px;justify-content:center;color:#c9d1da}
.vtitle{font-weight:600}
body.demo{font-size:17px}
body.demo header{padding:14px 20px 4px;max-width:none}
body.demo h1{font-size:24px}
body.demo main{max-width:none;padding:8px 20px 0;grid-template-columns:440px minmax(0,1fr) 580px;grid-template-areas:"left center right";height:calc(100vh - 74px);gap:16px}
body.demo main>section{overflow-y:auto;padding-bottom:16px}
body.demo video{max-height:56vh;width:auto;max-width:100%;margin:0 auto}
body.demo .card p,body.demo table{font-size:15px}
body.demo .small{font-size:14px}
body.demo #context,body.demo #provpanel,body.demo .srccol,body.demo .rule,body.demo #checkline .meaning{display:none}
body.demo .question{font-size:19px}
@media (max-width:900px){main{grid-template-columns:1fr;grid-template-areas:"left" "center" "right";padding:8px 16px 32px}header{padding:16px}}
</style>
</head>
<body>
<header>
  <h1>TraceGaze + Muse — retrospective activity review</h1>
  <div class="meta" id="recline"></div>
</header>
<main>
  <section id="col-left">
    <div class="panel">
      <h2>Researcher question</h2>
      <div class="question" id="question"></div>
      <div class="sec small" id="context"></div>
    </div>
    <div class="panel">
      <h2>Muse summary</h2>
      <div id="summary"></div>
      <div class="small sec" id="checkline"></div>
    </div>
    <div class="panel">
      <h2>Findings</h2>
      <div id="cards"></div>
    </div>
    <div class="panel">
      <h2>Remaining questions</h2>
      <ul id="remaining"></ul>
      <div id="reviewgeneral"></div>
    </div>
  </section>
  <section id="col-center">
    <div class="panel" id="vpanel">
      <div class="vbar">
        <button data-src="scene">Scene video</button>
        <button data-src="sam">SAM 3 objects</button>
        <button data-src="vis">Gaze + objects</button>
        <button id="theater" title="Enlarge video (Esc to close)">&#x2922; Enlarge</button>
        <span class="small" id="clock">0.00 s</span>
        <span class="small" id="playing"></span>
      </div>
      <video id="video" controls preload="auto" muted playsinline></video>
      <svg id="timeline" role="img" aria-label="Gaze episodes by object over time; candidate intervals marked"></svg>
      <div class="legend" id="legend"></div>
      <h2 class="sec" id="barsTitle">Where gaze went</h2>
      <div class="bars" id="bars"></div>
    </div>
  </section>
  <section id="col-right">
    <div class="panel" id="evidence"><h2>Evidence</h2><p class="small">Select a finding.</p></div>
    <div class="panel" id="provpanel">
      <h2>Checks and provenance</h2>
      <div id="prov" class="small"></div>
    </div>
  </section>
</main>
<div id="lightbox" role="dialog" aria-modal="true"><figure><img id="lbimg" alt=""><figcaption id="lbcap"></figcaption></figure></div>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const DEMO = new URLSearchParams(location.search).has('demo');
if (DEMO) document.body.classList.add('demo');
const P = D.package, F = D.findings, V = D.validation, R = D.review || null;
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const objName = Object.fromEntries(P.objects.map(o => [o.object_id, o.name]));
const objColor = Object.fromEntries(P.objects.map(o => [o.object_id, o.color]));
const facts = Object.fromEntries(P.facts.map(f => [f.fact_id, f]));
const ivs = Object.fromEntries(P.intervals.map(i => [i.interval_id, i]));
const frames = Object.fromEntries(P.frames.map(f => [f.frame_id, f]));
const eps = Object.fromEntries(P.episodes.map(e => [e.episode_id, e]));
const vres = Object.fromEntries(V.findings.map(r => [r.finding_id, r]));
const TYPE = {measured_pattern:'Measured pattern', visible_observation:'Visible observation', possible_interpretation:'Possible interpretation'};
const SUPPORT = {candidate_for_investigation:'Candidate for investigation', not_supported_by_evidence:'Not supported by this evidence', not_applicable:'Not applicable'};
const DUR = P.recording.duration_s;
const video = $('video');
const MEDIA = D.media || {};
let OFF = 0;                       // player time = clip time + first-frame PTS
let focusObj = null;
const fnum = Object.fromEntries(F.findings.map((f, i) => [f.finding_id, i + 1]));
const fname = fid => `Finding ${fnum[fid]}`;
const clipT = () => (video.currentTime || 0) - OFF;
let stopAt = null, selected = null;

// header & question
const ctx = P.context || {};
const pc = ctx.participant_context || {};
$('recline').innerHTML =
  `<span class="badge b-eng">${esc(ctx.example_label || 'engineering example')}</span>` +
  `<span class="mono">${esc(P.recording.recording_id)}</span> · ${esc(P.recording.source_system)} · ${DUR} s · ` +
  `participant age: ${esc(pc.age ?? 'unknown')} · older adult: ${esc(pc.older_adult ?? 'unknown')}`;
$('question').textContent = P.question;
const tc = ctx.task_context || {};
$('context').innerHTML = Object.entries(tc).map(([k,v]) => `<div><b>${esc(k.replaceAll('_',' '))}:</b> ${esc(v)}</div>`).join('');

$('summary').innerHTML = `<p>${esc(F.answer_summary)}</p>`;
const s = V.summary;
$('checkline').innerHTML = `&#10003; Automatic checks: ${s.references_verified}/${s.findings} findings have verified references and numbers` +
  (V.global_problems.length ? ` · <span class="err">${V.global_problems.length} global problem(s)</span>` : '') +
  `. <span class="meaning">${esc(V.meaning)}</span>`;

// findings list
function reviewOf(fid){ return R && R.findings ? R.findings[fid] : null; }
function cardHTML(f){
  const v = vres[f.finding_id] || {status:'needs_review', errors:['not validated']};
  const rv = reviewOf(f.finding_id);
  const rvBadge = rv ? `<span class="badge ${rv.verdict==='supported'?'b-ok':(rv.verdict==='unsupported'?'b-bad':'b-warn')}">manual review: ${esc(rv.verdict)}</span>` : '';
  return `<div class="card" data-f="${esc(f.finding_id)}" tabindex="0">
    <div><span class="badge t-${esc(f.evidence_type)}">${esc(TYPE[f.evidence_type]||f.evidence_type)}</span>
    <span class="badge ${v.status==='references_verified'?'b-ok':'b-bad'}">${v.status==='references_verified'?'references verified':'needs review'}</span>${rvBadge}</div>
    <h3>${esc(fname(f.finding_id))} · ${esc(f.title)} ${DEMO ? '' : `<span class="fid">${esc(f.finding_id)}</span>`}</h3>
    <p>${esc(f.statement)}</p>
    <p class="small">${(f.interval_ids||[]).map(i => ivs[i] ? `${i} (${ivs[i].start_s}–${ivs[i].end_s} s)` : i).join(' · ')}</p></div>`;
}
$('cards').innerHTML = F.findings.map(cardHTML).join('');
document.querySelectorAll('.card').forEach(c => {
  c.addEventListener('click', () => select(c.dataset.f));
  c.addEventListener('keydown', e => { if (e.key === 'Enter') select(c.dataset.f); });
});
$('remaining').innerHTML = (F.remaining_questions||[]).map(q => `<li>${esc(q)}</li>`).join('');
if (R && R.general) $('reviewgeneral').innerHTML = `<div class="sec"><h4>Manual review notes</h4><ul>${R.general.map(x=>`<li>${esc(x)}</li>`).join('')}</ul><div class="small">${esc(R.reviewer||'')} · ${esc(R.date||'')}</div></div>`;

// video
let current = null;
function setSource(which){
  if (!MEDIA[which]) return;
  const t = video.src ? clipT() : 0, wasPlaying = !video.paused;
  current = which; OFF = MEDIA[which].first_frame_pts_s || 0;
  video.src = MEDIA[which].src;
  video.addEventListener('loadedmetadata', () => { video.currentTime = t + OFF; if (wasPlaying) video.play(); }, {once:true});
  document.querySelectorAll('[data-src]').forEach(b => b.classList.toggle('on', b.dataset.src === which));
}
document.querySelectorAll('[data-src]').forEach(b => { if (!MEDIA[b.dataset.src]) b.style.display = 'none'; b.onclick = () => setSource(b.dataset.src); });
setSource(MEDIA.vis ? 'vis' : 'scene');
function setTheater(on){ $('vpanel').classList.toggle('theater', on); $('theater').innerHTML = on ? '&#x2715; Close' : '&#x2922; Enlarge'; drawTimeline(); }
$('theater').onclick = () => setTheater(!$('vpanel').classList.contains('theater'));
function seek(t, playUntil){
  stopAt = playUntil ?? null;
  const go = () => { video.currentTime = Math.max(0, Math.min(DUR, t)) + OFF; if (stopAt !== null) video.play(); };
  if (video.readyState >= 1) go(); else video.addEventListener('loadedmetadata', go, {once:true});
  $('playing').textContent = stopAt !== null ? `playing ${t.toFixed(2)}–${stopAt.toFixed(2)} s` : '';
}
video.addEventListener('timeupdate', () => {
  $('clock').textContent = clipT().toFixed(2) + ' s';
  drawHead();
  if (stopAt !== null && clipT() >= stopAt){ video.pause(); stopAt = null; $('playing').textContent=''; }
});

// timeline
const svg = $('timeline'); const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent){ const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent||svg).appendChild(e); return e; }
let head;
function drawTimeline(){
  svg.innerHTML = '';
  const W = svg.clientWidth || 800, X = t => 8 + (W-16) * t / DUR;
  svg.setAttribute('viewBox', `0 0 ${W} 96`);
  const sel = selected ? new Set(F.findings.find(f=>f.finding_id===selected).interval_ids) : new Set();
  for (const iv of P.intervals){
    const on = sel.has(iv.interval_id);
    el('rect', {x:X(iv.start_s), y:4, width:Math.max(2,X(iv.end_s)-X(iv.start_s)), height:60, rx:3,
      fill: on ? 'var(--accent-soft)' : 'transparent', stroke: on ? 'var(--accent)' : 'var(--line)', 'stroke-width': on?2:1});
    const t = el('text', {x:X(iv.start_s)+3, y:16, 'font-size':11, fill:'var(--muted)'}); t.textContent = iv.interval_id;
  }
  for (const e of P.episodes){
    const r = el('rect', {x:X(e.start_s), y:24, width:Math.max(1,X(e.start_s+e.duration_s)-X(e.start_s)), height:28,
      fill: e.target_id==='__no_tracked_object__' ? 'var(--line)' : (objColor[e.target_id]||'#888'),
      opacity: focusObj && e.target_id !== focusObj ? 0.12 : 1});
    const tt = el('title', {}, r); tt.textContent = `${e.episode_id} · ${objName[e.target_id]||e.target_id} · ${e.start_s}s +${e.duration_s}s · ${e.sample_count} sample(s), ${e.boundary_near_samples} boundary-near`;
    if (e.boundary_near_samples) el('rect', {x:X(e.start_s), y:54, width:Math.max(1,X(e.start_s+e.duration_s)-X(e.start_s)), height:4, fill:'var(--warn)', opacity: e.boundary_near_samples/e.sample_count});
  }
  for (let t = 0; t <= DUR; t += 5){
    el('line', {x1:X(t), x2:X(t), y1:64, y2:70, stroke:'var(--muted)'});
    const tx = el('text', {x:X(t), y:84, 'font-size':11, 'text-anchor':'middle', fill:'var(--muted)'}); tx.textContent = t + ' s';
  }
  head = el('line', {x1:0, x2:0, y1:2, y2:66, stroke:'var(--bad)', 'stroke-width':2});
  drawHead();
}
function drawHead(){ if (!head) return; const W = svg.clientWidth || 800; const x = 8 + (W-16) * Math.max(0, clipT()) / DUR; head.setAttribute('x1', x); head.setAttribute('x2', x); }
svg.addEventListener('click', ev => { const r = svg.getBoundingClientRect(); seek(DUR * (ev.clientX - r.left - 8) / (r.width - 16)); });
window.addEventListener('resize', drawTimeline);
const used = P.objects.filter(o => P.episodes.some(e => e.target_id === o.object_id));
$('legend').innerHTML = used.map(o => `<span><span class="sw" style="background:${o.object_id==='__no_tracked_object__'?'var(--line)':esc(o.color)}"></span>${esc(o.name)}</span>`).join('') +
  `<span><span class="sw" style="background:var(--warn)"></span>share of boundary-near samples (another target within 25 px)</span>`;

// where gaze went: deterministic per-object totals from the package facts
const totals = P.objects.map(o => ({id:o.object_id, name:o.name, color:o.object_id==='__no_tracked_object__'?'var(--line)':o.color,
  t:(facts[`object.${o.object_id}.total_attention_s`]||{}).value||0, share:(facts[`object.${o.object_id}.share_of_supported_time`]||{}).value||0,
  n:(facts[`object.${o.object_id}.episode_count`]||{}).value||0})).filter(o => o.t > 0).sort((a,b) => b.t - a.t).slice(0, 7);
const tmax = Math.max(...totals.map(o => o.t));
function drawBars(){
  $('bars').innerHTML = totals.map(o => `<button class="bar ${focusObj===o.id?'on':''}" data-obj="${esc(o.id)}" title="${o.n} episodes">
    <span>${esc(o.name)}</span><span class="track"><span class="fill" style="display:block;width:${(100*o.t/tmax).toFixed(1)}%;background:${esc(o.color)}"></span></span>
    <span class="val">${o.t} s · ${(o.share*100).toFixed(1)}%</span></button>`).join('');
  $('bars').querySelectorAll('[data-obj]').forEach(b => b.onclick = () => { focusObj = focusObj === b.dataset.obj ? null : b.dataset.obj; drawBars(); drawTimeline(); });
}
drawBars();

// evidence panel
function fmt(v){ return typeof v === 'number' ? String(v) : esc(v); }
function select(fid){
  selected = fid;
  document.querySelectorAll('.card').forEach(c => c.classList.toggle('sel', c.dataset.f === fid));
  const f = F.findings.find(x => x.finding_id === fid), v = vres[fid] || {errors:[],warnings:[],quantity_checks:[]};
  const rv = reviewOf(fid);
  const obs = Object.fromEntries((f.visible_observations||[]).map(o => [o.frame_id, o.observation]));
  const ivChips = (f.interval_ids||[]).map(i => ivs[i] ? `<button class="chip" data-iv="${i}">▶ ${i} · ${ivs[i].start_s}–${ivs[i].end_s} s · ${esc(ivs[i].kind)}</button>` : `<span class="chip err">${esc(i)} (unknown)</span>`).join('');
  const ivRules = (f.interval_ids||[]).filter(i=>ivs[i]).map(i => `<div class="small rule"><b>${i}</b>: ${esc(ivs[i].selection_rule)}</div>`).join('');
  const frameCards = (f.frame_ids||[]).map(id => { const fr = frames[id]; if (!fr) return `<div class="frame err">${esc(id)} (unknown)</div>`;
    return `<div class="frame" data-fr="${id}"><img src="frames/${esc(fr.file.split('/').pop())}" alt="Scene frame ${id} at ${fr.time_s} s"><div><b>${id}</b> · ${fr.time_s} s<br>gaze → ${esc(fr.assigned_target_name)}${fr.alternatives_within_px.length?`<br>within 25 px: ${fr.alternatives_within_px.map(a=>esc(objName[a]||a)).join(', ')}`:''}</div></div>`; }).join('');
  const qrows = (v.quantity_checks||[]).map(q => { const fc = facts[q.fact_id]||{};
    return `<tr><td class="mono">${esc(q.fact_id)}</td><td>${esc(fc.metric||'')}</td><td class="num">${fmt(q.deterministic)} ${esc(q.unit)}</td><td class="num">${fmt(q.claimed)} ${esc(q.claimed_unit)}</td><td>${q.match?'<span class="badge b-ok">match</span>':'<span class="badge b-bad">mismatch</span>'}</td><td class="mono small srccol">${esc(fc.source_ref||'')}</td></tr>`; }).join('');
  const otherFacts = (f.fact_ids||[]).filter(id => !(v.quantity_checks||[]).some(q=>q.fact_id===id)).map(id => { const fc = facts[id];
    return fc ? `<tr><td class="mono">${esc(id)}</td><td>${esc(fc.metric)}</td><td class="num">${fmt(fc.value)} ${esc(fc.unit)}</td><td class="small">cited, no number copied</td><td></td><td class="mono small srccol">${esc(fc.source_ref||'')}</td></tr>` : `<tr><td class="mono err">${esc(id)}</td><td colspan=5 class="err">unknown fact</td></tr>`; }).join('');
  const epRows = (f.episode_ids||[]).map(id => { const e = eps[id]; return e ? `<tr data-ep="${id}" style="cursor:pointer"><td class="mono">${id}</td><td><span class="sw" style="background:${e.target_id==='__no_tracked_object__'?'var(--line)':esc(objColor[e.target_id]||'#888')}"></span>${esc(objName[e.target_id]||e.target_id)}</td><td class="num">${e.start_s} s</td><td class="num">${e.duration_s} s</td><td class="num">${e.sample_count}</td><td class="num">${e.boundary_near_samples}</td></tr>` : `<tr><td class="err" colspan=6>${esc(id)} unknown</td></tr>`; }).join('');
  const so = f.support_opportunity || {};
  const list = a => (a&&a.length) ? `<ul>${a.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>` : '<p class="small">None given.</p>';
  $('evidence').innerHTML = `<h2>Evidence · ${esc(fname(fid))}</h2>
    <h3>${esc(f.title)}</h3>
    <div><span class="badge t-${esc(f.evidence_type)}">${esc(TYPE[f.evidence_type]||f.evidence_type)}</span><span class="badge">uncertainty: ${esc(f.uncertainty)}</span>
      <span class="badge wrap">objects: ${(f.object_ids||[]).map(o=>esc(objName[o]||o)).join(', ')||'—'}</span></div>
    <div class="sec"><h4>What the evidence shows</h4><div class="shows">${esc(f.statement)}</div></div>
    <div class="sec"><h4>Intervals</h4><div class="chips">${ivChips||'<span class="small">none cited</span>'}</div>${ivRules}</div>
    <div class="sec"><h4>Frames</h4><div class="frames">${frameCards||'<span class="small">none cited</span>'}</div></div>
    ${(f.visible_observations||[]).length ? `<div class="sec"><h4>Visible observations (Muse)</h4><ul>${f.visible_observations.map(o=>`<li><b class="mono">${esc(o.frame_id)}</b>: ${esc(o.observation)}</li>`).join('')}</ul></div>` : ''}
    <div class="sec"><h4>Measurements</h4>${(qrows||otherFacts) ? `<div class="scroll"><table><tr><th>fact</th><th>metric</th><th>deterministic</th><th>cited by Muse</th><th>check</th><th class="srccol">source</th></tr>${qrows}${otherFacts}</table></div>` : '<p class="small">No facts cited.</p>'}</div>
    ${epRows ? `<div class="sec"><h4>Episodes</h4><div class="scroll"><table><tr><th>episode</th><th>target</th><th>start</th><th>duration</th><th>samples</th><th>boundary-near</th></tr>${epRows}</table></div></div>` : ''}
    ${f.interpretation ? `<div class="sec"><h4>Possible interpretation</h4><div class="interp">${esc(f.interpretation)}</div></div>` : ''}
    <div class="sec"><h4>Alternative explanations</h4>${list(f.alternative_explanations)}</div>
    <div class="sec"><h4>Limitations</h4>${list(f.limitations)}</div>
    <div class="sec"><h4>Support opportunity</h4><div><span class="badge ${so.status==='candidate_for_investigation'?'b-warn':''}">${esc(SUPPORT[so.status]||so.status)}</span></div>
      ${so.description?`<p>${esc(so.description)}</p>`:''}${so.evidence_needed?`<p class="small"><b>Evidence needed:</b> ${esc(so.evidence_needed)}</p>`:''}</div>
    <div class="sec"><h4>Automatic checks</h4>${(v.errors||[]).length ? `<ul class="err">${v.errors.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>` : '<p class="small">All references exist; copied numbers match; no construct or personal-attribute words in evidence text.</p>'}
      ${(v.warnings||[]).length ? `<ul class="small">${v.warnings.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`:''}</div>
    ${rv ? `<div class="sec"><h4>Manual review</h4><p><span class="badge ${rv.verdict==='supported'?'b-ok':(rv.verdict==='unsupported'?'b-bad':'b-warn')}">${esc(rv.verdict)}</span> ${esc(rv.notes)}</p></div>` : ''}`;
  $('evidence').querySelectorAll('[data-iv]').forEach(b => b.onclick = () => { const iv = ivs[b.dataset.iv]; seek(iv.start_s, iv.end_s); });
  $('evidence').querySelectorAll('[data-fr]').forEach(c => c.onclick = () => openFrame(c.dataset.fr, obs[c.dataset.fr]));
  $('evidence').querySelectorAll('[data-ep]').forEach(r => r.onclick = () => { const e = eps[r.dataset.ep]; seek(e.start_s, e.start_s + Math.max(e.duration_s, 0.5)); });
  drawTimeline();
  $('col-right').scrollTop = 0;
  const first = (f.interval_ids||[]).find(i => ivs[i]); if (first) seek(ivs[first].start_s);
  history.replaceState(null, '', '#' + fid);
}
function openFrame(id, observation){
  const fr = frames[id];
  document.querySelectorAll('.frame').forEach(c => c.classList.toggle('sel', c.dataset.fr === id));
  video.pause(); seek((fr.frame_time_s ?? fr.time_s) + 0.01);
  $('lbimg').src = 'frames/' + fr.file.split('/').pop(); $('lbimg').alt = `Scene frame ${id}`;
  $('lbcap').innerHTML = `<b>${id}</b> · ${fr.time_s} s · interval ${fr.interval_id} · gaze sample ${fr.gaze_sample_index} → ${esc(fr.assigned_target_name)} · within 25 px: ${fr.alternatives_within_px.map(a=>esc(objName[a]||a)).join(', ')||'none'} (nearest ${fr.nearest_alternative_px ?? '—'} px)` +
    (observation ? `<br>Muse: ${esc(observation)}` : '') + '<br><span style="opacity:.7">Video moved to this frame. Click anywhere to close.</span>';
  $('lightbox').classList.add('on');
}
$('lightbox').onclick = () => $('lightbox').classList.remove('on');
document.addEventListener('keydown', e => { if (e.key === 'Escape') { $('lightbox').classList.remove('on'); setTheater(false); } });

// provenance
const run = D.run || {}, tamper = D.tamper;
const rd = V.rederivation || {};
$('prov').innerHTML = `<table>
  <tr><th>Muse model</th><td class="mono">${esc(run.returned_model || run.requested_model || '—')} (requested ${esc(run.requested_model||'—')})</td></tr>
  <tr><th>Response id</th><td class="mono">${esc(run.response_id||'—')}</td></tr>
  <tr><th>Request</th><td>${esc(run.started_utc||'')} · ${esc(run.endpoint||'')} · reasoning ${esc((run.request_parameters||{}).reasoning_effort||'')} · ${esc(run.elapsed_s||'')} s · finish ${esc(run.finish_reason||'')}</td></tr>
  <tr><th>Tokens</th><td class="mono">${esc(JSON.stringify(run.usage||{}))}</td></tr>
  <tr><th>Package</th><td>${P.facts.length} facts · ${P.intervals.length} intervals · ${P.episodes.length} episodes · ${P.frames.length} frames sent</td></tr>
  <tr><th>Fact re-derivation</th><td>${rd.checked ?? '—'} facts recomputed from trace CSVs · ${(rd.mismatches||[]).length} mismatches</td></tr>
  <tr><th>Tamper test</th><td>${tamper ? tamper.cases.map(c => `${esc(c.case)}: ${c.detected?'<span class="badge b-ok">detected</span>':'<span class="badge b-bad">missed</span>'}`).join(' ') : 'not run'}</td></tr>
  <tr><th>Artifacts</th><td class="mono">${Object.entries(P.recording.artifact_ids).map(([k,v])=>`${esc(k)}=${esc(v)}`).join('<br>')}</td></tr>
  <tr><th>Sources</th><td class="mono">${esc(P.source_refs.attention_trace_dir)}<br>${esc(P.source_refs.gaze_fusion_dir)}<br>${esc(P.source_refs.scene_video)}</td></tr>
</table>`;

drawTimeline();
const want = location.hash.slice(1);
if (want !== 'none') select(F.findings.some(f => f.finding_id === want) ? want : F.findings[0].finding_id);
else $('evidence').innerHTML = '<h2>Evidence</h2><p>Select a finding to open the moment behind it: video, frames and measurements.</p>';
</script>
</body>
</html>
"""
