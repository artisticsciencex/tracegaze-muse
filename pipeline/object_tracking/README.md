# object_tracking — select · name · mask · track · review · publish

GUI stage of the gaze pipeline: pick a NEON recording with a server-side
folder chooser, name arbitrary objects, select each with SAM3 point prompts,
verify the initial mask, track bidirectionally with persistent pipeline IDs,
review colored overlays, and publish a validated, portable
`tracked_objects` artifact for gaze fusion.

```bash
~/.venvs/neon-trim/bin/python pipeline/object_tracking/server.py   # → http://127.0.0.1:8002
~/.venvs/neon-trim/bin/python -m pytest pipeline/object_tracking/tests -q
```

GPU side runs in a separate env (`OBJTRACK_SAM3_PYTHON`, default
`~/.venvs/sam3/bin/python`: torch 2.11+cu128, `pip install -e sam3-main`,
einops psutil pandas pyarrow pycocotools opencv numpy<2). Gated weights need
`hf auth login` once. FlashAttention-3 stays disabled per project policy.

## Guided workflow (GUI)

The GUI walks a researcher through four steps — **1 Select objects → 2 Track
→ 3 Review → 4 Export** — with a state-driven stepper and a guidance card
that always offers the one obvious next action. Internal states stay
rigorous; the UI translates them (`mask_accepted` → "Ready to track",
`tracked` → "Needs review", `accepted` → "Approved ✓"). "Use this selection"
confirms the starting mask on ONE frame (full-video tracking has not run
yet); one "Track N ready objects" click starts the SAM3 worker automatically
(GPU controls live under "show advanced controls"); review is play-based
(tracked masks on, prompt points off by default) with include-checkboxes and
ONE global "Approve all included tracks" confirmation; Export stays hidden
until something is tracked and reviewed, and its guidance text explains what
is missing instead of raising "not publishable" errors. A re-accepted rerun
object keeps the track action reachable from the review/export steps.
Server-side validation is unchanged — untracked masks can never be
published.

## Configuration

| setting | default | override |
|---|---|---|
| browse roots | `<project>/neon-data` | `OBJTRACK_ROOTS` / repeatable `--root` |
| port | 8002 | `OBJTRACK_PORT` / `--port` |
| host | 127.0.0.1 (DNS-rebinding protected) | `--host` + `--allow-lan` |
| SAM3 python | `~/.venvs/sam3/bin/python` | `OBJTRACK_SAM3_PYTHON` |
| GPUs | single GPU `[0]` (see Blackwell note) | `OBJTRACK_GPUS=0,1` |
| state dir (drafts, tracks, recents) | `~/.cache/object_tracking` | `OBJTRACK_STATE_DIR` |

## Input types (auto-detected by the chooser)

trimmed NEON artifact (preferred) · prepared NEON recording · NEON CSV
export · native/raw NEON recording (structural detection → metadata panel →
`Prepare recording` action) · video-only (`gaze unavailable`, never
gaze-compatible) · tracked-object artifacts are recognized but not re-opened
as inputs.

**Native preparation** uses the official `pupil-labs-neon-recording` library
(never hand-decoded) and produces an export-shaped canonical artifact +
`prep_manifest.json`. Verified against the matching Cloud export of the SAME
recording (0000abcd…, participant_02 native vs participant_a export): native
scene = 6,403 frames ↔ 6,403 `.time` entries (1:1, PTS spread 0.44 ms);
Cloud adds 30 synthetic 50 ms-grid pad frames at the head and re-timestamps
real frames by ±5.5 µs (mapping is INDEX-based, never timestamp-equality);
native library gaze ≡ export gaze exactly (±0.0005 px CSV rounding) with the
recorded `gaze_offset` ALREADY applied — recorded as provenance, never
re-applied; 5 gaze tail samples past recording end are trimmed exactly as
Cloud does. Unresolvable sync blocks gaze-compatible publication with a
diagnostic — never silently forced.

## SAM3 integration

- prompt method: SAM2-style tracker path — positive/negative POINTS +
  `obj_id`, relative coordinates (the reliable instance path; the raw
  box-exemplar path is not used; a drawn rectangle only generates a movable
  center point and says so)
- single-frame preview via the `add_prompt` response (no propagation);
  preview and tracking use the SAME score threshold
- bidirectional: separate forward (`ref → end`) and backward (`ref−1 → 0`)
  propagation streams with independent per-direction status; reference frame
  appears exactly once; one reset pass per object (cost shown before launch)
- worker subprocess in the GPU env (JSON-lines protocol, cancel-safe,
  process-group SIGKILL cleanup, response-desync fail-stop); requires the
  vendored LOCAL PATCH (verified before spawn)
- **Blackwell note (RTX PRO 6000 / sm_120)**: cuDNN heuristic engine
  selection fails for SAM3's neck convs ("GET was unable to find an
  engine..."); the worker sets `cudnn.benchmark = True` and defaults to a
  single in-process GPU because these flags cannot reach SAM3's spawned
  multi-GPU rank workers.

## Artifact contract (`tracking_manifest.json`, schema 1.0)

`artifact_type: "tracked_objects"`; relative `files` (movable folder;
symlink-escape-checked on load); parent recorded as artifact id + labelled
absolute source path (video/gaze/world are NOT duplicated — self-contained
mode is intentionally not implemented and recorded as such);
`objects.parquet` columns: frame_index, timestamp_ns (nullable Int64, only
from VERIFIED sync), object_id, object_name, sam_local_id, reference_frame,
tracking_direction, score, bbox_x/y/w/h, mask_area, mask_rle (COCO RLE).
Missing rows = not visible (no fabricated zero-area rows). Optional
`tracking_overlay.mp4` is time-faithful (exact frame count/order, VFR spacing
validated ≤2 ms, constant muxer offset recorded, audio retained) and is never
a synchronization source. `load_tracked_artifact()` validates schema, files,
id uniqueness/immutability, deterministic colors, frame bounds, duplicate
rows, and table↔annotation consistency. Publication runs in an isolated
subprocess (`publish_cli.py` — also the CLI entry point) and is atomic.

## Process-isolation architecture (Arrow segfault hardening)

pyarrow crashed the GUI server twice (kernel-verified `segfault in
libarrow.so.2500`, once from `pandas.read_parquet` in a FastAPI thread). The
server process therefore NEVER imports pandas/pyarrow (regression-tested in
a pristine interpreter):

- track rows live in a **SQLite session store** (`trackstore.py`, WAL): one
  row per object/frame, PK-enforced no-duplicates, indexed range queries for
  bounded mask retrieval, and ATOMIC per-object replacement (a cancelled or
  failed rerun leaves the previous rows untouched)
- the SAM3 worker writes SQLite directly; post-track validation, review
  summaries, and mask serving are pure-SQL in the server
- **all** Parquet work is subprocess-isolated: `publish_cli.py` (publication,
  reads SQLite → writes `objects.parquet`), `table_io.py` (orphan-parquet
  import + artifact inspection). A helper segfault (exit 139) becomes a
  clear job error; the server keeps serving
- crash recovery: on load, completed-but-unregistered tracks (server died
  before registration) are validated and imported by the isolated helper —
  only genuinely unrecoverable objects need a rerun
- the worker suppresses shutdown `BrokenPipeError` (secondary to whatever
  killed the parent); the GUI detects server loss, says so explicitly, and
  reconnects the session automatically

## Frame-presentation truth (mask/frame sync hardening)

The GUI once derived the mask frame from `video.currentTime` and painted on
`timeupdate`/`seeked`/timer ticks — none of which prove the browser has
PRESENTED the expected VFR frame, so frame-N masks could overlay frame-N−1
pixels. Offline composites (source frame decoded by exact index + SQLite
masks) proved the STORED tracks were correctly indexed; the defect was
presentation-only. Now:

- `static/framesync.js` — explicit state machine (`requestedFrame`,
  `displayedFrame`, `displayedMediaTime`, `seekGeneration`,
  `framePresentationReady`); background, masks, points, readout and
  reference-frame operations commit ATOMICALLY per generation; stale
  image/mask responses are discarded; annotation is disabled while a seek is
  unresolved (node-tested: 10 race scenarios)
- paused review uses `GET /api/frame?dir&frame_index=N` — the exact frame by
  presentation order against the authoritative decoded-PTS table
  (`framesource.FrameDecoder`: indexed keyframe seek + forward decode,
  sequential roll-forward, bounded JPEG LRU; identity headers)
- playback uses `requestVideoFrameCallback` presented-frame `mediaTime`,
  mapped by nearest-PTS within half the smallest VFR gap
  (`timemap.FrameTimeMap`: tested frame ↔ PTS ↔ mediaTime ↔ world-ns
  conversions, constant video/world offsets preserved, e.g. the fixture's
  16 ms); browsers without rVFC get a clearly-labelled approximate fallback
- the status bar states "✓ frame N committed … screenshot-safe" only when
  image + masks + readout belong to one generation; a debug panel
  (checkbox) exposes all sync state

## Known limitations

- no multi-frame correction-point merging: the supported correction is a
  per-object rerun from a revised prompt/reference frame (recorded in the
  manifest)
- multi-GPU disabled by default on Blackwell (see note above)
- auxiliary NEON streams (IMU, eye state, events, …) are not converted by
  native preparation (listed in `omitted_streams`)
- if the PARENT recording folder is later moved, the artifact still loads
  but parent paths stop resolving (absolute provenance; surfaced by the
  loader's `parent_resolvable` flags)
