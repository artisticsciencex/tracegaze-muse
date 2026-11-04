# neon_trim — frame-accurate synchronized trimming of NEON recordings

A local browser GUI (QuickTime-style) to trim a Pupil Labs NEON recording:
pick the recording with the **in-app server-side folder chooser**, preview the
scene video, drag two timeline handles to pick the first/last included frame,
and export a **re-encoded trimmed MP4 + correctly synchronized `gaze.csv` /
`world_timestamps.csv` + a versioned `trim_manifest.json` artifact**.
Source recordings are never modified; exports are validated and written
atomically.

```bash
# one-time env (any python 3.12 with venv; see requirements.txt)
python3 -m venv --without-pip ~/.venvs/neon-trim
curl -sS https://bootstrap.pypa.io/get-pip.py | ~/.venvs/neon-trim/bin/python -
~/.venvs/neon-trim/bin/pip install -r pipeline/neon_trim/requirements.txt

# run the GUI
~/.venvs/neon-trim/bin/python pipeline/neon_trim/server.py     # → http://127.0.0.1:8001

# run the tests
~/.venvs/neon-trim/bin/python -m pytest pipeline/neon_trim/tests -q
```

(The `sam3` conda env on the GPU box also works after
`pip install av "uvicorn[standard]" fastapi imageio-ffmpeg pytest httpx`.)

## Configuration (documented defaults)

| setting | default | override |
|---|---|---|
| browse roots | `<project>/neon-data` | `NEON_TRIM_ROOTS` (":"-separated) and/or repeatable `--root DIR` |
| host | `127.0.0.1` | `--host` (non-loopback additionally requires `--allow-lan`) |
| port | `8001` | `NEON_TRIM_PORT` or `--port` |
| state dir (recents, last output parent) | `~/.cache/neon_trim` | `NEON_TRIM_STATE_DIR` |

The browse roots gate **every** server-side path operation — directory
listing, recording inspection, loading, video streaming, and output creation.
All requested paths are canonicalized with symlinks resolved *before* the
containment check; symlinks that point outside a root are rejected.

## Architecture (pipeline-ready layering)

```
config.py    AppConfig / load_config          — roots, port, state dir
fsbrowse.py  PathAuthorizer · list_dir · RecentsStore
intake.py    RecordingInspector → RecordingBundle   (fast structural inspection:
             file discovery, ambiguity, header-only video probe, newline-counted
             CSV rows, structural sync hint — NO packet sweep, NO gaze loading)
core.py      timestamp mapping · TrimPlan · verbatim CSV trim  (stdlib only)
media.py     PyAV probe · ffmpeg frame-accurate cut · output decode-verify
export.py    load_recording (authoritative sync check) · run_export ·
             trim_recording()  ← one-call library/CLI entry point
artifact.py  manifest schema 2.0 · load_artifact → TrimmedRecordingArtifact
server.py    create_app(cfg) — thin HTTP layer over the services above
```

The trim operation is invocable three ways without duplicated logic: the GUI
(`/api/export`), Python (`export.trim_recording(...)`), and any future CLI /
orchestrator — progress/cancel are a plain callback + `threading.Event`.

## Artifact contract (`trim_manifest.json`, schema 2.0)

`artifact_type: "trimmed_neon_recording"` with `schema_version`,
`artifact_id`, `parent_artifact` (set when re-trimming a trimmed artifact),
`created_utc`, `tool`, and a `files` section whose paths are **relative to the
artifact directory** — the folder can be moved/renamed safely. The absolute
source path appears only under `source` as labelled provenance. Downstream
stages (SAM3 tracking, gaze fusion) should open trims via
`artifact.load_artifact(dir_or_manifest)`, which validates the contract and
returns resolved `video_path` / `gaze_path` / `world_timestamps_path`.

Re-trim note (verified by test): ffmpeg's input `-ss` is relative to the
container `start_time`, which is nonzero for trimmed outputs — `run_cut`
compensates (`effective_cut_start_s`), so re-trimming an artifact stays
frame-accurate.

---

## The video ↔ world-timestamp relationship (investigated, do not assume)

Measured on the fixture
`neon-data/.../2024-01-01_00-00-00-0000abcd_participant_a/692fd7c7_0.0-214.935.mp4`:

| what | value |
|---|---|
| container duration | 214.935 s |
| container "nominal fps" | **meaningless** (reported ≈12.5/29.93 depending on tool) |
| decoded video frames | **6,433** (H.264 1600×1200, time_base 1/90000, VFR) |
| `world_timestamps.csv` rows | **6,433** |
| decoded PTS vs `(world_ts[i] − world_ts[0])` | equal, max deviation **0.000 ms** |
| frame intervals | 13.63–50 ms, median 33.34 ms (irregular, esp. at start) |
| audio | AAC mono 44.1 kHz (retained on export) |
| gaze | 42,422 rows @ ~200 Hz, starting **2.37 s after** frame 0 |

**Mapping (`world_timestamps_index_1to1`):** decoded frame *i* ↔ world row *i*;
the absolute time of frame *i* is its world `timestamp [ns]`. This is verified
per recording at load time (`core.check_sync`): equal counts + a constant
per-index PTS-vs-world offset (spread ≤ 5 ms). A constant offset is tolerated
because re-loaded trimmed outputs legitimately have one (the cut lands between
two source frames). If the check fails, **export is blocked** with a
diagnostic — the tool never guesses a gaze-per-frame ratio or a fixed fps.

Timestamps are handled as **int nanoseconds** throughout: absolute NEON times
(~1.7 × 10¹⁸) exceed float64's 2⁵³ exact-integer range.

## Selection semantics

A selection `[start_frame, end_frame]` **includes both endpoint frames** and is
represented as the half-open absolute interval
`[world_ts[start], world_ts[end+1])`, using the recording end
(`info.json start_time + duration`) after the final frame. Gaze rows are kept
iff their `timestamp [ns]` lies inside; `world_timestamps.csv` keeps rows
`start..end`. Kept CSV rows are written **byte-verbatim** (headers, column
order, precision, section/recording IDs preserved).

## Frame-accurate video cut

The MP4 is re-encoded (libx264 CRF 18, AAC audio, `+faststart`) with cut
boundaries at **PTS midpoints** between neighbouring frames, so ffmpeg's µs
rounding can never drop an endpoint frame or leak a neighbour. Two flags are
**essential** for VFR correctness (empirically verified — without them ffmpeg
silently rewrites the irregular timestamps onto a constant-fps grid, including
duplicated PTS): `-fps_mode passthrough -enc_time_base 1:90000`
(+ `-video_track_timescale 90000`).

## Export validation (all must pass before the output appears)

Sources byte-for-byte unchanged (sha256 before/after) · output decodes to
exactly the selected frame count · output PTS strictly increasing · VFR
inter-frame spacing matches source (≤2 ms) · world rows == selected frames
with exact endpoint timestamps · every gaze row inside the interval, none
missed · audio present and aligned · empty-gaze selections allowed but
reported. Everything is staged in a temp dir and `os.rename`d into place only
after validation; failures/cancels leave no partial output.

**Omitted on purpose:** `fixations.csv`, `saccades.csv`, `blinks.csv`,
`imu.csv`, `3d_eye_states.csv`, `events.csv`, `info.json`, `scene_camera.json`
are *not* copied — the output is a trim subset, not a fake-complete NEON
export. The manifest lists them under `omitted_files`.

## Tests

`tests/` — 65+ tests: synthetic-timestamp unit tests, end-to-end exports on a
generated VFR clip (boundaries, missing/empty gaze, irregular timing,
count-mismatch blocking, atomicity on failure/cancel), recording discovery
(valid / parent-with-children / missing files / ambiguous MP4s), authorization
(path traversal, symlink escape, unauthorized input+output — service level
and via HTTP TestClient), and artifact contract (v2 fields, relative
portability after moving the folder, loader rejection matrix, re-trim parent
chain).
