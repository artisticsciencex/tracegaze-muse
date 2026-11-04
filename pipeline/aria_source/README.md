# aria_source — Project Aria public-dataset adapter

Adds Aria as a **second source system** beside NEON. Aria never
masquerades as a NEON recording: it has its own artifact types, its own
manifests and its own timestamp domain, and both systems converge on one
logical interface — scene video, one authoritative timestamp per frame,
timestamped 2D gaze in scene pixels, provenance.

```bash
# in the unified workbench: Choose Aria folder → Prepare → Trim → Track → Fuse
~/.venvs/neon-trim/bin/python pipeline/workbench/server.py   # → :8000
~/.venvs/neon-trim/bin/python -m pytest pipeline/aria_source/tests -q
```

## Isolated native code

`projectaria_tools` (native VRS/calibration) is imported **only** by
`helper_cli.py`, run as a subprocess by a dedicated interpreter
(`ARIA_PYTHON`, default `~/.venvs/aria/bin/python`). It is never imported
by the workbench, trim, tracking, fusion servers or the SAM3 worker. A
helper crash, segfault or timeout becomes a typed `AriaHelperError`: the
GUI stays alive, the job shows a clear error, sources and any valid cache
are untouched, and no partial artifact is published.

## Supported source layouts

The lightweight folder checker resolves dataset-specific names into canonical
roles instead of requiring one downloader convention:

- **Reading in the Wild:** `_recording_vrs.zip`, full `_mps.zip`, optional
  metadata/preview, and `download_receipt.json` (required and validated).
- **Aria Everyday Activities:** direct `_main_recording.vrs`, gaze-only
  `_mps_eye_gaze.zip`, optional `_annotations.zip` / preview, and no receipt.
  Exact source SHA-256 hashes are recorded during preparation instead.

Mixed layouts, duplicate role files, and folders without both a VRS and an
explicit eye-gaze package are rejected rather than guessed.

## Cache (configurable: `ARIA_CACHE_DIR`, default `~/.cache/aria_source`)

Source files are only ever read. Preparation materializes `recording.vrs`
plus **only** `personalized_eye_gaze.csv`, `general_eye_gaze.csv`, and
`metadata.json`. A direct AEA VRS is copied atomically; a RITW VRS is
selectively extracted. Gaze CSVs may be nested under `mps/eye_gaze/` or at a
gaze-only ZIP's root. Semidense point clouds, SLAM observations, trajectories,
online calibration and hand tracking are never extracted. Entries are built
in a temp dir and renamed on success; a manifest that is missing, unparseable,
size-inconsistent or whose sources changed is rebuilt.

## Frames, timestamps, orientation

RGB stream `214-1/camera-rgb` is authoritative. Every frame's
`DEVICE_TIME` capture timestamp is read as an exact integer nanosecond
value (timestamp table, spot-checked against the frame records) and
written one row per exported frame to `world_timestamps.csv` — a
**compatibility filename** whose manifest states plainly that the values
are Aria DEVICE_TIME, not NEON world time. Strict ordering is verified
and irregular intervals are reported. One documented orientation
transform (`rot90_cw_upright`) is applied consistently to video frames,
camera calibration and gaze projection. `scene.mp4` is generated directly
from VRS RGB frames, so frame-for-frame correspondence with the timestamp
table holds by construction (the preview MP4 is not used). Audio is not
muxed in v1 and is reported `audio_present: false` with the reason.

## Gaze

`personalized_eye_gaze.csv` is preferred when present, non-empty,
parseable and temporally overlapping the VRS recording; otherwise the
general model is used and the reason is recorded. `tracking_timestamp_us`
is converted by exact integer ×1000. Projection uses the official
`get_gaze_vector_reprojection` with the device calibration and the
CW90-rotated RGB camera calibration (`make_upright=True`) — no pinhole
approximation, no image-size scaling. Per-row `depth_m` is used when
valid, otherwise a documented 1.0 m fallback, with `depth_source`
recorded. Out-of-image projections are kept unclamped; samples with no
projection get empty coordinates and
`projection_status = outside_field_of_view`. NEON-only `worn`, fixation
and blink values are never invented — they are absent and read as null
downstream.

## Artifacts

`prepared_aria_recording` and `trimmed_aria_recording` (schema 1.0), each
containing `scene.mp4`, `gaze.csv`, `world_timestamps.csv` and its
manifest. Trimming reuses the verified source-agnostic interval math and
frame-accurate cut (`neon_trim.core.plan_trim`, `common.media.run_cut`):
the UI selection stays inclusive `[start, end]` while the interval is
half-open. Downstream, `common.recordings` dispatches on manifest
`artifact_type` (never the filename, which both systems share) so object
tracking and gaze fusion accept either source without duplicating any
tracking, mask, hit-testing or rendering logic.

## Limitations (v1)

- audio from the VRS microphone stream is not muxed
- the preview MP4 is not used for browsing (correspondence not asserted)
- only Gen 1 RGB (`214-1`) is handled; SLAM/ET streams are not exported
- the cache holds a full copy of `recording.vrs` (~2.5 GB per recording)
