# gaze_object_fusion — fuse gaze samples with tracked object masks

Stage 3 of the pipeline (after `neon_trim` and `object_tracking`).
Computation lives in reusable non-GUI modules driven by a CLI; a small
local GUI orchestrates the same CLI in a subprocess (so the long-lived
FastAPI process stays pandas/pyarrow-free per the Arrow-isolation policy).

```bash
# CLI (from pipeline/, or with pipeline/ on PYTHONPATH)
~/.venvs/neon-trim/bin/python -m gaze_object_fusion \
    --artifact /path/to/<rec>_tracked_objects        # → <rec>_gaze_fusion

# GUI (or .claude/launch.json config "gaze-fusion")
~/.venvs/neon-trim/bin/python pipeline/gaze_object_fusion/server.py
    # → http://127.0.0.1:8003 — choose artifact folder (auto discovery +
    # validation), choose output, "Build gaze demo" with progress, then a
    # completion summary (sample counts, per-object hits) and an in-page
    # demo-video preview / open action

~/.venvs/neon-trim/bin/python -m pytest pipeline/gaze_object_fusion/tests -q
```

## Authoritative public table — `gaze_samples.csv` + `gaze_samples.parquet`

Both files carry the SAME rows, values, ordering, and null semantics
(tested field-by-field). Exact column order:

`gaze_sample_index, timestamp_ns, relative_time_s, frame_index,
frame_timestamp_ns, frame_offset_ms, gaze_x_px, gaze_y_px, gaze_pixel_x,
gaze_pixel_y, worn, gaze_valid, gaze_status, is_gaze_at_object,
primary_object_id, primary_object_name, hit_object_count, hit_object_ids,
hit_object_names, primary_mask_area, primary_track_score, fixation_id,
blink_id`

- `gaze_sample_index`: stable zero-based index in original gaze-row order
- `timestamp_ns` / `frame_timestamp_ns`: exact integer nanoseconds
  (`frame_timestamp_ns` = the assigned frame's authoritative world
  timestamp); `relative_time_s` is measured from the trimmed-video start
- `gaze_pixel_x/y`: the exact integer pixels used for the mask lookup
  (`floor`), null when coordinates are non-finite
- `gaze_status` ∈ {`on_object`, `background`, `invalid_gaze`,
  `outside_frame`, `no_matching_frame`} with documented precedence
  no_matching_frame > invalid_gaze > outside_frame > on_object/background
- `is_gaze_at_object` is true exactly when ≥1 accepted mask contains the
  valid gaze pixel; `hit_object_count == len(hit_object_ids)`
- CSV encoding: ints exact integer text, nulls empty, booleans True/False,
  floats shortest-roundtrip repr, list fields JSON arrays (parse back to
  the parquet lists); `fixation_id`/`blink_id` are nullable INTEGERS —
  never floats with misleading numeric NaN-neighbors
- `gaze_objects.parquet` is retained as an INTERNAL working table (extra
  columns, no stability promise)

## Inputs

A published `tracked_objects` artifact. Its manifest resolves the parent
**trimmed NEON artifact** (fusion v1 requires this parent type: the trim
manifest carries the validated interval boundaries). Before fusing, the
parent's video/gaze/world files are re-hashed and must match the hashes the
tracking stage recorded — a changed parent aborts the run.

## The gaze→frame assignment rule (exact, timestamp-defined)

No fixed global frame-distance threshold (16 ms, 20 ms, …) is used —
frame CELLS are defined purely by timestamps:

- frame *i* owns the half-open cell `[b[i], b[i+1])`
- interior boundaries: `b[i] = (world[i−1] + world[i]) // 2` — exact
  integer midpoints of adjacent `world_timestamps.csv` entries (floor
  division, int nanoseconds end to end; these values exceed 2^53 and must
  never pass through floats)
- first/final boundaries: the parent trim interval
  `selection.absolute_start_ns` / `selection.absolute_end_boundary_ns`
  (semantics `[start_frame_time, time_after_last_frame)`)
- samples outside the interval → `no_matching_frame`
- the signed `frame_offset_ns`/`frame_offset_ms` (gaze ts − assigned
  frame's world ts) is preserved on every row
- world-timestamp gaps above the documented sanity threshold (default
  200 ms ≈ 4× the largest normal NEON VFR interval of ~50 ms) are REPORTED
  in the manifest's `assignment_rule.anomalous_gaps`; in-range samples are
  still assigned by the rule, never silently dropped

## Hit lists (deterministic)

Gaze `(x, y)` lies in pixel `(floor(x), floor(y))`; out-of-frame
coordinates are flagged, never clamped. All masks containing that pixel
form the hit list, ordered by: **1) smaller mask area, 2) higher tracking
score, 3) stable object id** — the first entry is the primary object.
Rows with `worn == 0` or non-finite coordinates keep their frame
assignment but are marked invalid and never hit-tested.

## Demonstration video (`gaze_demo.mp4`)

Time-faithful like the tracking overlay (identical frame count/order,
source VFR PTS copied and validated ≤ 2 ms beyond the constant container
offset, audio retained). Per-frame semantics (tested pixel-level):

- cursor row = the valid fused row assigned to that EXACT frame with the
  smallest `|frame_offset_ms|` (tie → earliest timestamp); *valid* =
  `gaze_valid` (worn, finite) AND coordinates inside the frame; no valid
  row → no cursor; adjacent-frame samples are never borrowed
- ONE fixed high-contrast cursor for every valid sample, independent of
  what is under it: bright green ring (~18 px radius at 1600×1200,
  scaled by min(width, height); ~3.8 px thick) over a black outer halo,
  with a small red center dot ringed by a thin white edge; identical for
  gaze-on-object and gaze-on-background; no trail, pulse, smoothing, or
  interpolation
- object-hit meaning is conveyed EXCLUSIVELY by the mask treatment: the
  frame's PRIMARY gazed-at object (`primary_object_id` of the cursor
  row) gets a bright 50 % colored fill, a 3 px boundary, and a
  `name object_id` label near its bounding box; all OTHER tracked
  objects get a subtle 12 % fill + 1 px boundary in their own colors;
  background gaze → no brightly highlighted object
- invalid / out-of-frame gaze: no cursor (subtle masks only — the
  documented neutral treatment)

## What `background` means (and unlabeled objects)

`gaze_status = background` means: a **valid gaze sample inside the video
frame that intersects none of the accepted tracked-object masks** on its
assigned frame. It does **not** prove the participant viewed physical
background — the gaze may fall on an untracked or unlabeled object that
simply was not selected in the tracking stage.

Recommended treatment of unlabeled objects:

- any visible object can be selected manually in the object-tracking GUI
  with point prompts — the vocabulary is open, no predefined class list
  is required
- provisional names such as `unknown-tool-01` are perfectly valid; names
  can be edited later without changing the ID
- `object_id` values (`obj_NNNN`) are stable within one recording's
  tracked artifact — they are NOT automatically global across different
  recordings (cross-recording identity is a downstream mapping decision)
- an object discovered later (e.g. while reviewing the demo video)
  requires adding + tracking it in the object-tracking GUI and rerunning
  fusion; previous fusion outputs stay untouched (new output folder)

### Object-count behavior

The tracking GUI has **no artificial hard object limit**: objects are
tracked sequentially, one SAM3 pass per object, so runtime, storage, and
review effort scale approximately linearly with the number of objects.
The separate SAM3.1 `max_num_objects=16` multiplex setting applies to
SAM3's internal multi-object text-prompt path and does **not** cap this
point-prompt workflow.

## Artifact contract (`fusion_manifest.json`, schema 1.0)

`artifact_type: "gaze_object_fusion"`; files: `gaze_objects.parquet`
(one row per gaze sample) + optional `gaze_demo.mp4`. The manifest records
the full assignment rule (including interval bounds, sanity threshold and
its basis, anomalous gaps, pixel convention, hit order, demo rule), parent
linkage (tracked artifact id + manifest/table SHA-256, grandparent trim
artifact id, recording id), parent file hashes, counts, and a validation
record (all checks must be ok; the loader refuses otherwise).

Fused table columns: `gaze_ts_ns, assignment, frame_index,
frame_offset_ns, frame_offset_ms, gaze_x_px, gaze_y_px, gaze_valid,
invalid_reason, coords_in_frame, hit_object_ids, hit_object_names, n_hits,
primary_object_id, primary_object_name, primary_mask_area, primary_score,
demo_selected` + passthrough NEON columns (`worn`, `fixation_id`,
`blink_id`, `azimuth_deg`, `elevation_deg`) when present.

## Known limitations (v1)

- requires a trimmed-NEON-artifact parent (bare CSV exports / prepared
  native recordings carry no validated trim-interval boundaries yet)
- blink samples are passed through with their `blink id` — not invalidated
  (downstream stages decide; recorded here for transparency)
- no dwell/transition aggregation yet — that is the next stage on top of
  `gaze_objects.parquet`
