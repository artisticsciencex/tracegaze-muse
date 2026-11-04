# attention_trace — relational attention trace

Stage 4. Turns a validated `gaze_object_fusion` artifact into an
auditable, object-relational description of attention:

```bash
~/.venvs/neon-trim/bin/python -m attention_trace \
    --fusion /path/to/<rec>_gaze_fusion     # → <rec>_attention_trace
~/.venvs/neon-trim/bin/python -m pytest pipeline/attention_trace/tests -q
```

Run from `pipeline/` (or with `pipeline/` on `PYTHONPATH`). The parent
fusion artifact is only ever read; an existing output is refused.

## Evidence

The authoritative `gaze_samples.parquet` table is the **only** input. The
demonstration video and its presentation-only interpolated cursor
(`demo_presentation.csv`) are never read — no attention event is ever
inferred from rendered pixels. Accepted object IDs and manual names pass
through unchanged, and overlap ties keep whatever `primary_object_id`
gaze fusion already decided (smaller mask area → higher track score →
stable id); this stage never re-decides them.

`__no_tracked_object__` means exactly *a valid, in-frame gaze sample that
hit none of the accepted tracked masks*. It does not assert that the
participant looked at physical background and never implies an unlabeled
object.

## Temporal policy (exact integer nanoseconds)

Eligible samples are assigned to a frame, gaze-valid, in frame, and carry
a supporting status (`on_object` or `background`). `invalid_gaze`,
`outside_frame` and `no_matching_frame` rows carry no attention but do
take part in the midpoint tiling, so a blink's own territory becomes
unsupported and it splits the episode even when the same object resumes
immediately after it.

```
max_gap_ns = clamp(3 × median positive valid sampling interval, 50 ms, 250 ms)
```

Each eligible sample is supported from the midpoint to its previous
neighbour to the midpoint to its next neighbour **in the combined
timeline of eligible and breaking samples**, each side capped at
`max_gap_ns // 2` and clipped to the fusion artifact's validated
interval. Both boundaries use the same floor midpoint, so intervals tile
exactly for either parity of the gap (an asymmetric floor/ceil pair
overlaps by 1 ns on odd gaps). A gap longer than `max_gap_ns` leaves its
middle **unsupported** rather than silently bridged. Seconds appear only
as derived display values.

- **Episode** — a maximal run of consecutive eligible samples sharing a
  target whose support intervals touch. No smoothing, no merging across a
  break, no relabelling of short episodes.
- **Transition** — a link between consecutive episodes, carrying the
  unsupported gap between them and one of `object_to_object`,
  `object_to_no_hit`, `no_hit_to_object`, `no_hit_to_no_hit`.
- **Revisit** — a *return after leaving*: `episode_count - 1`, never the
  sample count.

## Outputs

`attention_trace_manifest.json` · `attention_episodes.csv` / `.parquet` ·
`attention_transitions.csv` / `.parquet` ·
`object_attention_summary.csv` · `attention_trace.json`

The manifest records parent fusion identity and file hashes (re-checked
after writing), source system, timestamp domain, both policies, row
counts, duration conservation checks and the validation record. Episodes
and transitions link back to concrete `gaze_sample_index` and
`frame_index` values, so every reported number can be traced to the exact
rows that produced it. `attention_trace.json` is the compact
machine-readable sequence (episodes, transitions, object dictionary,
policy, evidence links).

## Limitations

- one target per sample: overlapping masks contribute only through the
  primary object fusion already selected, so a "looked at both" reading
  is out of scope here
- support is estimated from sample midpoints, not from fixation
  detection; a saccade crossing an object boundary is attributed by
  timestamp, not by oculomotor event
- episodes are not merged across breaks even when the same object resumes
  immediately after an invalid sample — this is deliberate, and shows up
  as an `object_to_object` transition with a positive unsupported gap
- accepted objects that were never attended appear in the summary with
  zero dwell and zero episodes, so an absence of attention is visible
  rather than missing
- provenance (source system, timestamp domain) is inherited from the
  tracked-object manifest only when its artifact id matches the one the
  fusion artifact recorded; otherwise it is reported `unknown` rather
  than guessed
