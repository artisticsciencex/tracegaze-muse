# patches/ — vendored SAM3 local patch (known-good point-prompt behavior)

The object-tracking stage depends on ONE local modification to the vendored
SAM3 source. Without it, a point-selected new object makes
`propagate_in_video → _get_processing_order` raise
`"No prompts are received on any frames"`, and bidirectional tracking of
point-prompted instances is impossible. This directory makes that behavior
restorable without any network access.

## What is patched

- File: `sam3-main/sam3/model/sam3_video_inference.py`, method `add_prompt`
  (tracker / SAM2-style point+obj_id path).
- Change: capture the `add_tracker_new_points(...)` result instead of
  returning it directly, mark the prompted frame in
  `previous_stages_out` with the `"_THIS_FRAME_HAS_OUTPUTS_"` sentinel
  (exactly what upstream's semantic path does at line ~402), then return.
- The full change is `sam3_point_prompt.patch` (one hunk pair, 9 lines
  added, 1 line changed).

## Upstream / vendored source identity

- Upstream: https://github.com/facebookresearch/sam3 — vendored into
  `sam3-main/` as a plain source snapshot (NOT a git clone), pip metadata:
  `sam3 0.1.0`, "Last Updated: November 19, 2025", installed editable into
  `~/.venvs/sam3`.
- Verified 2026-08-08 against upstream `main`
  (blob sha256 `70d32ec094b3151aa5b5e73e1c0dd13bc5e7a06ab51fc6b1776909e9ad501aef`):
  after reverting this patch the vendored file is IDENTICAL to upstream
  except four `# pyrefly: ignore` comment lines upstream added after the
  snapshot date (the vendored tree contains no pyrefly comments anywhere).
- `sam3_vendored_source_hashes.sha256` records SHA-256 of every `.py` file
  in the vendored `sam3-main/sam3/` package (152 files) at milestone time —
  use it to detect any later drift of the vendored dependency.

## Files

| file | role |
|---|---|
| `sam3_point_prompt.patch` | minimal unified diff (pre-patch → patched) |
| `sam3_video_inference.known-good.py` | exact byte-for-byte copy of the patched file (sha256 `de8286270f4a1fb35e7ced5bca4441cbbc9a34f3d7f4239df576db4180ae53d7`) |
| `sam3_vendored_source_hashes.sha256` | identity manifest of the whole vendored package |

Pre-patch original sha256 (reconstructed, pyrefly-era upstream equivalent):
`bbcc8d1804efdc1290d7b68a9cbcd03d7dba88db54aef8cfe4bff5768bb9e5ac`.

## Restore / apply

Preferred (exact restore, no tooling):

```bash
cp patches/sam3_video_inference.known-good.py \
   sam3-main/sam3/model/sam3_video_inference.py
```

Or apply the diff to a clean upstream-equivalent file:

```bash
cd sam3-main && patch -p1 < ../patches/sam3_point_prompt.patch
```

No reinstall needed (`sam3` is an editable install).

## Verify

1. Marker present twice (upstream semantic path + this patch):
   `grep -c _THIS_FRAME_HAS_OUTPUTS_ sam3-main/sam3/model/sam3_video_inference.py`
   → `2`
2. Hash matches:
   `sha256sum sam3-main/sam3/model/sam3_video_inference.py`
   → `de8286…ae53d7`
3. Behavioral check (needs GPU + gated `facebook/sam3` weights): run the
   object-tracking GUI, add one point prompt on any frame, preview, accept,
   Track — both forward and backward propagation must complete. The
   pre-patch failure mode is an immediate
   `"No prompts are received on any frames"` error at track start.

Model weights are NOT stored in this repository (gated Hugging Face repo
`facebook/sam3`; authenticate with `hf auth login`).
