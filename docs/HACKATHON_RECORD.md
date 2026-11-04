# Hackathon record: first working example (2026-11-05)

What was reused, what was built, the Muse runs and the checks performed. Local paths are anonymised.

## Existing foundation (verified, reused unchanged)

| Component | Where | State found |
|---|---|---|
| Reviewed tracking + gaze fusion | `neon-data/AriaEverydayActivities_loc4_script3_seq4_rec1_prepared_aria_trim_f709-f1463_{namefix1_tracked_objects,gaze_fusion_v3}` | 16 accepted objects; 378 gaze samples, all valid |
| Source recording | `data/aria/…_trim_f709-f1463/` (scene.mp4, gaze.csv, world_timestamps.csv) | sha256 matches the fusion manifest |
| attention_trace stage | `pipeline/attention_trace/` | ran unchanged → 92 episodes, 91 transitions, 10/10 validations |
| Evidence-grounded analyst v1 | `pipeline/tracegaze_analyst/{facts,packet,validate,render,ollama,baseline}.py` | pre-hackathon; `facts.build_fact_catalog`, `facts.load_trace` and `validate.UNSUPPORTED_ASSERTION_FRAGMENTS` reused |
| Python env | `~/.venvs/neon-trim` (py 3.12.3; pandas, pyarrow, av, opencv, pycocotools) | used as is; nothing installed |
| Meta API config | none in the repository | key read from a local env file via `--env-file`; never written to outputs |
| Hardware | GPU0 RTX PRO 6000 Blackwell, GPU1 RTX PRO 5000 Blackwell | not needed: no segmentation was rerun |

## New in this task

- `support_package.py` — evidence package: deterministic facts (session,
  per-object incl. boundary-near counts, per-interval, per-10-s phase,
  transition pairs), rule-selected intervals, boundary proximity (25 px
  convention), 12 frames with the gaze ring, source references, video PTS
  offsets.
- `muse.py` — stdlib client for `POST https://api.meta.ai/v1/chat/completions`
  with JSON-schema output; key read from `MODEL_API_KEY` or `--env-file`,
  scrubbed from errors, never saved; image bytes hashed in saved requests.
- `findings.py` — findings schema (strict), system prompt, model view of the
  package, validator (references, exact quantities, numbers in prose,
  construct and personal-attribute gates, evidence-type obligations) and an
  independent re-derivation of every fact from the trace CSVs.
- `support_cli.py` — `support-package`, `muse-findings`, `check-findings`,
  `tamper-check`, `support-report`, `serve-report`.
- `support_report.py` — static report with the TraceGaze visualization and
  scene video (symlinked), clickable timeline, per-finding evidence panel,
  provenance; local server with HTTP range support.
- `examples/aea_loc4_script3_seq4_rec1.json` — question and explicitly
  supplied context (unknowns left unknown).
- `tests/test_support.py` — 10 tests (15 total with the existing 5).

## Muse runs (all `muse-spark-1.3`, standard mode, chat completions)

| Run | Settings | Result |
|---|---|---|
| `muse_run_attempt1` (`chatcmpl-01a1a7b4…`) | reasoning medium, 16k cap, `package_attempt1` | 1 finding, references verified; flagged "expected 2-4 findings" |
| `muse_run_attempt2_failed` | reasoning high, 16k cap, three-finding prompt | unparseable JSON; reply not kept (bug, fixed) |
| `muse_run` (`chatcmpl-01a1a7ba…`) — shown | reasoning high, 32k cap | 3 findings, 3 references verified; 52,249 prompt / 15,328 completion tokens; 152.5 s |

Between attempts 1 and 3 the package gained per-object `sample_count` and
`boundary_near_sample_count` facts (frames byte-identical), and the prompt
asked for three findings covering every part of the question.

## Checks performed

- Fact re-derivation: 220/220 package facts recomputed from the trace CSVs, 0 mismatches.
- Validator on the real response: 3/3 findings references_verified, no global problems.
- Tamper test on the real response/package: invalid frame ref, invalid interval
  ref, altered quantity, untraceable prose number, construct word in evidence,
  altered package fact — 6/6 detected (`muse_run/tamper_check.json`).
- Report in the browser: every interval link seeks to its start (±0.01 s);
  all 12 frame links show the pixel-matching scene-video frame (nearest
  neighbour frames differ); all measurement rows match the package; no
  console errors, no horizontal overflow at 705 px.
- Manual review (`muse_run/manual_review.json`): f1 supported; f2 and f3
  partially supported (f2 "before eating" not established; f3 overweights
  edge effects — two cited frames show genuine looks toward the partner and
  the window).
- Originals: 15 source files re-hashed identical (`originals_sha256_{before,after}.txt`).
- No saved file contains the key.

## Known gaps

- The validator's head-movement gate matches only the exact phrase
  "head movement"; f3 says "head and view movement" and passed.
- Hands are not tracked, so gaze on the wearer's own hand is "no accepted
  tracked-mask hit".
- One recording, not an older-adult recording; no accuracy, time-saving or
  benefit claims are made.
