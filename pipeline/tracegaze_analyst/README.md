# TraceGaze analyst v1

This is the minimal evidence-grounded LLM layer. It does four things:

1. converts a validated attention trace into deterministic, citable facts;
2. selects a bounded evidence packet for a researcher question;
3. requires an LLM response to separate measured facts, observed patterns,
   interpretations, hypotheses and limitations;
4. rejects unknown evidence references and renders a report for human checking.

Model use is optional. Packet construction, deterministic facts, response
validation, and report rendering do not depend on a model provider. A local
Ollama adapter is included for semantic analysis without paid API use or data
upload; externally obtained response JSON can be validated through the same
interface.

```bash
cd pipeline
python -m tracegaze_analyst prepare \
  --trace /path/to/attention_trace \
  --question "Which objects structured this session?" \
  --out /tmp/question_packet

python -m tracegaze_analyst baseline \
  --packet /tmp/question_packet/evidence_packet.json \
  --out /tmp/question_report

# After obtaining JSON from a chosen model:
python -m tracegaze_analyst validate \
  --packet /tmp/question_packet/evidence_packet.json \
  --response /path/to/model_response.json \
  --out /tmp/validated_report
```

The baseline is not presented as semantic analysis; it is a zero-API integration
test. The local provider adapter was exercised only after the deterministic path
passed on both NEON and Aria data.



## Local semantic baseline

```bash
python -m tracegaze_analyst analyze-local \
  --packet /tmp/question_packet/evidence_packet.json \
  --model qwen3.6:35b \
  --out /tmp/local_analysis
```

This uses the local Ollama service. It makes no paid API call and uploads no data.


## Everyday-activity support example (TraceGaze + Muse)

Retrospective analysis for researchers studying everyday activities and
designing assistance: recording + researcher question → bounded TraceGaze
evidence package → one Muse request → findings linked to source evidence →
local report. No live assistance; no claim of benefit for older adults.

```bash
cd pipeline
PY=~/.venvs/neon-trim/bin/python
E=../out/aea_meal_v1                       # gitignored output root

# 0) existing deterministic stage (CPU only; parent artifacts are read-only)
$PY -m attention_trace --fusion <..._gaze_fusion_v3> --out $E/attention_trace

# 1) evidence package: facts, rule-selected intervals, frames, boundary proximity
$PY -m tracegaze_analyst support-package --trace $E/attention_trace \
    --config tracegaze_analyst/examples/aea_loc4_script3_seq4_rec1.json --out $E/package

# 2) one Muse request (needs MODEL_API_KEY; --dry-run writes the request only)
$PY -m tracegaze_analyst muse-findings --package $E/package --out $E/muse_run

# 3) checks: references, copied numbers, prose numbers, language gates,
#    re-derivation of every package fact from the trace CSVs; tamper test
$PY -m tracegaze_analyst check-findings --package $E/package --run $E/muse_run
$PY -m tracegaze_analyst tamper-check  --package $E/package --run $E/muse_run

# 4) report (videos symlinked from the original artifacts, never copied)
$PY -m tracegaze_analyst support-report --example $E
$PY -m tracegaze_analyst serve-report  --example $E --port 8765
#   -> http://127.0.0.1:8765/report/index.html   (#F2 deep-links a finding)
```

### What is deterministic and what is Muse

| Deterministic (TraceGaze code) | Muse (`muse-spark-1.3`, Meta Model API) |
|---|---|
| episodes, transitions, revisits, durations (attention_trace) | which patterns answer the question |
| per-object, per-interval and per-10-s-phase totals as citable facts | what is visible in the supplied frames |
| candidate intervals by fixed rules | possible interpretations + alternatives |
| boundary proximity: other targets within 25 px of the gaze pixel | candidate support opportunities, or none |
| frame selection and extraction | remaining questions |

Muse may only copy numbers; `check-findings` rejects a copied value that
differs from the package and any number in prose that is not traceable to
the finding's cited facts, intervals, frames or episodes.

### Design decisions

- **Candidate moments are chosen by rule, not by the model**: the two
  densest 3-s switching windows, the two longest tracked-object episodes
  (±1 s), and the longest no-hit stretch (≥0.5 s); whole episodes, no
  overlaps. The rule text travels with each interval.
- **Ambiguity is measured, never resolved**: assignments are fusion's
  `primary_object_id` unchanged. A sample is *boundary-near* when another
  accepted mask (or, for on-object samples, the no-hit region) lies within
  25 px of the gaze pixel — the project's `--nearest-within-px 25`
  convention, explicitly not a gaze-accuracy bound.
- **Evidence vs interpretation**: each finding has an `evidence_type`
  (measured pattern / visible observation / possible interpretation);
  construct words (confusion, memory, difficulty, searching, intention,
  need for help …) are rejected outside `interpretation`, and any
  interpretation needs an alternative explanation and medium/high
  uncertainty. Personal-attribute words (age, gender pronouns …) are
  rejected in evidence text and interpretation.
- **"references_verified" is not "true"**: it means citations exist and
  numbers match. Interpretations are judged in `muse_run/manual_review.json`.
- **Data sent**: question, explicitly supplied context, the package without
  local paths/hashes, and 12 JPEG frames (1024 px). No credentials are
  stored; `request_redacted.json` replaces image bytes by their sha256.
- **Video timeline**: both MP4s start at PTS 0.025 s; the package records
  `video_first_frame_pts_s` and the report adds it when seeking, so a frame
  link shows exactly that frame (checked by pixel comparison).
