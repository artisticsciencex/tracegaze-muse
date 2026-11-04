"""attention_trace — relational attention trace from a gaze-fusion artifact.

Stage 4 of the TraceGaze pipeline. It turns an already validated
`gaze_object_fusion` artifact into an auditable, object-relational
description of where attention went and how it moved:

    episodes      maximal supported runs of attention on one target
    transitions   the moves between consecutive supported episodes
    summary       per-object dwell, episode and revisit counts

Two rules define everything here:

* the AUTHORITATIVE `gaze_samples.parquet` table is the only evidence.
  The demonstration video's linearly interpolated cursor and the
  `demo_presentation.csv` sidecar are presentation-only and are never
  read by this stage.
* `__no_tracked_object__` means exactly "no accepted tracked mask was
  hit". It is never described as physical background and never treated
  as an unlabeled object.

All temporal arithmetic is exact integer nanoseconds; seconds appear only
as derived display values.
"""

__version__ = "0.1.0"

ATTENTION_ARTIFACT_TYPE = "attention_trace"
ATTENTION_SCHEMA_VERSION = "1.0"
ATTENTION_MANIFEST = "attention_trace_manifest.json"

#: explicit "no accepted tracked-mask hit" target (never "background")
NO_TRACKED_OBJECT = "__no_tracked_object__"
NO_TRACKED_OBJECT_NAME = "No accepted tracked-mask hit"
