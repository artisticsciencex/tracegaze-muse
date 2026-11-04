"""workbench — NEON Gaze–Object pipeline launcher and guided workflow.

A thin local orchestrator (port 8000) that starts, reuses, and links the
three verified pipeline modules — neon_trim (8001), object_tracking
(8002), gaze_object_fusion (8003) — and hands validated artifacts from
one stage to the next.

Process isolation is the point: this package imports ONLY FastAPI and the
standard library (plus the pure-stdlib `common.config` / `common.fsbrowse`
helpers). It never imports the module servers or any computation library
(torch, sam3, pandas, pyarrow) — modules run as separate child processes,
so a crash in a module, in SAM3, or in Arrow cannot take the workbench
down with it.
"""

__version__ = "0.1.0"
