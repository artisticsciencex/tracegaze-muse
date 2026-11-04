"""gaze_object_fusion — fuse NEON gaze samples with tracked object masks.

Stage 3 of the gaze pipeline. Consumes a published `tracked_objects`
artifact (and, through its manifest, the parent trimmed NEON recording) and
produces a `gaze_object_fusion` artifact: one row per gaze sample with an
exact timestamp-defined frame assignment, a deterministic ordered object
hit list, and a time-faithful demonstration video.

The frame-assignment rule is timestamp-defined frame cells (midpoints
between adjacent world timestamps; trim-interval boundaries at the ends) —
never a fixed global frame-distance threshold. See `timecells.py`.
"""

__version__ = "0.1.0"
