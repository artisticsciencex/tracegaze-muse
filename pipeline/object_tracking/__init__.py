"""object_tracking — GUI stage for selecting, naming, masking, tracking,
reviewing, and publishing persistent objects from first-person NEON
recordings.

Modules:
    intake       input-type detection + canonical RecordingBundle (4 types)
    native_prep  native NEON recording -> canonical prepared artifact
                 (official pupil-labs-neon-recording library; no hand decoding)
    annotations  open-vocabulary object model: immutable IDs, names, points,
                 deterministic colors, revision history, review status
    tracking     SAM3 execution engine: subprocess worker, single-frame mask
                 preview, bidirectional propagation, per-object rerun
    artifact     tracked_objects artifact contract: publish + loader
    server       FastAPI app (thin HTTP layer over the services above)
"""

__version__ = "0.1.0"
