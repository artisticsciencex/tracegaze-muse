"""neon_trim — frame-accurate synchronized trimming of Pupil Labs NEON recordings.

Modules:
    core    pure-Python timestamp mapping + CSV trimming (stdlib only; reusable
            by gaze_fusion for gaze->frame index mapping)
    media   PyAV probing + ffmpeg frame-accurate cutting
    export  orchestrated, validated, atomic export
    server  FastAPI app serving the browser GUI (static/index.html)
"""

__version__ = "0.2.0"
