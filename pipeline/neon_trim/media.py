"""neon_trim.media — compatibility wrapper over pipeline common.media.

The canonical implementation was promoted to ``common.media``. Everything is
re-exported by name so ``neon_trim.media.X`` (including test monkeypatching of
``run_cut`` / ``decode_all_pts``) continues to work.
"""

from __future__ import annotations

from common.media import (  # noqa: F401  (re-exports)
    ENCODE_SETTINGS,
    VIDEO_EXTS,
    CutCancelled,
    CutError,
    VideoProbe,
    decode_all_pts,
    ffmpeg_exe,
    find_scene_video,
    probe_video,
    run_cut,
)
