"""neon_trim.fsbrowse — compatibility wrapper over pipeline common.fsbrowse.

The canonical implementation was promoted to ``common.fsbrowse``; this module
preserves the original neon_trim API (including the recording-badge
``list_dir``) so existing imports and tests keep working unchanged.
"""

from __future__ import annotations

from common.fsbrowse import (  # noqa: F401  (re-exports)
    DirEntry,
    DirListing,
    PathAuthorizer,
    PathNotAllowed,
    RecentsStore,
)
from common.fsbrowse import list_dir as _list_dir

from . import intake


def _neon_classifier(p):
    return "neon_recording" if intake.looks_like_recording(p) else None


def list_dir(auth: PathAuthorizer, raw_path, classifier=None) -> DirListing:
    """One directory level with NEON-recording badges (original behavior).

    `classifier` may override the badge function — the server passes one
    that also recognizes Project Aria folders.
    """
    return _list_dir(auth, raw_path,
                     classifier=classifier or _neon_classifier)
