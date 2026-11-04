"""Restricted server-side filesystem browsing (shared pipeline service).

Canonical implementation promoted from neon_trim (where it is fully tested;
neon_trim.fsbrowse re-exports from here). All FastAPI-free:

* :class:`PathAuthorizer` — canonicalizes any requested path (symlinks
  resolved) and enforces containment in the configured browse roots. EVERY
  server-side path operation (listing, inspection, loading, video streaming,
  output creation) must pass through :meth:`PathAuthorizer.resolve`.
* :func:`list_dir` — one directory level: folders first, pluggable per-entry
  `classifier` for input-type badges, distinct inaccessible/error states.
* :class:`RecentsStore` — small JSON state file remembering recently selected
  recordings and the last output parent directory.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


class PathNotAllowed(PermissionError):
    """Requested path escapes the configured browse roots."""


class PathAuthorizer:
    def __init__(self, roots):
        self.roots = [Path(r).expanduser().resolve() for r in roots]
        if not self.roots:
            raise ValueError("at least one browse root is required")

    def resolve(self, raw) -> Path:
        """Canonicalize `raw` (symlinks resolved, including for paths that do
        not fully exist yet) and return it iff inside a configured root.

        Raises PathNotAllowed otherwise. Never leaks whether an outside path
        exists — the same error is raised either way.
        """
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            raise PathNotAllowed(f"path must be absolute: {raw!r}")
        # resolve() canonicalizes '..' and follows symlinks in every existing
        # component; non-existent trailing components are appended literally
        rp = p.resolve()
        for root in self.roots:
            if rp == root or root in rp.parents:
                return rp
        raise PathNotAllowed(
            f"path is outside the configured browse roots: {raw}")

    def is_allowed(self, raw) -> bool:
        try:
            self.resolve(raw)
            return True
        except PathNotAllowed:
            return False

    def root_for(self, resolved: Path) -> Optional[Path]:
        for root in self.roots:
            if resolved == root or root in resolved.parents:
                return root
        return None


# ---------------------------------------------------------------------------
# directory listing
# ---------------------------------------------------------------------------

@dataclass
class DirEntry:
    name: str
    path: str
    is_dir: bool
    is_recording: bool = False       # structural quick check only (legacy flag)
    kind: Optional[str] = None       # classifier badge, e.g. 'trimmed_artifact'
    inaccessible: bool = False


@dataclass
class DirListing:
    path: str
    root: str                        # the browse root this path lives under
    parent: Optional[str]            # None at a root boundary
    breadcrumbs: list = field(default_factory=list)  # [{name, path}], root first
    entries: list = field(default_factory=list)      # [DirEntry]
    error: Optional[str] = None      # 'not_a_directory' | 'inaccessible' | ...

    def as_dict(self) -> dict:
        return {
            "path": self.path, "root": self.root, "parent": self.parent,
            "breadcrumbs": self.breadcrumbs,
            "entries": [vars(e) for e in self.entries],
            "error": self.error,
        }


def list_dir(auth: PathAuthorizer, raw_path,
             classifier: Optional[Callable[[Path], Optional[str]]] = None
             ) -> DirListing:
    """List one authorized directory level (folders first, badge flags).

    `classifier(path) -> Optional[str]` supplies a cheap structural badge for
    each sub-folder (stat-level only — expensive probing belongs to the
    inspect-on-highlight step, not the listing).
    """
    rp = auth.resolve(raw_path)                    # raises PathNotAllowed
    root = auth.root_for(rp)
    parent = str(rp.parent) if rp != root else None

    crumbs = [{"name": root.name or str(root), "path": str(root)}]
    if rp != root:
        rel = rp.relative_to(root)
        acc = root
        for part in rel.parts:
            acc = acc / part
            crumbs.append({"name": part, "path": str(acc)})

    listing = DirListing(path=str(rp), root=str(root), parent=parent,
                         breadcrumbs=crumbs)
    if not rp.exists():
        listing.error = "not_found"
        return listing
    if not rp.is_dir():
        listing.error = "not_a_directory"
        return listing
    try:
        children = sorted(rp.iterdir(), key=lambda p: p.name.lower())
    except PermissionError:
        listing.error = "inaccessible"
        return listing

    dirs, files = [], []
    for c in children:
        if c.name.startswith("."):
            continue
        try:
            if c.is_dir():
                kind = classifier(c) if classifier is not None else None
                dirs.append(DirEntry(
                    name=c.name, path=str(c), is_dir=True,
                    is_recording=bool(kind), kind=kind))
            else:
                files.append(DirEntry(name=c.name, path=str(c), is_dir=False))
        except PermissionError:
            dirs.append(DirEntry(name=c.name, path=str(c), is_dir=True,
                                 inaccessible=True))
    listing.entries = dirs + files                 # folders first
    return listing


# ---------------------------------------------------------------------------
# recents / last-output-parent state
# ---------------------------------------------------------------------------

class RecentsStore:
    """Tiny JSON state file: recent recordings + last output parent."""

    MAX_RECENTS = 8

    def __init__(self, state_dir: Path):
        self.path = Path(state_dir) / "state.json"
        self._lock = threading.Lock()

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.path)

    def recent_recordings(self) -> list:
        return list(self._read().get("recent_recordings", []))

    def add_recording(self, path: str) -> None:
        with self._lock:
            d = self._read()
            rec = [p for p in d.get("recent_recordings", []) if p != str(path)]
            rec.insert(0, str(path))
            d["recent_recordings"] = rec[: self.MAX_RECENTS]
            self._write(d)

    def last_out_parent(self) -> Optional[str]:
        return self._read().get("last_out_parent")

    def set_last_out_parent(self, path: str) -> None:
        with self._lock:
            d = self._read()
            d["last_out_parent"] = str(path)
            self._write(d)
