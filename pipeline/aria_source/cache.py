"""Aria working cache: selective extraction, validation, atomic renames.

Project Aria Tools needs a real VRS file on disk, and MPS eye-gaze CSVs
may live in a full MPS archive or a gaze-only archive. Required inputs are
materialized into a CONFIGURABLE cache (`ARIA_CACHE_DIR`, default
`~/.cache/aria_source`).

Rules:
- source files are only ever READ (never modified, moved or deleted)
- only the files this stage needs are extracted: `recording.vrs`, the two
  eye-gaze CSVs and `metadata.json`. Semidense point clouds, SLAM
  observations, trajectories and calibration jsonl are never extracted.
- a cache entry is built in a temporary directory and renamed into place
  only after every declared file is complete (atomic publication)
- a cache entry whose manifest is missing/unparseable, whose declared
  files are missing or the wrong size, or whose sources changed
  (size/mtime) is treated as stale and rebuilt
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

from . import __version__
from .detect import PKG_META, PKG_MPS, PKG_VRS, AriaPackage

CACHE_MANIFEST = "cache_manifest.json"
CACHE_SCHEMA = "1.0"
VRS_NAME = "recording.vrs"
PERSONALIZED = "personalized_eye_gaze.csv"
GENERAL = "general_eye_gaze.csv"
METADATA = "metadata.json"

# Preferred full-MPS locations. Gaze-only AEA archives put the same
# basenames at the ZIP root; `_gaze_member` accepts both without treating
# unrelated CSVs as gaze.
MPS_MEMBERS = {
    PERSONALIZED: "mps/eye_gaze/personalized_eye_gaze.csv",
    GENERAL: "mps/eye_gaze/general_eye_gaze.csv",
}
VRS_MEMBER = "recording.vrs"


def _gaze_member(names, out_name: str):
    preferred = MPS_MEMBERS[out_name]
    if preferred in names:
        return preferred
    matches = sorted(n for n in names if n.rsplit("/", 1)[-1] == out_name)
    return matches[0] if len(matches) == 1 else None


class CacheError(RuntimeError):
    pass


class CacheCancelled(CacheError):
    pass


def sha256_file(path: Path, chunk: int = 1 << 22,
                progress: Optional[Callable[[float], None]] = None) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size or 1
    done = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            done += len(b)
            if progress:
                progress(done / total)
    return h.hexdigest()


def _src_identity(p: Path) -> dict:
    st = p.stat()
    return {"name": p.name, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


class RecordingCache:
    """Cache entry for ONE Aria recording."""

    def __init__(self, recording_id: str, cache_root: Path):
        self.recording_id = recording_id
        self.root = Path(cache_root)
        self.dir = self.root / recording_id

    # ------------------------------------------------------------ paths
    @property
    def vrs(self) -> Path:
        return self.dir / VRS_NAME

    @property
    def manifest_path(self) -> Path:
        return self.dir / CACHE_MANIFEST

    def gaze_csv(self, model: str) -> Path:
        return self.dir / (PERSONALIZED if model == "personalized"
                           else GENERAL)

    # ------------------------------------------------------- validation
    def read_manifest(self) -> Optional[dict]:
        try:
            m = json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return m if isinstance(m, dict) else None

    def status(self, pkg: AriaPackage) -> dict:
        """Is this cache entry usable for `pkg`? Never raises."""
        m = self.read_manifest()
        if m is None:
            return {"valid": False, "reason": "no readable cache manifest",
                    "state": "absent" if not self.dir.exists() else "corrupt"}
        if m.get("schema_version") != CACHE_SCHEMA:
            return {"valid": False, "state": "stale",
                    "reason": f"cache schema {m.get('schema_version')} != "
                              f"{CACHE_SCHEMA}"}
        for name, info in (m.get("files") or {}).items():
            p = self.dir / name
            if not p.is_file():
                return {"valid": False, "state": "corrupt",
                        "reason": f"cached file missing: {name}"}
            if p.stat().st_size != info.get("bytes"):
                return {"valid": False, "state": "corrupt",
                        "reason": f"cached file truncated/modified: {name}"}
        for key, ident in (m.get("sources") or {}).items():
            src = pkg.files.get(key)
            if src is None or not src.is_file():
                return {"valid": False, "state": "stale",
                        "reason": f"source archive missing: {key}"}
            now = _src_identity(src)
            if (now["bytes"] != ident.get("bytes")
                    or now["mtime_ns"] != ident.get("mtime_ns")):
                return {"valid": False, "state": "stale",
                        "reason": f"source archive changed: {src.name}"}
        return {"valid": True, "state": "ready", "reason": "cache reusable",
                "manifest": m}

    # ---------------------------------------------------------- populate
    def populate(self, pkg: AriaPackage, *,
                 progress: Optional[Callable[[str, float], None]] = None,
                 cancel: Optional[Callable[[], bool]] = None,
                 hash_sources: bool = True) -> dict:
        """(Re)build this cache entry atomically. Returns its manifest."""
        def report(phase, frac):
            if progress:
                progress(phase, frac)

        def check_cancel():
            if cancel and cancel():
                raise CacheCancelled("cache preparation cancelled")

        for key in (PKG_VRS, PKG_MPS):
            if key not in pkg.files:
                raise CacheError(f"missing required Aria source: {key}")

        tmp = self.dir.with_name(self.dir.name
                                 + f".tmp_{int(time.time() * 1000)}")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            sources = {}
            hashes = {}
            source_items = sorted(pkg.files.items())
            source_count = max(len(source_items), 1)
            for key, source in source_items:
                sources[key] = _src_identity(source)
            if hash_sources:
                for i, (key, source) in enumerate(source_items):
                    check_cancel()
                    hashes[key] = sha256_file(
                        source, progress=lambda f, i=i, k=key: report(
                            f"hashing {k}",
                            0.02 + 0.18 * (i + f) / source_count))

            selected = []
            check_cancel()
            vrs_source = pkg.files[PKG_VRS]
            if pkg.file_formats.get(PKG_VRS) == "vrs":
                report("copying direct recording.vrs", 0.25)
                with open(vrs_source, "rb") as fsrc, \
                        open(tmp / VRS_NAME, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst, length=1 << 22)
                selected.append(f"source:{vrs_source.name}")
            else:
                report("extracting recording.vrs", 0.25)
                with zipfile.ZipFile(vrs_source) as z:
                    names = z.namelist()
                    member = (VRS_MEMBER if VRS_MEMBER in names else
                              next((n for n in names if n.endswith(".vrs")),
                                   None))
                    if member is None:
                        raise CacheError(
                            "no .vrs member inside the VRS archive")
                    with z.open(member) as fsrc, \
                            open(tmp / VRS_NAME, "wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst, length=1 << 22)
                    selected.append(member)
            check_cancel()

            report("extracting eye-gaze CSVs", 0.80)
            gaze_selected = []
            with zipfile.ZipFile(pkg.files[PKG_MPS]) as z:
                names = set(z.namelist())
                for out_name in MPS_MEMBERS:
                    member = _gaze_member(names, out_name)
                    if member is None:
                        continue
                    with z.open(member) as fsrc, \
                            open(tmp / out_name, "wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst, length=1 << 20)
                    gaze_selected.append(member)
                    selected.append(member)
            if not gaze_selected:
                raise CacheError(
                    "eye-gaze archive contains neither general_eye_gaze.csv "
                    "nor personalized_eye_gaze.csv")
            check_cancel()

            if PKG_META in pkg.files:
                report("extracting metadata", 0.92)
                try:
                    with zipfile.ZipFile(pkg.files[PKG_META]) as z:
                        member = next((n for n in z.namelist()
                                       if n.endswith("metadata.json")), None)
                        if member:
                            (tmp / METADATA).write_bytes(z.read(member))
                            selected.append(member)
                except zipfile.BadZipFile:
                    pass                        # metadata is optional

            # payload files only — the manifest must never record its own
            # size (writing the size into the file changes the size)
            files = {p.name: {"bytes": p.stat().st_size}
                     for p in sorted(tmp.iterdir())
                     if p.is_file() and p.name != CACHE_MANIFEST}
            manifest = {
                "schema_version": CACHE_SCHEMA,
                "recording_id": self.recording_id,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
                "tool": {"name": "aria_source", "version": __version__},
                "source_dir": str(pkg.path),
                "source_layout": pkg.layout,
                "source_formats": dict(pkg.file_formats),
                "sources": sources,
                "source_sha256": hashes,
                "selected_members": sorted(selected),
                "omitted_note":
                    "semidense point clouds, SLAM observations, "
                    "trajectories, online calibration and hand tracking are "
                    "deliberately NOT extracted — this stage does not use "
                    "them",
                "files": files,
            }
            (tmp / CACHE_MANIFEST).write_text(json.dumps(manifest, indent=2))

            report("publishing cache", 0.97)
            shutil.rmtree(self.dir, ignore_errors=True)
            tmp.replace(self.dir)                # atomic rename-on-success
            report("cache ready", 1.0)
            return manifest
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)   # never a partial cache
            raise

    def ensure(self, pkg: AriaPackage, **kw) -> dict:
        """Reuse a valid cache entry, otherwise rebuild it."""
        st = self.status(pkg)
        if st["valid"]:
            return {"reused": True, **st}
        m = self.populate(pkg, **kw)
        return {"reused": False, "state": "ready", "manifest": m,
                "reason": st["reason"]}
