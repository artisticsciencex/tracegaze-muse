"""Generic artifact-contract helpers shared by artifact types.

Every pipeline artifact (trimmed_neon_recording, prepared_neon_recording,
tracked_objects, ...) follows the same conventions established by neon_trim:

* a versioned JSON manifest at a fixed name inside the artifact directory
* `files` entries RELATIVE to the artifact directory (movable folder),
  validated against path/symlink escape on load
* absolute paths only under clearly-labelled source provenance
* atomic publication: stage into a hidden temp dir next to the destination,
  validate, then rename; failures leave no partial output
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Callable, Optional


class ArtifactError(RuntimeError):
    pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_manifest(path, manifest_name: str) -> tuple:
    """(artifact_dir, manifest dict) from a dir or manifest file path."""
    p = Path(path).expanduser()
    manifest_path = p if p.is_file() else p / manifest_name
    if not manifest_path.is_file():
        raise ArtifactError(f"no {manifest_name} found at {p}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except ValueError as e:
        raise ArtifactError(f"{manifest_path}: invalid JSON: {e}") from None
    except OSError as e:
        raise ArtifactError(f"{manifest_path}: cannot read: {e}") from None
    return manifest_path.parent, manifest


def require(manifest: dict, where: str, *, artifact_type: str,
            schema_major: str) -> None:
    atype = manifest.get("artifact_type")
    if atype != artifact_type:
        raise ArtifactError(f"{where}: artifact_type is {atype!r}, expected "
                            f"{artifact_type!r}")
    sv = str(manifest.get("schema_version", ""))
    if not sv.startswith(schema_major + "."):
        raise ArtifactError(f"{where}: unsupported schema_version {sv!r} "
                            f"(supported: {schema_major}.x)")
    if not manifest.get("artifact_id"):
        raise ArtifactError(f"{where}: missing artifact_id")


def resolve_declared_file(art_dir: Path, rel, *, key: str, where: str) -> Path:
    """Resolve a manifest-declared relative file, rejecting absolute paths,
    '..' traversal, and symlink escape out of the artifact directory."""
    if not rel or not isinstance(rel, str):
        raise ArtifactError(f"{where}: files.{key} missing or not a string "
                            f"(got {type(rel).__name__})")
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ArtifactError(f"{where}: files.{key} must be a plain relative "
                            f"path inside the artifact directory, got {rel!r}")
    fp = art_dir / rel_path
    art_real = os.path.realpath(art_dir)
    fp_real = os.path.realpath(fp)
    if fp_real != art_real and not fp_real.startswith(art_real + os.sep):
        raise ArtifactError(f"{where}: files.{key} resolves outside the "
                            f"artifact directory (symlink escape): {rel!r}")
    if not fp.is_file():
        raise ArtifactError(f"artifact incomplete: {rel!r} not found in "
                            f"{art_dir}")
    return fp


class AtomicPublisher:
    """Stage → validate → atomically publish an artifact directory.

    Mirrors the reviewed neon_trim.export pattern: claim the destination
    directory up front (fast conflict detection), stage into a hidden temp
    dir on the same filesystem, and rename over the claimed (empty) dir only
    after the caller's validations pass. Failure/cancel leaves no partial
    output and undoes any created parent chain.
    """

    def __init__(self, out_dir: Path, *, source_dir: Optional[Path] = None,
                 path_guard: Optional[Callable[[Path], bool]] = None,
                 tmp_prefix: str = ".pipeline_artifact_tmp_"):
        out_dir = Path(out_dir).resolve()
        if source_dir is not None:
            src = Path(source_dir).resolve()
            if out_dir == src or src in out_dir.parents:
                raise ArtifactError("output directory must be outside the "
                                    "source directory")
        self._created_parents = []
        p = out_dir.parent
        while not p.exists():
            self._created_parents.append(p)          # deepest first
            p = p.parent
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            parent_real = out_dir.parent.resolve(strict=True)
        except OSError as e:
            raise ArtifactError(f"cannot resolve output parent: {e}") from None
        out_dir = parent_real / out_dir.name
        if path_guard is not None and not (path_guard(parent_real)
                                           and path_guard(out_dir)):
            raise ArtifactError("output directory escaped the authorized "
                                "location (symlinked parent?) — refused")
        if source_dir is not None:
            src = Path(source_dir).resolve()
            if out_dir == src or src in out_dir.parents:
                raise ArtifactError("output directory must be outside the "
                                    "source directory")
        try:
            out_dir.mkdir()                           # atomic claim
        except FileExistsError:
            raise ArtifactError(f"output directory already exists: {out_dir} "
                                f"— refusing to overwrite") from None
        # from here on the claim is held: undo it if anything else in
        # construction fails, so a failed init never blocks retries
        try:
            self.out_dir = out_dir
            self.tmp_dir = out_dir.parent \
                / f"{tmp_prefix}{uuid.uuid4().hex[:8]}"
            self.tmp_dir.mkdir()
            self._published = False
        except BaseException:
            try:
                out_dir.rmdir()
            except OSError:
                pass
            for p in self._created_parents:
                try:
                    p.rmdir()
                except OSError:
                    break
            raise

    def publish(self) -> Path:
        try:
            os.rename(self.tmp_dir, self.out_dir)     # atomic on POSIX
        except OSError as e:
            raise ArtifactError(
                f"could not publish to {self.out_dir} (was something written "
                f"into it during staging?): {e}") from None
        self._published = True
        return self.out_dir

    def cleanup(self) -> None:
        """Idempotent: removes staging leftovers; undoes claim on failure."""
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        if not self._published:
            try:
                self.out_dir.rmdir()
            except OSError:
                pass
            for p in self._created_parents:
                try:
                    p.rmdir()                          # only if empty
                except OSError:
                    break
