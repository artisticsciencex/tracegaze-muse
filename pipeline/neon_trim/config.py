"""Configuration for neon_trim (documented defaults, env overrides).

Nothing here is FastAPI-specific; the same config drives GUI, library, and
future CLI/orchestrator use.

| setting        | default                                     | override |
|----------------|---------------------------------------------|----------|
| browse roots   | <project>/neon-data (see DEFAULT_ROOT)      | env NEON_TRIM_ROOTS (":"-separated) and/or --root CLI flags |
| host           | 127.0.0.1                                   | --host (non-loopback needs --allow-lan) |
| port           | 8001                                        | env NEON_TRIM_PORT or --port |
| state dir      | ~/.cache/neon_trim (recents, last out dir)  | env NEON_TRIM_STATE_DIR |

Browse roots gate EVERY path the server will touch: directory listing,
recording inspection/loading, video streaming, and output creation. Paths are
canonicalized (symlinks resolved) before the containment check.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# The project's NEON data folder. This is a *default*, not a hard-coded
# restriction — override or extend with NEON_TRIM_ROOTS / --root.
DEFAULT_ROOT = Path(__file__).resolve().parent.parent.parent / "neon-data"
DEFAULT_PORT = 8001
DEFAULT_HOST = "127.0.0.1"


@dataclass
class AppConfig:
    browse_roots: list = field(default_factory=list)   # list[Path], resolved
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    state_dir: Path = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.state_dir is None:
            self.state_dir = Path(
                os.environ.get("NEON_TRIM_STATE_DIR",
                               Path.home() / ".cache" / "neon_trim"))
        # canonicalize roots once; drop non-directories with a note
        resolved = []
        for r in self.browse_roots:
            p = Path(r).expanduser().resolve()
            if p not in resolved:
                resolved.append(p)
        self.browse_roots = resolved


def load_config(extra_roots: Optional[list] = None,
                port: Optional[int] = None,
                host: Optional[str] = None) -> AppConfig:
    """Build config from documented defaults + environment + explicit args."""
    roots = []
    env_roots = os.environ.get("NEON_TRIM_ROOTS", "")
    if env_roots:
        roots += [Path(p) for p in env_roots.split(":") if p.strip()]
    if extra_roots:
        roots += [Path(p) for p in extra_roots]
    if not roots:
        roots = [DEFAULT_ROOT]
    env_port = os.environ.get("NEON_TRIM_PORT")
    return AppConfig(
        browse_roots=roots,
        host=host or DEFAULT_HOST,
        port=port or (int(env_port) if env_port else DEFAULT_PORT),
    )
