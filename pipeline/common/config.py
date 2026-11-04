"""Shared configuration helpers for pipeline servers.

Each stage keeps its own env prefix (NEON_TRIM_*, OBJTRACK_*) but shares the
same defaults and semantics:

| setting       | default                          | override               |
|---------------|----------------------------------|------------------------|
| browse roots  | <project>/neon-data              | <PREFIX>_ROOTS (':'-sep) and/or --root |
| host          | 127.0.0.1                        | --host (+ --allow-lan) |
| port          | per-stage default                | <PREFIX>_PORT / --port |
| state dir     | ~/.cache/<stage>                 | <PREFIX>_STATE_DIR     |

Browse roots gate EVERY server-side path operation; paths are canonicalized
(symlinks resolved) before the containment check.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "neon-data"
DEFAULT_HOST = "127.0.0.1"


@dataclass
class StageConfig:
    stage: str                                     # e.g. "object_tracking"
    port: int
    browse_roots: list = field(default_factory=list)
    host: str = DEFAULT_HOST
    state_dir: Path = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.state_dir is None:
            self.state_dir = Path.home() / ".cache" / self.stage
        resolved = []
        for r in self.browse_roots:
            p = Path(r).expanduser().resolve()
            if p not in resolved:
                resolved.append(p)
        self.browse_roots = resolved


def load_stage_config(stage: str, env_prefix: str, default_port: int,
                      extra_roots: Optional[list] = None,
                      port: Optional[int] = None,
                      host: Optional[str] = None) -> StageConfig:
    roots = []
    env_roots = os.environ.get(f"{env_prefix}_ROOTS", "")
    if env_roots:
        roots += [Path(p) for p in env_roots.split(":") if p.strip()]
    if extra_roots:
        roots += [Path(p) for p in extra_roots]
    if not roots:
        roots = [DEFAULT_DATA_ROOT]
    env_port = os.environ.get(f"{env_prefix}_PORT")
    state_env = os.environ.get(f"{env_prefix}_STATE_DIR")
    cfg = StageConfig(
        stage=stage,
        browse_roots=roots,
        host=host or DEFAULT_HOST,
        port=port or (int(env_port) if env_port else default_port),
        state_dir=Path(state_env) if state_env else None,
    )
    return cfg
