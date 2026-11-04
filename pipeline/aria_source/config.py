"""Configuration for the Aria adapter — interpreter, cache root, roots.

Everything here is configurable; no recording id or dataset path is baked
into application logic.
"""

from __future__ import annotations

import os
from pathlib import Path

# dedicated interpreter that has projectaria_tools installed (never the
# NEON GUI env, never the SAM3 env)
DEFAULT_ARIA_PYTHON = Path.home() / ".venvs" / "aria" / "bin" / "python"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "aria_source"

ENV_PYTHON = "ARIA_PYTHON"
ENV_CACHE = "ARIA_CACHE_DIR"


def aria_python() -> Path:
    return Path(os.environ.get(ENV_PYTHON) or DEFAULT_ARIA_PYTHON)


def cache_dir() -> Path:
    return Path(os.environ.get(ENV_CACHE) or DEFAULT_CACHE_DIR)


def helper_path() -> Path:
    return Path(__file__).resolve().parent / "helper_cli.py"


def aria_env_available() -> bool:
    return aria_python().is_file()
