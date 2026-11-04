"""Module specifications, read from the authoritative launch config.

The launch commands are NOT guessed: they come from `.claude/launch.json`
(the same configurations used to run each module by hand, matching each
module README). If an entry is missing the workbench says so instead of
inventing an interpreter or command line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_JSON = PROJECT_ROOT / ".claude" / "launch.json"
WORKBENCH_PORT = 8000


class WorkbenchConfigError(RuntimeError):
    pass


@dataclass
class ModuleSpec:
    key: str                 # stable id, also the module's health identity
    title: str
    stage: int
    launch_name: str         # entry name in .claude/launch.json
    port: int
    executable: str
    args: list
    cwd: Path
    description: str
    input_desc: str
    output_desc: str
    # slot in the workbench state file this stage's OUTPUT fills
    output_slot: Optional[str] = None
    # slot providing this stage's INPUT (shown on the card)
    input_slot: Optional[str] = None
    # how to read the module's own /api/inspect verdict for a handoff path
    validator: str = "none"
    env: dict = field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    @property
    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/api/health"

    def inspect_url(self, path: str) -> str:
        from urllib.parse import quote
        return (f"http://127.0.0.1:{self.port}/api/inspect"
                f"?path={quote(str(path), safe='')}")

    def cmd(self) -> list:
        return [self.executable, *self.args]


# static, human-facing stage descriptions (the workbench never re-derives
# module behavior — it only presents it)
STAGES = {
    "neon_trim": {
        "title": "1 · Trim recording",
        "stage": 1,
        "launch_name": "neon-trim",
        "description": "Frame-accurate synchronized trimming of a raw "
                       "Pupil Labs NEON recording (timestamp-authoritative, "
                       "atomic validated export).",
        "input_desc": "raw NEON recording folder",
        "output_desc": "trimmed recording artifact",
        "output_slot": "trim_output",
        "validator": "recording",
    },
    "object_tracking": {
        "title": "2 · Track objects",
        "stage": 2,
        "launch_name": "object-tracking",
        "description": "Select, name, mask, track, review and publish "
                       "persistent objects with SAM3 point prompts "
                       "(GPU worker starts explicitly inside the module).",
        "input_desc": "trimmed recording artifact",
        "output_desc": "_tracked_objects artifact",
        "input_slot": "trim_output",
        "output_slot": "tracked_output",
        "validator": "trackable",
    },
    "gaze_object_fusion": {
        "title": "3 · Fuse gaze",
        "stage": 3,
        "launch_name": "gaze-fusion",
        "description": "Fuse NEON gaze samples with tracked object masks "
                       "and render the demonstration video.",
        "input_desc": "_tracked_objects artifact",
        "output_desc": "gaze_samples.csv / .parquet, fusion manifest, "
                       "demonstration video",
        "input_slot": "tracked_output",
        "output_slot": "fusion_output",
        "validator": "fusable",
    },
}

ORDER = ["neon_trim", "object_tracking", "gaze_object_fusion"]

# stage -> the next stage its "Continue" button targets
NEXT_STAGE = {"neon_trim": "object_tracking",
              "object_tracking": "gaze_object_fusion",
              "gaze_object_fusion": "home"}


def load_specs(launch_json: Path = None,
               project_root: Path = None) -> dict:
    """Build ModuleSpecs from the authoritative launch configuration."""
    launch_json = Path(launch_json or LAUNCH_JSON)
    root = Path(project_root or PROJECT_ROOT)
    if not launch_json.is_file():
        raise WorkbenchConfigError(
            f"launch configuration not found: {launch_json} — the workbench "
            f"reads module commands from it instead of guessing")
    try:
        data = json.loads(launch_json.read_text())
    except json.JSONDecodeError as e:
        raise WorkbenchConfigError(f"{launch_json}: invalid JSON ({e})")
    by_name = {c.get("name"): c for c in data.get("configurations", [])}

    specs = {}
    for key in ORDER:
        meta = STAGES[key]
        conf = by_name.get(meta["launch_name"])
        if conf is None:
            raise WorkbenchConfigError(
                f"{launch_json}: no configuration named "
                f"{meta['launch_name']!r} for stage {meta['title']!r}")
        exe = conf.get("runtimeExecutable")
        args = list(conf.get("runtimeArgs") or [])
        port = conf.get("port")
        if not exe or not args or not port:
            raise WorkbenchConfigError(
                f"{launch_json}: configuration {meta['launch_name']!r} must "
                f"define runtimeExecutable, runtimeArgs and port")
        specs[key] = ModuleSpec(
            key=key, title=meta["title"], stage=meta["stage"],
            launch_name=meta["launch_name"], port=int(port),
            executable=exe, args=args, cwd=root,
            description=meta["description"],
            input_desc=meta["input_desc"], output_desc=meta["output_desc"],
            output_slot=meta.get("output_slot"),
            input_slot=meta.get("input_slot"),
            validator=meta["validator"])
    return specs


def verdict(validator: str, payload: dict) -> tuple:
    """Interpret a module's OWN /api/inspect response for a handoff path.

    Returns (ok, detail). The workbench never re-implements intake or
    manifest validation — it asks the destination module and reports what
    the module said.
    """
    if not isinstance(payload, dict):
        return False, "module returned no inspection result"
    if validator == "recording":                       # neon_trim
        if payload.get("is_recording"):
            return True, "NEON recording accepted by neon_trim"
        return False, ("neon_trim rejected the folder: "
                       + "; ".join(payload.get("problems")
                                   or ["not a NEON recording"]))
    if validator == "trackable":                       # object_tracking
        kind = payload.get("kind")
        label = payload.get("kind_label") or kind
        if kind and kind != "tracked_objects":
            return True, f"object_tracking accepted the folder: {label}"
        if kind == "tracked_objects":
            return False, ("object_tracking does not re-open tracked-object "
                           "artifacts as inputs")
        return False, "object_tracking: incomplete or unsupported folder"
    if validator == "fusable":                         # gaze_object_fusion
        if payload.get("ok"):
            return True, "gaze_object_fusion accepted the tracked artifact"
        return False, ("gaze_object_fusion rejected the artifact: "
                       + "; ".join(payload.get("problems") or ["unknown"]))
    return True, "no destination validation configured"
