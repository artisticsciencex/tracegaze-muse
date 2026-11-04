# workbench — NEON Gaze–Object pipeline launcher

One local page that runs the whole verified pipeline: **trim → track →
fuse**. No commands, no ports to remember, no extra browser tabs.

```bash
~/.venvs/neon-trim/bin/python pipeline/workbench/server.py
#   → http://127.0.0.1:8000        (or the .claude/launch.json "workbench" config)
~/.venvs/neon-trim/bin/python -m pytest pipeline/workbench/tests -q
```

| stage | module | port | input → output |
|---|---|---|---|
| 1 Trim recording | `neon_trim` | 8001 | raw NEON recording → trimmed artifact |
| 2 Track objects | `object_tracking` | 8002 | trimmed artifact → `_tracked_objects` |
| 3 Fuse gaze | `gaze_object_fusion` | 8003 | `_tracked_objects` → gaze CSV/Parquet + manifest + demo video |
| — | workbench | 8000 | orchestration only |

Each card shows live status (`stopped` / `starting` / `ready` / `busy` /
`failed` / `port_conflict`), the module description, its expected input
and output, the latest known artifact for both, and the failure detail if
a start goes wrong. Every module page carries a **← Pipeline Home**
control, and all navigation happens in the same tab.

## Process architecture

The workbench is a launcher, not a library host. It imports FastAPI and
the standard library only — never a module server, never `torch`, `sam3`,
`pandas` or `pyarrow` (enforced by a pristine-interpreter test). The three
modules run as **separate child processes**, so a crash in a module, in
SAM3, or in Arrow cannot reach this process.

Launch commands are read from `.claude/launch.json` — the same
configurations used to start each module by hand (matching each module
README). Nothing about interpreters or command lines is guessed; a
missing entry is reported as a configuration error.

`Start and open` is deliberately conservative:

1. probe `GET /api/health` on the configured port
2. if it answers with the expected module identity → **reuse it** (no
   second process is ever launched, whoever started it)
3. if it answers with something else → **port conflict**: the workbench
   reports it and never kills a process it does not own
4. otherwise launch the configured command in its own process group and
   poll health until ready or a bounded timeout (default 40 s), reporting
   the child's output tail if it exits or hangs
5. navigate the current tab to the module

Only workbench-owned children may be stopped, and a module reporting
`busy` (an export, a tracking run, a fusion build) requires an explicit
confirmation first, so an active tracking job is never terminated
silently. On workbench shutdown, owned children are terminated
gracefully; anything started elsewhere is left alone.

Starting the tracking server does **not** load SAM3 or start the GPU
worker — that remains an explicit action inside the tracking module.

A bounded per-module diagnostic buffer (200 lines) is kept for processes
the workbench started, viewable under "Show launch diagnostics". It holds
the child's own stdout/stderr only; no environment or credential data is
collected or exposed, and the filesystem is never listed here.

## Artifact handoff

Handoff travels through the workbench as a URL-encoded query parameter,
so a stage never asks for a folder the previous stage already produced:

```
<module page>  "Continue to …"  →  http://127.0.0.1:8000/go?to=<stage>&handoff=<urlencoded path>
```

`/go` then:

1. **authorizes** the path against the workbench browse roots
   (`PathAuthorizer`: canonicalize, resolve symlinks, containment) — a
   query parameter can never bypass this; unauthorized paths get 403
2. starts or reuses the destination module
3. asks the **destination module's own** `GET /api/inspect` to validate
   the folder — neon_trim's recording intake, object_tracking's folder
   classification, gaze_object_fusion's tracked-artifact validation. The
   workbench re-implements none of it; it relays the module's verdict and
   refuses the handoff with the module's own problem list if invalid
4. remembers the accepted path in `~/.cache/workbench/workbench_state.json`
5. redirects the tab to the module with `?handoff=…&home=…`

The module loads the handoff through its ordinary load/inspect route, so
path authorization and manifest validation apply exactly as for a
manually chosen folder. Opening any module directly, without a handoff,
keeps working with its existing folder chooser.

Fusion completion returns to the workbench with its output recorded; the
stage-3 card's **Open latest result** then plays the demonstration video
through the fusion module's authorized `/api/demo` route.

## Limitations

- roots: the workbench authorizes against `WORKBENCH_ROOTS` (default
  `<project>/neon-data`). If you point a module at different roots, give
  the workbench the same ones or handoff will be refused.
- the workbench does not proxy module pages; it navigates to them, so
  each module keeps its own origin and port.
- `busy` is only as good as what a module exposes in `/api/health`
  (all three expose it today).
- artifacts are remembered per workbench state file, not per user
  session; a second browser window shares the same state.
