"""NEON Gaze–Object Workbench — one local page that runs the pipeline.

    ~/.venvs/neon-trim/bin/python pipeline/workbench/server.py
        → http://127.0.0.1:8000

What this process does: read the authoritative launch configuration,
start/reuse the three module servers as CHILD PROCESSES, show their
health, and hand a validated artifact path from one stage to the next.

What this process deliberately never does: import a module server or any
computation library (torch, sam3, pandas, pyarrow), open extra browser
tabs, or touch a process it did not start.
"""

from __future__ import annotations

import html
import ipaddress
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

try:
    from . import __version__
    from . import config as wb_config
    from . import procman
except ImportError:                       # executed directly as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from workbench import __version__
    from workbench import config as wb_config
    from workbench import procman

from common import config as common_config
from common import fsbrowse

from fastapi import FastAPI, HTTPException
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse)
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
STATE_FILE = "workbench_state.json"
SLOTS = ("trim_output", "tracked_output", "fusion_output")


class StartBody(BaseModel):
    key: str


class StopBody(BaseModel):
    key: str
    force: bool = False


class StateBody(BaseModel):
    slot: str
    path: str


def _error_page(title: str, detail: str, *, home: str = "/",
                extra_html: str = "") -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><meta charset='utf-8'>"
        "<title>Workbench — problem</title>"
        "<style>body{background:#111418;color:#dfe5ec;font:15px/1.5 "
        "system-ui,sans-serif;margin:0;padding:40px}"
        ".box{max-width:760px;margin:0 auto;background:#191e24;border:1px "
        "solid #6e2c2c;border-radius:10px;padding:20px}"
        "h1{font-size:19px;margin:0 0 10px;color:#ffb4b4}"
        "pre{white-space:pre-wrap;color:#dfe5ec;background:#12161b;"
        "padding:12px;border-radius:8px}"
        "a{color:#7fb0f2}</style>"
        f"<div class='box'><h1>{html.escape(title)}</h1>"
        f"<pre>{html.escape(detail)}</pre>{extra_html}"
        f"<p><a href='{html.escape(home)}'>← Pipeline Home</a></p></div>",
        status_code=409)


def create_app(cfg: common_config.StageConfig,
               specs: dict = None,
               *, start_timeout: float = procman.START_TIMEOUT) -> FastAPI:
    auth = fsbrowse.PathAuthorizer(cfg.browse_roots)
    if specs is None:
        specs = wb_config.load_specs()
    pm = procman.ProcessManager(specs, start_timeout=start_timeout)
    state_path = Path(cfg.state_dir) / STATE_FILE

    @asynccontextmanager
    async def _lifespan(app):
        yield
        pm.shutdown()                      # only workbench-owned children

    app = FastAPI(title="workbench", lifespan=_lifespan)
    app.state.cfg = cfg
    app.state.pm = pm
    app.state.specs = specs

    loopback = False
    try:
        loopback = ipaddress.ip_address(cfg.host).is_loopback
    except ValueError:
        loopback = cfg.host in ("localhost",)
    if loopback:
        from starlette.middleware.trustedhost import TrustedHostMiddleware
        app.add_middleware(TrustedHostMiddleware,
                           allowed_hosts=["localhost", "127.0.0.1", "[::1]"])

    home_url = f"http://{cfg.host}:{cfg.port}/"

    # ----------------------------------------------------------- state
    def read_state() -> dict:
        try:
            data = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        # never surface a remembered path that is no longer authorized
        return {k: v for k, v in data.items()
                if k in SLOTS and isinstance(v, str) and auth.is_allowed(v)}

    def write_slot(slot: str, path: str) -> None:
        if slot not in SLOTS:
            return
        data = read_state()
        data[slot] = str(path)
        data["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime())
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(state_path)

    def authorized(raw: str) -> Path:
        try:
            return auth.resolve(raw)
        except fsbrowse.PathNotAllowed as e:
            raise HTTPException(403, str(e))

    # ----------------------------------------------------------- pages
    @app.get("/")
    def index():
        return FileResponse(APP_DIR / "static" / "index.html")

    @app.get("/api/health")
    def api_health():
        return {"ok": True, "module": "workbench", "version": __version__,
                "busy": False}

    @app.get("/api/config")
    def api_config():
        return {"roots": [str(r) for r in auth.roots],
                "version": __version__, "home": home_url,
                "port": cfg.port}

    # --------------------------------------------------------- modules
    def module_view(key: str) -> dict:
        spec = specs[key]
        st = pm.status(key)
        state = read_state()
        return {
            **st,
            "title": spec.title, "stage": spec.stage,
            "description": spec.description,
            "input_desc": spec.input_desc, "output_desc": spec.output_desc,
            "input_artifact": (state.get(spec.input_slot)
                               if spec.input_slot else None),
            "output_artifact": (state.get(spec.output_slot)
                                if spec.output_slot else None),
            "next_stage": wb_config.NEXT_STAGE.get(key),
            "launch_name": spec.launch_name,
            "command": " ".join(spec.cmd()),
        }

    def ordered_keys() -> list:
        # pipeline order first, then anything else configured
        return ([k for k in wb_config.ORDER if k in specs]
                + [k for k in specs if k not in wb_config.ORDER])

    @app.get("/api/modules")
    def api_modules():
        return {"modules": [module_view(k) for k in ordered_keys()],
                "state": read_state(), "home": home_url}

    @app.post("/api/start")
    def api_start(body: StartBody):
        if body.key not in specs:
            raise HTTPException(404, f"unknown module {body.key!r}")
        try:
            st = pm.start(body.key)
        except procman.PortConflict as e:
            raise HTTPException(409, str(e))
        except procman.StartFailed as e:
            raise HTTPException(502, str(e))
        return {**module_view(body.key), "reused": st.get("reused", False)}

    @app.post("/api/stop")
    def api_stop(body: StopBody):
        if body.key not in specs:
            raise HTTPException(404, f"unknown module {body.key!r}")
        try:
            pm.stop(body.key, force=body.force)
        except procman.NotOwned as e:
            raise HTTPException(409, str(e))
        except procman.ModuleBusy as e:
            raise HTTPException(409, str(e))
        return module_view(body.key)

    @app.get("/api/logs/{key}")
    def api_logs(key: str):
        if key not in specs:
            raise HTTPException(404, f"unknown module {key!r}")
        # bounded, workbench-authored diagnostics only
        return {"key": key, "lines": pm.logs(key)[-procman.LOG_LINES:]}

    @app.post("/api/state")
    def api_state(body: StateBody):
        if body.slot not in SLOTS:
            raise HTTPException(400, f"unknown slot {body.slot!r}")
        rp = authorized(body.path)
        write_slot(body.slot, str(rp))
        return read_state()

    @app.get("/api/state")
    def api_state_get():
        return read_state()

    # ---------------------------------------------------------- handoff
    @app.get("/go")
    def go(to: str, handoff: str = None, slot: str = None):
        """Guided navigation: validate a handoff path with the DESTINATION
        module's own intake/manifest validation, remember it, make sure the
        module runs, then navigate this tab to it."""
        if to == "home":
            if handoff:
                rp = authorized(handoff)          # authz is never bypassed
                write_slot(slot or "fusion_output", str(rp))
            return RedirectResponse("/", status_code=303)
        if to not in specs:
            raise HTTPException(404, f"unknown module {to!r}")
        spec = specs[to]

        rp = None
        if handoff:
            rp = authorized(handoff)              # 403 outside browse roots
            if not rp.exists():
                return _error_page(
                    "Handoff path does not exist",
                    f"{rp}\n\nThe artifact may have been moved or renamed. "
                    f"Open {spec.title} and choose a folder manually.",
                    home="/",
                    extra_html=f"<p><a href='{spec.url}'>Open "
                               f"{html.escape(spec.title)} without a "
                               f"handoff</a></p>")
        try:
            pm.start(to)
        except procman.PortConflict as e:
            return _error_page("Port conflict", str(e))
        except procman.StartFailed as e:
            return _error_page(f"Could not start {to}", str(e))

        if rp is not None:
            # the destination module validates the path with its own
            # intake/manifest logic — the workbench only relays the verdict
            payload, note = procman.probe_health(spec.inspect_url(str(rp)),
                                                 timeout=20.0)
            ok, detail = wb_config.verdict(spec.validator, payload or {})
            if payload is None:
                ok, detail = False, f"module did not answer inspection ({note})"
            if not ok:
                return _error_page(
                    f"{spec.title} cannot use this folder", f"{rp}\n\n{detail}",
                    extra_html=f"<p><a href='{spec.url}'>Open "
                               f"{html.escape(spec.title)} and choose "
                               f"another folder</a></p>")
            if spec.input_slot:
                write_slot(spec.input_slot, str(rp))

        target = spec.url + f"?home={quote(home_url, safe='')}"
        if rp is not None:
            target += f"&handoff={quote(str(rp), safe='')}"
        return RedirectResponse(target, status_code=303)

    @app.get("/result")
    def result():
        """Open the latest fusion result's demonstration video, served by
        the fusion module itself (its own path authorization applies)."""
        out = read_state().get("fusion_output")
        if not out:
            return _error_page(
                "No fusion result yet",
                "Run stage 3 (Fuse gaze) first — the workbench remembers "
                "the result when you return here from the fusion module.")
        rp = authorized(out)
        demo = rp / "gaze_demo.mp4"
        if not demo.is_file():
            return _error_page(
                "That fusion result has no demonstration video",
                f"{rp}\n\nThe run may have been made with --no-demo.")
        spec = specs["gaze_object_fusion"]
        try:
            pm.start("gaze_object_fusion")
        except procman.PortConflict as e:
            return _error_page("Port conflict", str(e))
        except procman.StartFailed as e:
            return _error_page("Could not start gaze_object_fusion", str(e))
        return RedirectResponse(
            f"{spec.url}api/demo?path={quote(str(demo), safe='')}",
            status_code=303)

    @app.exception_handler(Exception)
    async def _unhandled(request, exc):                # noqa: ANN001
        # a failure while talking to a module must never take the
        # workbench process down
        return JSONResponse(
            {"detail": f"workbench error: {type(exc).__name__}: {exc}"},
            status_code=500)

    return app


if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="NEON Gaze–Object Workbench")
    ap.add_argument("--host", default=common_config.DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--root", action="append", default=None, metavar="DIR")
    ap.add_argument("--allow-lan", action="store_true")
    args = ap.parse_args()
    cfg = common_config.load_stage_config(
        "workbench", "WORKBENCH", wb_config.WORKBENCH_PORT,
        extra_roots=args.root, port=args.port, host=args.host)
    try:
        application = create_app(cfg)
    except wb_config.WorkbenchConfigError as e:
        ap.error(str(e))
    lb = False
    try:
        lb = ipaddress.ip_address(cfg.host).is_loopback
    except ValueError:
        lb = cfg.host in ("localhost",)
    if not lb and not args.allow_lan:
        ap.error(f"refusing to bind {cfg.host} without --allow-lan")
    print(f"NEON Gaze–Object Workbench  →  http://{cfg.host}:{cfg.port}")
    print("browse roots:", ", ".join(str(r) for r in cfg.browse_roots))
    uvicorn.run(application, host=cfg.host, port=cfg.port,
                log_level="warning")
