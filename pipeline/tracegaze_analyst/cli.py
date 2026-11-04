"""Command-line interface for the minimal TraceGaze analyst."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import make_baseline_response
from .ollama import analyze_local
from .packet import write_packet
from .render import render_report
from .support_cli import add_commands as add_support_commands
from .support_cli import run as run_support
from .validate import validate_response


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tracegaze_analyst")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="build a bounded evidence packet and prompt")
    p.add_argument("--trace", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--out", required=True)
    b = sub.add_parser("baseline", help="create and validate a zero-API deterministic report")
    b.add_argument("--packet", required=True)
    b.add_argument("--out", required=True)
    a = sub.add_parser("analyze-local", help="analyze with a local Ollama model")
    a.add_argument("--packet", required=True)
    a.add_argument("--model", default="qwen3.6:35b")
    a.add_argument("--out", required=True)
    v = sub.add_parser("validate", help="validate and render a structured model response")
    v.add_argument("--packet", required=True)
    v.add_argument("--response", required=True)
    v.add_argument("--out", required=True)
    add_support_commands(sub)
    args = ap.parse_args(argv)

    try:
        if args.command not in ("prepare", "baseline", "analyze-local", "validate"):
            return run_support(args)
        if args.command == "prepare":
            packet = write_packet(args.trace, args.question, args.out)
            print(f"prepared {len(packet['facts'])} facts, {len(packet['episodes'])} episodes, "
                  f"{len(packet['transitions'])} transitions → {Path(args.out).resolve()}")
            return 0
        packet = _load(args.packet)
        metadata = None
        if args.command == "baseline":
            response = make_baseline_response(packet)
        elif args.command == "analyze-local":
            response, metadata = analyze_local(packet, args.model)
        else:
            response = _load(args.response)
        validation = validate_response(packet, response)
        out = Path(args.out).expanduser().resolve()
        if out.exists():
            raise ValueError(f"output directory already exists: {out}")
        out.mkdir(parents=True)
        (out / "analyst_response.json").write_text(
            json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "validation.json").write_text(
            json.dumps(validation, indent=2) + "\n", encoding="utf-8")
        if metadata is not None:
            (out / "model_run.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        render_report(packet, response, validation, out / "report.html")
        print(f"{'verified' if validation['ok'] else 'needs review'} → {out}")
        return 0 if validation["ok"] else 2
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"error: {e}")
        return 1

