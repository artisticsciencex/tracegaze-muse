"""Commands for the everyday-activity support example (TraceGaze + Muse).

    support-package  trace + question/context  -> package/
    muse-findings    package/                  -> muse_run/ (one real request)
    check-findings   package/ + muse_run/      -> muse_run/validation.json
    tamper-check     package/ + muse_run/      -> muse_run/tamper_check.json
    support-report   example dir               -> report/index.html
    serve-report     example dir               -> http://127.0.0.1:PORT/report/
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .findings import (FINDINGS_SCHEMA, build_messages, rederive_facts,
                       validate_findings)


def _load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_commands(sub) -> None:
    p = sub.add_parser("support-package", help="build a bounded evidence package")
    p.add_argument("--trace", required=True, help="attention_trace artifact dir")
    p.add_argument("--config", required=True,
                   help="JSON with 'question' and 'context' (explicitly supplied)")
    p.add_argument("--out", required=True, help="new package directory")

    m = sub.add_parser("muse-findings", help="one Muse request -> findings")
    m.add_argument("--package", required=True)
    m.add_argument("--out", required=True, help="new run directory")
    m.add_argument("--model", default=None, help="default: muse-spark-1.3")
    m.add_argument("--reasoning-effort", default="medium")
    m.add_argument("--max-completion-tokens", type=int, default=16000,
                   help="output cap; reasoning tokens count against it")
    m.add_argument("--dry-run", action="store_true",
                   help="write the redacted request and prompt; send nothing")
    m.add_argument("--env-file", default=None,
                   help="dotenv file holding MODEL_API_KEY (read silently; "
                        "an already exported MODEL_API_KEY wins)")

    c = sub.add_parser("check-findings", help="validate findings + re-derive facts")
    c.add_argument("--package", required=True)
    c.add_argument("--run", required=True)

    t = sub.add_parser("tamper-check", help="prove the validator catches edits")
    t.add_argument("--package", required=True)
    t.add_argument("--run", required=True)

    r = sub.add_parser("support-report", help="render the local report")
    r.add_argument("--example", required=True,
                   help="directory holding package/ and muse_run/")

    s = sub.add_parser("serve-report", help="serve the report on 127.0.0.1")
    s.add_argument("--example", required=True)
    s.add_argument("--port", type=int, default=8765)


def run(args) -> int:
    cmd = args.command
    if cmd == "support-package":
        from .support_package import build_support_package
        cfg = _load(args.config)
        pkg = build_support_package(args.trace, cfg["question"], cfg["context"], args.out)
        print(f"package: {len(pkg['facts'])} facts, {len(pkg['intervals'])} intervals, "
              f"{len(pkg['episodes'])} episodes, {len(pkg['frames'])} frames "
              f"-> {Path(args.out).resolve()}")
        return 0

    if cmd == "muse-findings":
        return _muse_findings(args)

    if cmd == "check-findings":
        validation = check(Path(args.package), Path(args.run))
        s = validation["summary"]
        print(f"{s['references_verified']}/{s['findings']} findings references_verified; "
              f"re-derived {validation['rederivation']['checked']} facts, "
              f"{len(validation['rederivation']['mismatches'])} mismatches; "
              f"global problems: {validation['global_problems'] or 'none'}")
        return 0 if validation["ok"] else 2

    if cmd == "tamper-check":
        result = tamper_check(Path(args.package), Path(args.run))
        for case in result["cases"]:
            print(f"  {'DETECTED' if case['detected'] else 'MISSED  '} {case['case']}")
        return 0 if result["all_detected"] else 2

    if cmd == "support-report":
        from .support_report import build_report
        print(f"report -> {build_report(Path(args.example).resolve())}")
        return 0

    if cmd == "serve-report":
        from .support_report import serve
        serve(Path(args.example).resolve(), args.port)
        return 0
    raise ValueError(cmd)


def _muse_findings(args) -> int:
    from . import muse
    pkg_dir = Path(args.package).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise ValueError(f"output directory already exists: {out}")
    package = _load(pkg_dir / "support_package.json")
    messages, prompt_record = build_messages(package, pkg_dir)
    body = muse.chat_request(messages, FINDINGS_SCHEMA,
                             model=args.model or muse.DEFAULT_MODEL,
                             reasoning_effort=args.reasoning_effort,
                             max_completion_tokens=args.max_completion_tokens)
    out.mkdir(parents=True)
    _dump(muse.redact_images(body), out / "request_redacted.json")
    (out / "prompt.md").write_text(prompt_record, encoding="utf-8")
    inputs = {
        "package_sha256": _sha(pkg_dir / "support_package.json"),
        "frames_sha256": {f["frame_id"]: f["sha256"] for f in package["frames"]},
        "schema_sha256": hashlib.sha256(json.dumps(FINDINGS_SCHEMA, sort_keys=True)
                                        .encode()).hexdigest(),
    }
    if args.dry_run:
        _dump({"dry_run": True, **inputs, "request_parameters":
               {k: v for k, v in body.items() if k not in ("messages", "response_format")}},
              out / "run_metadata.json")
        print(f"dry run: request written to {out}")
        return 0
    if args.env_file:
        muse.load_env_file(args.env_file)
    try:
        findings, envelope, meta = muse.chat_json(body)
    except muse.MuseError as exc:
        if exc.envelope is not None:
            _dump(exc.envelope, out / "raw_response.json")
        _dump({"error": str(exc), **(exc.metadata or {}), **inputs}, out / "error.json")
        print(f"Muse request failed: {exc}")
        return 1
    _dump(envelope, out / "raw_response.json")
    _dump(findings, out / "findings.json")
    _dump({**meta, **inputs}, out / "run_metadata.json")
    validation = check(pkg_dir, out)
    s = validation["summary"]
    print(f"Muse {meta['returned_model']} ({meta['response_id']}): {s['findings']} findings, "
          f"{s['references_verified']} references_verified -> {out}")
    return 0


def check(pkg_dir: Path, run_dir: Path) -> dict:
    package = _load(pkg_dir / "support_package.json")
    findings = _load(run_dir / "findings.json")
    trace_dir = Path(package["source_refs"]["attention_trace_dir"])
    red = rederive_facts(package, trace_dir, pkg_dir / "boundary_proximity.json")
    validation = validate_findings(package, findings, red)
    _dump(validation, run_dir / "validation.json")
    return validation


def tamper_check(pkg_dir: Path, run_dir: Path) -> dict:
    """Corrupt the real response (and package) in memory; each must be caught."""
    package = _load(pkg_dir / "support_package.json")
    findings = _load(run_dir / "findings.json")
    trace_dir = Path(package["source_refs"]["attention_trace_dir"])
    prox = pkg_dir / "boundary_proximity.json"
    base = validate_findings(package, findings, rederive_facts(package, trace_dir, prox))
    base_errors = {(r["finding_id"], e) for r in base["findings"] for e in r["errors"]}
    cases = []

    def first_with(key):
        return next((i for i, f in enumerate(findings["findings"]) if f.get(key)), None)

    def run_case(name, resp, pkg=None, detail=""):
        pkg = pkg or package
        v = validate_findings(pkg, resp, rederive_facts(pkg, trace_dir, prox))
        found = {(r["finding_id"], e) for r in v["findings"] for e in r["errors"]}
        new = sorted(e for _, e in found - base_errors)
        new += [g for g in v["global_problems"] if g not in base["global_problems"]]
        cases.append({"case": name, "detail": detail, "detected": bool(new),
                      "new_problems": new})

    i = first_with("frame_ids")
    if i is not None:
        bad = copy.deepcopy(findings)
        bad["findings"][i]["frame_ids"][0] = "fr_9999"
        run_case("invalid frame reference", bad, detail=f"finding {i}: frame_ids[0] -> fr_9999")
    i = first_with("interval_ids")
    if i is not None:
        bad = copy.deepcopy(findings)
        bad["findings"][i]["interval_ids"].append("iv_99")
        run_case("invalid interval reference", bad, detail=f"finding {i}: + iv_99")
    i = first_with("quantities")
    if i is not None:
        bad = copy.deepcopy(findings)
        q = bad["findings"][i]["quantities"][0]
        q["value"] = q["value"] + 1 if isinstance(q["value"], int) else round(q["value"] * 1.5 + 0.1, 3)
        run_case("altered quantity", bad,
                 detail=f"finding {i}: {q['fact_id']} -> {q['value']}")
    i = first_with("statement")
    if i is not None:
        bad = copy.deepcopy(findings)
        bad["findings"][i]["statement"] += " This lasted 99.9 s."
        run_case("untraceable number in prose", bad, detail=f"finding {i}: + '99.9 s'")
        bad = copy.deepcopy(findings)
        bad["findings"][i]["statement"] += " The wearer appears confused."
        run_case("construct language in evidence text", bad,
                 detail=f"finding {i}: + 'appears confused'")
    bad_pkg = copy.deepcopy(package)
    target = next(f for f in bad_pkg["facts"] if f["fact_id"].endswith("total_attention_s"))
    target["value"] = round(target["value"] + 2.0, 3)
    run_case("altered package fact", findings, pkg=bad_pkg,
             detail=f"{target['fact_id']} -> {target['value']} (package only)")

    result = {"baseline_ok": base["ok"], "cases": cases,
              "all_detected": all(c["detected"] for c in cases)}
    _dump(result, run_dir / "tamper_check.json")
    return result
