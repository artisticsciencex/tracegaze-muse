"""A stand-in module server for workbench tests (stdlib only).

    python fake_module.py --port 9101 --module neon_trim [--version 1.2]
        [--busy] [--delay 5] [--exit-code 3] [--inspect-json '{"...":1}']

--delay N     wait N seconds before serving (startup-timeout tests)
--exit-code N exit immediately with that code (child-failure tests)
--busy        report busy=true in /api/health
"""

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--module", default="fake")
    ap.add_argument("--version", default="0.0.1")
    ap.add_argument("--busy", action="store_true")
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--exit-code", type=int, default=None)
    ap.add_argument("--inspect-json", default=None)
    ap.add_argument("--no-identity", action="store_true",
                    help="serve health WITHOUT the module identity "
                         "(simulates a foreign service on the port)")
    args = ap.parse_args()

    if args.exit_code is not None:
        print(f"fake module refusing to start (exit {args.exit_code})",
              flush=True)
        return args.exit_code
    if args.delay:
        print(f"fake module sleeping {args.delay}s before serving",
              flush=True)
        time.sleep(args.delay)

    health = ({"ok": True, "service": "something-else"} if args.no_identity
              else {"ok": True, "module": args.module,
                    "version": args.version, "busy": args.busy})
    inspect = json.loads(args.inspect_json) if args.inspect_json else {}

    class H(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):                       # noqa: N802
            u = urlparse(self.path)
            if u.path == "/api/health":
                return self._send(health)
            if u.path == "/api/inspect":
                q = parse_qs(u.query)
                return self._send({**inspect,
                                   "path": (q.get("path") or [""])[0]})
            if u.path == "/":
                body = b"<html><body>fake module page</body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            self._send({"detail": "not found"}, 404)

        def log_message(self, *a):              # keep test output clean
            pass

    srv = HTTPServer(("127.0.0.1", args.port), H)
    print(f"fake module {args.module} listening on {args.port}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
