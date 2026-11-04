"""Minimal Meta Model API client for Muse (stdlib only).

Official surface (dev.meta.ai/docs, checked 2026-11-05):
  POST https://api.meta.ai/v1/chat/completions, OpenAI-compatible
  Authorization: Bearer $MODEL_API_KEY
  images: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
  structured output: response_format = {"type": "json_schema", "json_schema": {...}}

The key is read from the environment at call time and never written to
disk, logs, metadata or exceptions. Saved request bodies replace image
bytes with their sha256 so a run can be reproduced from the frame files.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"
KEY_ENV = "MODEL_API_KEY"
BASE_URL_ENV = "MODEL_API_BASE_URL"

# response headers worth keeping for debugging; everything else is dropped
_SAFE_HEADERS = ("x-request-id", "request-id", "x-trace-id", "x-fb-trace-id",
                 "x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens",
                 "date", "content-type")


class MuseError(RuntimeError):
    """Request failure; `envelope`/`metadata` are set when a reply arrived."""

    def __init__(self, message: str, envelope: dict | None = None,
                 metadata: dict | None = None):
        super().__init__(message)
        self.envelope, self.metadata = envelope, metadata


def load_env_file(path) -> list[str]:
    """Load MODEL_API_KEY / MODEL_API_BASE_URL from a dotenv-style file.

    Only these two names are read, values are never printed, and variables
    already set in the environment win. Returns the names that were set.
    """
    loaded = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, sep, value = line.partition("=")
        name = name.strip()
        if sep and name in (KEY_ENV, BASE_URL_ENV) and not os.environ.get(name):
            os.environ[name] = value.strip().strip("'\"")
            loaded.append(name)
    return loaded


def _key() -> str:
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise MuseError(f"{KEY_ENV} is not set; export the Meta Model API key "
                        f"in the shell that runs this command")
    return key


def _base_url() -> str:
    return os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL).rstrip("/")


def _scrub(text: str) -> str:
    key = os.environ.get(KEY_ENV, "")
    return text.replace(key, "[REDACTED]") if key else text


def _call(method: str, path: str, body: dict | None, timeout_s: int):
    url = f"{_base_url()}{path}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {_key()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    started = time.monotonic()
    try:
        with urlopen(req, timeout=timeout_s) as reply:
            raw = reply.read().decode("utf-8")
            headers = {k.lower(): v for k, v in reply.headers.items()
                       if k.lower() in _SAFE_HEADERS}
            status = reply.status
    except HTTPError as exc:
        detail = _scrub(exc.read().decode("utf-8", "replace"))[:2000]
        raise MuseError(f"HTTP {exc.code} from {path}: {detail}") from None
    except (URLError, OSError) as exc:
        raise MuseError(f"request to {path} failed: {_scrub(str(exc))}") from None
    elapsed = time.monotonic() - started
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        raise MuseError(f"non-JSON reply from {path}: {_scrub(raw)[:500]}") from None
    return envelope, {"http_status": status, "elapsed_s": round(elapsed, 3),
                      "response_headers": headers, "url": url}


def list_models(timeout_s: int = 60) -> list[str]:
    envelope, _ = _call("GET", "/models", None, timeout_s)
    return sorted(m.get("id", "") for m in envelope.get("data", []))


def image_part(jpeg_bytes: bytes) -> dict:
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def redact_images(body: dict) -> dict:
    """Copy of a request body with image bytes replaced by their sha256."""
    out = copy.deepcopy(body)
    for msg in out.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image_url":
                url = part["image_url"]["url"]
                raw = base64.b64decode(url.split(",", 1)[1])
                part["image_url"] = {"url": "[image omitted]",
                                     "sha256": hashlib.sha256(raw).hexdigest(),
                                     "bytes": len(raw)}
    return out


def chat_request(messages: list, schema: dict, *, model: str = DEFAULT_MODEL,
                 schema_name: str = "tracegaze_findings", strict: bool = True,
                 reasoning_effort: str = "medium", seed: int | None = 7,
                 max_completion_tokens: int = 16000) -> dict:
    """Request body for one chat completion with a JSON-schema response."""
    body = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": schema_name, "schema": schema, "strict": strict}},
        "reasoning_effort": reasoning_effort,
        "max_completion_tokens": max_completion_tokens,
    }
    if seed is not None:
        body["seed"] = seed
    return body


def chat_json(body: dict, timeout_s: int = 600) -> tuple[dict, dict, dict]:
    """Send a chat_request body. Returns (parsed_json, envelope, metadata).

    Raises MuseError on any transport, HTTP or parsing failure; nothing is
    retried silently.
    """
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    envelope, transport = _call("POST", "/chat/completions", body, timeout_s)
    choice = (envelope.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    js = body["response_format"]["json_schema"]
    metadata = {
        "provider": "meta-model-api",
        "endpoint": "/chat/completions",
        "base_url": _base_url(),
        "requested_model": body["model"],
        "returned_model": envelope.get("model"),
        "response_id": envelope.get("id"),
        "finish_reason": choice.get("finish_reason"),
        "usage": envelope.get("usage"),
        "request_parameters": {k: v for k, v in body.items()
                               if k not in ("messages", "response_format")},
        "response_format": {"type": "json_schema", "name": js["name"],
                            "strict": js["strict"]},
        "request_sha256": hashlib.sha256(
            json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest(),
        "started_utc": started_utc,
        **transport,
    }
    if not isinstance(content, str) or not content.strip():
        raise MuseError(f"empty completion (finish_reason="
                        f"{choice.get('finish_reason')!r})", envelope, metadata)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise MuseError(f"completion is not valid JSON ({exc.msg} at char "
                        f"{exc.pos} of {len(content)}; finish_reason="
                        f"{choice.get('finish_reason')!r})",
                        envelope, metadata) from None
    return parsed, envelope, metadata
