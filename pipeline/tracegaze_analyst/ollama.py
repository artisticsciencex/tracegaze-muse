"""Minimal local Ollama client. No credentials and no network upload."""

from __future__ import annotations

import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

from .packet import make_prompt


def analyze_local(packet: dict, model: str,
                  endpoint: str | None = None,
                  timeout_s: int = 900) -> tuple[dict, dict]:
    endpoint = endpoint or os.environ.get(
        "TRACEGAZE_OLLAMA_ENDPOINT", "http://127.0.0.1:11434/api/generate")
    body = {
        "model": model,
        "prompt": make_prompt(packet),
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0, "num_ctx": 32768},
    }
    request = Request(endpoint, data=json.dumps(body).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout_s) as reply:
            envelope = json.loads(reply.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise ValueError(f"local Ollama request failed: {exc}") from exc
    try:
        response = json.loads(envelope["response"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Ollama did not return a valid JSON analyst response") from exc
    metadata = {
        "provider": "ollama-local",
        "model": envelope.get("model", model),
        "done_reason": envelope.get("done_reason"),
        "prompt_eval_count": envelope.get("prompt_eval_count"),
        "eval_count": envelope.get("eval_count"),
        "total_duration_ns": envelope.get("total_duration"),
    }
    return response, metadata
