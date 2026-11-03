"""
Regression test for a real, confirmed bug: a genuinely unhandled exception
produced a 500 response with NO Access-Control-Allow-Origin header,
because Starlette routes unhandled exceptions to ServerErrorMiddleware,
which sits OUTSIDE CORSMiddleware regardless of add_middleware() ordering
— and even a registered `@app.exception_handler(Exception)` doesn't fix
this, since that also attaches to ServerErrorMiddleware, not the inner
ExceptionMiddleware CORSMiddleware actually wraps.

A browser making a cross-origin request (the app shell at :5173 calling
the API at :8000) can't read a response with no CORS header at all — it
surfaces in JS as a generic "TypeError: Failed to fetch" with zero
indication anything reached the server or what went wrong. This is
EXACTLY what broke the chat builder in practice, before a *separate* fix
(see test_v9_fixes.py's Fix 4) turned that ONE specific failure mode
(missing ANTHROPIC_API_KEY) into a clean, handled 503 instead.

That later fix means the missing-API-key path no longer exercises a truly
UNHANDLED exception, so it can no longer double as this test's trigger.
This version instead injects a deliberately-raising fake store via
dependency override — a genuine, unhandled RuntimeError through a real
route on the REAL app object (not a synthetic test app, not a copy of the
middleware) — so this test keeps proving the general mechanism works for
*any* future unhandled exception, not just the one bug that originally
surfaced it.

Uses raw http.client (not httpx/requests) specifically so response headers
are inspected at the wire level, not normalized/hidden by a higher-level
client library.

Run:
    PYTHONPATH=. python backend/api/test_error_cors.py
"""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PORT = 8976

passed = 0
failed = 0


def check(condition: bool, message: str) -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"FAIL: {message}")


def wait_for_server(timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/agents/__probe__", timeout=1)
            return
        except urllib.error.HTTPError:
            return  # server responded, just 404'd — it's up
        except Exception:
            time.sleep(0.3)
    raise RuntimeError("server did not become ready in time")


# A tiny FastAPI "app" wouldn't exercise the REAL production middleware
# stack faithfully, so instead this runs the actual `backend.api.app:app`
# as a real subprocess, with a one-line env-var-triggered fault injected
# via a store method that raises — see the SIMULATE_UNHANDLED_ERROR check
# this test adds no permanent code path for; it's applied via monkeypatch
# in a tiny bootstrap script passed to the subprocess.
_BOOTSTRAP_SCRIPT = """
import sys
sys.path.insert(0, {root!r})
from backend.api.app import app, get_store
from backend.api.store import AgentStore

class _RaisingStore(AgentStore):
    def get(self, agent_id):
        raise RuntimeError("simulated unhandled bug — proves the CORS-on-error fix, not a real defect")

app.dependency_overrides[get_store] = lambda: _RaisingStore()

import uvicorn
uvicorn.run(app, host="127.0.0.1", port={port})
"""


def main() -> None:
    script = _BOOTSTRAP_SCRIPT.format(root=str(ROOT), port=PORT)
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server()
        print("server up, with a fault-injecting store override\n")

        # --- allowed origin: the CORS header must be present on the error ---
        print("=== genuinely unhandled exception, from an ALLOWED origin ===")
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=15)
        conn.request(
            "GET", "/agents/anything",
            headers={"origin": "http://localhost:5173"},
        )
        resp = conn.getresponse()
        headers = {k.lower(): v for k, v in resp.getheaders()}
        raw_body = resp.read()
        print(f"  status: {resp.status}")
        print(f"  Access-Control-Allow-Origin: {headers.get('access-control-allow-origin')}")

        check(resp.status == 500, f"expected a genuine unhandled exception to surface as 500, got {resp.status}")
        check(
            "access-control-allow-origin" in headers,
            "THE BUG: response is missing Access-Control-Allow-Origin — a browser would report this as "
            "'Failed to fetch' with no visibility into the real error",
        )
        check(
            headers.get("access-control-allow-origin") == "http://localhost:5173",
            f"CORS header should echo the allowed origin, got {headers.get('access-control-allow-origin')!r}",
        )
        check(b"detail" in raw_body, "error body should be structured JSON with a 'detail' field, not a bare crash")
        try:
            parsed = json.loads(raw_body)
            check("detail" in parsed and len(parsed["detail"]) > 0, "detail field is present and non-empty")
        except json.JSONDecodeError:
            check(False, "error response body should be valid JSON")

        # --- disallowed origin: must NOT leak a CORS header ---
        print("\n=== same failure from a DISALLOWED origin — must not leak CORS access ===")
        conn2 = http.client.HTTPConnection("127.0.0.1", PORT, timeout=15)
        conn2.request(
            "GET", "/agents/anything",
            headers={"origin": "http://evil.example.com"},
        )
        resp2 = conn2.getresponse()
        headers2 = {k.lower(): v for k, v in resp2.getheaders()}
        resp2.read()
        check(
            "access-control-allow-origin" not in headers2,
            "a disallowed origin must NOT get a CORS header on the error response either — "
            "the fix must not accidentally become an open CORS bypass",
        )
        print(f"  Access-Control-Allow-Origin present: {'access-control-allow-origin' in headers2} (correct: False)")

        print(f"\n{passed} passed, {failed} failed")
        if failed:
            raise SystemExit(1)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
