"""
Regression tests for the concrete, verifiable items from the V9 fix spec
(section 63) addressed this session — each independently confirmed against
real code/real HTTP before being fixed, not assumed:

  Fix 4 — NL compiler with no ANTHROPIC_API_KEY: was a raw Anthropic SDK
          exception surfacing as a generic 500. Now a clean 503 with an
          actionable message, checked in get_anthropic_client() so it
          can't accidentally fire for tests using a fake-client override.

  Fix 5 — dropping a tool-category node (search_web, send_email, etc.)
          from NodePalette with no tool binding: was structurally valid
          (Pydantic doesn't check this) but failed at COMPILE time with an
          opaque error only discovered when the user tried to run it. Now
          NodePalette drops auto-provision a placeholder ToolBinding
          ATOMICALLY with the node — this test proves the exact JSON batch
          the frontend now generates an explicit incomplete binding. The
          real backend must reject it until Inspector configuration selects
          a registry-recognized tool.

  Fix 8 — compile errors now carry a structured node_id (and use the
          node's human label, not its raw id, in the message) so the
          frontend can offer a "Configure X" action instead of just
          displaying a technical string.

Run:
    PYTHONPATH=. python backend/api/test_v9_fixes.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from backend.api.app import app

ROOT = Path(__file__).resolve().parents[2]

passed = 0
failed = 0


def check(condition: bool, message: str) -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"FAIL: {message}")


def load_spec_json(name: str) -> dict:
    return json.loads((ROOT / "examples" / name).read_text())


def test_fix4_ai_unavailable() -> None:
    print("=== Fix 4: clean 'AI Builder unavailable' UX with no API key ===")
    os.environ.pop("ANTHROPIC_API_KEY", None)
    client = TestClient(app)

    spec = load_spec_json("research-agent.json")
    spec["metadata"]["id"] = "agt_fix4_test"
    client.post("/agents", json={"spec": spec})

    resp = client.post("/agents/agt_fix4_test/instructions", json={"message": "add approval before summarize"})
    check(resp.status_code == 503, f"expected 503, got {resp.status_code}")
    body = resp.json()
    check(body["detail"]["reason"] == "no_api_key", "response carries a machine-readable reason the frontend can key off")
    check("ANTHROPIC_API_KEY" in body["detail"]["error"], "error message names the actual missing configuration")
    check(
        "Could not resolve authentication" not in json.dumps(body),
        "the raw Anthropic SDK exception text does NOT leak through — it's a clean, purpose-written message",
    )
    print(f"  status={resp.status_code}, reason={body['detail']['reason']!r}\n")


def test_fix4_fake_client_override_unaffected() -> None:
    print("=== Fix 4 regression guard: fake-client test override still bypasses the check ===")
    from backend.api.app import get_anthropic_client

    class _FakeMessages:
        def parse(self, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(
                parsed_output=SimpleNamespace(reasoning="ok", mutations=[]),
                stop_reason="end_turn",
            )

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

    app.dependency_overrides[get_anthropic_client] = lambda: _FakeClient()
    try:
        client = TestClient(app)
        spec = load_spec_json("research-agent.json")
        spec["metadata"]["id"] = "agt_fix4_override_test"
        client.post("/agents", json={"spec": spec})
        resp = client.post("/agents/agt_fix4_override_test/instructions", json={"message": "what does this do?"})
        check(
            resp.status_code == 200,
            f"a dependency-overridden fake client should NOT hit the no-API-key check at all — got {resp.status_code}: {resp.text}",
        )
        print(f"  status={resp.status_code} (override correctly bypassed the check)\n")
    finally:
        app.dependency_overrides.pop(get_anthropic_client, None)


def test_fix5_tool_binding_autoprovision() -> None:
    print("=== Fix 5: dropped tool node's placeholder blocks execution clearly ===")
    client = TestClient(app)
    spec = load_spec_json("research-agent.json")
    spec["metadata"]["id"] = "agt_fix5_test"
    resp = client.post("/agents", json={"spec": spec})
    version = resp.json()["metadata"]["version"]

    # The EXACT batch packages/canvas/src/specToFlow.ts's
    # catalogEntryToAddNodeMutation now generates for a dropped search_web node.
    resp = client.post(
        "/agents/agt_fix5_test/mutations",
        json={
            "mutations": [
                {
                    "kind": "add_tool",
                    "tool": {
                        "id": "tool_1",
                        "toolId": "unconfigured.search_web",
                        "displayName": "Search Web (not yet configured)",
                        "permissions": ["read"],
                    },
                },
                {
                    "kind": "add_node",
                    "node": {
                        "id": "n_dropped",
                        "type": "search_web",
                        "category": "tool",
                        "label": "Search Web",
                        "executionType": "deterministic",
                        "config": {},
                        "canvasPosition": {"x": 100, "y": 100},
                        "toolRefs": ["tool_1"],
                    },
                },
            ],
            "expected_version": version,
        },
    )
    check(resp.status_code == 200, f"mutation batch should apply cleanly: {resp.text}")

    resp = client.post("/agents/agt_fix5_test/compile")
    check(resp.status_code == 200, f"compile check request should 200: {resp.text}")
    body = resp.json()
    check(body["ok"] is False, f"the unconfigured dropped node must block compile, got: {body}")
    check(body["node_id"] == "n_dropped", f"error points to the unconfigured node: {body}")
    check("not registered" in body["error"], f"error explains the binding needs configuration: {body}")
    print(f"  compile check: ok={body['ok']}\n")


def test_fix8_structured_compile_error() -> None:
    print("=== Fix 8: compile errors carry node_id and use the human label ===")
    client = TestClient(app)
    spec = load_spec_json("research-agent.json")
    spec["metadata"]["id"] = "agt_fix8_test"
    spec["nodes"].append({
        "id": "n_broken",
        "type": "send_email",
        "category": "tool",
        "label": "Notify Customer",
        "config": {},
    })
    client.post("/agents", json={"spec": spec})

    resp = client.post("/agents/agt_fix8_test/compile")
    check(resp.status_code == 200, f"compile check should 200 even on failure: {resp.text}")
    body = resp.json()
    check(body["ok"] is False, "the broken node should correctly fail the check")
    check(body["node_id"] == "n_broken", f"error carries the specific node_id, got {body.get('node_id')!r}")
    check(
        "Notify Customer" in body["error"],
        f"error uses the node's human LABEL ('Notify Customer'), not just its raw id — got: {body['error']!r}",
    )
    print(f"  node_id={body['node_id']!r}, error={body['error']!r}\n")


if __name__ == "__main__":
    test_fix4_ai_unavailable()
    test_fix4_fake_client_override_unaffected()
    test_fix5_tool_binding_autoprovision()
    test_fix8_structured_compile_error()
    print(f"{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
