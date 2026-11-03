"""
End-to-end test of the API layer via FastAPI's TestClient — no real network
calls, no real Anthropic API key needed (the Anthropic client dependency is
overridden with a fake, same pattern as backend/run_nl_compiler.py).

Run:
    PYTHONPATH=. python backend/api/test_app.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from backend.api.app import app, get_anthropic_client
from backend.nl_compiler.schemas import MutationBatch

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FakeParsedResponse:
    parsed_output: MutationBatch
    stop_reason: str = "end_turn"


class FakeMessages:
    def __init__(self, batches: list[dict[str, Any]]):
        self._queue = [MutationBatch.model_validate(b) for b in batches]

    def parse(self, **kwargs) -> FakeParsedResponse:
        return FakeParsedResponse(parsed_output=self._queue.pop(0))


class FakeAnthropicClient:
    def __init__(self, batches: list[dict[str, Any]]):
        self.messages = FakeMessages(batches)


def load_spec_json(name: str) -> dict:
    return json.loads((ROOT / "examples" / name).read_text())


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    client = TestClient(app)

    # ---- create ----------------------------------------------------
    print("=== create agent ===")
    invoice_spec = load_spec_json("invoice-agent.json")
    invoice_spec["metadata"]["id"] = "agt_e2e_invoice"  # avoid clobbering across re-runs
    resp = client.post("/agents", json={"spec": invoice_spec})
    check(resp.status_code == 201, f"expected 201, got {resp.status_code}: {resp.text}")
    agent_id = resp.json()["metadata"]["id"]
    print(f"  created {agent_id}")

    resp = client.post("/agents", json={"spec": invoice_spec})
    check(resp.status_code == 409, "creating the same agent id twice should 409")
    print("  OK: duplicate create correctly rejected (409)")

    # ---- get ----------------------------------------------------
    resp = client.get(f"/agents/{agent_id}")
    check(resp.status_code == 200, "get should succeed")
    version = resp.json()["metadata"]["version"]
    print(f"  fetched, version={version}")

    resp = client.get("/agents/does_not_exist")
    check(resp.status_code == 404, "unknown agent should 404")
    print("  OK: unknown agent id correctly 404s")

    # ---- mutations ----------------------------------------------------
    print("\n=== apply mutations ===")
    resp = client.post(
        f"/agents/{agent_id}/mutations",
        json={
            "mutations": [{"kind": "update_node", "nodeId": "n_gen_email", "patch": {"canvasPosition": {"x": 300, "y": 500}}}],
            "expected_version": version,
        },
    )
    check(resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.text}")
    new_version = resp.json()["spec"]["metadata"]["version"]
    check(new_version != version, "version should bump after a successful mutation")
    print(f"  OK: mutation applied, version {version} -> {new_version}")

    # stale version -> 409
    resp = client.post(
        f"/agents/{agent_id}/mutations",
        json={"mutations": [{"kind": "update_goal", "goal": "stale edit"}], "expected_version": version},
    )
    check(resp.status_code == 409, f"stale version should 409, got {resp.status_code}")
    print("  OK: stale expected_version correctly rejected with 409 (concurrency conflict)")

    # malformed mutation -> 422
    resp = client.post(
        f"/agents/{agent_id}/mutations",
        json={"mutations": [{"kind": "not_a_real_kind"}]},
    )
    check(resp.status_code == 422, f"malformed mutation kind should 422, got {resp.status_code}")
    print("  OK: unrecognized mutation kind correctly rejected with 422")

    # ---- instructions (NL compiler via fake client) ----------------------
    print("\n=== NL instruction (fake Anthropic client) ===")
    fake_client = FakeAnthropicClient([{
        "reasoning": "Added a Slack notification after the invoice is saved.",
        "mutations": [
            {"kind": "add_node", "node": {"id": "n_notify", "type": "api_call", "category": "tool", "label": "Notify Slack"}},
            {"kind": "add_edge", "edge": {"id": "e_notify", "source": "n_save", "target": "n_notify"}},
        ],
    }])
    app.dependency_overrides[get_anthropic_client] = lambda: fake_client
    try:
        resp = client.post(f"/agents/{agent_id}/instructions", json={"message": "Notify me on Slack once saved."})
        check(resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.text}")
        body = resp.json()
        print(f"  reasoning: {body['reasoning']}")
        check(body["result"]["ok"], "instruction should apply cleanly")
        check(any(n["id"] == "n_notify" for n in body["result"]["spec"]["nodes"]), "new node should be present")
        print("  OK: NL instruction applied via the API, same mutation engine underneath")
    finally:
        app.dependency_overrides.pop(get_anthropic_client, None)

    # ---- compile check ----------------------------------------------------
    print("\n=== compile check ===")
    resp = client.post(f"/agents/{agent_id}/compile")
    check(resp.status_code == 200 and resp.json()["ok"], f"compile check should pass: {resp.text}")
    print("  OK: spec compiles cleanly")

    # ---- run + interrupt + resume (approve) ----------------------------------------------------
    print("\n=== run: pauses for approval, then resumes ===")
    resp = client.post(f"/agents/{agent_id}/run", json={"seed_variables": {}})
    check(resp.status_code == 200, f"run should 200: {resp.text}")
    run_body = resp.json()
    check(run_body["status"] == "paused", f"expected paused, got {run_body['status']}")
    check(run_body["interrupt"]["type"] == "human_approval", "should be paused at the human_approval node")
    print(f"  paused: {run_body['interrupt']}")

    resp = client.post(
        f"/agents/{agent_id}/resume",
        json={"thread_id": run_body["thread_id"], "resume": {"approved": True}},
    )
    check(resp.status_code == 200, f"resume should 200: {resp.text}")
    resumed = resp.json()
    check(resumed["status"] == "completed", f"expected completed, got {resumed['status']}")
    check(resumed["variables"].get("email_sent") is True, "approved run should have sent the email")
    print("  OK: resumed after approval, email_sent == True")

    # ---- publish ----------------------------------------------------
    print("\n=== publish ===")
    resp = client.get(f"/agents/{agent_id}")
    current_version = resp.json()["metadata"]["version"]
    resp = client.post(f"/agents/{agent_id}/publish")
    check(resp.status_code == 200, f"publish should 200: {resp.text}")
    pub_body = resp.json()
    check(pub_body["published_version"] == current_version, "publishedVersion should match current version")
    check(pub_body["spec"]["metadata"]["status"] == "published", "status should flip to published")
    print(f"  OK: published version {pub_body['published_version']}")

    print("\nAll API end-to-end checks passed.")


if __name__ == "__main__":
    main()
