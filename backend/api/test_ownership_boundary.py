"""
Regression tests for P1 "Security": the agent ownership boundary.

Run:
    PYTHONPATH=. python backend/api/test_ownership_boundary.py
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.api.app import app


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _minimal_spec(agent_id: str, claimed_owner: str | None) -> dict:
    spec = {
        "metadata": {
            "id": agent_id,
            "name": "Ownership test",
            "status": "draft",
            "version": "0.0.1",
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z",
        },
        "goal": "Test fixture for the ownership boundary.",
        "nodes": [{"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"}],
        "edges": [],
    }
    if claimed_owner is not None:
        spec["metadata"]["owner"] = claimed_owner
    return spec


def test_create_ignores_client_claimed_owner(client: TestClient) -> None:
    print("=== POST /agents: server-assigned owner overrides any client-claimed owner ===")
    resp = client.post(
        "/agents",
        json={"spec": _minimal_spec("agt_ownership_create_test", claimed_owner="someone_else")},
        headers={"X-User-Id": "real_requester"},
    )
    check(resp.status_code == 201, f"unexpected status: {resp.status_code} {resp.text}")
    check(
        resp.json()["metadata"]["owner"] == "real_requester",
        f"owner should be the real requester, not the client-claimed value, got {resp.json()['metadata']['owner']!r}",
    )
    print("  OK: stored owner is 'real_requester', client's claimed 'someone_else' was ignored")


def test_create_defaults_owner_to_default_principal(client: TestClient) -> None:
    print("\n=== POST /agents with no X-User-Id: owner defaults to 'default' (the stand-in principal) ===")
    resp = client.post("/agents", json={"spec": _minimal_spec("agt_ownership_default_test", claimed_owner=None)})
    check(resp.status_code == 201, f"unexpected status: {resp.status_code} {resp.text}")
    check(resp.json()["metadata"]["owner"] == "default", f"unexpected owner: {resp.json()['metadata']['owner']!r}")
    print("  OK: owner defaults consistently, matching get_owner_id's own default")


def test_mutation_cannot_reassign_owner(client: TestClient) -> None:
    print("\n=== update_metadata mutation cannot reassign owner ===")
    resp = client.post(
        "/agents",
        json={"spec": _minimal_spec("agt_ownership_mutation_test", claimed_owner=None)},
        headers={"X-User-Id": "real_requester"},
    )
    version = resp.json()["metadata"]["version"]

    resp = client.post(
        "/agents/agt_ownership_mutation_test/mutations",
        json={
            "mutations": [{"kind": "update_metadata", "patch": {"owner": "someone_else", "name": "Renamed"}}],
            "expected_version": version,
        },
        headers={"X-User-Id": "real_requester"},
    )
    check(resp.status_code == 422, f"expected 422, got {resp.status_code}: {resp.text}")
    check("owner" in resp.json()["detail"].lower(), f"error should mention ownership: {resp.json()}")
    print(f"  OK: mutation rejected with 422: {resp.json()['detail']}")

    current = client.get("/agents/agt_ownership_mutation_test").json()
    check(current["metadata"]["owner"] == "real_requester", f"owner should be unchanged, got {current['metadata']['owner']!r}")
    print("  OK: owner field is genuinely unchanged after the rejected attempt")


def test_legitimate_metadata_mutation_still_works(client: TestClient) -> None:
    print("\n=== a legitimate metadata mutation that doesn't touch owner still works ===")
    resp = client.post(
        "/agents",
        json={"spec": _minimal_spec("agt_ownership_legit_mutation_test", claimed_owner=None)},
        headers={"X-User-Id": "real_requester"},
    )
    version = resp.json()["metadata"]["version"]

    resp = client.post(
        "/agents/agt_ownership_legit_mutation_test/mutations",
        json={"mutations": [{"kind": "update_metadata", "patch": {"name": "A perfectly normal rename"}}], "expected_version": version},
        headers={"X-User-Id": "real_requester"},
    )
    check(resp.status_code == 200, f"a normal metadata edit should still succeed: {resp.status_code} {resp.text}")
    check(resp.json()["spec"]["metadata"]["name"] == "A perfectly normal rename", "the rename should have taken effect")
    print("  OK: unrelated metadata edits are unaffected by the ownership guard")


def test_ownership_actually_gates_credential_access(client: TestClient) -> None:
    print("\n=== end-to-end: server-assigned ownership genuinely gates credential access ===")
    resp = client.post(
        "/credentials",
        json={
            "provider": "gmail",
            "name": "Alice's Gmail",
            "secret": {"access_token": "a", "refresh_token": "r", "client_id": "c", "client_secret": "s"},
        },
        headers={"X-User-Id": "alice"},
    )
    check(resp.status_code == 201, f"credential creation failed: {resp.status_code} {resp.text}")
    cred_id = resp.json()["id"]

    spec = _minimal_spec("agt_ownership_credential_test", claimed_owner="alice")  # attempted impersonation again
    spec["nodes"].append(
        {"id": "n_send", "type": "send_email", "category": "tool", "label": "Send", "toolRefs": ["t1"]}
    )
    spec["edges"].append({"id": "e1", "source": "n_trigger", "target": "n_send"})
    spec["tools"] = [
        {"id": "t1", "toolId": "gmail.send", "permissions": ["send"], "mode": "connected", "credentialRef": cred_id}
    ]

    resp = client.post("/agents", json={"spec": spec}, headers={"X-User-Id": "mallory"})
    check(resp.status_code == 201, f"create failed: {resp.status_code} {resp.text}")
    check(resp.json()["metadata"]["owner"] == "mallory", f"owner should be mallory, got {resp.json()['metadata']['owner']!r}")

    resp = client.post("/agents/agt_ownership_credential_test/compile")
    check(resp.status_code == 200, f"compile-check route itself should 200: {resp.status_code}")
    body = resp.json()
    check(body["ok"] is False, f"compilation should fail -- mallory's agent cannot use alice's credential: {body}")
    check(
        "not owned" in body["error"].lower() or "does not exist" in body["error"].lower(),
        f"failure should specifically be the ownership/existence check, got: {body['error']}",
    )
    print(f"  OK: mallory's agent (impersonation attempt notwithstanding) cannot compile against alice's credential: {body['error']}")


def main() -> None:
    client = TestClient(app)
    test_create_ignores_client_claimed_owner(client)
    test_create_defaults_owner_to_default_principal(client)
    test_mutation_cannot_reassign_owner(client)
    test_legitimate_metadata_mutation_still_works(client)
    test_ownership_actually_gates_credential_access(client)
    print("\nAll ownership boundary checks passed.")


if __name__ == "__main__":
    main()
