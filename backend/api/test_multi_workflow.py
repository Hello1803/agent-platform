"""
Regression tests for multi-workflow support.

Run:
    PYTHONPATH=. python backend/api/test_multi_workflow.py
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.api.app import app


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_create_list_rename_duplicate_delete(client: TestClient) -> None:
    print("=== create / list / rename / duplicate / delete ===")
    agent_id = "agt_container_crud_test"
    headers = {"X-User-Id": "tester"}

    r = client.post(f"/agents/{agent_id}/workflows", json={"name": "Invoice Processing", "description": "handles invoices"}, headers=headers)
    check(r.status_code == 201, f"create failed: {r.status_code} {r.text}")
    wf_a = r.json()["metadata"]["id"]
    check(r.json()["metadata"]["agentId"] == agent_id, f"unexpected agentId: {r.json()['metadata']}")
    check(r.json()["metadata"]["enabled"] is True, "a new workflow should default to enabled")
    check(len(r.json()["nodes"]) == 1 and r.json()["nodes"][0]["type"] == "trigger_manual", "should start with a minimal one-trigger-node graph")
    print(f"  OK: created workflow {wf_a} under agent {agent_id}, minimal starter graph, enabled=True")

    r = client.post(f"/agents/{agent_id}/workflows", json={"name": "Weekly Report"}, headers=headers)
    wf_b = r.json()["metadata"]["id"]

    r = client.get(f"/agents/{agent_id}/workflows")
    check(r.status_code == 200, f"list failed: {r.status_code}")
    ids = {w["workflow_id"] for w in r.json()}
    check({wf_a, wf_b} <= ids, f"both workflows should be listed: {ids}")
    print(f"  OK: GET /agents/{agent_id}/workflows lists both workflows")

    r = client.post(f"/agents/{agent_id}/workflows/{wf_a}/rename", json={"name": "Invoice Processing v2"})
    check(r.status_code == 200 and r.json()["metadata"]["name"] == "Invoice Processing v2", f"rename failed: {r.status_code} {r.text}")
    print("  OK: rename applied through the real mutation engine")

    r = client.post(f"/agents/{agent_id}/workflows/{wf_a}/duplicate", headers=headers)
    check(r.status_code == 201, f"duplicate failed: {r.status_code} {r.text}")
    wf_a_copy = r.json()["metadata"]["id"]
    check(wf_a_copy != wf_a, "duplicate must get a NEW workflow id")
    check(r.json()["metadata"]["status"] == "draft", "a duplicate starts as its own fresh draft")
    check(r.json()["metadata"]["publishedVersion"] is None, "a duplicate should not inherit the source's published snapshot")
    check(len(r.json()["nodes"]) == 1, "the duplicate's graph content should match the source")
    print(f"  OK: duplicated {wf_a} -> {wf_a_copy}, independent id, fresh draft state")

    r = client.delete(f"/agents/{agent_id}/workflows/{wf_b}")
    check(r.status_code == 204, f"delete failed: {r.status_code}")
    r = client.get(f"/agents/{agent_id}/workflows/{wf_b}")
    check(r.status_code == 404, f"deleted workflow should 404, got {r.status_code}")
    r = client.get(f"/agents/{agent_id}/workflows")
    remaining_ids = {w["workflow_id"] for w in r.json()}
    check(wf_b not in remaining_ids and wf_a in remaining_ids and wf_a_copy in remaining_ids, f"unexpected remaining set: {remaining_ids}")
    print(f"  OK: deleted {wf_b}; the other two workflows are untouched")


def test_editing_one_workflow_never_mutates_another(client: TestClient) -> None:
    print("\n=== editing Workflow A never mutates Workflow B (the critical isolation test) ===")
    agent_id = "agt_isolation_test"
    headers = {"X-User-Id": "tester"}

    wf_a = client.post(f"/agents/{agent_id}/workflows", json={"name": "A"}, headers=headers).json()["metadata"]["id"]
    wf_b = client.post(f"/agents/{agent_id}/workflows", json={"name": "B"}, headers=headers).json()["metadata"]["id"]

    b_before = client.get(f"/agents/{agent_id}/workflows/{wf_b}").json()

    version = client.get(f"/agents/{agent_id}/workflows/{wf_a}").json()["metadata"]["version"]
    r = client.post(
        f"/agents/{wf_a}/mutations",
        json={
            "mutations": [
                {
                    "kind": "add_node",
                    "node": {"id": "n_new", "type": "ask_ai", "category": "ai", "label": "New node", "executionType": "probabilistic", "config": {"prompt": "hi"}},
                },
                {"kind": "add_edge", "edge": {"id": "e_new", "source": "n_trigger", "target": "n_new"}},
            ],
            "expected_version": version,
        },
    )
    check(r.status_code == 200, f"mutation on A failed: {r.status_code} {r.text}")
    a_after = r.json()["spec"]
    check(len(a_after["nodes"]) == 2, f"A should now have 2 nodes, got {len(a_after['nodes'])}")

    b_after = client.get(f"/agents/{agent_id}/workflows/{wf_b}").json()
    check(len(b_after["nodes"]) == 1, f"B must be COMPLETELY unaffected by A's mutation, got {len(b_after['nodes'])} nodes")
    check(b_after["metadata"]["version"] == b_before["metadata"]["version"], "B's version must not have changed")
    check(b_after == b_before, "B must be byte-for-byte identical to before A was edited")
    print(f"  OK: A now has 2 nodes; B is completely untouched (same version, same content)")

    version_b = b_after["metadata"]["version"]
    r = client.post(
        f"/agents/{wf_b}/mutations",
        json={"mutations": [{"kind": "update_metadata", "patch": {"description": "B was edited"}}], "expected_version": version_b},
    )
    check(r.status_code == 200, f"mutation on B failed: {r.status_code} {r.text}")
    a_still = client.get(f"/agents/{agent_id}/workflows/{wf_a}").json()
    check(a_still == a_after, "A must be unaffected by B's mutation")
    print("  OK: reverse direction also isolated -- editing B never touches A")


def test_independent_versions_and_publishing(client: TestClient) -> None:
    print("\n=== independent version history and independent publishing ===")
    agent_id = "agt_versioning_test"
    headers = {"X-User-Id": "tester"}

    wf_a = client.post(f"/agents/{agent_id}/workflows", json={"name": "A"}, headers=headers).json()["metadata"]["id"]
    wf_b = client.post(f"/agents/{agent_id}/workflows", json={"name": "B"}, headers=headers).json()["metadata"]["id"]

    r = client.post(f"/agents/{wf_a}/publish")
    check(r.status_code == 200, f"publish A failed: {r.status_code} {r.text}")
    published_version_a = r.json()["published_version"]

    r = client.get(f"/agents/{wf_b}/published")
    check(r.status_code == 404, f"B has never been published, should 404, got {r.status_code}")
    b_meta = client.get(f"/agents/{agent_id}/workflows/{wf_b}").json()["metadata"]
    check(b_meta["publishedVersion"] is None, f"B's publishedVersion should still be None, got {b_meta['publishedVersion']!r}")
    print(f"  OK: publishing A (-> version {published_version_a}) has no effect on B (still unpublished)")

    r = client.get(f"/agents/{wf_a}/versions")
    check(r.status_code == 200 and len(r.json()) >= 1, f"A should have version history: {r.status_code} {r.text}")
    r = client.get(f"/agents/{wf_b}/versions")
    check(r.status_code == 200, f"B should have its OWN (empty/independent) version list: {r.status_code}")
    print("  OK: version history is independent per workflow")


def test_workflow_aware_execution_and_trace(client: TestClient) -> None:
    print("\n=== workflow-aware execution: each workflow runs and traces independently ===")
    agent_id = "agt_execution_test"
    headers = {"X-User-Id": "tester"}

    wf_a = client.post(f"/agents/{agent_id}/workflows", json={"name": "A"}, headers=headers).json()["metadata"]["id"]
    wf_b = client.post(f"/agents/{agent_id}/workflows", json={"name": "B"}, headers=headers).json()["metadata"]["id"]

    r_a = client.post(f"/agents/{wf_a}/run", json={"seed_variables": {"marker": "A"}})
    r_b = client.post(f"/agents/{wf_b}/run", json={"seed_variables": {"marker": "B"}})
    check(r_a.status_code == 200 and r_b.status_code == 200, f"runs should succeed: {r_a.status_code} {r_b.status_code}")
    check(r_a.json()["thread_id"] != r_b.json()["thread_id"], "each workflow's run must get its own runId (thread_id)")
    check(r_a.json()["variables"]["marker"] == "A", f"A's run should reflect A's own seed variables: {r_a.json()['variables']}")
    check(r_b.json()["variables"]["marker"] == "B", f"B's run should reflect B's own seed variables: {r_b.json()['variables']}")
    print(f"  OK: A's run (thread {r_a.json()['thread_id']}) and B's run (thread {r_b.json()['thread_id']}) are independently scoped and traced")


def test_backward_compatibility_legacy_single_workflow_agent(client: TestClient) -> None:
    print("\n=== backward compatibility: a legacy agent with no agentId set still works ===")
    legacy_id = "agt_legacy_no_agentid_test"
    spec = {
        "metadata": {
            "id": legacy_id,
            "name": "Legacy agent",
            "status": "draft",
            "version": "0.0.1",
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z",
        },
        "goal": "A pre-multi-workflow spec with no agentId/enabled fields at all.",
        "nodes": [{"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"}],
        "edges": [],
    }
    r = client.post("/agents", json={"spec": spec}, headers={"X-User-Id": "tester"})
    check(r.status_code == 201, f"legacy create failed: {r.status_code} {r.text}")
    check(r.json()["metadata"]["agentId"] is None, "agentId should be None on the wire when never set")
    check(r.json()["metadata"]["enabled"] is True, "enabled should default to True")

    r = client.get(f"/agents/{legacy_id}/workflows")
    check(r.status_code == 200, f"unexpected: {r.status_code}")
    ids = {w["workflow_id"] for w in r.json()}
    check(legacy_id in ids, f"a legacy agent should be discoverable as a single-workflow agent under its own id, got {ids}")
    print(f"  OK: a legacy (no agentId) spec is still fully usable, and self-discoverable as a one-workflow agent")

    r = client.post(f"/agents/{legacy_id}/run", json={"seed_variables": {}})
    check(r.status_code == 200 and r.json()["status"] == "completed", f"legacy run should still work exactly as before: {r.status_code} {r.text}")
    print("  OK: legacy run/execution path completely unaffected")


def main() -> None:
    client = TestClient(app)
    test_create_list_rename_duplicate_delete(client)
    test_editing_one_workflow_never_mutates_another(client)
    test_independent_versions_and_publishing(client)
    test_workflow_aware_execution_and_trace(client)
    test_backward_compatibility_legacy_single_workflow_agent(client)
    print("\nAll multi-workflow checks passed.")


if __name__ == "__main__":
    main()
