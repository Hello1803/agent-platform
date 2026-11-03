"""
Regression test for trace/variables preservation on mid-run failure.

The bug: /run, /resume, and the streaming routes' final `done` event all
returned `trace=[], variables={}` whenever a LATER node failed after
EARLIER nodes had already succeeded -- discarding real, already-persisted
work. A run that classified an invoice, extracted its fields, and saved
them to the database, only to fail on the final send_email step, reported
back as if none of that had happened. The frontend then compounded it:
useAgentRunStream.ts's `setVariables(result.variables)` unconditionally
overwrote its variables state with that (empty) payload.

The fix (backend/api/app.py's `_partial_result_after_failure`) reads the
already-checkpointed state back via `graph.get_state(config)` instead of
returning empty literals, and reports the actual failed node via
LangGraph's own `state.next`.

Run:
    PYTHONPATH=. python backend/api/test_trace_preservation.py
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from backend.api.app import app, get_executor_overrides
from backend.models.agent_spec import NodeType


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _failing_send_email_executor(node, spec, variables) -> dict:
    raise ValueError("simulated SMTP failure")


def _spec_dict(agent_id: str) -> dict:
    return {
        "metadata": {
            "id": agent_id,
            "name": "Trace preservation test",
            "status": "draft",
            "version": "0.0.1",
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z",
        },
        "goal": "Test fixture: a node succeeds, then a later node fails.",
        "nodes": [
            {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
            {
                "id": "n_save",
                "type": "database",
                "category": "tool",
                "label": "Save record (will succeed)",
                "toolRefs": ["t_sheets"],
            },
            {
                "id": "n_send",
                "type": "send_email",
                "category": "tool",
                "label": "Send email (will fail)",
                "toolRefs": ["t_gmail"],
            },
        ],
        "edges": [
            {"id": "e1", "source": "n_trigger", "target": "n_save"},
            {"id": "e2", "source": "n_save", "target": "n_send"},
        ],
        "tools": [
            {"id": "t_sheets", "toolId": "google_sheets.append_row", "permissions": ["write"], "mode": "demo"},
            {"id": "t_gmail", "toolId": "gmail.send", "permissions": ["send"], "mode": "demo"},
        ],
    }


def test_sync_run_preserves_partial_state(client: TestClient) -> None:
    print("=== POST /run: partial trace/variables preserved when a later node fails ===")
    agent_id = "agt_trace_sync_test"
    resp = client.post("/agents", json={"spec": _spec_dict(agent_id)})
    check(resp.status_code == 201, f"create failed: {resp.status_code} {resp.text}")

    resp = client.post(f"/agents/{agent_id}/run", json={"seed_variables": {}})
    check(resp.status_code == 200, f"run request itself should succeed (the AGENT run fails, not the HTTP call): {resp.status_code} {resp.text}")
    body = resp.json()

    check(body["status"] == "failed", f"expected status=failed, got {body['status']}")
    check(body["error"] is not None and "simulated SMTP failure" in body["error"], f"unexpected error: {body['error']}")
    check(body["failed_node_id"] == "n_send", f"expected failed_node_id='n_send', got {body['failed_node_id']!r}")
    print(f"  OK: status=failed, failed_node_id={body['failed_node_id']!r}")

    node_ids_in_trace = [event["node_id"] for event in body["trace"]]
    check(node_ids_in_trace == ["n_trigger", "n_save"], f"expected trace for the two successful nodes, got {node_ids_in_trace}")
    print(f"  OK: trace preserved for both successful nodes: {node_ids_in_trace} (NOT the old empty [])")

    check(body["variables"].get("saved") is True, f"expected n_save's real output preserved, got {body['variables']}")
    print(f"  OK: variables preserved from the successful node: {body['variables']} (NOT the old empty {{}})")


def test_streaming_run_preserves_partial_state(client: TestClient) -> None:
    print("\n=== POST /run/stream: final 'done' event preserves partial trace/variables ===")
    agent_id = "agt_trace_stream_test"
    resp = client.post("/agents", json={"spec": _spec_dict(agent_id)})
    check(resp.status_code == 201, f"create failed: {resp.status_code} {resp.text}")

    events: list[tuple[str, dict]] = []
    with client.stream("POST", f"/agents/{agent_id}/run/stream", json={"seed_variables": {}}) as resp:
        check(resp.status_code == 200, f"stream request should succeed: {resp.status_code}")
        event_name = None
        for line in resp.iter_lines():
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: ") and event_name:
                events.append((event_name, json.loads(line[len("data: "):])))
                event_name = None

    event_names = [name for name, _ in events]
    check("done" in event_names, f"stream should end with a done event, got {event_names}")
    done_data = next(data for name, data in events if name == "done")

    check(done_data["status"] == "failed", f"expected status=failed, got {done_data['status']}")
    check(done_data["failed_node_id"] == "n_send", f"expected failed_node_id='n_send', got {done_data.get('failed_node_id')!r}")
    node_ids_in_trace = [event["node_id"] for event in done_data["trace"]]
    check(node_ids_in_trace == ["n_trigger", "n_save"], f"final done event should carry the preserved trace, got {node_ids_in_trace}")
    check(done_data["variables"].get("saved") is True, f"final done event should carry preserved variables, got {done_data['variables']}")
    print(f"  OK: final 'done' event on failure carries the real trace ({node_ids_in_trace}) and variables ({done_data['variables']})")

    live_trace_node_ids = [data["node_id"] for name, data in events if name == "trace"]
    check(live_trace_node_ids == ["n_trigger", "n_save"], f"live trace events should have streamed for both successful nodes: {live_trace_node_ids}")
    print("  OK: live 'trace' SSE events also streamed correctly for both successful nodes before the failure")

    check("node_failed" in event_names, f"a node_failed SSE event should have been forwarded, got {event_names}")
    print("  OK: a distinct 'node_failed' SSE event was forwarded (not mislabeled as 'node_started')")


def main() -> None:
    client = TestClient(app)
    app.dependency_overrides[get_executor_overrides] = lambda: {NodeType.send_email: _failing_send_email_executor}
    try:
        test_sync_run_preserves_partial_state(client)
        test_streaming_run_preserves_partial_state(client)
    finally:
        app.dependency_overrides.pop(get_executor_overrides, None)

    print("\nAll trace preservation checks passed.")


if __name__ == "__main__":
    main()
