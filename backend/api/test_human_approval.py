"""
Regression tests for P1 "Human Approval": complete pause/approve/reject/
resume behavior, verified both via the synchronous /run+/resume routes
and the SSE streaming routes.

Run:
    PYTHONPATH=. python backend/api/test_human_approval.py
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from backend.api.app import app


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _spec_dict(agent_id: str) -> dict:
    return {
        "metadata": {
            "id": agent_id,
            "name": "Human approval test",
            "status": "draft",
            "version": "0.0.1",
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z",
        },
        "goal": "Test fixture: trigger -> human_approval -> send_email (only if approved).",
        "nodes": [
            {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
            {
                "id": "n_approval",
                "type": "human_approval",
                "category": "human",
                "label": "Approve the send",
                "config": {"actionCategory": "send_message"},
            },
            {"id": "n_send", "type": "send_email", "category": "tool", "label": "Send", "toolRefs": ["t_gmail"]},
        ],
        "edges": [
            {"id": "e1", "source": "n_trigger", "target": "n_approval"},
            {"id": "e2", "source": "n_approval", "target": "n_send", "condition": {"expression": "approved == true", "kind": "deterministic"}},
        ],
        "tools": [{"id": "t_gmail", "toolId": "gmail.send", "permissions": ["send"], "mode": "demo"}],
    }


def test_sync_pause_and_approve(client: TestClient) -> None:
    print("=== sync /run + /resume: pause, then APPROVE ===")
    agent_id = "agt_approval_sync_approve"
    client.post("/agents", json={"spec": _spec_dict(agent_id)})

    resp = client.post(f"/agents/{agent_id}/run", json={"seed_variables": {}})
    check(resp.status_code == 200, f"run failed: {resp.status_code} {resp.text}")
    body = resp.json()
    check(body["status"] == "paused", f"expected paused, got {body['status']}")
    check(body["interrupt"] is not None, "paused response should carry the interrupt payload")
    check(body["interrupt"]["type"] == "human_approval", f"unexpected interrupt type: {body['interrupt']}")
    check(body["interrupt"]["node_id"] == "n_approval", f"interrupt should identify the pausing node: {body['interrupt']}")
    check(body["interrupt"]["action_category"] == "send_message", f"interrupt should carry the node's config: {body['interrupt']}")
    print(f"  OK: paused with a real, complete interrupt payload: {body['interrupt']}")

    thread_id = body["thread_id"]
    resp = client.post(f"/agents/{agent_id}/resume", json={"thread_id": thread_id, "resume": {"approved": True}})
    check(resp.status_code == 200, f"resume failed: {resp.status_code} {resp.text}")
    body = resp.json()
    check(body["status"] == "completed", f"expected completed after approval, got {body}")
    check(body["variables"].get("approved") is True, f"unexpected variables: {body['variables']}")
    check(body["variables"].get("email_sent") is True, f"the send_email node should have run after approval: {body['variables']}")
    print(f"  OK: approved and resumed -> completed, email actually sent: {body['variables']}")


def test_sync_pause_and_reject(client: TestClient) -> None:
    print("\n=== sync /run + /resume: pause, then REJECT ===")
    agent_id = "agt_approval_sync_reject"
    client.post("/agents", json={"spec": _spec_dict(agent_id)})

    resp = client.post(f"/agents/{agent_id}/run", json={"seed_variables": {}})
    body = resp.json()
    check(body["status"] == "paused", f"expected paused, got {body['status']}")
    thread_id = body["thread_id"]

    resp = client.post(
        f"/agents/{agent_id}/resume",
        json={"thread_id": thread_id, "resume": {"approved": False, "reason": "Not authorized"}},
    )
    check(resp.status_code == 200, f"resume failed: {resp.status_code} {resp.text}")
    body = resp.json()
    check(body["status"] == "completed", f"a rejection is still a completed run (of the branch that WAS taken), got {body}")
    check(body["variables"].get("approved") is False, f"unexpected variables: {body['variables']}")
    check(body["variables"].get("approval_reason") == "Not authorized", f"rejection reason should be preserved: {body['variables']}")
    check("email_sent" not in body["variables"], f"send_email must NOT have run after rejection: {body['variables']}")
    print(f"  OK: rejected and resumed -> completed, email correctly NOT sent: {body['variables']}")


def test_streaming_pause_and_approve(client: TestClient) -> None:
    print("\n=== SSE /run/stream + /resume/stream: pause, then APPROVE ===")
    agent_id = "agt_approval_stream_approve"
    client.post("/agents", json={"spec": _spec_dict(agent_id)})

    events: list[tuple[str, dict]] = []
    with client.stream("POST", f"/agents/{agent_id}/run/stream", json={"seed_variables": {}}) as resp:
        check(resp.status_code == 200, f"stream failed: {resp.status_code}")
        event_name = None
        for line in resp.iter_lines():
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: ") and event_name:
                events.append((event_name, json.loads(line[len("data: "):])))
                event_name = None

    done = next(data for name, data in events if name == "done")
    check(done["status"] == "paused", f"expected paused, got {done}")
    check(done["interrupt"]["type"] == "human_approval", f"unexpected interrupt: {done}")
    thread_id = done["thread_id"]
    print(f"  OK: stream paused with interrupt: {done['interrupt']}")

    resume_events: list[tuple[str, dict]] = []
    with client.stream(
        "POST", f"/agents/{agent_id}/resume/stream", json={"thread_id": thread_id, "resume": {"approved": True}}
    ) as resp:
        check(resp.status_code == 200, f"resume stream failed: {resp.status_code}")
        event_name = None
        for line in resp.iter_lines():
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: ") and event_name:
                resume_events.append((event_name, json.loads(line[len("data: "):])))
                event_name = None

    resume_done = next(data for name, data in resume_events if name == "done")
    check(resume_done["status"] == "completed", f"expected completed, got {resume_done}")
    check(resume_done["variables"].get("email_sent") is True, f"unexpected variables: {resume_done['variables']}")
    resume_trace_node_ids = [data["node_id"] for name, data in resume_events if name == "trace"]
    check(
        resume_trace_node_ids == ["n_approval", "n_send"],
        f"resume should stream ONLY the newly-executed nodes (not re-emit n_trigger), got {resume_trace_node_ids}",
    )
    print(f"  OK: resumed via SSE -> completed, only the 2 new nodes streamed: {resume_trace_node_ids}")


def test_streaming_pause_and_reject(client: TestClient) -> None:
    print("\n=== SSE /run/stream + /resume/stream: pause, then REJECT ===")
    agent_id = "agt_approval_stream_reject"
    client.post("/agents", json={"spec": _spec_dict(agent_id)})

    with client.stream("POST", f"/agents/{agent_id}/run/stream", json={"seed_variables": {}}) as resp:
        event_name = None
        events = []
        for line in resp.iter_lines():
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: ") and event_name:
                events.append((event_name, json.loads(line[len("data: "):])))
                event_name = None
    thread_id = next(data for name, data in events if name == "done")["thread_id"]

    resume_events: list[tuple[str, dict]] = []
    with client.stream(
        "POST", f"/agents/{agent_id}/resume/stream", json={"thread_id": thread_id, "resume": {"approved": False}}
    ) as resp:
        event_name = None
        for line in resp.iter_lines():
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: ") and event_name:
                resume_events.append((event_name, json.loads(line[len("data: "):])))
                event_name = None

    resume_done = next(data for name, data in resume_events if name == "done")
    check(resume_done["status"] == "completed", f"unexpected: {resume_done}")
    check("email_sent" not in resume_done["variables"], f"send_email must NOT have run: {resume_done['variables']}")
    resume_trace_node_ids = [data["node_id"] for name, data in resume_events if name == "trace"]
    check(resume_trace_node_ids == ["n_approval"], f"only the approval node should have run (the send edge's condition failed), got {resume_trace_node_ids}")
    print(f"  OK: resumed via SSE after rejection -> completed, send_email correctly never ran: trace={resume_trace_node_ids}")


def main() -> None:
    client = TestClient(app)
    test_sync_pause_and_approve(client)
    test_sync_pause_and_reject(client)
    test_streaming_pause_and_approve(client)
    test_streaming_pause_and_reject(client)
    print("\nAll Human Approval pause/approve/reject/resume checks passed (sync + streaming).")


if __name__ == "__main__":
    main()
