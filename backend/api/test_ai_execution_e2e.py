"""
End-to-end verification of the full real-AI-execution path: Manual
Trigger -> Ask AI -> real provider -> SSE trace -> result. Run
independently for Anthropic and for OpenAI.

No real ANTHROPIC_API_KEY/OPENAI_API_KEY is available in this sandbox, so
each provider's REAL AIProvider class (AnthropicProvider / OpenAIProvider
-- unchanged production code) is exercised against an injected fake SDK
client, then wired into the real API via
`app.dependency_overrides[get_executor_overrides]`.

Run:
    PYTHONPATH=. python backend/api/test_ai_execution_e2e.py
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.ai_providers.anthropic_provider import AnthropicProvider
from backend.ai_providers.openai_provider import OpenAIProvider
from backend.api.app import app, get_executor_overrides
from backend.real_executors.ai import build_ai_executors


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _spec_dict(agent_id: str) -> dict:
    return {
        "metadata": {
            "id": agent_id,
            "name": "Ask AI e2e test",
            "status": "draft",
            "version": "0.0.1",
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z",
        },
        "goal": "Manual trigger -> Ask AI end-to-end test fixture.",
        "nodes": [
            {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
            {
                "id": "n_ask",
                "type": "ask_ai",
                "category": "ai",
                "label": "Ask AI",
                "executionType": "probabilistic",
                "config": {"prompt": "What is 2+2?", "output_key": "ai_response"},
            },
        ],
        "edges": [{"id": "e1", "source": "n_trigger", "target": "n_ask"}],
    }


def _sync_run(client: TestClient, agent_id: str) -> dict:
    resp = client.post("/agents", json={"spec": _spec_dict(agent_id)})
    check(resp.status_code == 201, f"create failed: {resp.status_code} {resp.text}")
    resp = client.post(f"/agents/{agent_id}/run", json={"seed_variables": {}})
    check(resp.status_code == 200, f"run failed: {resp.status_code} {resp.text}")
    return resp.json()


def _stream_run(client: TestClient, agent_id: str) -> tuple[list[tuple[str, dict]], dict]:
    resp = client.post("/agents", json={"spec": _spec_dict(agent_id)})
    check(resp.status_code == 201, f"create failed: {resp.status_code} {resp.text}")

    events: list[tuple[str, dict]] = []
    with client.stream("POST", f"/agents/{agent_id}/run/stream", json={"seed_variables": {}}) as resp:
        check(resp.status_code == 200, f"stream request failed: {resp.status_code}")
        event_name = None
        for line in resp.iter_lines():
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: ") and event_name:
                events.append((event_name, json.loads(line[len("data: "):])))
                event_name = None
    done = next(data for name, data in events if name == "done")
    return events, done


class _AnthropicMessages:
    def create(self, **kwargs):
        check("system" in kwargs and "messages" in kwargs, f"unexpected Anthropic call shape: {kwargs}")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Anthropic says: 4")])


class _AnthropicClient:
    def __init__(self):
        self.messages = _AnthropicMessages()


def test_anthropic_end_to_end(client: TestClient) -> None:
    print("=== Anthropic: Manual Trigger -> Ask AI -> real provider -> SSE trace -> result ===")
    provider = AnthropicProvider(client=_AnthropicClient())
    app.dependency_overrides[get_executor_overrides] = lambda: build_ai_executors(provider)
    try:
        body = _sync_run(client, "agt_ask_ai_anthropic_sync")
        check(body["status"] == "completed", f"unexpected status: {body}")
        check(body["variables"]["ai_response"] == "Anthropic says: 4", f"unexpected result: {body['variables']}")
        print(f"  OK (sync /run): {body['variables']}")

        events, done = _stream_run(client, "agt_ask_ai_anthropic_stream")
        event_names = [name for name, _ in events]
        check("node_started" in event_names, f"missing node_started events: {event_names}")
        check("trace" in event_names, f"missing trace events: {event_names}")
        check(done["status"] == "completed", f"unexpected done: {done}")
        check(done["variables"]["ai_response"] == "Anthropic says: 4", f"unexpected result: {done['variables']}")
        trace_node_ids = [e["node_id"] for name, e in events if name == "trace"]
        check(trace_node_ids == ["n_trigger", "n_ask"], f"unexpected trace order: {trace_node_ids}")
        print(f"  OK (SSE stream): events={event_names}, final ai_response={done['variables']['ai_response']!r}")
    finally:
        app.dependency_overrides.pop(get_executor_overrides, None)


class _OpenAICompletions:
    def create(self, **kwargs):
        check("messages" in kwargs and kwargs["messages"][0]["role"] == "system", f"unexpected OpenAI call shape: {kwargs}")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OpenAI says: 4"))])


class _OpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_OpenAICompletions())


def test_openai_end_to_end(client: TestClient) -> None:
    print("\n=== OpenAI: Manual Trigger -> Ask AI -> real provider -> SSE trace -> result ===")
    provider = OpenAIProvider(client=_OpenAIClient())
    app.dependency_overrides[get_executor_overrides] = lambda: build_ai_executors(provider)
    try:
        body = _sync_run(client, "agt_ask_ai_openai_sync")
        check(body["status"] == "completed", f"unexpected status: {body}")
        check(body["variables"]["ai_response"] == "OpenAI says: 4", f"unexpected result: {body['variables']}")
        print(f"  OK (sync /run): {body['variables']}")

        events, done = _stream_run(client, "agt_ask_ai_openai_stream")
        event_names = [name for name, _ in events]
        check("node_started" in event_names, f"missing node_started events: {event_names}")
        check("trace" in event_names, f"missing trace events: {event_names}")
        check(done["status"] == "completed", f"unexpected done: {done}")
        check(done["variables"]["ai_response"] == "OpenAI says: 4", f"unexpected result: {done['variables']}")
        trace_node_ids = [e["node_id"] for name, e in events if name == "trace"]
        check(trace_node_ids == ["n_trigger", "n_ask"], f"unexpected trace order: {trace_node_ids}")
        print(f"  OK (SSE stream): events={event_names}, final ai_response={done['variables']['ai_response']!r}")
    finally:
        app.dependency_overrides.pop(get_executor_overrides, None)


def test_providers_are_not_interchangeable(client: TestClient) -> None:
    print("\n=== providers are not interchangeably mixed up ===")
    anthropic_provider = AnthropicProvider(client=_AnthropicClient())
    app.dependency_overrides[get_executor_overrides] = lambda: build_ai_executors(anthropic_provider)
    try:
        body = _sync_run(client, "agt_ask_ai_identity_check")
        check("Anthropic" in body["variables"]["ai_response"], f"expected the Anthropic fake's text, got {body['variables']}")
        check("OpenAI" not in body["variables"]["ai_response"], "OpenAI's response text should never appear here")
    finally:
        app.dependency_overrides.pop(get_executor_overrides, None)
    print("  OK: the Anthropic-injected run's result is genuinely Anthropic's response text, not OpenAI's")


def main() -> None:
    client = TestClient(app)
    test_anthropic_end_to_end(client)
    test_openai_end_to_end(client)
    test_providers_are_not_interchangeable(client)
    print("\nAll Manual Trigger -> Ask AI -> provider -> SSE -> result checks passed (both providers).")


if __name__ == "__main__":
    main()
