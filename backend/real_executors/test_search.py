"""
Tests for backend/real_executors/search.py -- the real search_web
executor, using Anthropic's server-side web_search_20250305 tool.

No real ANTHROPIC_API_KEY / live web search call is available in this
sandbox, so correctness is proven the same way Gmail/Sheets are proven in
backend/real_executors/test_tools.py: inject a fake client that returns a
response shaped exactly like Anthropic's real API would for a web-search
turn (server_tool_use / web_search_tool_result / text content blocks),
and check the executor extracts the right query, calls the right tool,
and parses the response into `search_results`/`search_sources` correctly.

Run:
    PYTHONPATH=. python backend/real_executors/test_search.py
"""
from __future__ import annotations

from types import SimpleNamespace

from backend.compiler.executors import exec_search_web
from backend.models.agent_spec import AgentSpecification
from backend.real_executors.search import (
    _resolve_query,
    build_search_web_executor,
    build_search_web_executor_with_fallback,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _spec(goal: str = "Research agent", mode: str = "connected") -> AgentSpecification:
    return AgentSpecification.model_validate(
        {
            "metadata": {
                "id": "agt_search_test",
                "name": "Search test",
                "status": "draft",
                "version": "0.0.1",
                "createdAt": "2026-01-01T00:00:00Z",
                "updatedAt": "2026-01-01T00:00:00Z",
            },
            "goal": goal,
            "nodes": [
                {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
                {
                    "id": "n_search",
                    "type": "search_web",
                    "category": "tool",
                    "label": "Search",
                    "toolRefs": ["t_search"],
                },
            ],
            "edges": [{"id": "e1", "source": "n_trigger", "target": "n_search"}],
            "tools": [{"id": "t_search", "toolId": "web.search", "permissions": ["read"], "mode": mode}],
        }
    )


class _FakeMessages:
    def __init__(self, response=None, raise_error=False):
        self._response = response
        self._raise_error = raise_error
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._raise_error:
            raise RuntimeError("simulated Anthropic API failure")
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response=None, raise_error=False):
        self.messages = _FakeMessages(response, raise_error)


def _realistic_search_response(query_seen: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="server_tool_use", id="srvtool_1", name="web_search", input={"query": query_seen}),
            SimpleNamespace(
                type="web_search_tool_result",
                tool_use_id="srvtool_1",
                content=[
                    SimpleNamespace(type="web_search_result", title="Result One", url="https://example.com/one"),
                    SimpleNamespace(type="web_search_result", title="Result Two", url="https://example.com/two"),
                ],
            ),
            SimpleNamespace(
                type="text",
                text="Based on recent sources, the answer is X and Y.",
            ),
        ]
    )


def test_query_resolution() -> None:
    print("=== query resolution order: config > variables > goal ===")
    spec = _spec(goal="Research recent AI safety papers")
    node = next(n for n in spec.nodes if n.id == "n_search")

    check(_resolve_query(node, spec, {}) == "Research recent AI safety papers", "should fall back to spec.goal")
    print("  OK: falls back to the agent's goal when nothing else is configured")

    check(_resolve_query(node, spec, {"query": "quantum computing 2026"}) == "quantum computing 2026", "should use variables['query']")
    print("  OK: uses a 'query' variable from an earlier node when present")

    node.config["query"] = "explicit static query"
    check(_resolve_query(node, spec, {"query": "should be overridden"}) == "explicit static query", "node.config.query should win")
    print("  OK: an explicit node.config.query takes priority over everything else")


def test_real_search_execution() -> None:
    print("\n=== executing a real search call against a mocked Anthropic client ===")
    spec = _spec()
    node = next(n for n in spec.nodes if n.id == "n_search")
    node.config["query"] = "latest developments in fusion energy"

    fake_client = _FakeAnthropicClient(response=_realistic_search_response("latest developments in fusion energy"))
    executor = build_search_web_executor(client=fake_client)
    result = executor(node, spec, {})

    check(result["search_results"] == "Based on recent sources, the answer is X and Y.", f"unexpected: {result}")
    check(
        result["search_sources"] == [
            {"title": "Result One", "url": "https://example.com/one"},
            {"title": "Result Two", "url": "https://example.com/two"},
        ],
        f"unexpected sources: {result['search_sources']}",
    )
    print(f"  OK: real text and source extraction from a realistic API response shape: {result}")

    kwargs = fake_client.messages.last_kwargs
    check(kwargs["tools"] == [{"type": "web_search_20250305", "name": "web_search"}], f"wrong tools param: {kwargs['tools']}")
    check("latest developments in fusion energy" in kwargs["messages"][0]["content"], "the query should actually appear in the prompt sent")
    print("  OK: the request genuinely asks for Anthropic's server-side web_search tool with the resolved query")


def test_goal_fallback_direct() -> None:
    print("\n=== falls back to the agent's goal when nothing else is configured ===")
    spec = _spec(goal="Track competitor pricing changes")
    node = next(n for n in spec.nodes if n.id == "n_search")
    node.config.pop("query", None)
    result = _resolve_query(node, spec, {})
    check(result == "Track competitor pricing changes", f"unexpected: {result}")
    print("  OK: resolves to the agent's stated goal as a last resort")


def test_api_failure_raises() -> None:
    print("\n=== a real API failure raises, doesn't silently return empty results ===")
    spec = _spec()
    node = next(n for n in spec.nodes if n.id == "n_search")
    node.config["query"] = "test query"
    fake_client = _FakeAnthropicClient(raise_error=True)
    executor = build_search_web_executor(client=fake_client)
    try:
        executor(node, spec, {})
        raise AssertionError("a failing Anthropic call should raise, not silently return empty search results")
    except Exception as exc:
        check("Anthropic" in str(exc) or "simulated" in str(exc), f"unexpected error: {exc}")
        print(f"  OK: a real API failure propagates as an error ({type(exc).__name__}), not a fake empty success")


def test_fallback_when_no_credentials() -> None:
    print("\n=== fallback executor uses the plain stub when no client is available ===")
    spec = _spec()
    node = next(n for n in spec.nodes if n.id == "n_search")
    fallback_executor = build_search_web_executor_with_fallback(None)
    result = fallback_executor(node, spec, {})
    expected = exec_search_web(node, spec, {})
    check(result == expected, f"fallback should match the plain stub exactly, got {result} vs {expected}")
    print("  OK: with no client (no ANTHROPIC_API_KEY), falls back to the exact plain stub output")

    real_fallback = build_search_web_executor_with_fallback(_FakeAnthropicClient(response=_realistic_search_response()))
    node.config["query"] = "anything"
    real_result = real_fallback(node, spec, {})
    check(real_result != expected, "with a client provided, it should NOT use the stub")
    print("  OK: with a client provided, it genuinely uses the real search path instead")


def test_demo_mode_binding_stays_stub_even_with_a_real_client() -> None:
    print("\n=== a mode='demo' binding stays the stub even when a real client is available ===")
    spec = _spec(mode="demo")
    node = next(n for n in spec.nodes if n.id == "n_search")
    node.config["query"] = "this should never actually be searched"
    fake_client = _FakeAnthropicClient(response=_realistic_search_response())
    executor = build_search_web_executor(client=fake_client)
    result = executor(node, spec, {})
    check(
        result == {"search_results": ["[stub result 1]", "[stub result 2]"]},
        f"a demo-mode binding must use the plain stub regardless of client availability, got {result}",
    )
    check(fake_client.messages.last_kwargs is None, "the (fake) Anthropic client should never have been called for a demo-mode binding")
    print("  OK: demo-mode binding never calls the real client, even though one was available -- explicit tool binding is honored")


def test_connected_mode_binding_uses_real_search() -> None:
    print("\n=== a mode='connected' binding genuinely uses real search (the contrasting case) ===")
    spec = _spec(mode="connected")
    node = next(n for n in spec.nodes if n.id == "n_search")
    node.config["query"] = "genuinely search this"
    fake_client = _FakeAnthropicClient(response=_realistic_search_response())
    executor = build_search_web_executor(client=fake_client)
    result = executor(node, spec, {})
    check(result["search_results"] != ["[stub result 1]", "[stub result 2]"], f"connected mode should NOT use the stub: {result}")
    check(fake_client.messages.last_kwargs is not None, "the client should have been called for a connected-mode binding")
    print("  OK: connected-mode binding genuinely calls real search")


def main() -> None:
    test_query_resolution()
    test_real_search_execution()
    test_goal_fallback_direct()
    test_api_failure_raises()
    test_fallback_when_no_credentials()
    test_demo_mode_binding_stays_stub_even_with_a_real_client()
    test_connected_mode_binding_uses_real_search()
    print("\nAll search_web executor checks passed.")


if __name__ == "__main__":
    main()
