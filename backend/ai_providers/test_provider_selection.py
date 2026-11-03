"""
Regression tests for the "AI provider/model configuration" P0 work.

Run:
    PYTHONPATH=. python backend/ai_providers/test_provider_selection.py
"""
from __future__ import annotations

from types import SimpleNamespace

from pydantic import ValidationError

from backend.ai_providers.anthropic_provider import AnthropicProvider
from backend.ai_providers.openai_provider import OpenAIProvider
from backend.compiler.compiler import CompileError, compile_agent
from backend.models.agent_spec import AgentNode, AgentSpecification
from backend.real_executors.ai import build_ai_executors
from fastapi.testclient import TestClient


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _spec_dict(node_config: dict) -> dict:
    return {
        "metadata": {
            "id": "agt_provider_selection_test",
            "name": "Provider selection test",
            "status": "draft",
            "version": "0.0.1",
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z",
        },
        "goal": "Test fixture for per-node AI provider selection.",
        "nodes": [
            {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
            {"id": "n_ask", "type": "ask_ai", "category": "ai", "label": "Ask AI", "config": node_config},
        ],
        "edges": [{"id": "e1", "source": "n_trigger", "target": "n_ask"}],
    }


def test_node_level_validation() -> None:
    print("=== AgentNode.config.provider/model validation ===")
    AgentNode.model_validate(
        {"id": "n1", "type": "ask_ai", "category": "ai", "label": "Ask", "config": {"provider": "openai", "model": "gpt-4o"}}
    )
    AgentNode.model_validate({"id": "n1", "type": "ask_ai", "category": "ai", "label": "Ask", "config": {}})
    print("  OK: a matching provider/model pair, and no provider at all, both validate")

    for bad_config, why in [
        ({"provider": "openai", "model": "claude-sonnet-5"}, "Anthropic model under openai"),
        ({"provider": "anthropic", "model": "gpt-4o"}, "OpenAI model under anthropic"),
        ({"provider": "gemini", "model": "gemini-2.5-pro"}, "planned-not-implemented provider"),
        ({"provider": "made_up_provider"}, "unknown provider"),
    ]:
        try:
            AgentNode.model_validate({"id": "n1", "type": "ask_ai", "category": "ai", "label": "Ask", "config": bad_config})
            raise AssertionError(f"should have rejected {bad_config!r} ({why})")
        except ValidationError:
            pass
    print("  OK: cross-provider models, planned providers, and unknown providers are all rejected")


def test_compile_time_strict_rejection_for_node_override() -> None:
    print("\n=== per-node explicit provider selection: missing credentials is a clear compile error ===")
    spec = AgentSpecification.model_validate(
        _spec_dict({"provider": "openai", "model": "gpt-4o", "prompt": "hi"})
    )
    try:
        compile_agent(spec)
        raise AssertionError("should have raised -- no OPENAI_API_KEY is configured in this test environment")
    except CompileError as e:
        check("openai" in str(e) and "OPENAI_API_KEY" in str(e), f"unexpected error: {e}")
        check(e.node_id == "n_ask", f"error should be attributed to the node, got node_id={e.node_id!r}")
        print(f"  OK: compile_agent raises a clear, node-attributed error instead of silently falling back: {e}")


def test_spec_wide_preference_stays_lenient() -> None:
    print("\n=== spec-WIDE runtimeConfig.model keeps its original lenient fallback (scope boundary) ===")
    spec_dict = _spec_dict({"prompt": "hi"})  # no per-node provider override
    spec_dict["runtimeConfig"] = {"model": "openai:gpt-4o"}  # spec-wide preference, no credentials configured
    spec = AgentSpecification.model_validate(spec_dict)
    compile_agent(spec)  # must NOT raise -- this is the pre-existing, deliberately lenient path
    print("  OK: a spec-wide provider preference with no matching credentials still compiles (falls back to server default)")


def test_executor_actually_uses_the_selected_provider() -> None:
    print("\n=== the executor genuinely uses the node's selected provider, not just validates it ===")

    spec = AgentSpecification.model_validate(_spec_dict({"provider": "openai", "prompt": "hi"}))
    node = next(n for n in spec.nodes if n.id == "n_ask")

    calls = []

    class _RecordingOpenAIClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    calls.append(kwargs)
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="openai said hi"))])

    class _RecordingAnthropicClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise AssertionError("the DEFAULT (anthropic) provider should NOT be called -- the node explicitly selected openai")

    # build_ai_executors is given ANTHROPIC as the default provider (as
    # get_executor_overrides would if AI_PROVIDER=anthropic) -- proving
    # the node's own config.provider="openai" overrides it at call time.
    default_provider = AnthropicProvider(client=_RecordingAnthropicClient())
    executors = build_ai_executors(default_provider)

    from backend.ai_providers import registry as registry_module

    real_build_provider = registry_module.build_provider

    def fake_build_provider(name, model=None):
        check(name == "openai", f"should resolve the node's selected provider 'openai', got {name!r}")
        return OpenAIProvider(client=_RecordingOpenAIClient(), model=model or "gpt-4o")

    registry_module.build_provider = fake_build_provider
    import os

    had_key = "OPENAI_API_KEY" in os.environ
    old_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "test-key-not-real"  # only need has_credentials() to see something present
    try:
        result = executors[node.type](node, spec, {})
    finally:
        registry_module.build_provider = real_build_provider
        if had_key:
            os.environ["OPENAI_API_KEY"] = old_key
        else:
            del os.environ["OPENAI_API_KEY"]

    check(result == {"ai_response": "openai said hi"}, f"unexpected result: {result}")
    check(len(calls) == 1, f"expected exactly one OpenAI call, got {len(calls)}")
    print(f"  OK: the node's explicitly-selected provider (openai) was genuinely used, not the default (anthropic): {result}")


def test_ai_providers_endpoint_exposes_models() -> None:
    print("\n=== /ai/providers exposes model information for the frontend ===")
    from backend.api.app import app

    client = TestClient(app)
    resp = client.get("/ai/providers")
    check(resp.status_code == 200, f"unexpected status: {resp.status_code}")
    body = resp.json()
    provider_ids = {p["id"] for p in body["providers"]}
    check({"anthropic", "openai", "gemini"} <= provider_ids, f"unexpected providers: {provider_ids}")
    for provider in body["providers"]:
        if provider["status"] == "available":
            check("models" in provider and len(provider["models"]) > 0, f"available provider missing models: {provider}")
            check(all("id" in m and "label" in m for m in provider["models"]), f"malformed models list: {provider}")
    print(f"  OK: /ai/providers response includes a models list per available provider: {[p['id'] for p in body['providers']]}")
    check("apiKey" not in resp.text and "api_key" not in resp.text, "response must never leak a key field name, let alone a value")
    print("  OK: no secret material or key field names anywhere in the response")


def main() -> None:
    test_node_level_validation()
    test_compile_time_strict_rejection_for_node_override()
    test_spec_wide_preference_stays_lenient()
    test_executor_actually_uses_the_selected_provider()
    test_ai_providers_endpoint_exposes_models()
    print("\nAll AI provider/model selection checks passed.")


if __name__ == "__main__":
    main()
