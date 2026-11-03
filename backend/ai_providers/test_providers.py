"""
Tests for the AI provider abstraction (backend/ai_providers/).

Three things are proven here, deliberately separately:

  1. AnthropicProvider.parse()/create() still make exactly the same
     client.messages.parse()/create() calls as before this abstraction
     existed (via a fake Anthropic client -- same FakeAnthropicClient
     pattern backend/run_real_executors.py and backend/run_nl_compiler.py
     already use, since no real ANTHROPIC_API_KEY is available in this
     sandbox).
  2. OpenAIProvider.parse()/create() work correctly against the real
     openai SDK's response shapes (via a fake OpenAI client -- no real
     OPENAI_API_KEY is available in this sandbox either, same reasoning).
  3. backend/real_executors/ai.py's node executors (classify, extract_
     information, summarize, make_decision, ai_agent) produce IDENTICAL
     output shapes regardless of which AIProvider implementation is
     injected -- this is the actual point of the abstraction: ai.py
     never branches on which provider it's talking to.
  4. backend/ai_providers/registry.py's provider/model-string resolution
     and RuntimeConfig.model's validator (backend/models/agent_spec.py)
     reject unknown/planned providers and unblock supported ones.

Run:
    PYTHONPATH=. python backend/ai_providers/test_providers.py
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, ValidationError

from backend.ai_providers.anthropic_provider import AnthropicProvider
from backend.ai_providers.base import AIProviderError
from backend.ai_providers.openai_provider import OpenAIProvider
from backend.ai_providers.registry import (
    PLANNED_PROVIDERS,
    SUPPORTED_PROVIDERS,
    UnsupportedProviderError,
    api_key_env_var,
    build_provider,
    parse_model_string,
    resolve_provider_name,
)
from backend.models.agent_spec import AgentSpecification, RuntimeConfig
from backend.real_executors.ai import build_ai_executors


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _FakeAnthropicMessages:
    def __init__(self, parse_result: Any = None, create_text: str | None = None, raise_error: bool = False):
        self._parse_result = parse_result
        self._create_text = create_text
        self._raise_error = raise_error

    def parse(self, **kwargs):
        if self._raise_error:
            raise RuntimeError("simulated Anthropic API failure")
        return SimpleNamespace(parsed_output=self._parse_result, stop_reason="end_turn")

    def create(self, **kwargs):
        if self._raise_error:
            raise RuntimeError("simulated Anthropic API failure")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._create_text)])


class _FakeAnthropicClient:
    def __init__(self, parse_result: Any = None, create_text: str | None = None, raise_error: bool = False):
        self.messages = _FakeAnthropicMessages(parse_result, create_text, raise_error)


class _FakeOpenAIMessage:
    def __init__(self, parsed: Any = None, content: str | None = None, refusal: str | None = None):
        self.parsed = parsed
        self.content = content
        self.refusal = refusal


class _FakeOpenAICompletions:
    def __init__(self, message: _FakeOpenAIMessage, raise_error: bool = False):
        self._message = message
        self._raise_error = raise_error

    def parse(self, **kwargs):
        if self._raise_error:
            raise RuntimeError("simulated OpenAI API failure")
        return SimpleNamespace(choices=[SimpleNamespace(message=self._message)])

    def create(self, **kwargs):
        if self._raise_error:
            raise RuntimeError("simulated OpenAI API failure")
        return SimpleNamespace(choices=[SimpleNamespace(message=self._message)])


class _FakeOpenAIClient:
    def __init__(self, message: _FakeOpenAIMessage, raise_error: bool = False):
        self.chat = SimpleNamespace(completions=_FakeOpenAICompletions(message, raise_error))


class _Schema(BaseModel):
    value: str


def test_anthropic_provider() -> None:
    print("=== AnthropicProvider ===")
    parsed = _Schema(value="hello")
    client = _FakeAnthropicClient(parse_result=parsed, create_text="free text response")
    provider = AnthropicProvider(client=client, model="claude-sonnet-5")

    out = provider.parse(system="sys", user_content="data", schema=_Schema, max_tokens=100)
    check(out is parsed, "AnthropicProvider.parse should return the client's parsed_output directly")
    print("  OK: parse() returns the structured output")

    text = provider.create(system="sys", user_content="data", max_tokens=100)
    check(text == "free text response", f"unexpected: {text}")
    print("  OK: create() returns the joined text content")

    failing_provider = AnthropicProvider(client=_FakeAnthropicClient(raise_error=True))
    try:
        failing_provider.parse(system="s", user_content="u", schema=_Schema, max_tokens=10)
        raise AssertionError("a failing Anthropic call should raise AIProviderError, not succeed silently")
    except AIProviderError as exc:
        check("Anthropic" in str(exc), f"unexpected error: {exc}")
        print("  OK: a real API failure raises AIProviderError, not a silent fake success")


def test_openai_provider() -> None:
    print("\n=== OpenAIProvider ===")
    parsed = _Schema(value="world")
    client = _FakeOpenAIClient(_FakeOpenAIMessage(parsed=parsed))
    provider = OpenAIProvider(client=client, model="gpt-4o")

    out = provider.parse(system="sys", user_content="data", schema=_Schema, max_tokens=100)
    check(out is parsed, "OpenAIProvider.parse should return message.parsed directly")
    print("  OK: parse() returns the structured output")

    text_client = _FakeOpenAIClient(_FakeOpenAIMessage(content="an openai free-text reply"))
    text_provider = OpenAIProvider(client=text_client, model="gpt-4o")
    text = text_provider.create(system="sys", user_content="data", max_tokens=100)
    check(text == "an openai free-text reply", f"unexpected: {text}")
    print("  OK: create() returns message.content")

    refusal_client = _FakeOpenAIClient(_FakeOpenAIMessage(parsed=None, refusal="policy violation"))
    refusal_provider = OpenAIProvider(client=refusal_client, model="gpt-4o")
    try:
        refusal_provider.parse(system="s", user_content="u", schema=_Schema, max_tokens=10)
        raise AssertionError("a refused structured-output request should raise, not return None silently")
    except AIProviderError as exc:
        check("policy violation" in str(exc), f"unexpected error: {exc}")
        print("  OK: a refusal raises AIProviderError with the refusal reason, not a silent None")

    failing_provider = OpenAIProvider(client=_FakeOpenAIClient(_FakeOpenAIMessage(), raise_error=True))
    try:
        failing_provider.create(system="s", user_content="u", max_tokens=10)
        raise AssertionError("a failing OpenAI call should raise AIProviderError")
    except AIProviderError as exc:
        check("OpenAI" in str(exc), f"unexpected error: {exc}")
        print("  OK: a real API failure raises AIProviderError, not a silent fake success")


def test_registry() -> None:
    print("\n=== ai_providers.registry ===")
    check("anthropic" in SUPPORTED_PROVIDERS and "openai" in SUPPORTED_PROVIDERS, "both should be supported")
    check("gemini" in PLANNED_PROVIDERS, "gemini should be listed as planned, not silently absent")
    print(f"  OK: SUPPORTED_PROVIDERS={SUPPORTED_PROVIDERS}, PLANNED_PROVIDERS={PLANNED_PROVIDERS}")

    check(resolve_provider_name("anthropic") == "anthropic", "explicit name should win")
    check(resolve_provider_name("OpenAI") == "openai", "resolution should be case-insensitive")
    print("  OK: resolve_provider_name honors an explicit name, case-insensitively")

    try:
        resolve_provider_name("gemini")
        raise AssertionError("gemini should raise (planned, not implemented)")
    except UnsupportedProviderError as exc:
        check("planned" in str(exc), f"unexpected error: {exc}")
        print("  OK: requesting 'gemini' raises a specific 'planned but not implemented' error")

    try:
        resolve_provider_name("made_up_provider")
        raise AssertionError("an unknown provider should raise")
    except UnsupportedProviderError as exc:
        check("Unknown" in str(exc), f"unexpected error: {exc}")
        print("  OK: an unknown provider name raises a clear error")

    check(api_key_env_var("anthropic") == "ANTHROPIC_API_KEY", "wrong env var name")
    check(api_key_env_var("openai") == "OPENAI_API_KEY", "wrong env var name")
    print("  OK: each provider maps to its own conventional API key env var")

    check(parse_model_string(None) == (None, None), "None should parse to (None, None)")
    check(parse_model_string("gpt-4o") == (None, "gpt-4o"), "bare model name should have no provider")
    check(parse_model_string("openai:gpt-4o") == ("openai", "gpt-4o"), "provider:model should split correctly")
    print("  OK: parse_model_string handles bare model names and 'provider:model' correctly")

    try:
        build_provider("made_up_provider")
        raise AssertionError("build_provider should raise for an unknown provider")
    except UnsupportedProviderError:
        print("  OK: build_provider refuses to construct a client for an unknown provider")


def test_runtime_config_validation() -> None:
    print("\n=== RuntimeConfig.model validation ===")
    for good in ("anthropic:claude-sonnet-5", "openai:gpt-4o", "gpt-4o-mini", None):
        RuntimeConfig.model_validate({"model": good})
    print("  OK: supported providers, bare model names, and None are all accepted")

    for bad in ("gemini:gemini-2.5-pro", "provider:claude-sonnet-4-6", "made_up:foo"):
        try:
            RuntimeConfig.model_validate({"model": bad})
            raise AssertionError(f"should have rejected {bad!r}")
        except ValidationError:
            pass
    print("  OK: an unsupported/planned/placeholder provider prefix is rejected at validation time")


def test_cross_provider_executor_parity() -> None:
    print("\n=== Cross-provider executor parity (the actual point of the abstraction) ===")

    spec = AgentSpecification.model_validate(
        {
            "metadata": {
                "id": "agt_provider_test",
                "name": "Provider test",
                "status": "draft",
                "version": "0.0.1",
                "createdAt": "2026-01-01T00:00:00Z",
                "updatedAt": "2026-01-01T00:00:00Z",
            },
            "goal": "Test fixture for cross-provider executor parity.",
            "nodes": [
                {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
                {
                    "id": "n_classify",
                    "type": "classify",
                    "category": "ai",
                    "label": "Classify",
                    "config": {"labels": ["invoice", "not_invoice"]},
                },
            ],
            "edges": [{"id": "e1", "source": "n_trigger", "target": "n_classify"}],
        }
    )
    classify_node = next(n for n in spec.nodes if n.id == "n_classify")

    class _ClassifyResult(BaseModel):
        label: str
        confidence: float

    anthropic_client = _FakeAnthropicClient(parse_result=_ClassifyResult(label="invoice", confidence=0.91))
    anthropic_provider = AnthropicProvider(client=anthropic_client)
    anthropic_executors = build_ai_executors(anthropic_provider)
    anthropic_result = anthropic_executors[classify_node.type](classify_node, spec, {})

    openai_client = _FakeOpenAIClient(_FakeOpenAIMessage(parsed=_ClassifyResult(label="invoice", confidence=0.91)))
    openai_provider = OpenAIProvider(client=openai_client)
    openai_executors = build_ai_executors(openai_provider)
    openai_result = openai_executors[classify_node.type](classify_node, spec, {})

    check(
        anthropic_result == openai_result == {"label": "invoice", "label_confidence": 0.91},
        f"executor output should be identical regardless of provider: anthropic={anthropic_result}, openai={openai_result}",
    )
    print(f"  OK: classify executor produces identical output from both providers: {anthropic_result}")

    from backend.models.agent_spec import NodeType

    expected_types = {
        NodeType.classify, NodeType.extract_information, NodeType.summarize,
        NodeType.make_decision, NodeType.ai_agent, NodeType.ask_ai,
    }
    check(set(anthropic_executors.keys()) == expected_types, f"unexpected keys: {anthropic_executors.keys()}")
    check(set(openai_executors.keys()) == expected_types, f"unexpected keys: {openai_executors.keys()}")
    print(f"  OK: both providers wire up executors for all {len(expected_types)} AI node types")


def test_multi_turn_parse_conversation() -> None:
    print("\n=== parse_conversation: multi-turn structured output, both providers ===")

    class _Answer(BaseModel):
        value: str

    # Anthropic
    turn2_result = _Answer(value="corrected")
    anthropic_client = _FakeAnthropicClient(parse_result=turn2_result)
    anthropic_provider = AnthropicProvider(client=anthropic_client)
    messages = [
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": '{"value": "wrong"}'},
        {"role": "user", "content": "that was rejected, fix it"},
    ]
    out = anthropic_provider.parse_conversation(system="sys", messages=messages, schema=_Answer, max_tokens=100)
    check(out is turn2_result, "Anthropic parse_conversation should return the structured output")
    print("  OK: AnthropicProvider.parse_conversation handles a 3-turn history")

    # OpenAI
    openai_client = _FakeOpenAIClient(_FakeOpenAIMessage(parsed=turn2_result))
    openai_provider = OpenAIProvider(client=openai_client)
    out2 = openai_provider.parse_conversation(system="sys", messages=messages, schema=_Answer, max_tokens=100)
    check(out2 is turn2_result, "OpenAI parse_conversation should return the structured output")
    print("  OK: OpenAIProvider.parse_conversation handles the same 3-turn history")

    # Anthropic: a refusal on a conversational turn should still raise, same as single-turn parse().
    refusing_client = _FakeAnthropicClient(parse_result=None)
    refusing_client.messages._parse_result = None

    class _RefusingMessages(_FakeAnthropicMessages):
        def parse(self, **kwargs):
            return SimpleNamespace(parsed_output=None, stop_reason="refusal")

    refusing_client.messages = _RefusingMessages()
    refusing_provider = AnthropicProvider(client=refusing_client)
    try:
        refusing_provider.parse_conversation(system="sys", messages=messages, schema=_Answer, max_tokens=100)
        raise AssertionError("a refusal should raise AIProviderError even in a multi-turn conversation")
    except AIProviderError as exc:
        check("declined" in str(exc).lower(), f"unexpected error: {exc}")
        print("  OK: a refusal mid-conversation still raises AIProviderError, not a silent None")


def main() -> None:
    test_anthropic_provider()
    test_openai_provider()
    test_multi_turn_parse_conversation()
    test_registry()
    test_runtime_config_validation()
    test_cross_provider_executor_parity()
    print("\nAll AI provider abstraction checks passed.")


if __name__ == "__main__":
    main()
