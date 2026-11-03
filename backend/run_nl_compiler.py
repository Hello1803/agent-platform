"""
Demonstrates the NL compiler end-to-end WITHOUT calling the real Anthropic
API (this sandbox has no API key configured) — a FakeAnthropicClient stands
in for `anthropic.Anthropic()`, returning pre-scripted responses shaped
exactly like `client.messages.parse()`'s real return value. Swapping to a
live model is only ever `compile_instruction(spec, msg)` with no `client=`
argument — everything downstream (prompt construction, structured-output
schema, mutation application, retry-on-error) is exercised identically
either way.

Run:
    PYTHONPATH=. python backend/run_nl_compiler.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.models.agent_spec import AgentSpecification
from backend.ai_providers.anthropic_provider import AnthropicProvider
from backend.nl_compiler.compiler import compile_instruction
from backend.nl_compiler.schemas import MutationBatch

ROOT = Path(__file__).resolve().parents[1]


def load_spec(name: str) -> AgentSpecification:
    return AgentSpecification.model_validate(json.loads((ROOT / "examples" / name).read_text()))


# ---- Fake Anthropic client -------------------------------------------

@dataclass
class FakeParsedResponse:
    parsed_output: MutationBatch
    stop_reason: str = "end_turn"


class FakeMessages:
    def __init__(self, scripted_batches: list[dict[str, Any]]):
        self._queue = [MutationBatch.model_validate(b) for b in scripted_batches]
        self.calls = 0

    def parse(self, **kwargs) -> FakeParsedResponse:
        self.calls += 1
        if not self._queue:
            raise RuntimeError("FakeMessages: no more scripted responses left.")
        return FakeParsedResponse(parsed_output=self._queue.pop(0))


class FakeAnthropicClient:
    def __init__(self, scripted_batches: list[dict[str, Any]]):
        self.messages = FakeMessages(scripted_batches)


# ---- Demos --------------------------------------------------------------

def demo_happy_path() -> None:
    print("\n=== NL instruction: 'Translate the summary into Spanish before finishing' ===")
    spec = load_spec("research-agent.json")

    # What we'd expect a well-behaved model to emit for this instruction,
    # scripted here in place of a live API call.
    fake_client = FakeAnthropicClient([{
        "reasoning": "Added a translation step after the summary, and wired it in as the new final node.",
        "mutations": [
            {
                "kind": "add_node",
                "node": {
                    "id": "n_translate",
                    "type": "ask_ai",
                    "category": "ai",
                    "label": "Translate to Spanish",
                    "executionType": "probabilistic",
                    "confidenceThreshold": 0.6,
                    "config": {"target_language": "es"},
                },
            },
            {"kind": "add_edge", "edge": {"id": "e3", "source": "n_summarize", "target": "n_translate"}},
        ],
    }])

    outcome = compile_instruction(
        spec, "Translate the summary into Spanish before finishing.", provider=AnthropicProvider(client=fake_client)
    )
    print(f"  reasoning: {outcome.reasoning}")
    print(f"  attempts: {outcome.attempts}, ok: {outcome.result.ok}")
    assert outcome.result.ok, outcome.result.errors
    new_spec = outcome.result.spec
    assert any(n.id == "n_translate" for n in new_spec.nodes)
    assert new_spec.metadata.version == "1.2.1"
    print(f"  OK: spec now has {len(new_spec.nodes)} nodes, version bumped to {new_spec.metadata.version}.")


def demo_retry_on_bad_reference() -> None:
    print("\n=== NL instruction with a model mistake: retries after validation catches it ===")
    spec = load_spec("invoice-agent.json")

    # First attempt: the model hallucinates an edge from a node id that
    # doesn't exist in this spec (a very real failure mode — e.g. it
    # confuses this agent with a different one, or misremembers an id).
    # Second attempt (after seeing the error): corrects itself.
    fake_client = FakeAnthropicClient([
        {
            "reasoning": "Adding a Slack notification after the invoice is saved.",
            "mutations": [
                {
                    "kind": "add_node",
                    "node": {"id": "n_notify", "type": "api_call", "category": "tool", "label": "Notify Slack"},
                },
                # WRONG: "n_saved" doesn't exist — the real node id is "n_save"
                {"kind": "add_edge", "edge": {"id": "e_notify", "source": "n_saved", "target": "n_notify"}},
            ],
        },
        {
            "reasoning": "Corrected: wired the Slack notification from the actual save node, n_save.",
            "mutations": [
                {
                    "kind": "add_node",
                    "node": {"id": "n_notify", "type": "api_call", "category": "tool", "label": "Notify Slack"},
                },
                {"kind": "add_edge", "edge": {"id": "e_notify", "source": "n_save", "target": "n_notify"}},
            ],
        },
    ])

    outcome = compile_instruction(
        spec, "Also notify me on Slack once the invoice is saved.", provider=AnthropicProvider(client=fake_client), max_attempts=2
    )
    print(f"  attempts used: {outcome.attempts}")
    assert outcome.attempts == 2, "should have needed the retry"
    assert outcome.result.ok, outcome.result.errors
    new_spec = outcome.result.spec
    assert any(e.id == "e_notify" and e.source == "n_save" for e in new_spec.edges)
    print(f"  reasoning (final): {outcome.reasoning}")
    print("  OK: first attempt's bad node reference was caught, model retried, second attempt applied cleanly.")


def demo_no_change_needed() -> None:
    print("\n=== NL instruction that's just a question, no spec change ===")
    spec = load_spec("research-agent.json")
    fake_client = FakeAnthropicClient([{
        "reasoning": "This agent searches the web for the topic you give it, then summarizes what it finds. No changes needed to answer this.",
        "mutations": [],
    }])
    outcome = compile_instruction(spec, "What does this agent do?", provider=AnthropicProvider(client=fake_client))
    print(f"  reasoning: {outcome.reasoning}")
    assert outcome.result.ok
    assert outcome.result.spec is None, "an empty mutation batch applies nothing, so there's no new spec to return"
    print("  OK: question answered via reasoning, spec left untouched.")


if __name__ == "__main__":
    demo_happy_path()
    demo_retry_on_bad_reference()
    demo_no_change_needed()
    print("\nAll NL compiler demos passed.")
