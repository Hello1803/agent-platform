"""
Regression tests for the Conversational Workflow Builder backend:
propose/preview/apply, workflow scoping, and provider-agnosticism.

Run:
    PYTHONPATH=. python backend/api/test_conversational_builder.py
"""
from __future__ import annotations

import os
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.ai_providers import registry as registry_module
from backend.ai_providers.anthropic_provider import AnthropicProvider
from backend.ai_providers.openai_provider import OpenAIProvider
from backend.api.app import app
from backend.nl_compiler.schemas import MutationBatch


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _FakeAnthropicMessages:
    def __init__(self, batches: list[dict]):
        self._queue = [MutationBatch.model_validate(b) for b in batches]

    def parse(self, **kwargs):
        return SimpleNamespace(parsed_output=self._queue.pop(0), stop_reason="end_turn")


class _FakeAnthropicClient:
    def __init__(self, batches: list[dict]):
        self.messages = _FakeAnthropicMessages(batches)


class _FakeOpenAICompletions:
    def __init__(self, batches: list[dict]):
        self._queue = [MutationBatch.model_validate(b) for b in batches]

    def parse(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=self._queue.pop(0), refusal=None))])


class _FakeOpenAIClient:
    def __init__(self, batches: list[dict]):
        self.chat = SimpleNamespace(completions=_FakeOpenAICompletions(batches))


_ADD_NODE_BATCH = {
    "reasoning": "Added an Ask AI node after the trigger.",
    "mutations": [
        {"kind": "add_node", "node": {"id": "n_new", "type": "ask_ai", "category": "ai", "label": "New", "executionType": "probabilistic", "config": {"prompt": "hi"}}},
        {"kind": "add_edge", "edge": {"id": "e_new", "source": "n_trigger", "target": "n_new"}},
    ],
}

_QUESTION_BATCH = {"reasoning": "This workflow just starts with a manual trigger and does nothing else yet.", "mutations": []}

_INVALID_BATCH = {
    "reasoning": "Connecting to a node that doesn't exist.",
    "mutations": [{"kind": "add_edge", "edge": {"id": "e_bad", "source": "n_trigger", "target": "n_does_not_exist"}}],
}


def _create_workflow(client: TestClient, agent_id: str, name: str) -> tuple[str, str]:
    r = client.post(f"/agents/{agent_id}/workflows", json={"name": name}, headers={"X-User-Id": "tester"})
    return r.json()["metadata"]["id"], r.json()["metadata"]["version"]


def test_propose_does_not_persist(client: TestClient) -> None:
    print("=== propose: a valid proposal is NOT persisted until apply ===")
    registry_module.build_provider = lambda name=None, model=None: AnthropicProvider(client=_FakeAnthropicClient([_ADD_NODE_BATCH]))
    agent_id = "agt_propose_test"
    wf, _ = _create_workflow(client, agent_id, "Propose test")

    r = client.post(f"/agents/{agent_id}/workflows/{wf}/chat/propose", json={"message": "add an ask ai node"})
    check(r.status_code == 200, f"unexpected: {r.status_code} {r.text}")
    body = r.json()
    check(body["ok"] is True, f"expected ok=True: {body}")
    check(len(body["mutations"]) == 2, f"unexpected mutations: {body['mutations']}")
    check(body["preview_spec"] is not None and len(body["preview_spec"]["nodes"]) == 2, f"preview should show 2 nodes: {body.get('preview_spec')}")

    current = client.get(f"/agents/{agent_id}/workflows/{wf}").json()
    check(len(current["nodes"]) == 1, f"the actual stored workflow must be untouched by propose, got {len(current['nodes'])} nodes")
    print("  OK: propose returns a preview with 2 nodes; the real stored workflow still has 1 (untouched)")


def test_apply_commits_through_the_real_mutation_engine(client: TestClient) -> None:
    print("\n=== apply: commits through apply_mutations + compare_and_set, same as any other edit ===")
    registry_module.build_provider = lambda name=None, model=None: AnthropicProvider(client=_FakeAnthropicClient([_ADD_NODE_BATCH]))
    agent_id = "agt_apply_test"
    wf, _ = _create_workflow(client, agent_id, "Apply test")

    proposal = client.post(f"/agents/{agent_id}/workflows/{wf}/chat/propose", json={"message": "add a node"}).json()
    r = client.post(f"/agents/{agent_id}/workflows/{wf}/chat/apply", json={"mutations": proposal["mutations"], "expected_version": proposal["base_version"]})
    check(r.status_code == 200, f"apply failed: {r.status_code} {r.text}")
    check(len(r.json()["nodes"]) == 2, f"unexpected: {r.json()['nodes']}")

    current = client.get(f"/agents/{agent_id}/workflows/{wf}").json()
    check(len(current["nodes"]) == 2, "the stored workflow should now genuinely reflect the applied mutations")
    print("  OK: apply genuinely persisted the change via the real mutation engine")


def test_cancel_leaves_workflow_unchanged(client: TestClient) -> None:
    print("\n=== cancel (never calling apply) leaves the workflow completely unchanged ===")
    registry_module.build_provider = lambda name=None, model=None: AnthropicProvider(client=_FakeAnthropicClient([_ADD_NODE_BATCH]))
    agent_id = "agt_cancel_test"
    wf, _ = _create_workflow(client, agent_id, "Cancel test")
    before = client.get(f"/agents/{agent_id}/workflows/{wf}").json()

    client.post(f"/agents/{agent_id}/workflows/{wf}/chat/propose", json={"message": "add a node"})

    after = client.get(f"/agents/{agent_id}/workflows/{wf}").json()
    check(after == before, "cancel must leave the workflow byte-for-byte unchanged")
    print("  OK: no apply call was ever made; the workflow is untouched")


def test_invalid_proposal_never_mutates(client: TestClient) -> None:
    print("\n=== an invalid AI proposal is rejected without modifying the workflow ===")
    registry_module.build_provider = lambda name=None, model=None: AnthropicProvider(client=_FakeAnthropicClient([_INVALID_BATCH, _INVALID_BATCH]))
    agent_id = "agt_invalid_test"
    wf, _ = _create_workflow(client, agent_id, "Invalid test")

    r = client.post(f"/agents/{agent_id}/workflows/{wf}/chat/propose", json={"message": "connect to a bogus node"})
    check(r.status_code == 200, f"the PROPOSE call itself should succeed even though the proposal is invalid: {r.status_code}")
    body = r.json()
    check(body["ok"] is False, f"expected ok=False: {body}")
    check(len(body["errors"]) >= 1, f"expected validation errors: {body}")
    check(body["preview_spec"] is None, "no preview should be offered for an invalid proposal")

    current = client.get(f"/agents/{agent_id}/workflows/{wf}").json()
    check(len(current["edges"]) == 0, "the workflow must be untouched by an invalid proposal")
    print(f"  OK: invalid proposal correctly rejected (ok=False, {len(body['errors'])} error(s)), workflow untouched")


def test_apply_rejects_a_mutation_not_actually_valid_against_current_state(client: TestClient) -> None:
    print("\n=== apply independently re-validates -- doesn't just trust the echoed-back mutations ===")
    registry_module.build_provider = lambda name=None, model=None: AnthropicProvider(client=_FakeAnthropicClient([_ADD_NODE_BATCH]))
    agent_id = "agt_apply_revalidate_test"
    wf, _ = _create_workflow(client, agent_id, "Revalidate test")
    proposal = client.post(f"/agents/{agent_id}/workflows/{wf}/chat/propose", json={"message": "add a node"}).json()

    tampered = list(proposal["mutations"])
    tampered[1]["edge"]["target"] = "n_does_not_exist"
    r = client.post(f"/agents/{agent_id}/workflows/{wf}/chat/apply", json={"mutations": tampered, "expected_version": proposal["base_version"]})
    check(r.status_code == 422, f"a tampered/invalid mutation set should be rejected at apply time too, got {r.status_code}")

    current = client.get(f"/agents/{agent_id}/workflows/{wf}").json()
    check(len(current["nodes"]) == 1, "workflow must remain untouched")
    print("  OK: apply re-validates independently and rejects an invalid mutation set, workflow untouched")


def test_question_gets_answered_with_no_mutation(client: TestClient) -> None:
    print("\n=== chat inspection: a question produces reasoning with zero mutations ===")
    registry_module.build_provider = lambda name=None, model=None: AnthropicProvider(client=_FakeAnthropicClient([_QUESTION_BATCH]))
    agent_id = "agt_question_test"
    wf, _ = _create_workflow(client, agent_id, "Question test")

    r = client.post(f"/agents/{agent_id}/workflows/{wf}/chat/propose", json={"message": "what does this workflow do?"})
    body = r.json()
    check(body["ok"] is True and len(body["mutations"]) == 0, f"a question should produce zero mutations: {body}")
    check("manual trigger" in body["reasoning"].lower(), f"reasoning should be the actual answer: {body['reasoning']}")
    print(f"  OK: question answered via reasoning ('{body['reasoning']}'), no mutation, nothing to apply")


def test_openai_provider_end_to_end(client: TestClient) -> None:
    print("\n=== OpenAI provider works identically through the SAME propose/apply routes ===")
    registry_module.build_provider = lambda name=None, model=None: OpenAIProvider(client=_FakeOpenAIClient([_ADD_NODE_BATCH]))
    agent_id = "agt_openai_chat_test"
    wf, _ = _create_workflow(client, agent_id, "OpenAI test")

    r = client.post(f"/agents/{agent_id}/workflows/{wf}/chat/propose", json={"message": "add a node"})
    check(r.status_code == 200 and r.json()["ok"] is True, f"unexpected: {r.status_code} {r.text}")
    body = r.json()
    r2 = client.post(f"/agents/{agent_id}/workflows/{wf}/chat/apply", json={"mutations": body["mutations"], "expected_version": body["base_version"]})
    check(r2.status_code == 200 and len(r2.json()["nodes"]) == 2, f"unexpected: {r2.status_code} {r2.text}")
    print("  OK: propose + apply both work correctly with OpenAI as the underlying provider, no code path difference")


def test_explicit_provider_without_credentials_is_a_clear_error(client: TestClient) -> None:
    print("\n=== explicitly requesting a provider with no configured credentials is a clear 422, not a silent fallback ===")
    agent_id = "agt_no_creds_test"
    wf, _ = _create_workflow(client, agent_id, "No creds test")
    had_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        r = client.post(f"/agents/{agent_id}/workflows/{wf}/chat/propose", json={"message": "add a node", "provider": "openai"})
        check(r.status_code == 422, f"expected 422, got {r.status_code}: {r.text}")
        check("OPENAI_API_KEY" in r.json()["detail"], f"error should name the missing env var: {r.json()}")
        print(f"  OK: {r.json()['detail']}")
    finally:
        if had_key is not None:
            os.environ["OPENAI_API_KEY"] = had_key


def test_scoped_to_current_workflow_only(client: TestClient) -> None:
    print("\n=== chat is scoped to the CURRENT workflow -- proposing against A never touches B ===")
    registry_module.build_provider = lambda name=None, model=None: AnthropicProvider(client=_FakeAnthropicClient([_ADD_NODE_BATCH]))
    agent_id = "agt_chat_scoping_test"
    wf_a, ver_a = _create_workflow(client, agent_id, "A")
    wf_b, _ = _create_workflow(client, agent_id, "B")

    proposal = client.post(f"/agents/{agent_id}/workflows/{wf_a}/chat/propose", json={"message": "add a node"}).json()
    client.post(f"/agents/{agent_id}/workflows/{wf_a}/chat/apply", json={"mutations": proposal["mutations"], "expected_version": proposal["base_version"]})

    a_after = client.get(f"/agents/{agent_id}/workflows/{wf_a}").json()
    b_after = client.get(f"/agents/{agent_id}/workflows/{wf_b}").json()
    check(len(a_after["nodes"]) == 2, f"A should have the applied change: {a_after['nodes']}")
    check(len(b_after["nodes"]) == 1, f"B must be completely untouched by a chat interaction scoped to A: {b_after['nodes']}")
    print("  OK: A changed, B is byte-for-byte untouched -- chat mutations are correctly workflow-scoped")


def test_chat_scoped_to_wrong_agent_404s(client: TestClient) -> None:
    print("\n=== chat routes are also agent-scoped, not just workflow-scoped ===")
    registry_module.build_provider = lambda name=None, model=None: AnthropicProvider(client=_FakeAnthropicClient([_ADD_NODE_BATCH]))
    agent_id = "agt_chat_owner_test"
    wf, _ = _create_workflow(client, agent_id, "A")

    r = client.post(f"/agents/some_other_agent/workflows/{wf}/chat/propose", json={"message": "add a node"})
    check(r.status_code == 404, f"expected 404 for a workflow accessed under the wrong agent_id, got {r.status_code}")
    print("  OK: proposing against the right workflow under the WRONG agent_id 404s")


def main() -> None:
    os.environ["ANTHROPIC_API_KEY"] = "fake-for-test"
    os.environ["OPENAI_API_KEY"] = "fake-for-test"
    real_build_provider = registry_module.build_provider
    client = TestClient(app)
    try:
        test_propose_does_not_persist(client)
        test_apply_commits_through_the_real_mutation_engine(client)
        test_cancel_leaves_workflow_unchanged(client)
        test_invalid_proposal_never_mutates(client)
        test_apply_rejects_a_mutation_not_actually_valid_against_current_state(client)
        test_question_gets_answered_with_no_mutation(client)
        test_openai_provider_end_to_end(client)
        test_explicit_provider_without_credentials_is_a_clear_error(client)
        test_scoped_to_current_workflow_only(client)
        test_chat_scoped_to_wrong_agent_404s(client)
    finally:
        registry_module.build_provider = real_build_provider

    print("\nAll conversational workflow builder checks passed.")


if __name__ == "__main__":
    main()
