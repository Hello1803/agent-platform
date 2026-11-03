"""
Demonstrates the mutation engine, including the scenario architecture doc
section 23 calls out explicitly: the user drags a node on the canvas while
simultaneously telling the chat builder "add an approval step here."

Run:
    PYTHONPATH=. python backend/run_mutations.py
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.models.agent_spec import AgentSpecification, parse_mutation
from backend.mutation.engine import apply_mutation, apply_mutations

ROOT = Path(__file__).resolve().parents[1]


def load_spec(name: str) -> AgentSpecification:
    data = json.loads((ROOT / "examples" / name).read_text())
    return AgentSpecification.model_validate(data)


def demo_batch_add_node_and_edge() -> None:
    print("\n=== Atomic batch: add a node + wire it in one step ===")
    spec = load_spec("research-agent.json")
    print(f"  before: {len(spec.nodes)} nodes, {len(spec.edges)} edges, version {spec.metadata.version}")

    mutations = [
        parse_mutation({
            "kind": "add_node",
            "node": {
                "id": "n_translate",
                "type": "ask_ai",
                "category": "ai",
                "label": "Translate Summary",
                "executionType": "probabilistic",
                "confidenceThreshold": 0.6,
            },
        }),
        parse_mutation({
            "kind": "add_edge",
            "edge": {"id": "e3", "source": "n_summarize", "target": "n_translate"},
        }),
    ]
    result = apply_mutations(spec, mutations, expected_version=spec.metadata.version)
    assert result.ok, result.errors
    new_spec = result.spec
    print(f"  after:  {len(new_spec.nodes)} nodes, {len(new_spec.edges)} edges, version {new_spec.metadata.version}")
    assert len(new_spec.nodes) == len(spec.nodes) + 1
    assert spec.metadata.version == "1.2.0", "original spec object must be untouched"
    print("  OK: batch applied atomically, version bumped, original spec object unmodified.")


def demo_atomicity_on_failure() -> None:
    print("\n=== Atomic batch: partial failure rolls back the WHOLE batch ===")
    spec = load_spec("research-agent.json")
    mutations = [
        parse_mutation({
            "kind": "add_node",
            "node": {"id": "n_valid", "type": "ask_ai", "category": "ai", "label": "Valid Node"},
        }),
        # this one references a node that doesn't exist yet -> should fail
        parse_mutation({
            "kind": "add_edge",
            "edge": {"id": "e_bad", "source": "n_valid", "target": "does_not_exist"},
        }),
    ]
    result = apply_mutations(spec, mutations)
    assert not result.ok
    print("  correctly rejected:", result.errors[0].message)
    print("  OK: neither the node nor the edge was added (spec left as if nothing happened).")


def demo_cascading_delete() -> None:
    print("\n=== remove_node cascades to its edges and approval rules ===")
    spec = load_spec("invoice-agent.json")
    before_edges = len(spec.edges)
    result = apply_mutation(spec, parse_mutation({"kind": "remove_node", "node_id": "n_approval"}))
    assert result.ok, result.errors
    new_spec = result.spec
    print(f"  edges before: {before_edges}, after removing n_approval: {len(new_spec.edges)}")
    assert all(e.source != "n_approval" and e.target != "n_approval" for e in new_spec.edges)
    print("  OK: dangling edges into/out of the removed node were cleaned up automatically.")


def demo_concurrent_edit_conflict() -> None:
    print("\n=== Concurrent modification: canvas drag vs. chat edit racing ===")
    spec = load_spec("invoice-agent.json")
    base_version = spec.metadata.version
    print(f"  both edits are based on version '{base_version}'")

    # "Canvas" edit: user drags a node, changing its position.
    canvas_mutation = parse_mutation({
        "kind": "update_node",
        "node_id": "n_gen_email",
        "patch": {"canvasPosition": {"x": 240, "y": 500}},
    })
    canvas_result = apply_mutations(spec, [canvas_mutation], expected_version=base_version)
    assert canvas_result.ok
    print(f"  canvas edit landed first -> new version '{canvas_result.spec.metadata.version}'")

    # "Chat" edit: user tells the builder "add an approval step here",
    # computed against the SAME original spec/version, arriving second —
    # but by the time it's applied, the server-side spec has already moved
    # on to the canvas edit's result.
    chat_mutation = parse_mutation({
        "kind": "add_approval_rule",
        "rule": {
            "id": "appr_extra",
            "appliesTo": {"actionCategory": "delete_data"},
            "autonomyLevel": "ask_permission",
        },
    })
    chat_result = apply_mutations(canvas_result.spec, [chat_mutation], expected_version=base_version)
    assert not chat_result.ok, "the second edit against a stale version must be rejected"
    print("  chat edit (also based on stale version) correctly rejected:")
    print(f"    {chat_result.errors[0].message}")

    # Correct flow: chat re-fetches the CURRENT spec (post-canvas-edit) and
    # retries its mutation against the up-to-date version.
    retry_result = apply_mutations(
        canvas_result.spec, [chat_mutation], expected_version=canvas_result.spec.metadata.version
    )
    assert retry_result.ok, retry_result.errors
    print(f"  retried against current version -> succeeds, both edits now present, version '{retry_result.spec.metadata.version}'")
    assert retry_result.spec.nodes[4].canvas_position.x == 240  # canvas edit preserved
    assert any(r.id == "appr_extra" for r in retry_result.spec.human_approval)  # chat edit applied
    print("  OK: no silent overwrite — both edits ended up in the final spec.")


if __name__ == "__main__":
    demo_batch_add_node_and_edge()
    demo_atomicity_on_failure()
    demo_cascading_delete()
    demo_concurrent_edit_conflict()
    print("\nAll mutation engine demos passed.")
