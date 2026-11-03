"""
Regression tests for backend/compiler/compiler.py's `policies.tool_allowlist`
check (`_check_run_readiness`).

The bug: the check used to iterate every entry in `spec.tools` regardless
of whether any node actually used it. spec.tools legitimately accumulates
bindings nothing currently references (e.g. a search_web node gets
deleted from the canvas but its tool binding is left behind — tool
bindings are meant to be reusable, so node deletion doesn't prune them).
That meant an unrelated, unused, disallowed binding could block a run
that never touches it — reported as a plain trigger -> read_file graph
failing to compile over 'web.search', a tool the graph never calls.

Run:
    PYTHONPATH=. python backend/compiler/test_policy.py
"""
from __future__ import annotations

from backend.compiler.compiler import CompileError, compile_agent
from backend.models.agent_spec import AgentSpecification


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _base_spec_dict() -> dict:
    return {
        "metadata": {
            "id": "agt_policy_test",
            "name": "Policy test",
            "status": "draft",
            "version": "0.0.1",
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z",
        },
        "goal": "Test fixture for the tool_allowlist policy check.",
        "nodes": [
            {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
            {"id": "n_read", "type": "read_file", "category": "tool", "label": "Read", "toolRefs": ["t_files"]},
        ],
        "edges": [{"id": "e1", "source": "n_trigger", "target": "n_read"}],
        "tools": [{"id": "t_files", "toolId": "files.read", "permissions": ["read"], "mode": "demo"}],
    }


def main() -> None:
    print("=== an unused, disallowed orphan tool binding no longer blocks compilation ===")
    spec_dict = _base_spec_dict()
    spec_dict["tools"].append({"id": "t_search", "toolId": "web.search", "permissions": ["search"], "mode": "demo"})
    spec_dict["policies"] = {"toolAllowlist": ["files.read"]}  # deliberately excludes web.search
    spec = AgentSpecification.model_validate(spec_dict)
    compile_agent(spec)  # should not raise
    print("  OK: a plain trigger -> read_file graph compiles despite an unused, disallowed 'web.search' binding")

    print("\n=== a tool a node actually uses is still correctly blocked when disallowed ===")
    spec_dict2 = _base_spec_dict()
    spec_dict2["policies"] = {"toolAllowlist": ["web.search"]}  # files.read (actually used) is NOT allowlisted
    spec2 = AgentSpecification.model_validate(spec_dict2)
    try:
        compile_agent(spec2)
        raise AssertionError("a used, disallowed tool should still block compilation")
    except CompileError as exc:
        check("files.read" in str(exc), f"unexpected error: {exc}")
        check(exc.node_id == "n_read", f"error should be attributed to the node using the disallowed tool, got node_id={exc.node_id!r}")
        print(f"  OK: correctly blocked, and attributed to the actual node (node_id={exc.node_id!r}), not null")

    print("\n=== an allowed, used tool still compiles fine ===")
    spec_dict3 = _base_spec_dict()
    spec_dict3["policies"] = {"toolAllowlist": ["files.read"]}
    spec3 = AgentSpecification.model_validate(spec_dict3)
    compile_agent(spec3)
    print("  OK: an allowlisted, actually-used tool still compiles")

    print("\n=== no allowlist configured at all: nothing is blocked ===")
    spec_dict4 = _base_spec_dict()
    spec4 = AgentSpecification.model_validate(spec_dict4)
    compile_agent(spec4)
    print("  OK: with no tool_allowlist set, compilation proceeds as before")

    print("\nAll tool_allowlist policy checks passed.")


if __name__ == "__main__":
    main()
