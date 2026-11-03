"""
Regression tests for P0 "Tool execution":

  - AI-category nodes (classify, extract_information, summarize,
    make_decision, ai_agent, ask_ai) never implicitly receive tool
    access -- no entry in compiler.py's REQUIRED_CAPABILITY, compile fine
    with zero tool bindings and an empty spec.tools, and their real
    executors (real_executors/ai.py) never read node.tool_refs/spec.tools
    at all.
  - Web Search's registry metadata (backend/tools/registry.py) now
    accurately reflects that it has a real connected-mode executor
    (backend/real_executors/search.py), gated by the tool binding's own
    `mode` -- not silently real regardless of what a binding says (see
    backend/real_executors/test_search.py's mode-awareness tests for the
    executor side of this same fix).

Run:
    PYTHONPATH=. python backend/api/test_ai_tool_isolation.py
"""
from __future__ import annotations

from backend.compiler.compiler import REQUIRED_CAPABILITY, compile_agent
from backend.models.agent_spec import AgentSpecification, NodeType
from backend.tools.registry import get_tool


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


AI_NODE_TYPES = [
    NodeType.classify,
    NodeType.extract_information,
    NodeType.summarize,
    NodeType.make_decision,
    NodeType.ai_agent,
    NodeType.ask_ai,
]


def test_ai_node_types_have_no_required_capability() -> None:
    print("=== AI-category node types have no entry in REQUIRED_CAPABILITY ===")
    for node_type in AI_NODE_TYPES:
        check(node_type not in REQUIRED_CAPABILITY, f"{node_type} should not require any tool capability")
    print(f"  OK: none of {[t.value for t in AI_NODE_TYPES]} require a tool binding")


def _ai_node_config(node_type: str) -> dict:
    if node_type == "classify":
        return {"labels": ["a", "b"]}
    if node_type == "extract_information":
        return {"fields": ["x"]}
    if node_type == "make_decision":
        return {"options": ["yes", "no"]}
    return {"prompt": "hi"}


def test_ai_nodes_compile_with_zero_tool_bindings() -> None:
    print("\n=== every AI node type compiles fine with NO tool bindings at all in the spec ===")
    for node_type in AI_NODE_TYPES:
        spec = AgentSpecification.model_validate(
            {
                "metadata": {
                    "id": f"agt_ai_isolation_{node_type.value}",
                    "name": "AI isolation test",
                    "status": "draft",
                    "version": "0.0.1",
                    "createdAt": "2026-01-01T00:00:00Z",
                    "updatedAt": "2026-01-01T00:00:00Z",
                },
                "goal": "Test fixture: an AI node with zero tool bindings anywhere in the spec.",
                "nodes": [
                    {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
                    {
                        "id": "n_ai",
                        "type": node_type.value,
                        "category": "ai",
                        "label": f"AI node ({node_type.value})",
                        "executionType": "probabilistic",
                        "config": _ai_node_config(node_type.value),
                    },
                ],
                "edges": [{"id": "e1", "source": "n_trigger", "target": "n_ai"}],
                "tools": [],  # deliberately empty -- proves no implicit tool grant is needed
            }
        )
        compile_agent(spec)  # must not raise
    print(f"  OK: all {len(AI_NODE_TYPES)} AI node types compile with an empty spec.tools list")


def test_ai_executors_never_read_tool_refs_or_spec_tools() -> None:
    print("\n=== ai.py's real executors never consult node.tool_refs or spec.tools ===")
    import inspect

    from backend.real_executors import ai as ai_module

    source = inspect.getsource(ai_module)
    check("tool_refs" not in source, "real_executors/ai.py should never reference node.tool_refs")
    check(".tools" not in source and "spec.tools" not in source, "real_executors/ai.py should never reference spec.tools")
    print("  OK: real_executors/ai.py's source contains no reference to tool_refs or spec.tools whatsoever")


def test_web_search_registry_metadata_accuracy() -> None:
    print("\n=== Web Search registry metadata accurately reflects its real executor ===")
    tool = get_tool("web.search")
    check(tool is not None, "web.search should be registered")
    check(
        tool.execution == "connected",
        f"web.search now has a real connected-mode executor (real_executors/search.py) -- "
        f"registry metadata should say 'connected', not '{tool.execution}'",
    )
    check(
        tool.credential_required is False,
        "web.search's connected execution uses the platform's own ANTHROPIC_API_KEY, not a "
        "per-binding credentialRef -- credential_required should be False",
    )
    print(f"  OK: web.search registry entry: execution={tool.execution!r}, credential_required={tool.credential_required}")


def main() -> None:
    test_ai_node_types_have_no_required_capability()
    test_ai_nodes_compile_with_zero_tool_bindings()
    test_ai_executors_never_read_tool_refs_or_spec_tools()
    test_web_search_registry_metadata_accuracy()
    print("\nAll AI-node/tool-isolation and Web Search registry accuracy checks passed.")


if __name__ == "__main__":
    main()
