"""Focused regression coverage for registry-backed binding validation."""
from __future__ import annotations

import json
from pathlib import Path

from backend.compiler.compiler import CompileError, compile_agent
from backend.models.agent_spec import AgentSpecification, parse_mutation
from backend.mutation.engine import apply_mutations

ROOT = Path(__file__).resolve().parents[2]


def spec() -> AgentSpecification:
    raw = json.loads((ROOT / "examples" / "research-agent.json").read_text())
    return AgentSpecification.model_validate(raw)


def expect_compile_error(candidate: AgentSpecification, text: str) -> None:
    try:
        compile_agent(candidate)
    except CompileError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError("expected compile failure")


def main() -> None:
    valid = spec()
    compile_agent(valid)

    bad_tool = valid.model_copy(deep=True)
    bad_tool.tools[0].tool_id = "unconfigured.search_web"
    expect_compile_error(bad_tool, "not registered")

    incompatible = valid.model_copy(deep=True)
    incompatible.tools[0].tool_id = "local.sqlite.append_row"
    incompatible.tools[0].permissions = ["read"]
    expect_compile_error(incompatible, "not compatible")

    disconnected = valid.model_copy(deep=True)
    result = apply_mutations(disconnected, [parse_mutation({"kind": "add_node", "node": {"id": "n_orphan", "type": "summarize", "category": "ai", "label": "Orphan", "executionType": "probabilistic"}})])
    assert result.ok and result.spec
    expect_compile_error(result.spec, "not connected")
    print("tool registry validation: passed")


if __name__ == "__main__":
    main()
