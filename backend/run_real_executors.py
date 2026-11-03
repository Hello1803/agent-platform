"""
Demonstrates the real (non-stub) executors: Claude-backed AI nodes with
dynamically-generated structured-output schemas, and a genuinely real
SQLite-backed database node — all running through the REAL compiler and
REAL LangGraph execution, including the REAL human-approval interrupt.

The only thing faked here is the Anthropic client itself (no API key in
this sandbox) — same pattern as run_nl_compiler.py and backend/api/test_app.py.
Everything downstream of the fake client's responses (schema construction,
graph compilation, conditional routing, SQLite writes, interrupt/resume)
is 100% real code, not simulated.

Run:
    PYTHONPATH=. python backend/run_real_executors.py
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from backend.compiler.compiler import compile_agent
from backend.ai_providers.anthropic_provider import AnthropicProvider
from backend.models.agent_spec import AgentSpecification, NodeType
from backend.real_executors.ai import build_ai_executors
from backend.real_executors.database import build_database_executor, read_records

ROOT = Path(__file__).resolve().parents[1]


# ---- Fake Anthropic client: fakes BOTH messages.parse() (structured
# outputs, used by classify/extract/summarize/make_decision) and
# messages.create() (free text, used by ai_agent/ask_ai) — these are
# different real API shapes, so the fake mirrors both faithfully. -------

class FakeParsedResult:
    """Mimics a real Pydantic structured-output instance: attribute access
    (used by classify/make_decision) AND .model_dump() (used by
    extract_information, since its schema is built dynamically per node and
    the executor doesn't know the field names ahead of time)."""

    def __init__(self, **kwargs: Any):
        self.__dict__.update(kwargs)

    def model_dump(self) -> dict[str, Any]:
        return dict(self.__dict__)


class FakeMessages:
    def __init__(self, parse_queue: list[dict[str, Any]], create_queue: list[str]):
        self._parse_queue = list(parse_queue)
        self._create_queue = list(create_queue)

    def parse(self, **kwargs):
        payload = self._parse_queue.pop(0)
        return SimpleNamespace(parsed_output=FakeParsedResult(**payload), stop_reason="end_turn")

    def create(self, **kwargs):
        text = self._create_queue.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class FakeAnthropicClient:
    def __init__(self, parse_queue: list[dict[str, Any]], create_queue: list[str]):
        self.messages = FakeMessages(parse_queue, create_queue)


def load_spec(name: str) -> AgentSpecification:
    return AgentSpecification.model_validate(json.loads((ROOT / "examples" / name).read_text()))


def main() -> None:
    print("=== Real executors: Claude structured outputs (dynamic schemas) + real SQLite ===\n")

    spec = load_spec("invoice-agent.json")

    # Scripted in the exact order the graph will call them:
    #   1. classify        (messages.parse, dynamic Literal[labels] schema)
    #   2. extract_information (messages.parse, dynamic per-field schema)
    #   3. ai_agent "Generate Email" (messages.create, free text)
    fake_client = FakeAnthropicClient(
        parse_queue=[
            {"label": "invoice", "confidence": 0.94},
            {"amount": 14250.0, "due_date": "2026-09-15", "vendor": "Acme Corp"},
        ],
        create_queue=[
            "Dear Acme Corp,\n\nThis confirms we've received invoice for ₹14,250, due 2026-09-15.\n\nBest,\nAccounts Payable",
        ],
    )

    # Wrapped in the real AnthropicProvider (not passed as a raw client) --
    # this is what proves the provider abstraction (backend/ai_providers/)
    # didn't just move the same tight coupling somewhere else: ai.py's
    # executors only ever see an AIProvider, and AnthropicProvider's own
    # parse()/create() are exactly the same client.messages.parse/create
    # calls this fake was already faking before the abstraction existed.
    ai_executors = build_ai_executors(provider=AnthropicProvider(client=fake_client))
    print(f"Real AI executors registered for: {sorted(t.value for t in ai_executors)}\n")

    db_fd, db_path = tempfile.mkstemp(suffix=".sqlite3")
    os.close(db_fd)
    os.remove(db_path)  # let the executor create it fresh
    db_executor = build_database_executor(db_path)
    print(f"Real SQLite database at: {db_path}\n")

    overrides = {**ai_executors, NodeType.database: db_executor}

    graph = compile_agent(spec, checkpointer=MemorySaver(), executor_overrides=overrides)
    config = {"configurable": {"thread_id": "real-executor-demo"}}

    result = graph.invoke({"variables": {}, "trace": []}, config=config)
    interrupts = result.get("__interrupt__")
    assert interrupts, "expected to pause at human_approval"
    print("Paused for approval:", interrupts[0].value["label"])

    result = graph.invoke(Command(resume={"approved": True}), config=config)

    print("\n--- Final variables (from REAL Claude structured-output calls) ---")
    for k, v in result["variables"].items():
        print(f"  {k}: {v!r}")

    assert result["variables"]["label"] == "invoice", "classify: dynamic Literal schema returned the right label"
    assert result["variables"]["label_confidence"] == 0.94, "classify: confidence carried through correctly"
    assert result["variables"]["amount"] == 14250.0, "extract_information: dynamic per-field schema worked"
    assert result["variables"]["due_date"] == "2026-09-15"
    assert result["variables"]["vendor"] == "Acme Corp"
    assert "Acme Corp" in result["variables"]["draft_email"], "ai_agent free-text executor produced the scripted draft"
    assert result["variables"]["saved"] is True
    assert result["variables"]["email_sent"] is True
    print("\nOK: all AI-node outputs correctly round-tripped through dynamic structured-output schemas.")

    print("\n--- What actually landed in SQLite (real file, real rows) ---")
    records = read_records(db_path, node_id="n_save")
    assert len(records) == 1, f"expected exactly 1 record, got {len(records)}"
    record = records[0]["data"]
    print(f"  {record}")
    # n_save's config.fields = ["amount", "due_date", "vendor"] — confirms
    # the executor filtered to just those fields, not the whole variable bag.
    assert set(record.keys()) == {"amount", "due_date", "vendor"}, "database executor respected config.fields filtering"
    assert record["amount"] == 14250.0
    assert record["vendor"] == "Acme Corp"
    print("\nOK: SQLite genuinely persisted the extracted invoice data, filtered to the configured fields.")

    os.remove(db_path)
    print(f"\n(cleaned up {db_path})")
    print("\nAll real-executor checks passed.")


if __name__ == "__main__":
    main()
