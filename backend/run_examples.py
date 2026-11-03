"""
Demonstrates the full pipeline: load an AgentSpecification JSON file,
validate it (Pydantic), compile it (compiler.py) into a LangGraph graph,
and run it — including pausing for human approval and resuming.

Run:
    python backend/run_examples.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from backend.compiler.compiler import compile_agent
from backend.models.agent_spec import AgentSpecification

ROOT = Path(__file__).resolve().parents[1]


def load_spec(name: str) -> AgentSpecification:
    data = json.loads((ROOT / "examples" / name).read_text())
    return AgentSpecification.model_validate(data)


def print_trace(result: dict) -> None:
    for event in result.get("trace", []):
        print(f"  [{event['status'].upper():7}] {event['node_label']:20} {event['detail']}")


def run_research_agent() -> None:
    print("\n=== Research Assistant (no approval steps) ===")
    spec = load_spec("research-agent.json")
    graph = compile_agent(spec, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = graph.invoke({"variables": {"topic": "quantum error correction"}, "trace": []}, config=config)

    print_trace(result)
    print("  final variables:", result["variables"])


def run_invoice_agent_approved() -> None:
    print("\n=== Invoice Processor — approval GRANTED ===")
    spec = load_spec("invoice-agent.json")
    graph = compile_agent(spec, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = graph.invoke({"variables": {}, "trace": []}, config=config)
    print_trace(result)

    interrupts = result.get("__interrupt__")
    assert interrupts, "expected the graph to pause at the human_approval node"
    payload = interrupts[0].value
    print("  >> PAUSED, awaiting approval. Payload shown to the human:")
    print(f"     {payload}")

    result = graph.invoke(Command(resume={"approved": True}), config=config)
    print_trace(result)
    print("  final variables:", result["variables"])
    assert result["variables"]["email_sent"] is True, "approved run should have sent the email"
    print("  OK: email was sent after approval.")


def run_invoice_agent_rejected() -> None:
    print("\n=== Invoice Processor — approval REJECTED ===")
    spec = load_spec("invoice-agent.json")
    graph = compile_agent(spec, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = graph.invoke({"variables": {}, "trace": []}, config=config)
    assert result.get("__interrupt__"), "expected the graph to pause at the human_approval node"

    result = graph.invoke(Command(resume={"approved": False, "reason": "Vendor not recognized"}), config=config)
    print_trace(result)
    print("  final variables:", result["variables"])
    assert "email_sent" not in result["variables"], "rejected run must NOT have sent the email"
    print("  OK: email was correctly NOT sent after rejection.")


if __name__ == "__main__":
    run_research_agent()
    run_invoice_agent_approved()
    run_invoice_agent_rejected()
    print("\nAll demo runs completed.")
