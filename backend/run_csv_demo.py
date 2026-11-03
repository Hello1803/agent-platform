"""
End-to-end demo: a real agent spec that reads examples/sample_invoices.csv
and genuinely extracts/summarizes it -- no stub anywhere in the tool path,
no external credential or network access needed (this is exactly why
read_file was the first real tool worth adding after Gmail/Sheets/HTTP:
it needs nothing this sandbox can't provide).

Run:
    PYTHONPATH=. python backend/run_csv_demo.py

What actually happens, for real:
  1. A `read_file` node, bound to a mode="connected" files.read tool, is
     compiled and executed through the REAL LangGraph graph.
  2. The executor (backend/real_executors/tools.py) genuinely parses
     examples/sample_invoices.csv with pandas -- real columns, a real row
     count, and real per-column numeric stats computed from the actual
     file on disk.
  3. If ANTHROPIC_API_KEY is set, a real Claude call summarizes the
     extracted table and the script prints its actual output. Without a
     key, that step is skipped explicitly (not silently faked) and the
     script still proves the extraction itself is real by printing the
     computed numbers.

Try it on your own file: copy any .csv/.xlsx into examples/ (or point
AGENT_FILES_DIR at a different directory) and change SAMPLE_PATH below.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from backend.compiler.compiler import compile_agent
from backend.models.agent_spec import AgentSpecification, NodeType
from backend.real_executors.tools import build_read_file_executor

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = "sample_invoices.csv"  # relative to AGENT_FILES_DIR, examples/ in this demo


def build_spec() -> AgentSpecification:
    return AgentSpecification.model_validate(
        {
            "metadata": {
                "id": "agt_csv_demo",
                "name": "Invoice CSV Summarizer",
                "status": "draft",
                "version": "0.0.1",
                "createdAt": "2026-01-01T00:00:00Z",
                "updatedAt": "2026-01-01T00:00:00Z",
                "owner": "demo",
            },
            "goal": "Read a CSV of invoices and summarize what's owed.",
            "nodes": [
                {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
                {
                    "id": "n_read",
                    "type": "read_file",
                    "category": "tool",
                    "label": "Read invoices.csv",
                    "toolRefs": ["t_files"],
                    "config": {"path": SAMPLE_PATH},
                },
            ],
            "edges": [{"id": "e1", "source": "n_trigger", "target": "n_read"}],
            "tools": [
                {
                    "id": "t_files",
                    "toolId": "files.read",
                    "displayName": "Local file storage",
                    "permissions": ["read"],
                    "mode": "connected",
                }
            ],
        }
    )


def main() -> None:
    print(f"=== Reading {SAMPLE_PATH} through the REAL compiled graph ===\n")

    # `files.read` is credential_required=False (backend/tools/registry.py)
    # -- the real executor does local, sandboxed disk I/O and genuinely
    # never needs a secret to do it, so a plain mode="connected" binding
    # with no credentialRef at all is enough. (It used to require one here
    # even though nothing ever read it -- fixed alongside this demo.)
    spec = build_spec()
    # AGENT_FILES_DIR points at examples/ so the demo runs against the
    # real file shipped in this repo without needing to copy it anywhere.
    read_executor = build_read_file_executor(base_dir=str(ROOT / "examples"))
    graph = compile_agent(
        spec,
        checkpointer=MemorySaver(),
        executor_overrides={NodeType.read_file: read_executor},
    )

    final_state = graph.invoke(
        {"variables": {}, "trace": []},
        config={"configurable": {"thread_id": "csv-demo"}},
    )
    variables = final_state["variables"]

    print(f"Columns:    {variables['table_columns']}")
    print(f"Row count:  {variables['table_row_count']}")
    print(f"Truncated:  {variables['table_truncated']}")
    print("\nFirst rows (real data from the file):")
    for row in variables["table_preview"]:
        print(f"  {row}")

    print("\nComputed numeric summary (genuinely calculated from the file, not canned):")
    for column, stats in variables["table_summary"].items():
        print(f"  {column}: {stats}")

    amount_stats = variables["table_summary"].get("amount")
    if amount_stats:
        unpaid_total = sum(
            row["amount"] for row in variables["rows"] if row.get("status") in ("unpaid", "overdue")
        )
        print(f"\nTotal outstanding (unpaid + overdue), computed from the real rows: {unpaid_total:,.2f}")

    print("\n=== Optional: a REAL Claude summary of the same data ===")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set, so this step is explicitly skipped rather than\n"
            "faked -- the extraction above is fully real either way. Set ANTHROPIC_API_KEY\n"
            "and rerun to see a real Claude-generated summary of this exact table."
        )
    else:
        from anthropic import Anthropic

        client = Anthropic()
        prompt = (
            "Here is an invoices table extracted from a CSV file:\n\n"
            f"{json.dumps(variables['rows'], indent=2)}\n\n"
            "In two sentences, summarize what's owed and to whom, in plain English."
        )
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        summary_text = "".join(block.text for block in response.content if hasattr(block, "text"))
        print(summary_text)

    print("\nDone -- every number above came from examples/sample_invoices.csv, not a stub.")


if __name__ == "__main__":
    main()
