"""
A real (non-stub) implementation of the `database` node type, backed by
SQLite. Unlike Gmail or Google Sheets, this needs no external credentials
this sandbox can't provide, so it's the one tool node in this project with
a genuinely working implementation instead of a documented stub — proof
that the executor interface (node, spec, variables) -> dict really is
enough to plug in a real integration without touching the compiler.

Swapping this for a real Google Sheets executor later means writing a
function with the exact same signature that calls the Sheets API instead
of sqlite3 — nothing else in the compiler, executors registry, or graph
changes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from backend.compiler.executors import NodeExecutor, ToolPermissionError
from backend.models.agent_spec import AgentNode, AgentSpecification

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def build_database_executor(db_path: str) -> NodeExecutor:
    """
    `db_path` is a real SQLite file path (or ":memory:" for a fully
    in-process store, though note ":memory:" doesn't survive across
    separate sqlite3.connect() calls — pass a real temp file path for
    anything that needs to persist across the executor's calls).
    """
    _init_db(db_path)
    lock = threading.Lock()

    def execute(node: AgentNode, spec: AgentSpecification, variables: dict[str, Any]) -> dict:
        # Same permission re-check pattern as the stub executors — the
        # compiler already verified this at compile time, this is
        # defense-in-depth for a spec whose permissions changed since compile.
        if not node.tool_refs:
            raise ToolPermissionError(f"Node '{node.id}' has no tool binding but tried to write to the database.")
        bindings = {t.id for t in spec.tools if "write" in t.permissions}
        if not any(ref in bindings for ref in node.tool_refs):
            raise ToolPermissionError(f"Node '{node.id}' is not permitted to write (tool_refs={node.tool_refs}).")

        fields_to_save = node.config.get("fields") or list(variables.keys())
        record = {f: variables.get(f) for f in fields_to_save}

        with lock:
            conn = sqlite3.connect(db_path)
            try:
                cursor = conn.execute(
                    "INSERT INTO agent_records (node_id, data, created_at) VALUES (?, ?, ?)",
                    (node.id, json.dumps(record, default=str), datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
                record_id = cursor.lastrowid
            finally:
                conn.close()

        return {"saved": True, "record_id": record_id}

    return execute


def read_records(db_path: str, node_id: str | None = None) -> list[dict[str, Any]]:
    """Test/debug helper: read back what a database executor actually wrote."""
    conn = sqlite3.connect(db_path)
    try:
        if node_id:
            rows = conn.execute(
                "SELECT id, node_id, data, created_at FROM agent_records WHERE node_id = ? ORDER BY id", (node_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT id, node_id, data, created_at FROM agent_records ORDER BY id").fetchall()
    finally:
        conn.close()
    return [
        {"id": r[0], "node_id": r[1], "data": json.loads(r[2]), "created_at": r[3]}
        for r in rows
    ]
