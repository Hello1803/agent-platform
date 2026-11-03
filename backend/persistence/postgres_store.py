"""
A real, Postgres-backed implementation of the AgentStore interface from
backend/api/store.py, so `backend/api/app.py` doesn't lose every agent
spec and every in-flight (paused-on-approval) run the moment the process
restarts.

Two tables:

    agents(id, spec_json, updated_at)
        The current AgentSpecification per agent, as JSONB. One row per
        agent — mutations overwrite the row (the mutation engine already
        gives us a validated, versioned spec; this table doesn't need its
        own history, since `metadata.version` IS the history marker).

    checkpoints / checkpoint_*  (owned by LangGraph itself)
        Created and managed by langgraph-checkpoint-postgres's
        PostgresSaver — this is what makes a paused human_approval
        interrupt survive a server restart: the run's state lives in
        Postgres, not in a process-local MemorySaver dict.

Same public interface as api.store.AgentStore (create/get/set/checkpointer/
exists) on purpose — backend/api/app.py doesn't need to change to use this
instead of the in-memory version; only the object it imports as `store`
changes (see PostgresAgentStore.checkpointer() below and __main__ usage).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

from backend.api.store import AgentNotFoundError
from backend.models.agent_spec import AgentSpecification

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    spec_json JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS agent_versions (
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    version TEXT NOT NULL,
    spec_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (agent_id, version)
);
"""


class PostgresAgentStore:
    """
    Real persistence, same shape as api.store.AgentStore. `conn_string` is
    a standard libpq connection string, e.g.
    "postgresql://agent_platform:agent_platform_dev@localhost/agent_platform".

    A single PostgresSaver (LangGraph's checkpointer) is shared across all
    agents — checkpoints are already namespaced by thread_id internally,
    so per-agent isolation isn't needed at this layer; per-agent isolation
    of the SPEC itself is what the `agents` table provides.
    """

    def __init__(self, conn_string: str):
        self.conn_string = conn_string
        self._init_schema()
        self._checkpointer_cm = None
        self._checkpointer = None

    def _init_schema(self) -> None:
        with psycopg.connect(self.conn_string, autocommit=True) as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[psycopg.Connection]:
        with psycopg.connect(self.conn_string, row_factory=dict_row) as conn:
            yield conn

    def create(self, spec: AgentSpecification) -> None:
        payload = json.dumps(spec.model_dump(mode="json", by_alias=True))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM agents WHERE id = %s", (spec.metadata.id,))
                if cur.fetchone():
                    raise ValueError(f"Agent '{spec.metadata.id}' already exists.")
                cur.execute(
                    "INSERT INTO agents (id, spec_json, updated_at) VALUES (%s, %s, now())",
                    (spec.metadata.id, payload),
                )
                cur.execute("INSERT INTO agent_versions (agent_id, version, spec_json) VALUES (%s, %s, %s)", (spec.metadata.id, spec.metadata.version, payload))
            conn.commit()

    def get(self, agent_id: str) -> AgentSpecification:
        with self._connect() as conn:
            row = conn.execute("SELECT spec_json FROM agents WHERE id = %s", (agent_id,)).fetchone()
        if row is None:
            raise AgentNotFoundError(agent_id)
        return AgentSpecification.model_validate(row["spec_json"])

    def set(self, spec: AgentSpecification) -> None:
        payload = json.dumps(spec.model_dump(mode="json", by_alias=True))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE agents SET spec_json = %s, updated_at = now() WHERE id = %s",
                    (payload, spec.metadata.id),
                )
                if cur.rowcount == 0:
                    raise AgentNotFoundError(spec.metadata.id)
            conn.commit()

    def compare_and_set(self, agent_id: str, expected_version: str, new_spec: AgentSpecification) -> bool:
        """
        Atomically: if the version currently stored in Postgres for this
        agent is still `expected_version`, replace it with `new_spec` and
        return True; otherwise leave the row untouched and return False.

        Unlike the in-memory store, where a Python `threading.Lock` can
        make read-check-write one critical section, Postgres needs the
        check done AS PART OF the same statement as the write, or two
        concurrent connections can genuinely both pass a
        SELECT-then-UPDATE check (classic TOCTOU) — a Python-level
        pre-check (apply_mutations() checking expected_version against an
        already-fetched spec) is not enough on its own to prevent a lost
        update when two requests race. The WHERE clause here makes
        Postgres itself the sole arbiter: the UPDATE only matches (and
        only takes effect) if the row's version is still what we expect,
        checked and written in one atomic statement, and `cur.rowcount`
        tells us definitively whether we won that race.
        """
        payload = json.dumps(new_spec.model_dump(mode="json", by_alias=True))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE agents
                    SET spec_json = %s, updated_at = now()
                    WHERE id = %s AND spec_json->'metadata'->>'version' = %s
                    """,
                    (payload, agent_id, expected_version),
                )
                won = cur.rowcount == 1
                if won:
                    cur.execute(
                        "INSERT INTO agent_versions (agent_id, version, spec_json) VALUES (%s, %s, %s) ON CONFLICT (agent_id, version) DO NOTHING",
                        (agent_id, new_spec.metadata.version, payload),
                    )
            conn.commit()
        if not won:
            # Distinguish "agent doesn't exist at all" from "it exists but
            # the version moved" so callers get an accurate error either way.
            if not self.exists(agent_id):
                raise AgentNotFoundError(agent_id)
        return won

    def list_agents(self) -> list[AgentSpecification]:
        with self._connect() as conn:
            rows = conn.execute("SELECT spec_json FROM agents ORDER BY updated_at DESC").fetchall()
        return [AgentSpecification.model_validate(row["spec_json"]) for row in rows]

    def list_versions(self, agent_id: str) -> list[AgentSpecification]:
        with self._connect() as conn:
            rows = conn.execute("SELECT spec_json FROM agent_versions WHERE agent_id = %s ORDER BY created_at", (agent_id,)).fetchall()
        if not rows:
            raise AgentNotFoundError(agent_id)
        return [AgentSpecification.model_validate(row["spec_json"]) for row in rows]

    def get_version(self, agent_id: str, version: str) -> AgentSpecification:
        with self._connect() as conn:
            row = conn.execute("SELECT spec_json FROM agent_versions WHERE agent_id = %s AND version = %s", (agent_id, version)).fetchone()
        if row is None:
            raise AgentNotFoundError(f"{agent_id}@{version}")
        return AgentSpecification.model_validate(row["spec_json"])

    def get_published(self, agent_id: str) -> AgentSpecification:
        current = self.get(agent_id)
        if not current.metadata.published_version:
            raise AgentNotFoundError(f"{agent_id}@published")
        return self.get_version(agent_id, current.metadata.published_version)

    def exists(self, agent_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM agents WHERE id = %s", (agent_id,)).fetchone()
        return row is not None

    def checkpointer(self, agent_id: str):
        """
        Returns a real PostgresSaver — checkpoints for a paused (interrupted)
        run genuinely persist across process restarts, unlike
        MemorySaver's process-local dict. `agent_id` isn't used to
        namespace this (LangGraph already namespaces by thread_id), it's
        accepted only to keep this method's signature identical to
        api.store.AgentStore.checkpointer(), so app.py doesn't need to
        change which store it imports.
        """
        if self._checkpointer is None:
            from langgraph.checkpoint.postgres import PostgresSaver

            self._checkpointer_cm = PostgresSaver.from_conn_string(self.conn_string)
            self._checkpointer = self._checkpointer_cm.__enter__()
            self._checkpointer.setup()  # idempotent — creates LangGraph's own checkpoint tables
        return self._checkpointer

    def close(self) -> None:
        if self._checkpointer_cm is not None:
            self._checkpointer_cm.__exit__(None, None, None)
            self._checkpointer_cm = None
            self._checkpointer = None
