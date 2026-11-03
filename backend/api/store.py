"""
In-memory storage for agent specs and their run checkpointers.

This is deliberately the simplest thing that works — a dict behind a
lock — so the API layer's request/response contracts can be nailed down
and tested before anyone has to make a database decision. Swapping this
for real persistence (Postgres for specs, Redis/Postgres-backed
checkpointer for LangGraph state) later only touches this file; the
route handlers in app.py talk to AgentStore's interface, not to dicts
directly.
"""

from __future__ import annotations

import threading

from langgraph.checkpoint.memory import MemorySaver

from backend.models.agent_spec import AgentSpecification


class AgentNotFoundError(KeyError):
    pass


class AgentStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._specs: dict[str, AgentSpecification] = {}
        self._versions: dict[str, dict[str, AgentSpecification]] = {}
        self._checkpointers: dict[str, MemorySaver] = {}

    def create(self, spec: AgentSpecification) -> None:
        with self._lock:
            self._specs[spec.metadata.id] = spec
            self._versions[spec.metadata.id] = {spec.metadata.version: spec}
            self._checkpointers[spec.metadata.id] = MemorySaver()

    def get(self, agent_id: str) -> AgentSpecification:
        with self._lock:
            if agent_id not in self._specs:
                raise AgentNotFoundError(agent_id)
            return self._specs[agent_id]

    def set(self, spec: AgentSpecification) -> None:
        with self._lock:
            if spec.metadata.id not in self._specs:
                raise AgentNotFoundError(spec.metadata.id)
            self._specs[spec.metadata.id] = spec

    def compare_and_set(self, agent_id: str, expected_version: str, new_spec: AgentSpecification) -> bool:
        """
        Atomically: if the CURRENTLY STORED spec's version is still
        `expected_version`, replace it with `new_spec` and return True;
        otherwise leave the store untouched and return False.

        This closes a real race that a Python-level "check version, then
        set()" sequence doesn't: apply_mutations() checks `expected_version`
        against whatever spec the caller already fetched, but two concurrent
        requests can both fetch the same version, both pass that check, and
        both then call set() — the second one silently overwrites the
        first's write with no error, a genuine lost update. Because this
        method does the read-check-write as ONE critical section under the
        store's lock (never releasing it between the version check and the
        write), two concurrent callers can't both succeed: whichever
        acquires the lock first wins and moves the stored version forward;
        the second sees a version that no longer matches and gets False.
        """
        with self._lock:
            current = self._specs.get(agent_id)
            if current is None:
                raise AgentNotFoundError(agent_id)
            if current.metadata.version != expected_version:
                return False
            self._specs[agent_id] = new_spec
            self._versions[agent_id][new_spec.metadata.version] = new_spec
            return True

    def list_agents(self) -> list[AgentSpecification]:
        with self._lock:
            return list(self._specs.values())

    def list_versions(self, agent_id: str) -> list[AgentSpecification]:
        with self._lock:
            if agent_id not in self._versions:
                raise AgentNotFoundError(agent_id)
            return sorted(self._versions[agent_id].values(), key=lambda spec: spec.metadata.updated_at)

    def get_version(self, agent_id: str, version: str) -> AgentSpecification:
        with self._lock:
            try:
                return self._versions[agent_id][version]
            except KeyError as exc:
                raise AgentNotFoundError(f"{agent_id}@{version}") from exc

    def get_published(self, agent_id: str) -> AgentSpecification:
        current = self.get(agent_id)
        if not current.metadata.published_version:
            raise AgentNotFoundError(f"{agent_id}@published")
        return self.get_version(agent_id, current.metadata.published_version)

    def checkpointer(self, agent_id: str) -> MemorySaver:
        with self._lock:
            if agent_id not in self._checkpointers:
                raise AgentNotFoundError(agent_id)
            return self._checkpointers[agent_id]

    def exists(self, agent_id: str) -> bool:
        with self._lock:
            return agent_id in self._specs

    def delete(self, agent_id: str) -> None:
        """Deletes a workflow (spec) and everything scoped to it — its
        version history and its checkpointer (in-run execution state).
        Does NOT touch any other workflow, by construction: everything
        here is keyed by `agent_id` (== workflowId), so deleting one
        entry can never reach into another's dict slot."""
        with self._lock:
            if agent_id not in self._specs:
                raise AgentNotFoundError(agent_id)
            del self._specs[agent_id]
            self._versions.pop(agent_id, None)
            self._checkpointers.pop(agent_id, None)

    def list_by_agent(self, agent_id: str) -> list[AgentSpecification]:
        """All workflows belonging to the given Agent container —
        matches on AgentMetadata.effective_agent_id, so a legacy spec
        with no explicit `agentId` set (effective_agent_id falls back to
        its own id) is correctly found when `agent_id` equals ITS OWN id,
        preserving single-workflow-agent lookups with zero migration."""
        with self._lock:
            return [s for s in self._specs.values() if s.metadata.effective_agent_id == agent_id]


# Process-wide singleton. Fine for a single-worker dev server; a real
# deployment behind multiple workers needs the DB-backed swap described
# above regardless of how this is structured.
store = AgentStore()
