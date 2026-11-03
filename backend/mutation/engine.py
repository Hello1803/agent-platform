"""
The Workflow Mutation Engine.

Neither the chat builder nor the visual canvas is allowed to write to an
AgentSpecification directly. Both propose AgentSpecMutation objects; this
module is the single place that applies them, and it always re-validates
the whole spec (via Pydantic) before returning success. This is the
concrete mechanism behind architecture doc section 23's answer to
"Concurrent Modifications": *make both operations mutations of the Agent
Specification and apply validation/merging before recompilation.*

Two properties matter here:

1. Atomicity — apply_mutations() takes a LIST and applies it as one unit.
   If mutation 3 of 5 is invalid, mutations 1 and 2 are rolled back too;
   the caller never sees a half-applied spec. (e.g. the NL compiler often
   needs to add a node AND wire an edge to it in the same turn — either
   both land or neither does.)

2. Optimistic concurrency — every batch can be tagged with the
   `metadata.version` it was computed against. If the spec has already
   moved past that version (someone else's edit landed first — the classic
   "user drags a node while chat says 'add an approval step'" race), the
   batch is rejected with a clear conflict instead of silently clobbering
   the other edit. The caller re-fetches the latest spec, re-derives its
   mutation against the new state, and retries.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from backend.models.agent_spec import (
    AgentSpecification,
    AgentSpecMutation,
    MutationResult,
    ValidationError,
)


class MutationApplyError(Exception):
    def __init__(self, message: str, path: str = "<mutation>"):
        super().__init__(message)
        self.path = path


def _find(items: list[dict], item_id: str, kind: str) -> dict:
    for item in items:
        if item["id"] == item_id:
            return item
    raise MutationApplyError(f"No {kind} with id '{item_id}' to update.", path=f"{kind}.{item_id}")


def _assert_unique(items: list[dict], item_id: str, kind: str) -> None:
    if any(item["id"] == item_id for item in items):
        raise MutationApplyError(f"A {kind} with id '{item_id}' already exists.", path=f"{kind}.{item_id}")


def _bump_version(version: str) -> str:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
    if not match:
        return version  # non-semver version string: leave it to the caller to manage
    major, minor, patch = match.groups()
    return f"{major}.{minor}.{int(patch) + 1}"


def _apply_one(working: dict[str, Any], mutation: AgentSpecMutation) -> dict[str, Any]:
    kind = mutation.kind

    if kind == "add_node":
        _assert_unique(working["nodes"], mutation.node.id, "node")
        working["nodes"].append(mutation.node.model_dump(mode="json", by_alias=True))

    elif kind == "update_node":
        node = _find(working["nodes"], mutation.node_id, "node")
        node.update(mutation.patch)

    elif kind == "remove_node":
        node_id = mutation.node_id
        working["nodes"] = [n for n in working["nodes"] if n["id"] != node_id]
        # Cascade: a node with no edges pointing at it is a common source of
        # dangling references the graph-integrity validator would otherwise
        # reject as an opaque error, so we clean up here instead of making
        # every caller do it.
        working["edges"] = [
            e for e in working["edges"] if e["source"] != node_id and e["target"] != node_id
        ]
        working["humanApproval"] = [
            r for r in working.get("humanApproval", [])
            if (r.get("appliesTo") or {}).get("nodeId") != node_id
        ]

    elif kind == "add_edge":
        _assert_unique(working["edges"], mutation.edge.id, "edge")
        working["edges"].append(mutation.edge.model_dump(mode="json", by_alias=True))

    elif kind == "update_edge":
        edge = _find(working["edges"], mutation.edge_id, "edge")
        edge.update(mutation.patch)

    elif kind == "remove_edge":
        working["edges"] = [e for e in working["edges"] if e["id"] != mutation.edge_id]

    elif kind == "update_metadata":
        working["metadata"].update(mutation.patch)

    elif kind == "update_goal":
        working["goal"] = mutation.goal

    elif kind == "update_instructions":
        working["instructions"] = mutation.instructions

    elif kind == "add_tool":
        _assert_unique(working.get("tools", []), mutation.tool.id, "tool")
        working.setdefault("tools", []).append(mutation.tool.model_dump(mode="json", by_alias=True))

    elif kind == "remove_tool":
        tool_id = mutation.tool_id
        working["tools"] = [t for t in working.get("tools", []) if t["id"] != tool_id]
        # Cascade: strip the now-dangling reference from any node that had it,
        # rather than leaving nodes pointing at a tool binding that no longer
        # exists (which graph-integrity validation would reject anyway).
        for n in working["nodes"]:
            if n.get("toolRefs"):
                n["toolRefs"] = [r for r in n["toolRefs"] if r != tool_id]

    elif kind == "update_tool":
        tool = _find(working.get("tools", []), mutation.tool_id, "tool")
        tool.update(mutation.patch)

    elif kind == "update_permissions":
        working["permissions"] = {**(working.get("permissions") or {}), **mutation.patch}

    elif kind == "add_approval_rule":
        working.setdefault("humanApproval", []).append(mutation.rule.model_dump(mode="json", by_alias=True))

    elif kind == "update_runtime_config":
        working["runtimeConfig"] = {**(working.get("runtimeConfig") or {}), **mutation.patch}

    else:  # pragma: no cover - the discriminated union prevents this at parse time
        raise MutationApplyError(f"Unhandled mutation kind: '{kind}'.")

    return working


def apply_mutations(
    spec: AgentSpecification,
    mutations: list[AgentSpecMutation],
    expected_version: str | None = None,
) -> MutationResult:
    """
    Apply a batch of mutations atomically. Returns a fresh, validated
    AgentSpecification on success — the input `spec` is never modified in
    place, so a caller holding a reference to the old spec still has a
    consistent view of it even after a failed or successful apply.
    """
    if expected_version is not None and spec.metadata.version != expected_version:
        return MutationResult(
            ok=False,
            errors=[
                ValidationError(
                    path="metadata.version",
                    message=(
                        f"Version conflict: this edit was based on version '{expected_version}', "
                        f"but the spec is now at '{spec.metadata.version}' (someone else's edit "
                        f"landed first). Re-fetch the latest spec and retry."
                    ),
                    severity="error",
                )
            ],
        )

    working = spec.model_dump(mode="json", by_alias=True)

    try:
        for mutation in mutations:
            working = _apply_one(working, mutation)
    except MutationApplyError as e:
        return MutationResult(ok=False, errors=[ValidationError(path=e.path, message=str(e), severity="error")])

    working["metadata"]["version"] = _bump_version(working["metadata"]["version"])
    # A published snapshot remains immutable; every subsequent workflow edit
    # is a new draft revision, not an in-place change to what users run.
    if working["metadata"].get("publishedVersion"):
        working["metadata"]["status"] = "draft"
    working["metadata"]["updatedAt"] = datetime.now(timezone.utc).isoformat()

    try:
        new_spec = AgentSpecification.model_validate(working)
    except PydanticValidationError as e:
        errors = [
            ValidationError(path=".".join(str(p) for p in err["loc"]) or "<root>", message=err["msg"])
            for err in e.errors()
        ]
        return MutationResult(ok=False, errors=errors)

    return MutationResult(ok=True, spec=new_spec)


def apply_mutation(
    spec: AgentSpecification,
    mutation: AgentSpecMutation,
    expected_version: str | None = None,
) -> MutationResult:
    return apply_mutations(spec, [mutation], expected_version=expected_version)


def publish(spec: AgentSpecification, expected_version: str | None = None) -> MutationResult:
    """
    Promotes the current draft to published, per doc section 16
    (Published: v1.4 / Draft: v1.5). Deliberately does NOT go through
    apply_mutations' automatic version bump — publishing is a status
    transition on the current draft, not a new edit to it. The version
    that becomes `metadata.publishedVersion` is exactly the version that
    was live when this was called.
    """
    if expected_version is not None and spec.metadata.version != expected_version:
        return MutationResult(
            ok=False,
            errors=[
                ValidationError(
                    path="metadata.version",
                    message=(
                        f"Version conflict: expected '{expected_version}' but spec is at "
                        f"'{spec.metadata.version}'. Re-fetch and retry."
                    ),
                    severity="error",
                )
            ],
        )

    working = spec.model_dump(mode="json", by_alias=True)
    working["metadata"]["publishedVersion"] = spec.metadata.version
    working["metadata"]["status"] = "published"
    working["metadata"]["updatedAt"] = datetime.now(timezone.utc).isoformat()

    try:
        new_spec = AgentSpecification.model_validate(working)
    except PydanticValidationError as e:
        errors = [
            ValidationError(path=".".join(str(p) for p in err["loc"]) or "<root>", message=err["msg"])
            for err in e.errors()
        ]
        return MutationResult(ok=False, errors=errors)

    return MutationResult(ok=True, spec=new_spec)
