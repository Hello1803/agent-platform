"""Request/response models for the API layer. Reuses the real domain
models (AgentSpecification, MutationResult) as response bodies wherever
possible, rather than hand-maintaining parallel API-shape models that
could drift from what the mutation engine and compiler actually return.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.compiler.state import TraceEvent
from backend.models.agent_spec import AgentSpecification, MutationResult


class CreateAgentRequest(BaseModel):
    spec: AgentSpecification


class CreateWorkflowRequest(BaseModel):
    name: str = Field(min_length=1)
    description: str | None = None


class RenameWorkflowRequest(BaseModel):
    name: str = Field(min_length=1)


class WorkflowSummary(BaseModel):
    """Lightweight listing shape for GET /agents/{agent_id}/workflows —
    everything a workflow selector UI needs without shipping every
    node/edge/tool in the full spec for every workflow just to populate a
    dropdown."""

    workflow_id: str
    agent_id: str
    name: str
    description: str | None = None
    enabled: bool
    status: str
    version: str
    published_version: str | None = None
    updated_at: str


class MutationsRequest(BaseModel):
    mutations: list[dict[str, Any]] = Field(description="Raw mutation objects, each with a 'kind' field.")
    expected_version: str | None = Field(
        default=None,
        description="Optimistic-concurrency guard — the metadata.version this batch was computed against.",
    )


class InstructionRequest(BaseModel):
    message: str


class InstructionResponse(BaseModel):
    reasoning: str
    attempts: int
    result: MutationResult


class ChatProposeRequest(BaseModel):
    message: str
    # Optional per-request provider override ("openai", "anthropic") --
    # same "<provider>[:<model>]" convention as RuntimeConfig.model and
    # the Ask AI inspector's per-node config.provider (see
    # backend/ai_providers/registry.py). None means "use the server's
    # configured default provider."
    provider: str | None = None
    model: str | None = None


class ChatProposalResponse(BaseModel):
    """A PROPOSED change — nothing has been persisted yet. `mutations` is
    the raw batch the model produced; send it back verbatim to
    POST .../chat/apply (along with `base_version`) to actually commit
    it. `ok=False` means the mutations didn't validate even after the NL
    compiler's internal retry loop — the workflow is guaranteed
    unchanged either way, since apply_mutations never touches the store."""

    reasoning: str
    mutations: list[dict[str, Any]]
    ok: bool
    errors: list[dict[str, Any]] = Field(default_factory=list)
    attempts: int
    base_version: str
    provider: str
    model: str | None = None
    # A full preview of what the workflow WOULD look like if applied —
    # lets the UI render an actual diff/preview rather than just a
    # mutation list, without persisting anything.
    preview_spec: dict[str, Any] | None = None


class ChatApplyRequest(BaseModel):
    mutations: list[dict[str, Any]] = Field(description="The exact mutations echoed back from a prior ChatProposalResponse.")
    expected_version: str = Field(description="The base_version from that same proposal — optimistic-concurrency guard.")
    force: bool = Field(
        default=False,
        description="If true, apply against the CURRENT stored version (ignore expected_version mismatch). "
        "Used when the user confirms overwrite after moving nodes or other concurrent edits.",
    )


class CompileCheckResponse(BaseModel):
    ok: bool
    error: str | None = None
    # Node id the error is about, if it's about a specific node — lets the
    # frontend jump straight to that node's inspector with a "Configure X"
    # action instead of just displaying a raw error string.
    node_id: str | None = None


class RunRequest(BaseModel):
    seed_variables: dict[str, Any] = Field(default_factory=dict)


class ResumeRequest(BaseModel):
    thread_id: str
    resume: dict[str, Any]


class RunResponse(BaseModel):
    thread_id: str
    status: str  # "completed" | "paused" | "failed"
    trace: list[TraceEvent]
    variables: dict[str, Any]
    interrupt: dict[str, Any] | None = None
    error: str | None = None
    failed_node_id: str | None = None


class PublishResponse(BaseModel):
    spec: AgentSpecification
    published_version: str
