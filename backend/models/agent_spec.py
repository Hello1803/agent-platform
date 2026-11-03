"""
Pydantic models for the AgentSpecification IR.

Mirrors schema/agent-spec.schema.json and packages/shared-types/src/agent-spec.ts.
All three must stay in sync — see backend/validation/check_examples.py for a
cross-check that validates example specs against both the JSON Schema and
these models.

This module is imported by:
  - the Agent Compiler (validates + turns a spec into a LangGraph execution graph)
  - the Workflow Mutation Engine (applies mutations, re-validates)
  - the NL compiler (parses LLM output into a spec)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator

AGENT_SPEC_VERSION: Literal["1.0.0"] = "1.0.0"


class AgentStatus(str, Enum):
    draft = "draft"
    published = "published"
    archived = "archived"


class AgentMetadata(BaseModel):
    id: str
    name: str = Field(min_length=1)
    description: Optional[str] = None
    status: AgentStatus
    version: str
    published_version: Optional[str] = Field(default=None, alias="publishedVersion")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    owner: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    # --- Multi-workflow fields (additive; every existing single-workflow
    # spec keeps working unchanged) ---
    # `id` above IS the workflowId — this was already the one thing every
    # part of the system (AgentStore, the compiler, the mutation engine,
    # credential ownership, versioning, the checkpointer) keys execution
    # and persistence by, so a "workflow" in the new multi-workflow model
    # is simply what this codebase already called an "agent": one
    # AgentSpecification, unchanged. What's new is `agent_id`: the
    # lightweight CONTAINER a workflow belongs to (backend/models/agent.py's
    # Agent) — several workflows can share one. Defaults to None on the
    # wire; `effective_agent_id` below is what every route actually reads,
    # so an unset value transparently means "this workflow is its own
    # single-workflow agent," matching every spec that existed before this
    # field did, with zero migration.
    agent_id: Optional[str] = Field(default=None, alias="agentId")
    # Whether this workflow is active — e.g. a schedule/webhook trigger
    # should not fire for a disabled workflow. Manual runs and the canvas
    # remain usable either way; enforcement of what "disabled" blocks is a
    # caller-level concern (see backend/api/app.py's run routes), not
    # something baked into the schema itself.
    enabled: bool = True

    model_config = {"populate_by_name": True}

    @property
    def effective_agent_id(self) -> str:
        """The agent container this workflow belongs to, defaulting to
        its own id when unset — see the `agent_id` field's docstring."""
        return self.agent_id or self.id


class TriggerType(str, Enum):
    manual = "manual"
    schedule = "schedule"
    webhook = "webhook"
    email = "email"
    file_change = "file_change"
    external_event = "external_event"
    api = "api"


class Trigger(BaseModel):
    id: str
    type: TriggerType
    config: dict[str, Any] = Field(default_factory=dict)


class NodeType(str, Enum):
    # AI / Agent nodes
    ai_agent = "ai_agent"
    ask_ai = "ask_ai"
    classify = "classify"
    extract_information = "extract_information"
    summarize = "summarize"
    make_decision = "make_decision"
    run_agent = "run_agent"
    # Tool nodes
    search_web = "search_web"
    open_website = "open_website"
    read_email = "read_email"
    send_email = "send_email"
    search_documents = "search_documents"
    read_file = "read_file"
    write_file = "write_file"
    database = "database"
    api_call = "api_call"
    code = "code"
    preview = "preview"
    rag_index = "rag_index"
    rag_query = "rag_query"
    browser = "browser"
    # Control nodes
    if_else = "if_else"
    switch = "switch"
    loop = "loop"
    parallel = "parallel"
    merge = "merge"
    wait = "wait"
    retry = "retry"
    # Human nodes
    human_approval = "human_approval"
    ask_user = "ask_user"
    review = "review"
    confirm = "confirm"
    # Trigger nodes
    trigger_manual = "trigger_manual"
    trigger_schedule = "trigger_schedule"
    trigger_email = "trigger_email"
    trigger_webhook = "trigger_webhook"
    trigger_file_change = "trigger_file_change"
    trigger_external_event = "trigger_external_event"


class NodeCategory(str, Enum):
    ai = "ai"
    tool = "tool"
    control = "control"
    human = "human"
    trigger = "trigger"


class DataType(str, Enum):
    string = "string"
    number = "number"
    boolean = "boolean"
    object = "object"
    array = "array"
    file = "file"
    any = "any"


class Port(BaseModel):
    id: str
    name: str
    data_type: Optional[DataType] = Field(default=None, alias="dataType")
    required: bool = False

    model_config = {"populate_by_name": True}


class CanvasPosition(BaseModel):
    x: float
    y: float


class SubgraphRef(BaseModel):
    agent_id: str = Field(alias="agentId")
    version_pin: Optional[str] = Field(default=None, alias="versionPin")

    model_config = {"populate_by_name": True}


class ExecutionType(str, Enum):
    deterministic = "deterministic"
    probabilistic = "probabilistic"


# Node types that require an AI call at runtime; used to enforce that
# executionType="probabilistic" lines up with reality (see validator.py).
PROBABILISTIC_CAPABLE_TYPES = {
    NodeType.ai_agent,
    NodeType.ask_ai,
    NodeType.classify,
    NodeType.extract_information,
    NodeType.summarize,
    NodeType.make_decision,
    NodeType.run_agent,
}


class AgentNode(BaseModel):
    id: str
    type: NodeType
    category: NodeCategory
    label: str
    execution_type: ExecutionType = Field(default=ExecutionType.deterministic, alias="executionType")
    confidence_threshold: Optional[float] = Field(default=None, alias="confidenceThreshold", ge=0, le=1)
    config: dict[str, Any] = Field(default_factory=dict)
    inputs: list[Port] = Field(default_factory=list)
    outputs: list[Port] = Field(default_factory=list)
    tool_refs: list[str] = Field(default_factory=list, alias="toolRefs")
    subgraph: Optional[SubgraphRef] = None
    canvas_position: Optional[CanvasPosition] = Field(default=None, alias="canvasPosition")

    model_config = {"populate_by_name": True}

    @model_validator(mode="before")
    @classmethod
    def _coerce_execution_type(cls, data: Any) -> Any:
        """LLMs sometimes mark non-AI nodes as probabilistic. Coerce those
        to deterministic instead of rejecting the whole mutation batch."""
        if not isinstance(data, dict):
            return data
        node_type = data.get("type")
        type_val = node_type.value if hasattr(node_type, "value") else str(node_type) if node_type is not None else None
        et = data.get("executionType", data.get("execution_type"))
        if et == "probabilistic" and type_val is not None:
            capable = {t.value if hasattr(t, "value") else str(t) for t in PROBABILISTIC_CAPABLE_TYPES}
            if type_val not in capable:
                data = {**data, "executionType": "deterministic", "execution_type": "deterministic"}
        # Local file/code nodes need no tool binding — NL often invents
        # toolRefs like "read_file" which are not binding ids.
        if type_val in ("read_file", "write_file", "code", "api_call", "rag_index", "rag_query", "preview"):
            data = {**data, "toolRefs": [], "tool_refs": []}
        return data

    @model_validator(mode="after")
    def _check_probabilistic_flag(self) -> "AgentNode":
        if self.execution_type == ExecutionType.probabilistic and self.type not in PROBABILISTIC_CAPABLE_TYPES:
            object.__setattr__(self, "execution_type", ExecutionType.deterministic)
        return self

    @model_validator(mode="after")
    def _check_ai_provider_config(self) -> "AgentNode":
        """A node's `config.provider`/`config.model` (set by the Ask AI
        inspector's provider/model dropdowns — see
        packages/canvas/src/NodeInspector.tsx) is the per-node counterpart
        to RuntimeConfig.model's spec-wide preference, and is validated
        exactly the same way: an unsupported/planned provider, or a
        provider paired with a model that unambiguously belongs to a
        DIFFERENT provider (e.g. `{"provider": "openai", "model":
        "claude-sonnet-5"}`), is rejected here at spec-construction time
        rather than discovered later at compile or run time. Irrelevant
        (and skipped) for any node without a `config.provider` — most
        nodes, and even most AI nodes, never set this at all and keep
        using the spec/server default.
        """
        provider = self.config.get("provider") if isinstance(self.config, dict) else None
        if not provider:
            return self

        from backend.ai_providers.registry import (
            PLANNED_PROVIDERS,
            SUPPORTED_PROVIDERS,
            ModelValidationError,
            validate_provider_model,
        )

        provider = str(provider).strip().lower()
        if provider in PLANNED_PROVIDERS:
            raise ValueError(
                f"Node '{self.id}' config.provider references '{provider}', which is planned but not "
                f"implemented yet. Supported providers: {', '.join(SUPPORTED_PROVIDERS)}."
            )
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Node '{self.id}' config.provider references unknown provider '{provider}'. "
                f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}."
            )
        model = self.config.get("model")
        if model:
            try:
                validate_provider_model(provider, str(model))
            except ModelValidationError as exc:
                raise ValueError(f"Node '{self.id}' has an invalid provider/model combination: {exc}") from None
        return self


class EdgeConditionKind(str, Enum):
    deterministic = "deterministic"
    ai_decision = "ai_decision"


class EdgeCondition(BaseModel):
    expression: Optional[str] = None
    kind: EdgeConditionKind = EdgeConditionKind.deterministic


class AgentEdge(BaseModel):
    id: str
    source: str
    source_handle: Optional[str] = Field(default=None, alias="sourceHandle")
    target: str
    target_handle: Optional[str] = Field(default=None, alias="targetHandle")
    condition: Optional[EdgeCondition] = None
    priority: Optional[int] = None
    label: Optional[str] = None

    model_config = {"populate_by_name": True}

    @model_validator(mode="before")
    @classmethod
    def _coerce_handles(cls, data: Any) -> Any:
        """NL models often invent handle names (outputs, file_content, …).
        This canvas only has output → input; coerce anything else."""
        if not isinstance(data, dict):
            return data
        src = data.get("sourceHandle", data.get("source_handle"))
        tgt = data.get("targetHandle", data.get("target_handle"))
        # None is fine (defaults applied later); only fix explicit wrong values
        if src is not None and src != "output":
            data = {**data, "sourceHandle": "output", "source_handle": "output"}
        if tgt is not None and tgt != "input":
            data = {**data, "targetHandle": "input", "target_handle": "input"}
        return data


class ToolBinding(BaseModel):
    id: str
    tool_id: str = Field(alias="toolId")
    display_name: Optional[str] = Field(default=None, alias="displayName")
    credential_ref: Optional[str] = Field(default=None, alias="credentialRef")
    permissions: list[str] = Field(default_factory=list)
    rate_limit: Optional[dict[str, int]] = Field(default=None, alias="rateLimit")
    # Demo bindings are explicitly non-production and may use stub executors.
    # Connected bindings require a real registry tool and credential reference.
    mode: Literal["demo", "connected"] = "demo"

    model_config = {"populate_by_name": True}


class MemoryType(str, Enum):
    none = "none"
    conversation_buffer = "conversation_buffer"
    vector_store = "vector_store"
    key_value = "key_value"
    summary = "summary"


class MemoryScope(str, Enum):
    run = "run"
    session = "session"
    agent_global = "agent_global"


class MemoryConfig(BaseModel):
    enabled: bool = False
    type: MemoryType = MemoryType.none
    scope: MemoryScope = MemoryScope.run
    retention_days: Optional[int] = Field(default=None, alias="retentionDays")

    model_config = {"populate_by_name": True}


class PermissionSet(BaseModel):
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class AutonomyLevel(str, Enum):
    suggest = "suggest"
    ask_permission = "ask_permission"
    automatic = "automatic"


class ApprovalActionCategory(str, Enum):
    money = "money"
    send_message = "send_message"
    delete_data = "delete_data"
    account_change = "account_change"
    external_publish = "external_publish"
    sensitive_info = "sensitive_info"
    other = "other"


class ApprovalAppliesTo(BaseModel):
    node_id: Optional[str] = Field(default=None, alias="nodeId")
    action_category: Optional[ApprovalActionCategory] = Field(default=None, alias="actionCategory")

    model_config = {"populate_by_name": True}


class ApprovalRule(BaseModel):
    id: str
    applies_to: ApprovalAppliesTo = Field(alias="appliesTo")
    autonomy_level: AutonomyLevel = Field(alias="autonomyLevel")
    approver_role: str = Field(default="owner", alias="approverRole")

    model_config = {"populate_by_name": True}


class Policies(BaseModel):
    tool_allowlist: list[str] = Field(default_factory=list, alias="toolAllowlist")
    data_access_boundaries: list[str] = Field(default_factory=list, alias="dataAccessBoundaries")
    prompt_injection_defenses: bool = Field(default=True, alias="promptInjectionDefenses")
    audit_log: bool = Field(default=True, alias="auditLog")
    max_run_duration_seconds: Optional[int] = Field(default=None, alias="maxRunDurationSeconds")
    max_cost_per_run: Optional[float] = Field(default=None, alias="maxCostPerRun")

    model_config = {"populate_by_name": True}


class RetryPolicy(BaseModel):
    max_retries: int = Field(default=2, alias="maxRetries")
    backoff_seconds: int = Field(default=5, alias="backoffSeconds")

    model_config = {"populate_by_name": True}


class RuntimeConfig(BaseModel):
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, alias="maxTokens")
    checkpointing: bool = True
    retry_policy: Optional[RetryPolicy] = Field(default=None, alias="retryPolicy")
    timeout_seconds: Optional[int] = Field(default=None, alias="timeoutSeconds")

    model_config = {"populate_by_name": True}

    @field_validator("model")
    @classmethod
    def _validate_model_provider(cls, value: Optional[str]) -> Optional[str]:
        """`model` is the ONE place an AgentSpec can express an AI provider
        preference, using the "<provider>:<model>" convention (e.g.
        "openai:gpt-4o", "anthropic:claude-sonnet-5") — see
        backend/ai_providers/registry.py's parse_model_string(). It is
        validated here, at spec-construction time, against
        SUPPORTED_PROVIDERS so a spec can never silently reference a
        provider that doesn't exist or isn't implemented yet — better to
        reject "gemini:gemini-2.5-pro" at save time with a clear message
        than have it compile fine and only fail at run time deep inside
        get_executor_overrides.

        Both the field and this validator are non-secret by construction:
        this only ever holds a provider *name* and a model *name*, never a
        credential — the actual API key resolution happens entirely
        server-side (backend/ai_providers/registry.py), independent of
        this field's value.
        """
        if not value:
            return value
        # Local import: backend/ai_providers has no dependency on this
        # module, so this is one-directional and safe, but importing at
        # call time (not module load time) keeps agent_spec.py from
        # requiring the ai_providers package to exist just to be imported
        # for schema-only purposes (e.g. the JSON-schema export step).
        from backend.ai_providers.registry import (
            PLANNED_PROVIDERS,
            SUPPORTED_PROVIDERS,
            ModelValidationError,
            parse_model_string,
            validate_provider_model,
        )

        provider, model = parse_model_string(value)
        if provider is None:
            # A bare model name with no "provider:" prefix — valid; the
            # provider is whatever AI_PROVIDER resolves to at run time.
            return value
        if provider in PLANNED_PROVIDERS:
            raise ValueError(
                f"runtimeConfig.model references provider '{provider}', which is planned but not "
                f"implemented yet. Supported providers: {', '.join(SUPPORTED_PROVIDERS)}."
            )
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"runtimeConfig.model references unknown provider '{provider}'. "
                f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}."
            )
        if model:
            # The actual "never allow an OpenAI provider with an
            # Anthropic model or vice versa" check — e.g.
            # "openai:claude-sonnet-5" has a valid provider AND a valid
            # (Anthropic) model name individually, so the checks above
            # alone wouldn't catch the mismatch between them.
            try:
                validate_provider_model(provider, model)
            except ModelValidationError as exc:
                raise ValueError(f"runtimeConfig.model is invalid: {exc}") from None
        return value


class AgentSpecification(BaseModel):
    spec_version: Literal["1.0.0"] = Field(default=AGENT_SPEC_VERSION, alias="specVersion")
    metadata: AgentMetadata
    goal: str = Field(min_length=1)
    instructions: Optional[str] = None
    triggers: list[Trigger] = Field(default_factory=list)
    nodes: list[AgentNode] = Field(min_length=1)
    edges: list[AgentEdge] = Field(default_factory=list)
    tools: list[ToolBinding] = Field(default_factory=list)
    memory: Optional[MemoryConfig] = None
    permissions: Optional[PermissionSet] = None
    human_approval: list[ApprovalRule] = Field(default_factory=list, alias="humanApproval")
    policies: Optional[Policies] = None
    runtime_config: Optional[RuntimeConfig] = Field(default=None, alias="runtimeConfig")

    model_config = {"populate_by_name": True}

    # --- Structural validation the JSON Schema alone can't express ---

    @model_validator(mode="after")
    def _check_graph_integrity(self) -> "AgentSpecification":
        node_ids = {n.id for n in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("Duplicate node ids in 'nodes'.")

        edge_ids = {e.id for e in self.edges}
        if len(edge_ids) != len(self.edges):
            raise ValueError("Duplicate edge ids in 'edges'.")

        nodes_by_id = {n.id: n for n in self.nodes}

        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(f"Edge '{edge.id}' references unknown source node '{edge.source}'.")
            if edge.target not in node_ids:
                raise ValueError(f"Edge '{edge.id}' references unknown target node '{edge.target}'.")

        # --- Edge semantics: independent of (and a backstop for) the
        # canvas's own connection validation (packages/canvas/src/specToFlow.ts
        # isValidAgentConnection). Every AgentEdge submitted through the API —
        # via POST /agents, an add_edge mutation, or the NL compiler — passes
        # through this same validator (AgentSpecification.model_validate),
        # so a hand-crafted HTTP request bypassing the canvas UI entirely
        # gets the identical checks a drag-and-drop connection does.

        # 1. No self-loops: a node's own output can't feed its own input.
        for edge in self.edges:
            if edge.source == edge.target:
                raise ValueError(f"Edge '{edge.id}' connects node '{edge.source}' to itself.")

        # 2. No duplicate edges: the same (source, target) pair wired twice
        # is never meaningful here — every node has exactly one output and
        # one input handle, so a second edge between the same pair can only
        # be an accidental double-connection, not a distinct routing.
        seen_pairs: set[tuple[str, str]] = set()
        for edge in self.edges:
            pair = (edge.source, edge.target)
            if pair in seen_pairs:
                raise ValueError(
                    f"Edge '{edge.id}' duplicates an existing connection from "
                    f"'{edge.source}' to '{edge.target}'."
                )
            seen_pairs.add(pair)

        # 3. Handle direction: a node's single output handle is always
        # conventionally id'd "output", its single input handle "input"
        # (see AgentNodeCard.tsx). An edge is only well-formed as
        # output -> input; anything else (an explicit "input" on the
        # source side or "output" on the target side) means an
        # output->output or input->input connection got through, which
        # should never be constructible via the canvas but is rejected
        # here too in case a client submits a spec directly.
        for edge in self.edges:
            if edge.source_handle is not None and edge.source_handle != "output":
                raise ValueError(
                    f"Edge '{edge.id}' has sourceHandle '{edge.source_handle}' — "
                    "an edge must originate from an output handle."
                )
            if edge.target_handle is not None and edge.target_handle != "input":
                raise ValueError(
                    f"Edge '{edge.id}' has targetHandle '{edge.target_handle}' — "
                    "an edge must terminate at an input handle."
                )

        # 4. Node/category compatibility: a trigger node is the graph's
        # entry point by definition (architecture doc — triggers start a
        # run, they don't react to one) and must never be the *target* of
        # an edge. Mirrors the identical rule in specToFlow.ts's
        # isValidAgentConnection so the canvas's live UI feedback and this
        # backstop agree on what's valid.
        for edge in self.edges:
            target_node = nodes_by_id.get(edge.target)
            if target_node is not None and target_node.category == NodeCategory.trigger:
                raise ValueError(
                    f"Edge '{edge.id}' targets '{edge.target}', a trigger node — "
                    "trigger nodes start a run and cannot receive incoming edges."
                )

        # 5. Uncontrolled cycles: a cycle made ENTIRELY of unconditional
        # edges can never terminate — nothing on that path ever routes
        # anywhere else, so a run entering it loops forever until
        # LangGraph's recursion limit kills it with an opaque error, not a
        # deliberate control-flow choice. This is NOT a blanket "no
        # cycles" rule — a `loop` node's whole mechanism (see
        # backend/compiler/compiler.py's conditional-edge routing) IS a
        # cycle, just one with a conditional "continue" edge that also has
        # a way out. So the rule is specifically: every cycle must include
        # at least one conditional edge (its exit). A cycle built purely
        # from plain connections — e.g. dragging web_search -> database ->
        # web_search on the canvas, none of which carry a condition — has
        # no such exit and is rejected; the same cycle with one leg turned
        # into a conditional edge (e.g. via an update_edge mutation, which
        # is how the NL compiler and a future condition editor both build
        # real loops) is fine. Mirrors
        # packages/canvas/src/specToFlow.ts's isValidAgentConnection,
        # which applies the identical rule to a proposed new connection
        # before it's even added.
        unconditional_adjacency: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for edge in self.edges:
            if not (edge.condition and edge.condition.expression):
                unconditional_adjacency[edge.source].append(edge.target)

        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n.id: WHITE for n in self.nodes}

        def _has_uncontrolled_cycle(node_id: str) -> bool:
            color[node_id] = GRAY
            for child in unconditional_adjacency[node_id]:
                if color[child] == GRAY:
                    return True
                if color[child] == WHITE and _has_uncontrolled_cycle(child):
                    return True
            color[node_id] = BLACK
            return False

        for node in self.nodes:
            if color[node.id] == WHITE and _has_uncontrolled_cycle(node.id):
                raise ValueError(
                    "Graph contains a cycle made entirely of unconditional edges "
                    f"(reachable from node '{node.id}') — a loop needs at least one "
                    "conditional edge to ever exit; otherwise a run would loop forever."
                )

        tool_ids = {t.id for t in self.tools}
        for node in self.nodes:
            # Drop dangling refs rather than rejecting the whole batch —
            # the NL compiler often invents binding ids that were never added.
            cleaned = [ref for ref in node.tool_refs if ref in tool_ids]
            if cleaned != list(node.tool_refs):
                object.__setattr__(node, "tool_refs", cleaned)

        for rule in self.human_approval:
            if rule.applies_to.node_id and rule.applies_to.node_id not in node_ids:
                raise ValueError(
                    f"Approval rule '{rule.id}' references unknown node '{rule.applies_to.node_id}'."
                )

        return self


# --- Mutations: the only sanctioned way for the chat builder or the visual
# editor to change a spec. Applied by the Workflow Mutation Engine, which
# re-runs full validation before recompiling (see Concurrent Modifications).

class AddNodeMutation(BaseModel):
    kind: Literal["add_node"] = "add_node"
    node: AgentNode


class UpdateNodeMutation(BaseModel):
    kind: Literal["update_node"] = "update_node"
    node_id: str = Field(alias="nodeId")
    patch: dict[str, Any]

    model_config = {"populate_by_name": True}


class RemoveNodeMutation(BaseModel):
    kind: Literal["remove_node"] = "remove_node"
    node_id: str = Field(alias="nodeId")

    model_config = {"populate_by_name": True}


class AddEdgeMutation(BaseModel):
    kind: Literal["add_edge"] = "add_edge"
    edge: AgentEdge


class RemoveEdgeMutation(BaseModel):
    kind: Literal["remove_edge"] = "remove_edge"
    edge_id: str = Field(alias="edgeId")

    model_config = {"populate_by_name": True}


class UpdateGoalMutation(BaseModel):
    kind: Literal["update_goal"] = "update_goal"
    goal: str


class AddApprovalRuleMutation(BaseModel):
    kind: Literal["add_approval_rule"] = "add_approval_rule"
    rule: ApprovalRule


class UpdateEdgeMutation(BaseModel):
    kind: Literal["update_edge"] = "update_edge"
    edge_id: str = Field(alias="edgeId")
    patch: dict[str, Any]

    model_config = {"populate_by_name": True}


class UpdateMetadataMutation(BaseModel):
    kind: Literal["update_metadata"] = "update_metadata"
    patch: dict[str, Any]


class UpdateInstructionsMutation(BaseModel):
    kind: Literal["update_instructions"] = "update_instructions"
    instructions: str


class AddToolMutation(BaseModel):
    kind: Literal["add_tool"] = "add_tool"
    tool: ToolBinding


class RemoveToolMutation(BaseModel):
    kind: Literal["remove_tool"] = "remove_tool"
    tool_id: str = Field(alias="toolId")

    model_config = {"populate_by_name": True}


class UpdateToolMutation(BaseModel):
    kind: Literal["update_tool"] = "update_tool"
    tool_id: str = Field(alias="toolId")
    patch: dict[str, Any]

    model_config = {"populate_by_name": True}


class UpdatePermissionsMutation(BaseModel):
    kind: Literal["update_permissions"] = "update_permissions"
    patch: dict[str, Any]


class UpdateRuntimeConfigMutation(BaseModel):
    kind: Literal["update_runtime_config"] = "update_runtime_config"
    patch: dict[str, Any]


AgentSpecMutation = Union[
    AddNodeMutation,
    UpdateNodeMutation,
    RemoveNodeMutation,
    AddEdgeMutation,
    UpdateEdgeMutation,
    RemoveEdgeMutation,
    UpdateMetadataMutation,
    UpdateGoalMutation,
    UpdateInstructionsMutation,
    AddToolMutation,
    RemoveToolMutation,
    UpdateToolMutation,
    UpdatePermissionsMutation,
    AddApprovalRuleMutation,
    UpdateRuntimeConfigMutation,
]

# Discriminated-union adapter: parses a raw mutation dict (as sent by the
# chat builder or the canvas over the wire) into the correct concrete
# mutation type based on its "kind" field, rejecting anything that doesn't
# match one of the sanctioned mutation shapes above.
MutationAdapter: TypeAdapter[AgentSpecMutation] = TypeAdapter(
    Annotated[AgentSpecMutation, Field(discriminator="kind")]
)


def parse_mutation(raw: dict[str, Any]) -> AgentSpecMutation:
    return MutationAdapter.validate_python(raw)


class ValidationError(BaseModel):
    path: str
    message: str
    severity: Literal["error", "warning"] = "error"


class MutationResult(BaseModel):
    ok: bool
    spec: Optional[AgentSpecification] = None
    errors: list[ValidationError] = Field(default_factory=list)
