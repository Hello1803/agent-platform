"""
The Agent Compiler: AgentSpecification -> compiled LangGraph graph.

    AgentSpecification
          |
    (already validated by Pydantic model_validate — graph integrity,
     node/edge id uniqueness, tool_ref existence)
          |
    compile-time checks specific to compilation (tool capability match)
          |
    StateGraph construction (nodes + conditional/unconditional edges)
          |
    CompiledStateGraph (LangGraph), checkpointed for interrupt/resume

This module intentionally knows nothing about the visual editor or the NL
compiler — it only consumes AgentSpecification. Swapping LangGraph for a
different runtime later means rewriting this file and nothing upstream of
it (see architecture doc section 8, "treat LangGraph as an implementation
detail").
"""

from __future__ import annotations

import time

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.errors import GraphInterrupt
from langgraph.types import StreamWriter

from backend.compiler.executors import NodeExecutor, get_executor
from backend.compiler.safe_eval import UnsafeExpressionError, safe_eval
from backend.compiler.state import AgentState, make_trace_event
from backend.credentials.providers import CredentialValidationError
from backend.credentials.store import (
    CredentialAccessError,
    CredentialNotFoundError,
    CredentialStore,
    credential_store as default_credential_store,
)
from backend.models.agent_spec import AgentEdge, AgentNode, AgentSpecification, NodeType, RetryPolicy
from backend.tools.registry import get_tool

# Node types whose executor calls a tool that needs a specific permission
# scope on its bound ToolBinding. Checked at compile time so a misconfigured
# agent (e.g. a send_email node wired to a read-only tool binding) fails
# fast, before it ever runs — rather than failing mid-execution at the
# executor's runtime `_require_tool` check.
REQUIRED_CAPABILITY: dict[NodeType, str] = {
    NodeType.send_email: "send",
    NodeType.database: "write",
    NodeType.read_email: "read",
    NodeType.search_web: "read",
    # read_file / write_file are local scoped disk I/O — no tool binding required
}


class CompileError(ValueError):
    """
    Raised when a spec fails compile-time checks (tool permission
    mismatches, denied tools, no entry point). `node_id`, when the error
    is about a specific node, lets the API surface an actionable
    "Configure {label}" link instead of just a raw string — see
    CompileCheckResponse in backend/api/schemas.py and the corresponding
    UI in App.tsx's compile-error banner.
    """

    def __init__(self, message: str, node_id: str | None = None):
        super().__init__(message)
        self.node_id = node_id


def _check_tool_permissions(spec: AgentSpecification) -> None:
    bindings_by_id = {t.id: t for t in spec.tools}
    for node in spec.nodes:
        required = REQUIRED_CAPABILITY.get(node.type)
        if required is None:
            continue
        if not node.tool_refs:
            raise CompileError(
                f"'{node.label}' needs a tool binding with '{required}' permission "
                f"but has no toolRefs configured.",
                node_id=node.id,
            )
        matching = [
            bindings_by_id[ref] for ref in node.tool_refs
            if ref in bindings_by_id and required in bindings_by_id[ref].permissions
        ]
        if not matching:
            raise CompileError(
                f"'{node.label}' requires capability '{required}' but none of its "
                f"tool bindings grant it.",
                node_id=node.id,
            )
        valid_binding_found = False
        reasons: list[str] = []
        for binding in matching:
            tool = get_tool(binding.tool_id)
            if tool is None:
                reasons.append(f"'{binding.tool_id}' is not registered")
                continue
            if node.type not in tool.node_types:
                reasons.append(f"'{binding.tool_id}' is not compatible with {node.type.value}")
                continue
            if required not in tool.permissions:
                reasons.append(f"'{binding.tool_id}' does not grant {required}")
                continue
            if binding.mode == "connected" and tool.credential_required and not binding.credential_ref:
                reasons.append(f"'{binding.tool_id}' requires a connected credential")
                continue
            if binding.mode == "connected" and tool.execution != "connected":
                reasons.append(f"'{binding.tool_id}' has no connected executor")
                continue
            valid_binding_found = True
        if not valid_binding_found:
            raise CompileError(
                f"'{node.label}' has no executable tool binding: {'; '.join(reasons)}.",
                node_id=node.id,
            )


def _check_deny_overrides(spec: AgentSpecification) -> None:
    """Deny list always wins — refuse to compile a spec where a tool binding's
    permissions conflict with an explicit top-level deny (see PermissionSet
    docs: 'Deny overrides allow')."""
    if not spec.permissions or not spec.permissions.deny:
        return
    denied = set(spec.permissions.deny)
    for tool in spec.tools:
        if tool.tool_id in denied:
            raise CompileError(
                f"Tool binding '{tool.id}' uses '{tool.tool_id}', which is explicitly "
                f"denied in spec.permissions.deny."
            )


def _check_credential_refs(spec: AgentSpecification, store: CredentialStore) -> None:
    """Every 'connected' tool binding with a credential_ref must point at a
    credential that actually exists, is owned by this spec's owner, and
    belongs to the provider the tool requires.

    This replaces the old `not binding.credential_ref` non-empty check in
    `_check_tool_permissions` with a real one, backed by
    `backend/credentials/store.py`. Deliberately a separate compile-time
    check rather than folded into `_check_tool_permissions` — that
    function is about *capability* (does some binding grant the
    permission this node needs), this one is about *credential validity*
    for whichever binding was actually configured, independent of
    whether that binding happens to be the one satisfying a capability
    check elsewhere.
    """
    owner_id = spec.metadata.owner or "default"
    for binding in spec.tools:
        if binding.mode != "connected" or not binding.credential_ref:
            continue
        try:
            store.validate_credential_ref(
                credential_id=binding.credential_ref,
                tool_id=binding.tool_id,
                owner_id=owner_id,
            )
        except CredentialNotFoundError:
            raise CompileError(
                f"Tool binding '{binding.id}' references credential "
                f"'{binding.credential_ref}', which does not exist."
            )
        except CredentialAccessError:
            raise CompileError(
                f"Tool binding '{binding.id}' references credential "
                f"'{binding.credential_ref}', which is not owned by this agent's owner."
            )
        except CredentialValidationError as exc:
            raise CompileError(
                f"Tool binding '{binding.id}' has an invalid credential reference: {exc}"
            )


def _check_ai_provider_availability(spec: AgentSpecification) -> None:
    """Every AI-category node whose config EXPLICITLY names a `provider`
    (set by the Ask AI inspector's provider/model dropdowns — see
    packages/canvas/src/NodeInspector.tsx — or a hand-authored spec) must
    have that provider's credentials configured on THIS server, checked
    here at compile time so a missing key fails fast and clearly instead
    of being discovered deep inside an executor mid-run.

    This is the concrete fix for "do not silently fall back from a
    selected connected provider to another provider or to a stub": before
    this check existed, a node explicitly asking for 'openai' with no
    OPENAI_API_KEY configured would compile fine and then, at run time,
    real_executors/ai.py would (before this same changeset) have had
    nothing built in to stop it from just running with the SERVER's
    unrelated default provider — the exact silent substitution this
    guards against.

    Deliberately scoped to per-NODE `config.provider` only — NOT
    `spec.runtimeConfig.model` (the spec-WIDE preference). That field
    predates this feature, is set by many existing specs simply to
    document a default preference (often naming whatever the server's
    own default already is), and keeps its original, more lenient
    fallback-to-default behavior (see get_executor_overrides /
    _apply_spec_provider_override in backend/api/app.py) — retroactively
    making it strict broke every spec that merely names its already-
    active default provider without a key configured in this environment.
    The strict "must have credentials or a clear error" behavior is for
    the NEW, deliberate, per-node override this session added, where a
    person explicitly picked a specific provider for one specific node.
    """
    from backend.ai_providers.registry import SUPPORTED_PROVIDERS, api_key_env_var, has_credentials

    def _check(provider_raw: object, node_id: str | None, node_label: str) -> None:
        if not provider_raw:
            return
        provider = str(provider_raw).strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            return  # unknown/planned provider names are already rejected by the Pydantic model validators
        if not has_credentials(provider):
            env_var = api_key_env_var(provider)
            raise CompileError(
                f"'{node_label}' explicitly selects AI provider '{provider}', but {env_var} is not "
                f"configured on this server. Set {env_var}, or remove the explicit provider "
                f"selection to use the server's default provider instead.",
                node_id=node_id,
            )

    for node in spec.nodes:
        provider = node.config.get("provider") if isinstance(node.config, dict) else None
        _check(provider, node.id, node.label)


def _check_run_readiness(spec: AgentSpecification) -> None:
    """Checks that are intentionally allowed during Build but block Test/Run."""
    if len(spec.nodes) > 1:
        adjacent = {node.id: set() for node in spec.nodes}
        for edge in spec.edges:
            adjacent[edge.source].add(edge.target)
            adjacent[edge.target].add(edge.source)
        seen: set[str] = set()
        components: list[set[str]] = []
        for node_id in adjacent:
            if node_id in seen:
                continue
            stack, component = [node_id], set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current); seen.add(current)
                stack.extend(adjacent[current] - component)
            components.append(component)
        if len(components) > 1:
            isolated = next(component for component in components if len(component) == min(map(len, components)))
            node = next(node for node in spec.nodes if node.id in isolated)
            raise CompileError(f"'{node.label}' is not connected to the workflow. Connect it before running.", node_id=node.id)

    if spec.policies and spec.policies.tool_allowlist:
        allowed = set(spec.policies.tool_allowlist)
        bindings_by_id = {t.id: t for t in spec.tools}
        # Only check bindings a node in THIS graph actually uses. spec.tools
        # can (legitimately) carry bindings nothing currently references —
        # e.g. a search_web node was deleted from the canvas but its tool
        # binding was left behind, which is normal (bindings are meant to
        # be reusable across nodes, so node deletion doesn't prune them).
        # Checking every entry in spec.tools regardless of use meant an
        # unrelated, unused binding (say, a leftover 'web.search' one)
        # could block a run that never touches it at all — reported as
        # e.g. a plain trigger -> read_file graph failing to compile over
        # a tool it never calls. The allowlist's job is to gate what a run
        # can actually DO, not what happens to be sitting in spec.tools.
        for node in spec.nodes:
            for ref in node.tool_refs or []:
                binding = bindings_by_id.get(ref)
                if binding and binding.tool_id not in allowed:
                    raise CompileError(
                        f"Tool '{binding.tool_id}' is not allowed by this agent's policy.",
                        node_id=node.id,
                    )


def _requires_approval(node: AgentNode, spec: AgentSpecification) -> bool:
    if node.type != NodeType.send_email:
        return False
    return any(
        rule.autonomy_level.value == "ask_permission"
        and (rule.applies_to.node_id == node.id or rule.applies_to.action_category and rule.applies_to.action_category.value == "send_message")
        for rule in spec.human_approval
    )


def _make_node_fn(
    node: AgentNode,
    spec: AgentSpecification,
    executor_overrides: dict[NodeType, "NodeExecutor"] | None = None,
    retry_policy: RetryPolicy | None = None,
):
    executor = (executor_overrides or {}).get(node.type) or get_executor(node.type)

    def node_fn(state: AgentState, writer: StreamWriter) -> dict:
        # Fires the instant this node begins — BEFORE the (possibly slow,
        # possibly model-calling) executor runs. This is what closes the
        # gap documented in useAgentRunStream.ts: without this, a client
        # streaming the run has no signal a node is "in progress," only
        # that it eventually finished. LangGraph's custom stream mode
        # delivers this genuinely live (verified: a writer() call inside a
        # node is yielded to .stream(stream_mode="custom") immediately,
        # not buffered until the node's return value is ready — see the
        # timing proof referenced in the README's Execution Trace UI section).
        writer({"type": "node_started", "node_id": node.id, "node_label": node.label})
        try:
            if _requires_approval(node, spec) and state["variables"].get("approved") is not True:
                raise PermissionError(f"'{node.label}' requires human approval before it can run.")
            updates = _execute_with_retry(node, spec, state["variables"], executor, retry_policy, writer)
        except GraphInterrupt:
            # Not a real failure — LangGraph's own control-flow signal for
            # human-in-the-loop pause/resume (see executors.py human nodes).
            # Must propagate unmodified or the runtime can't pause/resume.
            raise
        except Exception as e:
            writer({"type": "node_failed", "node_id": node.id, "node_label": node.label, "error": str(e)})
            raise RuntimeError(f"Node '{node.id}' ({node.label}) failed: {e}") from e
        event = make_trace_event(node.id, node.label, "success", f"executed {node.type.value}")
        return {"variables": updates, "trace": [event]}

    return node_fn


def _execute_with_retry(node, spec, variables, executor, retry_policy, writer):
    """Real retry-with-backoff, applied when this node is downstream of a
    `retry`-category node (see compile_agent's per-node retry_policy
    lookup below, and NodeType.retry in backend/models/agent_spec.py).
    The `retry` node itself stays a plain passthrough — same architecture
    as `if_else`/`switch`/`loop`, where a control-flow node's real
    semantics live in how it affects the node(s) it's wired to, not in
    its own (no-op) executor — this is what makes it a real retry policy
    application rather than the no-op exec_passthrough it used to be for
    every downstream node regardless of a retry node upstream.

    No retry_policy (the common case — most nodes have no retry node
    feeding into them) means exactly one attempt, no behavior change from
    before this existed.
    """
    if retry_policy is None:
        return executor(node, spec, variables)

    attempts = retry_policy.max_retries + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return executor(node, spec, variables)
        except GraphInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 — deliberately broad: any executor failure is retryable here
            last_error = e
            if attempt >= attempts:
                break
            writer(
                {
                    "type": "node_retrying",
                    "node_id": node.id,
                    "node_label": node.label,
                    "attempt": attempt,
                    "max_attempts": attempts,
                    "error": str(e),
                    "backoff_seconds": retry_policy.backoff_seconds,
                }
            )
            if retry_policy.backoff_seconds > 0:
                time.sleep(retry_policy.backoff_seconds)
    raise last_error


def _make_router(conditional_edges: list[AgentEdge], default_edge: AgentEdge | None):
    def router(state: AgentState) -> str:
        ctx = state["variables"]
        for edge in conditional_edges:
            expr = edge.condition.expression if edge.condition else None
            if not expr:
                continue
            try:
                if safe_eval(expr, ctx):
                    return edge.target
            except UnsafeExpressionError as ex:
                raise RuntimeError(f"Edge '{edge.id}' condition failed to evaluate: {ex}") from ex
        if default_edge is not None:
            return default_edge.target
        return END

    return router


def compile_agent(
    spec: AgentSpecification,
    checkpointer=None,
    executor_overrides: dict[NodeType, NodeExecutor] | None = None,
    credential_store: CredentialStore | None = None,
) -> CompiledStateGraph:
    """
    Compile a validated AgentSpecification into a runnable LangGraph graph.
    `spec` is assumed to already be a valid AgentSpecification instance
    (i.e. it passed Pydantic validation on construction) — this function
    adds the compile-specific checks the Pydantic model can't express.

    `executor_overrides` lets a caller swap in real implementations for
    specific node types (e.g. a Claude-backed `classify` executor, a real
    database-backed `database` executor) without touching this file or the
    default stub registry in executors.py — see backend/real_executors/.

    `credential_store` defaults to the process-wide singleton
    (`backend.credentials.store.credential_store`); overridable for tests
    that want an isolated store rather than sharing global state.
    """
    _check_run_readiness(spec)
    _check_tool_permissions(spec)
    _check_deny_overrides(spec)
    _check_credential_refs(spec, credential_store or default_credential_store)
    _check_ai_provider_availability(spec)

    graph = StateGraph(AgentState)

    # A node downstream of a `retry`-category node gets that retry node's
    # policy applied to ITS OWN executor calls — see _execute_with_retry's
    # docstring for why the retry node itself stays a passthrough (same
    # pattern as if_else/switch/loop: the control-flow node's semantics
    # live in how it affects what it's wired to). "Downstream" here means
    # directly downstream — a retry node's own outgoing edge target(s).
    nodes_by_id = {n.id: n for n in spec.nodes}
    retry_policy_for: dict[str, RetryPolicy] = {}
    for edge in spec.edges:
        source_node = nodes_by_id.get(edge.source)
        if source_node is not None and source_node.type == NodeType.retry:
            config = source_node.config or {}
            retry_policy_for[edge.target] = RetryPolicy(
                maxRetries=config.get("maxRetries", config.get("max_retries", 2)),
                backoffSeconds=config.get("backoffSeconds", config.get("backoff_seconds", 5)),
            )

    for node in spec.nodes:
        graph.add_node(node.id, _make_node_fn(node, spec, executor_overrides, retry_policy_for.get(node.id)))

    incoming_count = {n.id: 0 for n in spec.nodes}
    outgoing: dict[str, list[AgentEdge]] = {n.id: [] for n in spec.nodes}
    for edge in spec.edges:
        outgoing[edge.source].append(edge)
        incoming_count[edge.target] += 1

    entry_node_ids = [n.id for n in spec.nodes if incoming_count[n.id] == 0]
    if not entry_node_ids:
        raise CompileError("Graph has no entry point — every node has an incoming edge (a cycle with no start).")
    for node_id in entry_node_ids:
        graph.add_edge(START, node_id)

    for node in spec.nodes:
        edges = outgoing[node.id]
        if not edges:
            graph.add_edge(node.id, END)
            continue

        conditional = [e for e in edges if e.condition and e.condition.expression]
        unconditional = [e for e in edges if not (e.condition and e.condition.expression)]

        if not conditional:
            for edge in edges:
                graph.add_edge(node.id, edge.target)
            continue

        conditional_sorted = sorted(conditional, key=lambda e: e.priority if e.priority is not None else 999)
        default_edge = unconditional[0] if unconditional else None

        path_map = {e.target: e.target for e in conditional_sorted}
        if default_edge is not None:
            path_map[default_edge.target] = default_edge.target
        else:
            path_map[END] = END

        graph.add_conditional_edges(node.id, _make_router(conditional_sorted, default_edge), path_map)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
