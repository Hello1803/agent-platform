/**
 * Pure conversions between AgentSpecification (the source of truth) and
 * React Flow's node/edge shapes (what the canvas actually renders).
 *
 * Kept deliberately free of React and of any mutation-application logic —
 * these functions only ever produce AgentSpecMutation *objects*; applying
 * them (validating, versioning, persisting) is the caller's job via
 * backend/mutation/engine.py over the wire. This file being pure and
 * side-effect-free is what makes it unit-testable without a browser (see
 * __tests__/specToFlow.test.ts).
 */

import type { Edge as FlowEdge, Node as FlowNode, NodeChange, EdgeChange, Connection, XYPosition } from "@xyflow/react";
import type { AgentEdge, AgentNode, AgentSpecMutation } from "@agent-platform/shared-types";
import { NODE_CATALOG, REQUIRED_CAPABILITY } from "./nodeCatalog";
import { TOOL_DEFINITIONS } from "./toolRegistry";

export const AGENT_NODE_TYPE = "agentNode";
export const CONDITION_EDGE_TYPE = "conditionEdge";

/** Vertical fallback spacing for nodes that have never been placed on the canvas. */
const FALLBACK_ROW_HEIGHT = 140;

export type TraceStatus = "success" | "failed" | "waiting" | "running";

export interface TraceEventLike {
  node_id: string;
  node_label: string;
  status: "success" | "failed" | "waiting";
  detail: string;
  timestamp: string;
}

export interface AgentFlowNodeData extends Record<string, unknown> {
  agentNode: AgentNode;
  /** Set by applyTraceToNodes — undefined means "not part of the current/last run's executed path." */
  traceStatus?: TraceStatus;
  /** True on EVERY node once applyTraceToNodes has been called with a non-empty trace/activeNodeId — lets a node distinguish "no run has happened yet" (full opacity, normal editing view) from "a run happened and I specifically wasn't on the executed path" (dimmed). */
  traceActive?: boolean;
  /** Set by AgentCanvas from computeToolBindingWarnings — a client-side, no-round-trip signal that this tool node will fail to compile, shown as a warning badge before the user ever hits Run. */
  toolBindingWarning?: string;
}

export interface AgentFlowEdgeData extends Record<string, unknown> {
  agentEdge: AgentEdge;
  /** Set by AgentCanvas from its own `readOnly` prop so ConditionEdge can
   * hide the disconnect button instead of offering an action that would
   * only ever silently no-op (AgentCanvas's onEdgesChange already drops
   * mutations when readOnly, but the local flow-state change still fires
   * first — see AgentCanvas.tsx for that pre-existing distinction). */
  readOnly?: boolean;
}

export type AgentFlowNode = FlowNode<AgentFlowNodeData>;
export type AgentFlowEdge = FlowEdge<AgentFlowEdgeData>;

export function toFlowNodes(nodes: AgentNode[]): AgentFlowNode[] {
  return nodes.map((node, index) => ({
    id: node.id,
    type: AGENT_NODE_TYPE,
    // A node with no canvasPosition (e.g. one the NL compiler just added)
    // falls back to a simple vertical stack rather than piling every new
    // node at (0,0). Real auto-layout (dagre/elk) is a natural upgrade
    // here — see architecture doc section 23, "Visual Clutter" — but a
    // stable, deterministic fallback is enough for a first pass.
    position: node.canvasPosition ?? { x: 0, y: index * FALLBACK_ROW_HEIGHT },
    data: { agentNode: node },
  }));
}

/** Every AgentNode has exactly one output handle and one input handle
 * (AgentNodeCard.tsx), explicitly id'd so connection validation has
 * something stable to check against (see isValidAgentConnection below). */
export const OUTPUT_HANDLE_ID = "output";
export const INPUT_HANDLE_ID = "input";

export function toFlowEdges(edges: AgentEdge[]): AgentFlowEdge[] {
  return edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    // Specs saved before handles had explicit ids carry sourceHandle/
    // targetHandle: undefined — default those to the (only) real handle
    // id so old data keeps rendering/reconnecting correctly against the
    // now-explicitly-id'd Handle components.
    sourceHandle: edge.sourceHandle ?? OUTPUT_HANDLE_ID,
    targetHandle: edge.targetHandle ?? INPUT_HANDLE_ID,
    type: CONDITION_EDGE_TYPE,
    label: edge.label,
    data: { agentEdge: edge },
  }));
}

/**
 * Turns React Flow's NodeChange events into spec mutations. Only two kinds
 * of change translate into a mutation the spec cares about:
 *   - a completed drag (position change with dragging: false) -> update_node
 *   - a deletion -> remove_node
 * Everything else (selection, dimension measurement) is canvas-local UI
 * state that never touches the spec.
 */
export function nodeChangesToMutations(changes: NodeChange<AgentFlowNode>[]): AgentSpecMutation[] {
  const mutations: AgentSpecMutation[] = [];
  for (const change of changes) {
    if (change.type === "position" && change.position && change.dragging === false) {
      mutations.push({
        kind: "update_node",
        nodeId: change.id,
        patch: { canvasPosition: { x: change.position.x, y: change.position.y } },
      });
    } else if (change.type === "remove") {
      mutations.push({ kind: "remove_node", nodeId: change.id });
    }
  }
  return mutations;
}

export function edgeChangesToMutations(changes: EdgeChange<AgentFlowEdge>[]): AgentSpecMutation[] {
  const mutations: AgentSpecMutation[] = [];
  for (const change of changes) {
    if (change.type === "remove") {
      mutations.push({ kind: "remove_edge", edgeId: change.id });
    }
  }
  return mutations;
}

/** Generates a short, unique id with the given prefix, avoiding collisions with existingIds. */
/**
 * TS mirror of backend/ai_providers/registry.py's parse_model_string --
 * parses RuntimeConfig.model's "<provider>:<model>" convention. Kept in
 * sync by hand (same cross-language boundary as the rest of this file's
 * mirrored backend logic, e.g. isValidAgentConnection's cycle check).
 * Used by AiProviderSelector.tsx so the inspector's provider dropdown
 * reads/writes the exact same field the backend actually resolves a
 * provider from.
 */
export function parseModelString(value: string | null | undefined): [provider: string | null, model: string | null] {
  if (!value) return [null, null];
  const colonIndex = value.indexOf(":");
  if (colonIndex === -1) return [null, value.trim() || null];
  const provider = value.slice(0, colonIndex).trim().toLowerCase();
  const model = value.slice(colonIndex + 1).trim();
  return [provider || null, model || null];
}

/**
 * TS mirror of backend/ai_providers/registry.py's validate_provider_model
 * -- rejects a model that unambiguously belongs to a DIFFERENT provider
 * (e.g. an Anthropic "claude-*" model paired with "openai"). Used by
 * NodeInspector.tsx's Ask AI provider/model dropdowns so a mismatched
 * pair can never even be selected client-side, not just rejected after
 * the fact by the server.
 */
const PROVIDER_MODEL_PREFIXES: Record<string, string[]> = {
  anthropic: ["claude-"],
  openai: ["gpt-", "o1-", "o1", "o3-", "o3", "o4-", "chatgpt-"],
  lmstudio: [],
};

export function isModelCompatibleWithProvider(providerId: string, modelId: string): boolean {
  if (providerId === "lmstudio") return true;
  const otherPrefixes = Object.entries(PROVIDER_MODEL_PREFIXES).filter(
    ([id]) => id !== providerId && id !== "lmstudio"
  );
  const ownPrefixes = PROVIDER_MODEL_PREFIXES[providerId] ?? [];
  for (const [, prefixes] of otherPrefixes) {
    if (prefixes.some((p) => modelId.startsWith(p)) && !ownPrefixes.some((p) => modelId.startsWith(p))) {
      return false;
    }
  }
  return true;
}

export function generateId(prefix: "n" | "e" | "tool", existingIds: Iterable<string>): string {
  const taken = new Set(existingIds);
  for (let i = 1; i <= taken.size + 1; i++) {
    const candidate = `${prefix}_${i}`;
    if (!taken.has(candidate)) return candidate;
  }
  // Astronomically unlikely fallback if the sequential scan above somehow
  // collides on every candidate (e.g. ids not following the n_N/e_N convention).
  return `${prefix}_${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * Whether a proposed connection (from React Flow's onConnect / the
 * isValidConnection callback) is a semantically valid edge — the single
 * source of truth both AgentCanvas's `isValidConnection` prop (blocks the
 * drag gesture itself, so an invalid connection is never even added to
 * React Flow's local edge state) and `connectionToAddEdgeMutation`
 * (defense-in-depth before a mutation is even built) call into. Mirrors
 * backend/models/agent_spec.py's `_check_graph_integrity` checks exactly —
 * if the two ever disagree, the backend wins (it's what actually gates
 * what a spec can contain), but keeping the rules identical means the
 * canvas never shows a connection succeeding only to have it rejected a
 * moment later by the mutation round-trip.
 *
 *   1. Both ends must resolve to real, distinct nodes (no self-connection).
 *   2. The connection must run output-handle -> input-handle — rejects
 *      output->output and input->input, which is possible even with React
 *      Flow's default "strict" connectionMode if a caller (or a future
 *      node type with more than one handle) doesn't set handle ids that
 *      encode direction the way AgentNodeCard's do.
 *   3. No duplicate of an edge that already exists between this exact
 *      (source, target) pair.
 *   4. Node/category compatibility: a trigger node is a graph's entry
 *      point and can never be a connection's target.
 *   5. Uncontrolled cycles: a new connection is always unconditional (see
 *      the check's own comment below for why), so it must not close a
 *      cycle made entirely of unconditional edges — that cycle could
 *      never exit. Deliberately NOT a ban on cycles in general; a `loop`
 *      node's mechanism is a cycle with a conditional exit edge.
 */
export function isValidAgentConnection(
  connection: Connection,
  nodes: Pick<AgentFlowNode, "id" | "data">[],
  edges: Pick<AgentEdge, "source" | "target" | "condition">[]
): boolean {
  const { source, target, sourceHandle, targetHandle } = connection;
  if (!source || !target) return false;

  // 1. No self-connections.
  if (source === target) return false;

  // 2. Handle direction: an edge only ever makes sense as output -> input.
  // Handles default to their sole real id when a caller (e.g. a
  // programmatic connection with no explicit handle) leaves them unset,
  // matching toFlowEdges' fallback above.
  const effectiveSourceHandle = sourceHandle ?? OUTPUT_HANDLE_ID;
  const effectiveTargetHandle = targetHandle ?? INPUT_HANDLE_ID;
  if (effectiveSourceHandle !== OUTPUT_HANDLE_ID) return false;
  if (effectiveTargetHandle !== INPUT_HANDLE_ID) return false;

  // 3. No duplicate edges between the same pair.
  if (edges.some((e) => e.source === source && e.target === target)) return false;

  // 4. A trigger node starts a run — it can't also be fed by one.
  const targetNode = nodes.find((n) => n.id === target);
  if (targetNode?.data.agentNode.category === "trigger") return false;

  // 5. Uncontrolled cycles: dragging a connection on the canvas always
  // creates a plain, unconditional edge (there's no way to attach a
  // condition from a drag gesture — that only happens via an update_edge
  // mutation, e.g. from the NL compiler). So if `target` can already
  // reach `source` using only OTHER unconditional edges, adding this one
  // would close a cycle with no conditional edge anywhere in it — nothing
  // on that path would ever route elsewhere, so a run entering it loops
  // forever. This mirrors backend/models/agent_spec.py's
  // _check_graph_integrity exactly (search "Uncontrolled cycles" there
  // for the full rationale, including why this does NOT ban cycles in
  // general — a `loop` node's whole mechanism IS a cycle, just one with a
  // conditional exit, which can still be built via update_edge even
  // though it can't be dragged into existence).
  const unconditionalAdjacency = new Map<string, string[]>();
  for (const e of edges) {
    if (!(e.condition && e.condition.expression)) {
      const list = unconditionalAdjacency.get(e.source) ?? [];
      list.push(e.target);
      unconditionalAdjacency.set(e.source, list);
    }
  }
  const visited = new Set<string>([target]);
  const queue = [target];
  while (queue.length > 0) {
    const current = queue.shift()!;
    if (current === source) return false; // target can already reach source unconditionally
    for (const next of unconditionalAdjacency.get(current) ?? []) {
      if (!visited.has(next)) {
        visited.add(next);
        queue.push(next);
      }
    }
  }

  return true;
}

/**
 * Turns a React Flow onConnect event into an add_edge mutation. Does NOT
 * mutate the spec itself — the caller applies this via the mutation engine
 * and re-renders from the resulting authoritative spec.
 *
 * Re-validates with isValidAgentConnection as defense-in-depth: normally
 * AgentCanvas's `isValidConnection` prop already stops React Flow from
 * ever firing onConnect for an invalid pair, but this function has no way
 * to know that's true for every caller, so it refuses to produce a
 * mutation for an invalid connection regardless of how it was invoked.
 */
export function connectionToAddEdgeMutation(
  connection: Connection,
  existingEdgeIds: Iterable<string>,
  nodes: Pick<AgentFlowNode, "id" | "data">[] = [],
  edges: Pick<AgentEdge, "source" | "target" | "condition">[] = []
): AgentSpecMutation | null {
  if (!connection.source || !connection.target) return null;
  if (!isValidAgentConnection(connection, nodes, edges)) return null;
  return {
    kind: "add_edge",
    edge: {
      id: generateId("e", existingEdgeIds),
      source: connection.source,
      target: connection.target,
      sourceHandle: connection.sourceHandle ?? OUTPUT_HANDLE_ID,
      targetHandle: connection.targetHandle ?? INPUT_HANDLE_ID,
    },
  };
}

/**
 * Turns a NodePalette drag-drop into a mutation BATCH — always an array,
 * even for the common case of just one add_node — using the catalog
 * entry's defaults (labels for classify, fields for extract, etc.) so a
 * freshly-dropped node isn't immediately invalid. Pure and separated from
 * AgentCanvas's onDrop handler specifically so this mapping is
 * unit-testable without a browser or a ReactFlow context —
 * screenToFlowPosition (the coordinate math) stays in the component,
 * everything else is here.
 *
 * For a tool-category node that needs a capability (search_web, send_email,
 * database, etc. — see REQUIRED_CAPABILITY), this is where a REAL bug got
 * fixed: dropping such a node with no tool binding used to produce a spec
 * that validates fine (Pydantic doesn't check this) but fails at COMPILE
 * time with an opaque error, only discovered when the user tries to run
 * it. Now a placeholder ToolBinding with the exact permission the node
 * type needs is added ATOMICALLY in the same batch as the node — the
 * mutation engine either applies both or neither, so there's never a
 * moment where a tool node exists without something satisfying its
 * compile-time permission check. The placeholder's toolId is an obvious
 * "not yet configured" stand-in the user is expected to edit via
 * NodeInspector once they've actually connected a real service — this is
 * "automatic compatible defaults where safe," not a real integration.
 */
export function catalogEntryToAddNodeMutation(
  nodeType: string,
  position: XYPosition,
  existingNodeIds: Iterable<string>,
  existingToolIds: Iterable<string> = []
): AgentSpecMutation[] {
  const entry = NODE_CATALOG.find((e) => e.type === nodeType);
  if (!entry) return [];

  const nodeId = generateId("n", existingNodeIds);
  const mutations: AgentSpecMutation[] = [];
  let toolRefs: string[] | undefined;

  const requiredCapability = REQUIRED_CAPABILITY[entry.type];
  if (requiredCapability) {
    const toolId = generateId("tool", existingToolIds);
    const localConnected = entry.type === "read_file" || entry.type === "write_file" || entry.type === "database";
    mutations.push({
      kind: "add_tool",
      tool: {
        id: toolId,
        toolId: localConnected ? `local.${entry.type}` : `unconfigured.${entry.type}`,
        displayName: localConnected ? entry.label : `${entry.label} (not yet configured)`,
        permissions: [requiredCapability],
        mode: localConnected ? "connected" : "demo",
      },
    });
    toolRefs = [toolId];
  }

  mutations.push({
    kind: "add_node",
    node: {
      id: nodeId,
      type: entry.type,
      category: entry.category,
      label: entry.label,
      executionType: entry.defaultExecutionType ?? "deterministic",
      config: entry.defaultConfig ?? {},
      canvasPosition: { x: position.x, y: position.y },
      ...(toolRefs ? { toolRefs } : {}),
    },
  });

  return mutations;
}

/**
 * Client-side, no-round-trip check for the exact bug this session fixed:
 * a tool-category node needing a capability (search_web, send_email,
 * database, etc.) with no tool binding actually granting it. Mirrors
 * backend/compiler/compiler.py's REQUIRED_CAPABILITY + _check_tool_permissions
 * logic exactly, computed client-side so the warning shows up on the node
 * card IMMEDIATELY — the moment a node is dropped or a binding removed —
 * rather than only after an explicit compile-check round-trip. The
 * backend's compile_agent() remains the authoritative check (this is a UX
 * convenience, not a replacement); if the two ever disagree, the backend
 * wins, since it's what actually gates execution.
 *
 * Also flags the OTHER way a node's tool binding can fail to compile:
 * `spec.policies.toolAllowlist` rejecting the bound tool outright (see
 * compiler.py's `_check_run_readiness`). That check used to be
 * discoverable only by hitting Run and getting a 422 — there was, and
 * still is, no UI to view or edit the allowlist itself, so a user adding
 * a new tool node to an agent whose policy predates it had no way to see
 * why the run failed until it failed. Surfacing it here at least turns
 * that into an immediate Build-time warning instead of a surprise.
 */
export function computeToolBindingWarnings(
  nodes: AgentNode[],
  tools: { id: string; toolId: string; permissions: string[] }[],
  toolAllowlist?: string[]
): Map<string, string> {
  const warnings = new Map<string, string>();
  const bindingsById = new Map(tools.map((t) => [t.id, t]));
  const allowlist = toolAllowlist && toolAllowlist.length > 0 ? new Set(toolAllowlist) : null;

  for (const node of nodes) {
    const required = REQUIRED_CAPABILITY[node.type];
    if (!required) continue;

    if (!node.toolRefs || node.toolRefs.length === 0) {
      warnings.set(node.id, `Missing tool binding (needs '${required}' permission)`);
      continue;
    }
    const hasMatch = node.toolRefs.some((ref) => {
      const binding = bindingsById.get(ref);
      return !!binding && binding.permissions.includes(required) && TOOL_DEFINITIONS.some((tool) => tool.id === binding.toolId && tool.permissions.includes(required));
    });
    if (!hasMatch) {
      warnings.set(node.id, `Tool binding is missing or not configured for '${required}'`);
      continue;
    }
    if (allowlist) {
      const disallowed = node.toolRefs
        .map((ref) => bindingsById.get(ref))
        .find((binding) => binding && !allowlist.has(binding.toolId));
      if (disallowed) {
        warnings.set(node.id, `Tool '${disallowed.toolId}' is not on this agent's policy allowlist — it will fail to run`);
      }
    }
  }

  return warnings;
}

/**
 * Pure function computing each node's execution-trace status for
 * rendering, given the trace log so far and (while a run is actively
 * streaming) which node is currently executing. This is what
 * `AgentCanvas`'s `trace` prop feeds into — see useAgentRunStream.ts for
 * where `activeNodeId` comes from during a live SSE stream.
 *
 * A node's status is its LATEST trace entry (a node can appear more than
 * once in a trace across loops/retries) — later entries override earlier
 * ones, and `activeNodeId` (mid-stream, not yet in the trace at all)
 * always wins as "running."
 */
export function applyTraceToNodes(
  nodes: AgentFlowNode[],
  trace: TraceEventLike[],
  activeNodeId?: string | null
): AgentFlowNode[] {
  const statusByNode = new Map<string, TraceStatus>();
  for (const event of trace) {
    statusByNode.set(event.node_id, event.status);
  }
  if (activeNodeId) {
    statusByNode.set(activeNodeId, "running");
  }
  const traceActive = trace.length > 0 || !!activeNodeId;
  return nodes.map((node) => {
    const traceStatus = statusByNode.get(node.id);
    if (traceStatus === node.data.traceStatus && traceActive === node.data.traceActive) return node; // stable reference when nothing changed
    return { ...node, data: { ...node.data, traceStatus, traceActive } };
  });
}
