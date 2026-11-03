/**
 * Plain assertion-based tests for specToFlow.ts — no test framework, no
 * browser/DOM needed, since these functions are pure. Run with:
 *   npx tsx src/__tests__/specToFlow.test.ts
 */

import {
  toFlowNodes,
  toFlowEdges,
  nodeChangesToMutations,
  edgeChangesToMutations,
  connectionToAddEdgeMutation,
  isValidAgentConnection,
  generateId,
  parseModelString,
  isModelCompatibleWithProvider,
  applyTraceToNodes,
  catalogEntryToAddNodeMutation,
  computeToolBindingWarnings,
} from "../specToFlow";
import type { AgentNode, AgentEdge } from "@agent-platform/shared-types";
import type { NodeChange, EdgeChange, Connection } from "@xyflow/react";

let passed = 0;
let failed = 0;

function assert(condition: boolean, message: string) {
  if (condition) {
    passed++;
  } else {
    failed++;
    console.error(`FAIL: ${message}`);
  }
}

function assertEqual<T>(actual: T, expected: T, message: string) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  assert(ok, `${message} — expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
}

// ---- toFlowNodes ----------------------------------------------------

{
  const nodes: AgentNode[] = [
    { id: "n1", type: "trigger_manual", category: "trigger", label: "Start", canvasPosition: { x: 10, y: 20 } },
    { id: "n2", type: "classify", category: "ai", label: "Classify" }, // no canvasPosition
  ];
  const flowNodes = toFlowNodes(nodes);
  assertEqual(flowNodes[0].position, { x: 10, y: 20 }, "toFlowNodes: preserves existing canvasPosition");
  assertEqual(flowNodes[1].position, { x: 0, y: 140 }, "toFlowNodes: falls back to deterministic vertical stack");
  assert(flowNodes[0].data.agentNode.id === "n1", "toFlowNodes: carries the original AgentNode in data");
}

// ---- toFlowEdges ----------------------------------------------------

{
  const edges: AgentEdge[] = [
    { id: "e1", source: "n1", target: "n2", condition: { expression: "amount > 10000", kind: "deterministic" } },
  ];
  const flowEdges = toFlowEdges(edges);
  assertEqual(flowEdges[0].id, "e1", "toFlowEdges: preserves id");
  assert(flowEdges[0].data?.agentEdge.condition?.expression === "amount > 10000", "toFlowEdges: carries condition data");
  assertEqual(
    flowEdges[0].sourceHandle,
    "output",
    "toFlowEdges: defaults a persisted edge's missing sourceHandle to 'output' (back-compat with pre-handle-id specs)"
  );
  assertEqual(
    flowEdges[0].targetHandle,
    "input",
    "toFlowEdges: defaults a persisted edge's missing targetHandle to 'input' (back-compat with pre-handle-id specs)"
  );
}

// ---- nodeChangesToMutations ------------------------------------------

{
  const changes: NodeChange<any>[] = [
    { id: "n1", type: "position", position: { x: 50, y: 60 }, dragging: false },
    { id: "n2", type: "position", position: { x: 5, y: 5 }, dragging: true }, // mid-drag — should NOT emit
    { id: "n3", type: "select", selected: true }, // selection — should NOT emit
    { id: "n4", type: "remove" },
  ];
  const mutations = nodeChangesToMutations(changes);
  assertEqual(mutations.length, 2, "nodeChangesToMutations: only completed drags and removals produce mutations");
  assertEqual(
    mutations[0],
    { kind: "update_node", nodeId: "n1", patch: { canvasPosition: { x: 50, y: 60 } } },
    "nodeChangesToMutations: completed drag -> update_node with canvasPosition patch"
  );
  assertEqual(mutations[1], { kind: "remove_node", nodeId: "n4" }, "nodeChangesToMutations: remove -> remove_node");
}

// ---- edgeChangesToMutations ------------------------------------------

{
  const changes: EdgeChange<any>[] = [
    { id: "e1", type: "remove" },
    { id: "e2", type: "select", selected: true },
  ];
  const mutations = edgeChangesToMutations(changes);
  assertEqual(mutations, [{ kind: "remove_edge", edgeId: "e1" }], "edgeChangesToMutations: only removals produce mutations");
}

// ---- generateId -------------------------------------------------------

{
  const id1 = generateId("n", ["n_1", "n_2"]);
  assertEqual(id1, "n_3", "generateId: finds the next unused sequential id");

  const id2 = generateId("e", []);
  assertEqual(id2, "e_1", "generateId: starts at 1 for an empty existing set");
}

// ---- connectionToAddEdgeMutation ---------------------------------------

{
  const connection: Connection = { source: "n1", target: "n2", sourceHandle: null, targetHandle: null };
  const mutation = connectionToAddEdgeMutation(connection, ["e_1"]);
  assert(mutation !== null, "connectionToAddEdgeMutation: produces a mutation for a valid connection");
  if (mutation && mutation.kind === "add_edge") {
    assertEqual(mutation.edge.id, "e_2", "connectionToAddEdgeMutation: generates a non-colliding edge id");
    assertEqual(mutation.edge.source, "n1", "connectionToAddEdgeMutation: preserves source");
    assertEqual(mutation.edge.target, "n2", "connectionToAddEdgeMutation: preserves target");
  } else {
    failed++;
    console.error("FAIL: mutation was not an add_edge kind");
  }

  const incomplete: Connection = { source: null as any, target: "n2", sourceHandle: null, targetHandle: null };
  assertEqual(
    connectionToAddEdgeMutation(incomplete, []),
    null,
    "connectionToAddEdgeMutation: returns null for an incomplete connection"
  );
}

// ---- isValidAgentConnection ---------------------------------------------
// Regression coverage for the canvas edge-validation fix: only a real
// output(source) -> input(target) connection between two distinct nodes,
// not already connected, with a non-trigger target, is valid. Mirrors
// backend/models/agent_spec.py's _check_graph_integrity exactly.

{
  const nodes: AgentNode[] = [
    { id: "n_trigger", type: "trigger_manual", category: "trigger", label: "Start" },
    { id: "n_a", type: "classify", category: "ai", label: "Classify" },
    { id: "n_b", type: "summarize", category: "ai", label: "Summarize" },
  ];
  const flowNodes = toFlowNodes(nodes);
  const existingEdges: AgentEdge[] = [{ id: "e_existing", source: "n_trigger", target: "n_a" }];

  // 1. A well-formed output -> input connection between two unconnected nodes is valid.
  const valid: Connection = { source: "n_a", target: "n_b", sourceHandle: "output", targetHandle: "input" };
  assert(
    isValidAgentConnection(valid, flowNodes, existingEdges),
    "isValidAgentConnection: accepts a well-formed output -> input connection"
  );

  // 2. output -> output is rejected (both handles claim to be an output).
  const outOut: Connection = { source: "n_a", target: "n_b", sourceHandle: "output", targetHandle: "output" };
  assert(
    !isValidAgentConnection(outOut, flowNodes, existingEdges),
    "isValidAgentConnection: rejects output -> output"
  );

  // 3. input -> input is rejected (both handles claim to be an input).
  const inIn: Connection = { source: "n_a", target: "n_b", sourceHandle: "input", targetHandle: "input" };
  assert(
    !isValidAgentConnection(inIn, flowNodes, existingEdges),
    "isValidAgentConnection: rejects input -> input"
  );

  // 4. Self-connections are rejected.
  const selfConn: Connection = { source: "n_a", target: "n_a", sourceHandle: "output", targetHandle: "input" };
  assert(
    !isValidAgentConnection(selfConn, flowNodes, existingEdges),
    "isValidAgentConnection: rejects a self-connection"
  );

  // 5. A duplicate of an already-existing edge is rejected.
  const duplicate: Connection = { source: "n_trigger", target: "n_a", sourceHandle: "output", targetHandle: "input" };
  assert(
    !isValidAgentConnection(duplicate, flowNodes, existingEdges),
    "isValidAgentConnection: rejects a duplicate of an existing edge"
  );

  // 6. A connection into a trigger node is rejected (triggers are entry points).
  const intoTrigger: Connection = { source: "n_b", target: "n_trigger", sourceHandle: "output", targetHandle: "input" };
  assert(
    !isValidAgentConnection(intoTrigger, flowNodes, existingEdges),
    "isValidAgentConnection: rejects a connection targeting a trigger node"
  );

  // 6b. Uncontrolled cycle: n_a -> n_b already exists unconditionally
  // (added just above conceptually — use a fresh edge list to make this
  // block self-contained); connecting n_b back to n_a would close a cycle
  // with no conditional edge anywhere in it.
  const cycleEdges: AgentEdge[] = [{ id: "e_a_b", source: "n_a", target: "n_b" }];
  const closesCycle: Connection = { source: "n_b", target: "n_a", sourceHandle: "output", targetHandle: "input" };
  assert(
    !isValidAgentConnection(closesCycle, flowNodes, cycleEdges),
    "isValidAgentConnection: rejects a connection that would close an all-unconditional cycle"
  );

  // 6c. Same shape, but the existing edge is conditional (a controlled
  // loop's exit) — connecting back should be allowed.
  const controlledLoopEdges: AgentEdge[] = [
    { id: "e_a_b", source: "n_a", target: "n_b", condition: { expression: "not done", kind: "deterministic" } },
  ];
  assert(
    isValidAgentConnection(closesCycle, flowNodes, controlledLoopEdges),
    "isValidAgentConnection: allows closing a cycle when an existing edge in the path is conditional"
  );

  // 6d. A longer chain: n_trigger -> n_a -> n_b already unconditional;
  // connecting n_b back to n_trigger is separately blocked by the
  // trigger-target rule (4), but connecting n_b back to n_a should be
  // blocked by the cycle rule even though n_a isn't the direct source of
  // any existing edge FROM n_b.
  const chainEdges: AgentEdge[] = [
    { id: "e_t_a", source: "n_trigger", target: "n_a" },
    { id: "e_a_b", source: "n_a", target: "n_b" },
  ];
  const closesLongerCycle: Connection = { source: "n_b", target: "n_a", sourceHandle: "output", targetHandle: "input" };
  assert(
    !isValidAgentConnection(closesLongerCycle, flowNodes, chainEdges),
    "isValidAgentConnection: rejects a cycle closed via a multi-hop unconditional path"
  );

  // 7. Missing source/target (an incomplete drag) is rejected.
  assert(
    !isValidAgentConnection({ source: null as any, target: "n_b", sourceHandle: "output", targetHandle: "input" }, flowNodes, existingEdges),
    "isValidAgentConnection: rejects a connection missing a source"
  );

  // 8. connectionToAddEdgeMutation refuses to build a mutation for any of
  // the invalid shapes above — the same rule enforced at the point a
  // mutation would actually be constructed, not just at the gesture layer.
  assertEqual(
    connectionToAddEdgeMutation(outOut, ["e_existing"], flowNodes, existingEdges),
    null,
    "connectionToAddEdgeMutation: refuses to build a mutation for an output -> output connection"
  );
  assertEqual(
    connectionToAddEdgeMutation(selfConn, ["e_existing"], flowNodes, existingEdges),
    null,
    "connectionToAddEdgeMutation: refuses to build a mutation for a self-connection"
  );
  assertEqual(
    connectionToAddEdgeMutation(duplicate, ["e_existing"], flowNodes, existingEdges),
    null,
    "connectionToAddEdgeMutation: refuses to build a mutation for a duplicate connection"
  );
  assertEqual(
    connectionToAddEdgeMutation(intoTrigger, ["e_existing"], flowNodes, existingEdges),
    null,
    "connectionToAddEdgeMutation: refuses to build a mutation for a connection into a trigger node"
  );

  // 9. A genuinely valid connection still produces a real add_edge mutation
  // with the expected handle ids — confirms the fix doesn't break the
  // happy path.
  const validMutation = connectionToAddEdgeMutation(valid, ["e_existing"], flowNodes, existingEdges);
  assert(validMutation !== null, "connectionToAddEdgeMutation: still produces a mutation for a valid connection");
  if (validMutation && validMutation.kind === "add_edge") {
    assertEqual(validMutation.edge.sourceHandle, "output", "connectionToAddEdgeMutation: preserves the output handle id");
    assertEqual(validMutation.edge.targetHandle, "input", "connectionToAddEdgeMutation: preserves the input handle id");
  }
}

// ---- applyTraceToNodes -------------------------------------------------

{
  const nodes: AgentNode[] = [
    { id: "n1", type: "trigger_manual", category: "trigger", label: "Start" },
    { id: "n2", type: "classify", category: "ai", label: "Classify" },
    { id: "n3", type: "send_email", category: "tool", label: "Send" },
  ];
  const flowNodes = toFlowNodes(nodes);

  const trace = [
    { node_id: "n1", node_label: "Start", status: "success" as const, detail: "", timestamp: "t1" },
    { node_id: "n2", node_label: "Classify", status: "success" as const, detail: "", timestamp: "t2" },
  ];

  const withTrace = applyTraceToNodes(flowNodes, trace, "n3");
  assertEqual(withTrace.find((n) => n.id === "n1")!.data.traceStatus, "success", "applyTraceToNodes: completed node gets its trace status");
  assertEqual(withTrace.find((n) => n.id === "n2")!.data.traceStatus, "success", "applyTraceToNodes: second completed node too");
  assertEqual(withTrace.find((n) => n.id === "n3")!.data.traceStatus, "running", "applyTraceToNodes: activeNodeId overrides as running");

  const idle = applyTraceToNodes(flowNodes, [], undefined);
  assert(idle.every((n) => n.data.traceStatus === undefined), "applyTraceToNodes: no trace + no active node = all undefined");
  assert(idle.every((n) => n.data.traceActive === false), "applyTraceToNodes: traceActive is false when idle (no dimming in normal editing view)");
  assert(withTrace.every((n) => n.data.traceActive === true), "applyTraceToNodes: traceActive is true for ALL nodes once a run has happened, even ones not on the executed path");

  // Later trace entries for the same node override earlier ones (loops/retries).
  const traceWithRetry = [
    { node_id: "n1", node_label: "Start", status: "failed" as const, detail: "", timestamp: "t1" },
    { node_id: "n1", node_label: "Start", status: "success" as const, detail: "", timestamp: "t2" },
  ];
  const retried = applyTraceToNodes(flowNodes, traceWithRetry, undefined);
  assertEqual(retried.find((n) => n.id === "n1")!.data.traceStatus, "success", "applyTraceToNodes: latest trace entry wins on repeat execution");

  // Stable reference when nothing changed — important so React Flow doesn't re-render everything every poll.
  const first = applyTraceToNodes(flowNodes, trace, "n3");
  const second = applyTraceToNodes(first, trace, "n3");
  assert(first.find((n) => n.id === "n1") === second.find((n) => n.id === "n1"), "applyTraceToNodes: unchanged node keeps the same object reference");
}

// ---- catalogEntryToAddNodeMutation ---------------------------------------

{
  // A node type with NO required capability (classify is 'ai' category,
  // not in REQUIRED_CAPABILITY) -> just one add_node mutation, no tool binding needed.
  const mutations = catalogEntryToAddNodeMutation("classify", { x: 100, y: 200 }, ["n_1", "n_2"]);
  assertEqual(mutations.length, 1, "catalogEntryToAddNodeMutation: a non-tool type produces exactly one mutation");
  const [mutation] = mutations;
  if (mutation.kind === "add_node") {
    assertEqual(mutation.node.type, "classify", "catalogEntryToAddNodeMutation: preserves the node type");
    assertEqual(mutation.node.category, "ai", "catalogEntryToAddNodeMutation: uses the catalog entry's category");
    assertEqual(mutation.node.executionType, "probabilistic", "catalogEntryToAddNodeMutation: classify defaults to probabilistic");
    assertEqual(mutation.node.canvasPosition, { x: 100, y: 200 }, "catalogEntryToAddNodeMutation: places the node at the drop position");
    assert(
      Array.isArray((mutation.node.config as any)?.labels),
      "catalogEntryToAddNodeMutation: classify gets a sensible default config.labels so it isn't immediately invalid"
    );
    assert(mutation.node.id === "n_3", "catalogEntryToAddNodeMutation: generates a non-colliding id");
  } else {
    failed++;
    console.error("FAIL: mutation was not an add_node kind");
  }
}

{
  const mutations = catalogEntryToAddNodeMutation("if_else", { x: 0, y: 0 }, []);
  assertEqual(mutations.length, 1, "catalogEntryToAddNodeMutation: if_else (control, no capability) produces one mutation");
  const [mutation] = mutations;
  if (mutation.kind === "add_node") {
    assertEqual(mutation.node.executionType, "deterministic", "catalogEntryToAddNodeMutation: non-AI types default to deterministic");
  } else {
    failed++;
    console.error("FAIL: if_else should produce an add_node mutation");
  }
}

{
  // THE actual bug fix: search_web needs a 'read' capability — dropping it
  // must ATOMICALLY include a tool binding, not just the bare node,
  // otherwise the spec compiles-fails the moment the user tries to run it
  // (see backend/compiler/compiler.py's REQUIRED_CAPABILITY check).
  const mutations = catalogEntryToAddNodeMutation("search_web", { x: 0, y: 0 }, [], []);
  assertEqual(mutations.length, 2, "catalogEntryToAddNodeMutation: a tool-capability node produces TWO mutations (tool binding + node)");
  const [toolMutation, nodeMutation] = mutations;
  assertEqual(toolMutation.kind, "add_tool", "catalogEntryToAddNodeMutation: the tool binding is added FIRST in the batch");
  if (toolMutation.kind === "add_tool") {
    assert(toolMutation.tool.permissions.includes("read"), "catalogEntryToAddNodeMutation: search_web's auto-provisioned tool grants exactly the 'read' capability it needs");
  }
  assertEqual(nodeMutation.kind, "add_node", "catalogEntryToAddNodeMutation: the node mutation follows the tool mutation");
  if (nodeMutation.kind === "add_node" && toolMutation.kind === "add_tool") {
    assertEqual(
      nodeMutation.node.toolRefs,
      [toolMutation.tool.id],
      "catalogEntryToAddNodeMutation: the new node's toolRefs points at the newly-created tool binding, not left empty"
    );
  }
}

{
  // send_email needs 'send', not 'read' — confirms the capability mapping is actually per-type, not hardcoded to one value.
  const mutations = catalogEntryToAddNodeMutation("send_email", { x: 0, y: 0 }, [], []);
  const toolMutation = mutations.find((m) => m.kind === "add_tool");
  assert(
    toolMutation?.kind === "add_tool" && toolMutation.tool.permissions.includes("send"),
    "catalogEntryToAddNodeMutation: send_email's auto-provisioned tool grants 'send', not 'read'"
  );
}

{
  // Tool id generation avoids collisions with EXISTING tool bindings, not just node ids.
  const mutations = catalogEntryToAddNodeMutation("database", { x: 0, y: 0 }, [], ["tool_1"]);
  const toolMutation = mutations.find((m) => m.kind === "add_tool");
  assert(
    toolMutation?.kind === "add_tool" && toolMutation.tool.id === "tool_2",
    `catalogEntryToAddNodeMutation: generated tool id avoids colliding with existing tool ids — got ${
      toolMutation?.kind === "add_tool" ? toolMutation.tool.id : "n/a"
    }`
  );
}

{
  const mutations = catalogEntryToAddNodeMutation("not_a_real_node_type", { x: 0, y: 0 }, []);
  assertEqual(mutations, [], "catalogEntryToAddNodeMutation: returns an empty array for an unknown node type rather than throwing");
}

// ---- computeToolBindingWarnings -----------------------------------------

{
  const nodes: AgentNode[] = [
    { id: "n1", type: "search_web", category: "tool", label: "Search Web" }, // no toolRefs at all
    { id: "n2", type: "database", category: "tool", label: "Save Data", toolRefs: ["t1"] }, // has a ref, but wrong permission
    { id: "n3", type: "send_email", category: "tool", label: "Notify", toolRefs: ["t2"] }, // correctly bound
    { id: "n4", type: "classify", category: "ai", label: "Classify" }, // not a capability-requiring type at all
  ];
  const tools = [
    { id: "t1", toolId: "google_sheets.append_row", permissions: ["read"] }, // n2 needs 'write', this only grants 'read'
    { id: "t2", toolId: "gmail.send", permissions: ["send"] }, // n3 needs 'send' — correct
  ];

  const warnings = computeToolBindingWarnings(nodes, tools);
  assert(warnings.has("n1"), "computeToolBindingWarnings: flags a tool node with no toolRefs at all");
  assert(warnings.get("n1")!.includes("read"), "computeToolBindingWarnings: mentions the specific missing permission");
  assert(warnings.has("n2"), "computeToolBindingWarnings: flags a tool node whose binding doesn't grant the needed permission");
  assert(!warnings.has("n3"), "computeToolBindingWarnings: does NOT flag a correctly-bound node");
  assert(!warnings.has("n4"), "computeToolBindingWarnings: does NOT flag a non-tool node type (classify has no capability requirement)");
  assertEqual(warnings.size, 2, "computeToolBindingWarnings: exactly the two broken nodes are flagged, nothing else");
}

{
  // Empty spec (no tools at all) shouldn't throw, and should flag every capability-requiring node.
  const nodes: AgentNode[] = [{ id: "n1", type: "read_file", category: "tool", label: "Read" }];
  const warnings = computeToolBindingWarnings(nodes, []);
  assert(warnings.has("n1"), "computeToolBindingWarnings: works correctly with an empty tools array");
}

{
  // Regression for the reported bug: a correctly-bound tool node whose
  // toolId isn't on the agent's policy allowlist compiles-fails at Run
  // time with no prior warning (backend/compiler/compiler.py's
  // _check_run_readiness). This should now show up here instead, the
  // moment the node/binding exists, not only after hitting Run.
  const nodes: AgentNode[] = [{ id: "n1", type: "read_file", category: "tool", label: "Read", toolRefs: ["t1"] }];
  const tools = [{ id: "t1", toolId: "files.read", permissions: ["read"] }];

  const noAllowlist = computeToolBindingWarnings(nodes, tools);
  assert(!noAllowlist.has("n1"), "computeToolBindingWarnings: no allowlist configured means nothing is flagged on policy grounds");

  const passingAllowlist = computeToolBindingWarnings(nodes, tools, ["files.read", "web.search"]);
  assert(!passingAllowlist.has("n1"), "computeToolBindingWarnings: a correctly-bound tool that IS on the allowlist is not flagged");

  const failingAllowlist = computeToolBindingWarnings(nodes, tools, ["gmail.send"]);
  assert(failingAllowlist.has("n1"), "computeToolBindingWarnings: a correctly-bound tool NOT on the allowlist is flagged");
  assert(
    failingAllowlist.get("n1")!.includes("files.read") && failingAllowlist.get("n1")!.includes("allowlist"),
    "computeToolBindingWarnings: the allowlist warning names the specific disallowed tool"
  );

  const emptyAllowlist = computeToolBindingWarnings(nodes, tools, []);
  assert(!emptyAllowlist.has("n1"), "computeToolBindingWarnings: an empty allowlist array means 'no restriction', matching the backend's falsy check");
}

// ---- parseModelString ---------------------------------------------------
// TS mirror of backend/ai_providers/registry.py's parse_model_string --
// used by AiProviderSelector.tsx.

{
  assertEqual(JSON.stringify(parseModelString(undefined)), JSON.stringify([null, null]), "parseModelString: undefined -> [null, null]");
  assertEqual(JSON.stringify(parseModelString(null)), JSON.stringify([null, null]), "parseModelString: null -> [null, null]");
  assertEqual(JSON.stringify(parseModelString("")), JSON.stringify([null, null]), "parseModelString: empty string -> [null, null]");
  assertEqual(JSON.stringify(parseModelString("gpt-4o-mini")), JSON.stringify([null, "gpt-4o-mini"]), "parseModelString: bare model name has no provider");
  assertEqual(JSON.stringify(parseModelString("openai:gpt-4o")), JSON.stringify(["openai", "gpt-4o"]), "parseModelString: splits provider:model correctly");
  assertEqual(JSON.stringify(parseModelString("Anthropic:Claude-Sonnet-5")), JSON.stringify(["anthropic", "Claude-Sonnet-5"]), "parseModelString: provider is lowercased, model case is preserved");
}

// ---- isModelCompatibleWithProvider ---------------------------------------
// TS mirror of backend/ai_providers/registry.py's validate_provider_model --
// used by NodeInspector.tsx's Ask AI provider dropdown to auto-replace an
// incompatible model when the provider is switched.

{
  assert(isModelCompatibleWithProvider("anthropic", "claude-sonnet-5"), "isModelCompatibleWithProvider: anthropic + claude model is compatible");
  assert(isModelCompatibleWithProvider("openai", "gpt-4o"), "isModelCompatibleWithProvider: openai + gpt model is compatible");
  assert(!isModelCompatibleWithProvider("openai", "claude-sonnet-5"), "isModelCompatibleWithProvider: rejects an Anthropic model under openai");
  assert(!isModelCompatibleWithProvider("anthropic", "gpt-4o"), "isModelCompatibleWithProvider: rejects an OpenAI model under anthropic");
  assert(isModelCompatibleWithProvider("openai", "o3-mini"), "isModelCompatibleWithProvider: openai + o3-mini is compatible");
  assert(isModelCompatibleWithProvider("anthropic", "some-future-model"), "isModelCompatibleWithProvider: an unrecognized model name (no provider's prefix) is not rejected outright");
}

// ---- summary ----------------------------------------------------------

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
