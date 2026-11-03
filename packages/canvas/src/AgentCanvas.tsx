import { useCallback, useEffect, useMemo } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  Controls,
  Panel,
  useNodesState,
  useEdgesState,
  useReactFlow,
  type OnConnect,
  type OnNodesChange,
  type OnEdgesChange,
  type Node as FlowNodeType,
  type Connection,
  type IsValidConnection,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import type { AgentSpecification, AgentSpecMutation, NodeCategory } from "@agent-platform/shared-types";
import {
  toFlowNodes,
  toFlowEdges,
  nodeChangesToMutations,
  edgeChangesToMutations,
  connectionToAddEdgeMutation,
  isValidAgentConnection,
  catalogEntryToAddNodeMutation,
  applyTraceToNodes,
  computeToolBindingWarnings,
  AGENT_NODE_TYPE,
  CONDITION_EDGE_TYPE,
  type AgentFlowNode,
  type AgentFlowEdge,
  type AgentFlowEdgeData,
  type TraceEventLike,
} from "./specToFlow";
import { AgentNodeCard } from "./nodes/AgentNodeCard";
import { ConditionEdge } from "./edges/ConditionEdge";
import { DRAG_DATA_TYPE } from "./NodePalette";
import "./tokens.css";

const nodeTypes = { [AGENT_NODE_TYPE]: AgentNodeCard };
const edgeTypes = { [CONDITION_EDGE_TYPE]: ConditionEdge };

const LEGEND: { category: NodeCategory; label: string }[] = [
  { category: "trigger", label: "trigger" },
  { category: "ai", label: "ai — dashed = probabilistic" },
  { category: "control", label: "control — deterministic" },
  { category: "human", label: "human" },
  { category: "tool", label: "tool" },
];

export interface AgentCanvasProps {
  /** The authoritative spec — this component NEVER writes to it directly. */
  spec: AgentSpecification;
  /**
   * Called with a BATCH of mutations whenever the user does something that
   * should change the spec (drag a node to rest, connect two nodes,
   * delete one or more nodes/edges). Always an array, and always the whole
   * set from one user action in a single call — e.g. deleting 3 selected
   * nodes produces one call with 3 remove_node mutations, not 3 separate
   * calls. This matters: the mutation engine's atomic batch mechanism
   * exists specifically so a multi-part edit either fully applies or
   * fully fails together, based on ONE expected_version check — firing
   * mutations individually would have them race each other on the same
   * starting version, with only the first landing and the rest getting
   * spurious 409 conflicts even though nothing else touched the spec in
   * between. The caller is responsible for applying the batch via the
   * mutation engine and re-rendering with the resulting spec — this
   * component only ever proposes changes, exactly like the chat builder
   * does via the NL compiler. Neither interface is allowed to be a second
   * source of truth.
   */
  onMutation: (mutations: AgentSpecMutation[]) => void;
  /**
   * Bump this (e.g. a counter) to force a resync from `spec` even when
   * `spec.metadata.version` hasn't changed. Needed for a REJECTED
   * mutation: the server didn't change anything, so version is identical
   * to before, but the user's optimistic local drag/connect still needs
   * to be discarded. See useAgentSession.ts for the canonical caller.
   */
  resyncNonce?: number;
  readOnly?: boolean;
  /**
   * The execution trace to overlay on the canvas (doc section 12): nodes
   * that have run so far get a status-colored ring, everything else dims.
   * Pass the live-growing trace from useAgentRunStream while a run is
   * streaming, or the final trace from a completed/paused RunResult to
   * inspect a past run. Omit entirely for the normal editing view.
   */
  trace?: TraceEventLike[];
  /** The node currently executing, if a run is actively streaming — rendered as "running" even though it has no trace entry yet. */
  activeNodeId?: string | null;
  /**
   * Fired when the user clicks a node (or clicks empty canvas, with
   * `null`) — pair with NodeInspector to show the clicked node's details.
   * AgentCanvas doesn't track "selected node" as its own concept beyond
   * this passthrough; the caller owns that state.
   */
  onSelectNode?: (nodeId: string | null) => void;
  onSelectEdge?: (edgeId: string | null) => void;
}

function AgentCanvasInner({ spec, onMutation, resyncNonce, readOnly, trace, activeNodeId, onSelectNode, onSelectEdge }: AgentCanvasProps) {
  const [flowNodes, setFlowNodes, onNodesChangeInternal] = useNodesState<AgentFlowNode>(toFlowNodes(spec.nodes));
  const [flowEdges, setFlowEdges, onEdgesChangeInternal] = useEdgesState<AgentFlowEdge>(toFlowEdges(spec.edges));
  const { screenToFlowPosition } = useReactFlow();

  // The spec is the source of truth. Resync local canvas state whenever
  // either (a) the spec's version actually changed — a successful mutation
  // from this canvas or a concurrent chat edit via the NL compiler both
  // land here — or (b) resyncNonce was explicitly bumped, which is how a
  // REJECTED mutation (version unchanged, since nothing happened
  // server-side) still gets its optimistic local change discarded.
  useEffect(() => {
    setFlowNodes(toFlowNodes(spec.nodes));
    setFlowEdges(toFlowEdges(spec.edges));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec.metadata.version, resyncNonce]);

  const onNodesChange: OnNodesChange<AgentFlowNode> = useCallback(
    (changes) => {
      onNodesChangeInternal(changes); // smooth local drag/selection feedback
      if (readOnly) return;
      const mutations = nodeChangesToMutations(changes);
      if (mutations.length > 0) onMutation(mutations);
    },
    [onNodesChangeInternal, onMutation, readOnly]
  );

  const onEdgesChange: OnEdgesChange<AgentFlowEdge> = useCallback(
    (changes) => {
      onEdgesChangeInternal(changes);
      if (readOnly) return;
      const mutations = edgeChangesToMutations(changes);
      if (mutations.length > 0) onMutation(mutations);
    },
    [onEdgesChangeInternal, onMutation, readOnly]
  );

  const onConnect: OnConnect = useCallback(
    (connection) => {
      if (readOnly) return;
      // Deliberately NOT optimistically drawn locally — the new edge
      // appears once the mutation round-trips and `spec` updates, keeping
      // a single rendering path (spec -> flow) instead of two.
      //
      // isValidConnection (passed to <ReactFlow> below) already stops
      // React Flow from calling onConnect at all for a gesture like
      // output->output, input->input, or a self-connection — so in the
      // overwhelming majority of cases this handler only ever sees
      // already-valid connections. connectionToAddEdgeMutation
      // re-validates anyway (same isValidAgentConnection check) as
      // defense-in-depth, so nothing here has to trust that invariant.
      const existingIds = flowEdges.map((e) => e.id);
      const mutation = connectionToAddEdgeMutation(connection, existingIds, flowNodes, spec.edges);
      if (mutation) onMutation([mutation]);
    },
    [flowEdges, flowNodes, spec.edges, onMutation, readOnly]
  );

  // Gates the CONNECTION GESTURE ITSELF: when this returns false, React
  // Flow refuses to complete the drag — no edge preview snaps into place,
  // onConnect never fires, and nothing is added to React Flow's local
  // edge state (flowEdges) even transiently. This is what actually
  // satisfies "invalid connections must not be added to React Flow
  // state" — onConnect re-validating afterward would be too late, since
  // by then the user has already seen the (misleading) connection
  // succeed visually.
  const isValidConnection: IsValidConnection<AgentFlowEdge> = useCallback(
    (edgeOrConnection) => isValidAgentConnection(edgeOrConnection as Connection, flowNodes, spec.edges),
    [flowNodes, spec.edges]
  );

  // Drop handling for NodePalette drags: reads the dragged node type,
  // resolves it against NODE_CATALOG for sensible defaults (labels for
  // classify, fields for extract, etc. — so a freshly-dropped node isn't
  // immediately invalid), and computes the exact drop position in canvas
  // coordinates via screenToFlowPosition so it lands where the user
  // actually dropped it, not at some arbitrary default.
  const onDragOver = useCallback(
    (event: React.DragEvent) => {
      if (readOnly) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
    },
    [readOnly]
  );

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      if (readOnly) return;
      event.preventDefault();
      const nodeType = event.dataTransfer.getData(DRAG_DATA_TYPE);
      if (!nodeType) return;

      const position = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      const existingNodeIds = flowNodes.map((n) => n.id);
      const existingToolIds = spec.tools?.map((t) => t.id) ?? [];
      const mutations = catalogEntryToAddNodeMutation(nodeType, position, existingNodeIds, existingToolIds);
      if (mutations.length > 0) onMutation(mutations);
    },
    [readOnly, screenToFlowPosition, flowNodes, spec.tools, onMutation]
  );

  const onPaneClick = useCallback(() => {
    onSelectNode?.(null);
    onSelectEdge?.(null);
  }, [onSelectNode, onSelectEdge]);
  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: FlowNodeType) => {
      onSelectNode?.(node.id);
      onSelectEdge?.(null);
    },
    [onSelectNode, onSelectEdge]
  );
  const onEdgeClick = useCallback(
    (_event: React.MouseEvent, edge: { id: string }) => {
      onSelectEdge?.(edge.id);
      onSelectNode?.(null);
    },
    [onSelectEdge, onSelectNode]
  );

  // The trace overlay is a DERIVED view layer on top of the canvas's own
  // node state — drag positions and selection live in flowNodes/setFlowNodes
  // as usual; trace status is computed fresh each render and never written
  // back into flowNodes, so overlaying a trace can never fight with an
  // in-progress drag or get "stuck" after the trace prop changes.
  const renderedNodes = useMemo(() => {
    const withTrace = applyTraceToNodes(flowNodes, trace ?? [], activeNodeId);
    const warnings = computeToolBindingWarnings(spec.nodes, spec.tools ?? [], spec.policies?.toolAllowlist);
    if (warnings.size === 0) return withTrace;
    return withTrace.map((node) => {
      const warning = warnings.get(node.id);
      if (warning === node.data.toolBindingWarning) return node;
      return { ...node, data: { ...node.data, toolBindingWarning: warning } };
    });
  }, [flowNodes, trace, activeNodeId, spec.nodes, spec.tools, spec.policies]);

  const legendChips = useMemo(
    () =>
      LEGEND.map(({ category, label }) => (
        <span
          key={category}
          style={{
            fontSize: 11,
            padding: "2px 8px",
            borderRadius: 10,
            background: `var(--ac-${category}-tint)`,
            color: `var(--ac-${category}-ink)`,
            marginRight: 6,
          }}
        >
          {label}
        </span>
      )),
    []
  );

  const renderedEdges = useMemo(
    () => (readOnly ? flowEdges.map((edge) => ({ ...edge, data: { ...edge.data, readOnly } as AgentFlowEdgeData })) : flowEdges),
    [flowEdges, readOnly]
  );

  return (
    <div className="agent-canvas" style={{ width: "100%", height: "100%" }} onDragOver={onDragOver} onDrop={onDrop}>
      <ReactFlow
        nodes={renderedNodes}
        edges={renderedEdges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        isValidConnection={isValidConnection}
        onNodeClick={onNodeClick}
        onEdgeClick={onEdgeClick}
        onPaneClick={onPaneClick}
        nodesDraggable={!readOnly}
        nodesConnectable={!readOnly}
        elementsSelectable={!readOnly}
        fitView
      >
        <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="var(--ac-grid-dot)" />
        <Controls showInteractive={!readOnly} />
        <Panel position="top-left">
          <div style={{ background: "var(--ac-surface)", border: "1px solid var(--ac-border)", borderRadius: 8, padding: "6px 8px" }}>
            {legendChips}
          </div>
        </Panel>
      </ReactFlow>
    </div>
  );
}

/** Public entry point — wraps the inner component in ReactFlowProvider so it can be dropped anywhere without the consumer needing to know about that. */
export function AgentCanvas(props: AgentCanvasProps) {
  return (
    <ReactFlowProvider>
      <AgentCanvasInner {...props} />
    </ReactFlowProvider>
  );
}
