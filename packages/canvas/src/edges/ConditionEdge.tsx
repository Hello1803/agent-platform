import { BaseEdge, EdgeLabelRenderer, getSmoothStepPath, useReactFlow, type EdgeProps } from "@xyflow/react";
import type { AgentFlowEdge } from "../specToFlow";

/**
 * Renders the plain connector for unconditional edges, and a small
 * pill label carrying the condition expression for conditional ones —
 * this is the edge-side half of the deterministic/probabilistic
 * distinction from architecture doc section 11: a condition with
 * kind "ai_decision" reads differently from a plain deterministic one.
 *
 * Also renders a small "×" disconnect button at the edge's midpoint,
 * always visible rather than hover-only — the same discoverability fix
 * NodeInspector.tsx applied to node deletion ("previously the only way
 * was knowing to select a node and press Backspace, which... [worked]
 * but with no visible affordance telling anyone that"). Accidentally
 * wiring two nodes together and having no obvious way to undo it short of
 * knowing the edge-select-then-Backspace gesture was the exact complaint
 * this fixes. Clicking it calls useReactFlow().deleteElements(), which
 * raises the same 'remove' EdgeChange AgentCanvas's onEdgesChange already
 * turns into a remove_edge mutation (specToFlow.ts's
 * edgeChangesToMutations) — no new mutation plumbing needed, just a
 * button that triggers the path that already existed but was invisible.
 */
export function ConditionEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps<AgentFlowEdge>) {
  const { deleteElements } = useReactFlow();
  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    borderRadius: 8,
  });

  const condition = data?.agentEdge.condition;
  const expression = condition?.expression;
  const isAiDecision = condition?.kind === "ai_decision";

  // The condition pill (when present) and the delete button both want the
  // edge midpoint, so the button sits just below/beside it rather than on
  // top of it.
  const buttonOffsetY = expression ? 18 : 0;

  return (
    <>
      <BaseEdge id={id} path={edgePath} style={{ stroke: "var(--ac-edge-default)", strokeWidth: 1.5 }} />
      {expression && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              fontSize: 11,
              fontFamily: "var(--ac-font-mono)",
              padding: "2px 8px",
              borderRadius: isAiDecision ? 10 : 4,
              background: "var(--ac-edge-label-bg)",
              color: "var(--ac-edge-label-ink)",
              border: isAiDecision ? "1px dashed var(--ac-ai-base)" : "1px solid var(--ac-border)",
              pointerEvents: "none",
              whiteSpace: "nowrap",
            }}
          >
            {isAiDecision ? "✦ " : ""}
            {expression}
          </div>
        </EdgeLabelRenderer>
      )}
      {!data?.readOnly && (
        <EdgeLabelRenderer>
          <button
            onClick={(event) => {
              event.stopPropagation();
              deleteElements({ edges: [{ id }] });
            }}
            title="Disconnect this edge"
            aria-label="Disconnect edge"
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY + buttonOffsetY}px)`,
              width: 18,
              height: 18,
              borderRadius: "50%",
              border: "1px solid var(--ac-border)",
              background: "var(--ac-surface, #fff)",
              color: "var(--ac-ink-muted, #6b7280)",
              fontSize: 11,
              lineHeight: 1,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              padding: 0,
              // EdgeLabelRenderer's wrapper sets pointer-events: none by
              // default (so hovering a label doesn't block canvas
              // panning/edge-hover elsewhere) — anything inside it meant
              // to be clickable has to opt back in explicitly, or every
              // click silently passes through to whatever's underneath.
              // This one line is the actual fix for "the button is there
              // but nothing happens when I click it": the button
              // rendered, looked clickable, and never received a single
              // pointer event.
              pointerEvents: "all",
            }}
          >
            ×
          </button>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
