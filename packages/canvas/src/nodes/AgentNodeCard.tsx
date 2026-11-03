import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { AgentFlowNode } from "../specToFlow";
import type { NodeCategory } from "@agent-platform/shared-types";

interface CategoryStyle {
  base: string;
  tint: string;
  ink: string;
  icon: string;
}

const CATEGORY_STYLES: Record<NodeCategory, CategoryStyle> = {
  trigger: { base: "var(--ac-trigger-base)", tint: "var(--ac-trigger-tint)", ink: "var(--ac-trigger-ink)", icon: "⚡" },
  ai: { base: "var(--ac-ai-base)", tint: "var(--ac-ai-tint)", ink: "var(--ac-ai-ink)", icon: "✦" },
  tool: { base: "var(--ac-tool-base)", tint: "var(--ac-tool-tint)", ink: "var(--ac-tool-ink)", icon: "⚙" },
  control: { base: "var(--ac-control-base)", tint: "var(--ac-control-tint)", ink: "var(--ac-control-ink)", icon: "⌥" },
  human: { base: "var(--ac-human-base)", tint: "var(--ac-human-tint)", ink: "var(--ac-human-ink)", icon: "✋" },
};

/**
 * The core visual decision of this whole package: executionType controls
 * BORDER STYLE (solid/sharp = deterministic, dashed/gradient = probabilistic),
 * while category controls COLOR. These are independent axes on purpose —
 * conflating them (e.g. "AI nodes are always dashed purple") would make it
 * impossible to tell, at a glance, whether a given AI node's output is
 * being treated as a hard fact or a probabilistic guess downstream.
 */
export function AgentNodeCard({ data, selected }: NodeProps<AgentFlowNode>) {
  const node = data.agentNode;
  const style = CATEGORY_STYLES[node.category];
  const isProbabilistic = node.executionType === "probabilistic";
  const traceStatus = data.traceStatus;

  const traceRingColor =
    traceStatus === "running" ? "#2563eb" :
    traceStatus === "success" ? "#16a34a" :
    traceStatus === "failed" ? "#dc2626" :
    traceStatus === "waiting" ? "#d97706" :
    null;

  return (
    <div
      style={{
        minWidth: 168,
        maxWidth: 220,
        background: "var(--ac-surface)",
        borderRadius: isProbabilistic ? 14 : 8,
        border: isProbabilistic ? `1.5px dashed ${style.base}` : `1px solid var(--ac-border)`,
        borderLeft: `3px solid ${style.base}`,
        boxShadow: traceRingColor
          ? `0 0 0 2px ${traceRingColor}${traceStatus === "running" ? ", 0 0 8px 1px " + traceRingColor + "55" : ""}`
          : selected
          ? `0 0 0 2px ${style.base}`
          : "0 1px 2px rgba(22, 24, 26, 0.06)",
        padding: "10px 12px",
        fontFamily: "var(--ac-font-body)",
        opacity: data.traceActive && !traceStatus ? 0.45 : 1,
        transition: "box-shadow 150ms ease, opacity 150ms ease",
      }}
    >
      {/*
        Explicit handle ids ("input"/"output") rather than the implicit
        default (id=null) are what make connection validation possible at
        all: React Flow's onConnect/isValidConnection callbacks receive a
        Connection object carrying sourceHandle/targetHandle *ids*, not a
        "this was a source-type handle" flag. Without a stable id here,
        isValidAgentConnection (specToFlow.ts) would have no reliable way
        to tell an input->input drag from a legitimate output->input one.
        toFlowEdges() defaults any persisted edge's missing handle id to
        these same values, so specs saved before this change keep working.
      */}
      <Handle id="input" type="target" position={Position.Top} style={{ background: style.base }} />

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          fontSize: 11,
          color: style.ink,
          marginBottom: 4,
          fontFamily: "var(--ac-font-display)",
          textTransform: "uppercase",
          letterSpacing: "0.03em",
        }}
      >
        <span aria-hidden="true">{style.icon}</span>
        <span>{node.type.replace(/_/g, " ")}</span>
        {traceStatus === "running" && (
          <span aria-label="running" style={{ marginLeft: "auto", fontSize: 10, color: traceRingColor! }}>
            ●
          </span>
        )}
      </div>

      <div style={{ fontSize: 13, fontWeight: 600, color: "var(--ac-ink)", marginBottom: isProbabilistic ? 6 : 0 }}>
        {node.label}
      </div>

      {isProbabilistic && (
        <span
          style={{
            display: "inline-block",
            fontSize: 11,
            fontFamily: "var(--ac-font-mono)",
            padding: "1px 8px",
            borderRadius: 10,
            background: style.tint,
            color: style.ink,
          }}
        >
          {node.confidenceThreshold != null
            ? `min confidence ${Math.round(node.confidenceThreshold * 100)}%`
            : "model call"}
        </span>
      )}

      {data.toolBindingWarning && (
        <div
          title={data.toolBindingWarning}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 4,
            marginTop: 6,
            fontSize: 11,
            color: "var(--ac-human-ink, #8a2e08)",
            background: "var(--ac-human-tint, #fbeee6)",
            padding: "2px 7px",
            borderRadius: 6,
          }}
        >
          <span aria-hidden="true">⚠</span>
          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {data.toolBindingWarning}
          </span>
        </div>
      )}

      <Handle id="output" type="source" position={Position.Bottom} style={{ background: style.base }} />
    </div>
  );
}
