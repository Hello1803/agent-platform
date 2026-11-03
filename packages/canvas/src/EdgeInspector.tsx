import { useEffect, useState } from "react";
import type { AgentEdge, AgentSpecMutation } from "@agent-platform/shared-types";

export interface EdgeInspectorProps {
  edge: AgentEdge | null;
  sourceNodeType?: string | null;
  onMutate: (mutations: AgentSpecMutation[]) => void;
  onClose?: () => void;
}

export function EdgeInspector({ edge, sourceNodeType, onMutate, onClose }: EdgeInspectorProps) {
  const [expression, setExpression] = useState("");
  const [kind, setKind] = useState<"deterministic" | "ai_decision">("deterministic");
  const [label, setLabel] = useState("");

  useEffect(() => {
    if (edge) {
      setExpression(edge.condition?.expression ?? "");
      setKind((edge.condition?.kind as "deterministic" | "ai_decision") || "deterministic");
      setLabel(edge.label ?? "");
    }
  }, [edge?.id]);

  if (!edge) {
    return (
      <div style={{ padding: 16, fontSize: 13, color: "var(--ac-ink-muted, #6b7280)", fontFamily: "var(--ac-font-body, system-ui)" }}>
        Select an edge to edit its condition.
      </div>
    );
  }

  const isBranchSource = sourceNodeType === "if_else" || sourceNodeType === "switch";

  const apply = () => {
    const trimmed = expression.trim();
    onMutate([{
      kind: "update_edge",
      edgeId: edge.id,
      patch: {
        label: label.trim() || null,
        condition: trimmed ? { expression: trimmed, kind } : null,
      },
    }]);
  };

  return (
    <div style={{ padding: 16, fontFamily: "var(--ac-font-body, system-ui)", fontSize: 13 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <strong>Edge condition</strong>
        {onClose && (
          <button onClick={onClose} style={{ border: "none", background: "transparent", cursor: "pointer", fontSize: 16 }} aria-label="Close">×</button>
        )}
      </div>
      <div style={{ fontSize: 12, color: "var(--ac-ink-muted, #6b7280)", marginBottom: 10 }}>
        {edge.source} → {edge.target}
        {isBranchSource && (
          <div style={{ marginTop: 4, color: "var(--ac-tool-ink, #0f4f9c)" }}>
            Leaving a {sourceNodeType} node — set a condition (e.g. amount &gt; 1000).
          </div>
        )}
      </div>
      <label style={{ display: "block", marginBottom: 4, fontWeight: 600 }}>Condition expression</label>
      <input value={expression} onChange={(e) => setExpression(e.target.value)} placeholder='e.g. amount > 1000'
        style={{ width: "100%", boxSizing: "border-box", padding: "6px 8px", borderRadius: 6, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, fontFamily: "ui-monospace, monospace", marginBottom: 10 }} />
      <label style={{ display: "block", marginBottom: 4, fontWeight: 600 }}>Kind</label>
      <select value={kind} onChange={(e) => setKind(e.target.value as "deterministic" | "ai_decision")}
        style={{ width: "100%", padding: "6px 8px", borderRadius: 6, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, marginBottom: 10 }}>
        <option value="deterministic">Deterministic</option>
        <option value="ai_decision">AI decision</option>
      </select>
      <label style={{ display: "block", marginBottom: 4, fontWeight: 600 }}>Label (optional)</label>
      <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Shown on the edge"
        style={{ width: "100%", boxSizing: "border-box", padding: "6px 8px", borderRadius: 6, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, marginBottom: 12 }} />
      <div style={{ display: "flex", gap: 8 }}>
        <button onClick={apply} style={{ padding: "6px 12px", borderRadius: 6, border: "none", background: "var(--ac-tool-base, #1e6fd9)", color: "#fff", fontSize: 13, cursor: "pointer" }}>Save condition</button>
        <button onClick={() => { setExpression(""); onMutate([{ kind: "update_edge", edgeId: edge.id, patch: { condition: null } }]); }}
          style={{ padding: "6px 12px", borderRadius: 6, border: "1px solid var(--ac-border, #d8dad5)", background: "transparent", fontSize: 13, cursor: "pointer" }}>Clear</button>
      </div>
    </div>
  );
}
