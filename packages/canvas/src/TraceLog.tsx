import type { TraceEventLike } from "./specToFlow";

export interface TraceLogProps {
  trace: TraceEventLike[];
  /** Called when the user clicks a trace entry — a real app uses this to scroll/highlight the corresponding node on AgentCanvas (doc section 12's "why did this execute?"). */
  onSelectEvent?: (event: TraceEventLike) => void;
  selectedNodeId?: string | null;
}

const STATUS_COLOR: Record<TraceEventLike["status"], string> = {
  success: "#16a34a",
  failed: "#dc2626",
  waiting: "#d97706",
};

const STATUS_SYMBOL: Record<TraceEventLike["status"], string> = {
  success: "✓",
  failed: "✗",
  waiting: "⏸",
};

/** Renders the doc section 12 style trace: "✓ Trigger received / ✓ Email retrieved / ⏸ Waiting for human approval." */
export function TraceLog({ trace, onSelectEvent, selectedNodeId }: TraceLogProps) {
  if (trace.length === 0) {
    return (
      <div style={{ padding: 12, fontSize: 13, color: "var(--ac-ink-muted, #6b7280)", fontFamily: "var(--ac-font-body, system-ui)" }}>
        No run yet.
      </div>
    );
  }

  return (
    <div style={{ fontFamily: "var(--ac-font-body, system-ui)", fontSize: 13 }}>
      {trace.map((event, i) => {
        const isSelected = event.node_id === selectedNodeId;
        return (
          <button
            key={`${event.node_id}-${event.timestamp}-${i}`}
            onClick={() => onSelectEvent?.(event)}
            style={{
              display: "flex",
              alignItems: "flex-start",
              gap: 8,
              width: "100%",
              textAlign: "left",
              padding: "6px 10px",
              border: "none",
              borderLeft: isSelected ? "2px solid var(--ac-tool-base, #1e6fd9)" : "2px solid transparent",
              background: isSelected ? "var(--ac-tool-tint, #e8f1fc)" : "transparent",
              cursor: onSelectEvent ? "pointer" : "default",
              font: "inherit",
            }}
          >
            <span style={{ color: STATUS_COLOR[event.status], fontFamily: "var(--ac-font-mono, monospace)", flexShrink: 0 }}>
              {STATUS_SYMBOL[event.status]}
            </span>
            <span style={{ flex: 1, minWidth: 0 }}>
              <span style={{ fontWeight: 600 }}>{event.node_label}</span>
              <span style={{ color: "var(--ac-ink-muted, #6b7280)" }}> — {event.detail}</span>
            </span>
          </button>
        );
      })}
    </div>
  );
}
