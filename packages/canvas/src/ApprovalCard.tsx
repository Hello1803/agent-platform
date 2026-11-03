import { useState } from "react";

export interface ApprovalCardProps {
  /** The interrupt payload from a paused run — see executors.py's exec_human_approval/exec_ask_user/exec_confirm/exec_review. */
  interrupt: Record<string, unknown>;
  onApprove: (reason?: string) => void;
  onReject: (reason?: string) => void;
  busy?: boolean;
}

const ACTION_CATEGORY_LABEL: Record<string, string> = {
  money: "involves money",
  send_message: "sends a message",
  delete_data: "deletes data",
  account_change: "changes an account",
  external_publish: "publishes externally",
  sensitive_info: "handles sensitive info",
};

/**
 * Renders doc section 13's exact format:
 *   Agent wants to:
 *   Refund ₹4,999
 *   Customer: XXXXX
 *   Reason: Returned product
 *   [Approve] [Reject]
 * `interrupt.context` (set by exec_human_approval) becomes the key/value
 * lines; `interrupt.action_category` becomes the small warning label.
 */
export function ApprovalCard({ interrupt, onApprove, onReject, busy }: ApprovalCardProps) {
  const [reason, setReason] = useState("");
  const type = interrupt.type as string | undefined;
  const label = (interrupt.label as string | undefined) ?? "this action";
  const category = interrupt.action_category as string | undefined;
  const context = (interrupt.context as Record<string, unknown> | undefined) ?? {};
  const prompt = interrupt.prompt as string | undefined; // ask_user

  return (
    <div
      style={{
        border: "1px solid var(--ac-human-base, #c2410c)",
        borderRadius: 10,
        background: "var(--ac-human-tint, #fbeee6)",
        padding: 14,
        fontFamily: "var(--ac-font-body, system-ui)",
        maxWidth: 360,
      }}
    >
      <div style={{ fontSize: 12, fontWeight: 600, color: "var(--ac-human-ink, #8a2e08)", marginBottom: 6 }}>
        {type === "ask_user" ? "Agent is asking:" : "Agent wants to:"}
      </div>

      {type === "ask_user" ? (
        <div style={{ fontSize: 14, marginBottom: 10 }}>{prompt || label}</div>
      ) : (
        <>
          <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 4 }}>{label}</div>
          {category && (
            <div style={{ fontSize: 11, color: "var(--ac-human-ink, #8a2e08)", marginBottom: 8 }}>
              ⚠ {ACTION_CATEGORY_LABEL[category] ?? category}
            </div>
          )}
          {Object.entries(context).map(([key, value]) => (
            <div key={key} style={{ fontSize: 13, color: "var(--ac-ink, #16181a)" }}>
              <span style={{ color: "var(--ac-ink-muted, #6b7280)" }}>{key}: </span>
              {String(value)}
            </div>
          ))}
        </>
      )}

      <textarea
        value={reason}
        onChange={(e) => setReason(e.target.value)}
        placeholder="Optional note (shown in the run's audit log)"
        rows={2}
        style={{
          width: "100%",
          marginTop: 10,
          marginBottom: 10,
          resize: "none",
          fontSize: 12,
          padding: 6,
          borderRadius: 6,
          border: "1px solid var(--ac-border, #d8dad5)",
          fontFamily: "inherit",
          boxSizing: "border-box",
        }}
      />

      <div style={{ display: "flex", gap: 8 }}>
        <button
          onClick={() => onApprove(reason || undefined)}
          disabled={busy}
          style={{
            flex: 1,
            padding: "6px 0",
            borderRadius: 6,
            border: "none",
            background: "var(--ac-trigger-base, #1b8a5a)",
            color: "#fff",
            fontSize: 13,
            cursor: busy ? "default" : "pointer",
            opacity: busy ? 0.6 : 1,
          }}
        >
          Approve
        </button>
        <button
          onClick={() => onReject(reason || undefined)}
          disabled={busy}
          style={{
            flex: 1,
            padding: "6px 0",
            borderRadius: 6,
            border: "1px solid var(--ac-human-base, #c2410c)",
            background: "transparent",
            color: "var(--ac-human-ink, #8a2e08)",
            fontSize: 13,
            cursor: busy ? "default" : "pointer",
            opacity: busy ? 0.6 : 1,
          }}
        >
          Reject
        </button>
      </div>
    </div>
  );
}
