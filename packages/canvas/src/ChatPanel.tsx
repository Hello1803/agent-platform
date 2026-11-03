import { useEffect, useState, type KeyboardEvent } from "react";
import type { ChatMessage } from "./chatMessages";
import { summarizeMutation } from "./chatMessages";

export interface ChatPanelProps {
  messages: ChatMessage[];
  onSend: (text: string, options?: { provider?: string; model?: string }) => void;
  client?: import("./apiClient").AgentApiClient;
  defaultProvider?: string | null;
  defaultModel?: string | null;
  /** Commits a pending proposal (the message's `proposalStatus === "pending"`) through the mutation engine. */
  onApply?: (messageId: string) => void;
  /** Discards a pending proposal — purely local, nothing to undo server-side since propose() never persisted anything. */
  onCancel?: (messageId: string) => void;
  busy?: boolean;
  placeholder?: string;
}

/**
 * The chat half of the natural-language/visual-editor pair (architecture
 * doc's core concept). Deliberately presentational — all state lives in
 * useAgentSession, this component only renders `messages` and calls
 * `onSend`/`onApply`/`onCancel`. Pair with AgentCanvas reading the SAME
 * session's `spec`: type "ask me before sending the email", review the
 * proposed human_approval node in the preview below, click Apply, watch
 * it appear on the canvas — with no separate state to keep in sync, and
 * nothing touching the real workflow until Apply is clicked.
 */
export function ChatPanel({ messages, onSend, onApply, onCancel, busy, placeholder, client, defaultProvider, defaultModel }: ChatPanelProps) {
  const [draft, setDraft] = useState("");
  const [providerStatus, setProviderStatus] = useState<import("./apiClient").AiProvidersResponse | null>(null);
  const [provider, setProvider] = useState<string>(defaultProvider || "");
  const [model, setModel] = useState<string>(defaultModel || "");

  useEffect(() => {
    if (defaultProvider) setProvider(defaultProvider);
    if (defaultModel) setModel(defaultModel);
  }, [defaultProvider, defaultModel]);

  useEffect(() => {
    if (!client) return;
    let cancelled = false;
    client.getAiProviders().then((r) => { if (!cancelled) setProviderStatus(r); }).catch(() => {});
    return () => { cancelled = true; };
  }, [client]);


  const submit = () => {
    const text = draft.trim();
    if (!text || busy) return;
    const options = provider || model ? { provider: provider || undefined, model: model || undefined } : undefined;
    onSend(text, options);
    setDraft("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        fontFamily: "var(--ac-font-body, system-ui)",
        background: "var(--ac-surface, #fff)",
        border: "1px solid var(--ac-border, #d8dad5)",
        borderRadius: 8,
      }}
    >
      <div style={{ flex: 1, overflowY: "auto", padding: 12 }}>
        {messages.length === 0 && (
          <div style={{ fontSize: 13, color: "var(--ac-ink-muted, #6b7280)" }}>
            Describe what you want this workflow to do, or ask how it works.
          </div>
        )}
        {messages.map((msg) => (
          <div
            key={msg.id}
            style={{
              marginBottom: 10,
              display: "flex",
              flexDirection: "column",
              alignItems: msg.role === "user" ? "flex-end" : "flex-start",
            }}
          >
            <div
              style={{
                maxWidth: "85%",
                padding: "8px 12px",
                borderRadius: 12,
                fontSize: 13,
                lineHeight: 1.4,
                background:
                  msg.role === "user"
                    ? "var(--ac-tool-base, #1e6fd9)"
                    : msg.status === "error"
                    ? "var(--ac-human-tint, #fbeee6)"
                    : "var(--ac-ai-tint, #f1edff)",
                color:
                  msg.role === "user"
                    ? "#fff"
                    : msg.status === "error"
                    ? "var(--ac-human-ink, #8a2e08)"
                    : "var(--ac-ink, #16181a)",
                whiteSpace: "pre-wrap",
              }}
            >
              {msg.content}
            </div>

            {/* Proposal preview: a real, structured summary of every
                mutation the AI is proposing — not just its prose
                reasoning — so what [Apply Changes] would actually do is
                never a surprise. Nothing here has touched the workflow;
                the buttons are the only way anything gets applied. */}
            {msg.proposal && msg.proposal.mutations.length > 0 && (
              <div
                style={{
                  maxWidth: "85%",
                  marginTop: 4,
                  padding: "8px 10px",
                  borderRadius: 8,
                  border: "1px solid var(--ac-border, #d8dad5)",
                  background: "var(--ac-surface, #fff)",
                  fontSize: 12,
                }}
              >
                <div style={{ display: "flex", flexDirection: "column", gap: 3, fontFamily: "var(--ac-font-mono, monospace)" }}>
                  {msg.proposal.mutations.map((m, i) => (
                    <div key={i}>{summarizeMutation(m)}</div>
                  ))}
                </div>

                {msg.proposalStatus === "pending" && (
                  <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                    <button
                      onClick={() => onCancel?.(msg.id)}
                      disabled={busy}
                      style={{
                        padding: "5px 12px",
                        borderRadius: 6,
                        border: "1px solid var(--ac-border, #d8dad5)",
                        background: "transparent",
                        fontSize: 12,
                        cursor: busy ? "default" : "pointer",
                      }}
                    >
                      Cancel
                    </button>
                    <button
                      onClick={() => onApply?.(msg.id)}
                      disabled={busy}
                      style={{
                        padding: "5px 12px",
                        borderRadius: 6,
                        border: "none",
                        background: "var(--ac-tool-base, #1e6fd9)",
                        color: "#fff",
                        fontSize: 12,
                        cursor: busy ? "default" : "pointer",
                        opacity: busy ? 0.6 : 1,
                      }}
                    >
                      Apply Changes
                    </button>
                  </div>
                )}
                {msg.proposalStatus === "applied" && (
                  <div style={{ fontSize: 11, color: "var(--ac-tool-ink, #0f4f9c)", marginTop: 6, fontWeight: 600 }}>✓ Applied</div>
                )}
                {msg.proposalStatus === "cancelled" && (
                  <div style={{ fontSize: 11, color: "var(--ac-ink-muted, #6b7280)", marginTop: 6, fontWeight: 600 }}>Cancelled — no changes made</div>
                )}
              </div>
            )}

            {msg.role === "assistant" && msg.status === "ok" && !msg.proposal && (
              <span style={{ fontSize: 11, color: "var(--ac-ink-muted, #6b7280)", marginTop: 2 }}>No changes made</span>
            )}
          </div>
        ))}
        {busy && (
          <div style={{ fontSize: 12, color: "var(--ac-ink-muted, #6b7280)", fontStyle: "italic" }}>Thinking…</div>
        )}
      </div>

      {client && (
        <div style={{ display: "flex", gap: 8, padding: "6px 10px", borderTop: "1px solid var(--ac-border, #d8dad5)", alignItems: "center", fontSize: 12 }}>
          <span style={{ color: "var(--ac-ink-muted, #6b7280)" }}>Chat AI</span>
          <select value={provider} onChange={(e) => { const next = e.target.value; setProvider(next); setModel(providerStatus?.providers.find((p) => p.id === next)?.models?.[0]?.id || ""); }} disabled={busy} style={{ fontSize: 12, padding: "2px 6px", borderRadius: 4, border: "1px solid var(--ac-border, #d8dad5)" }}>
            <option value="">Server default</option>
            {(providerStatus?.providers ?? []).map((p) => (
              <option key={p.id} value={p.id} disabled={!p.hasCredentials}>{p.id}{!p.hasCredentials ? " (not configured)" : ""}</option>
            ))}
          </select>
          {provider && (
            (providerStatus?.providers.find((p) => p.id === provider)?.models?.length ?? 0) > 0 ? (
              <select value={model} onChange={(e) => setModel(e.target.value)} disabled={busy} style={{ fontSize: 12, padding: "2px 6px", borderRadius: 4, border: "1px solid var(--ac-border, #d8dad5)", flex: 1 }}>
                {(providerStatus?.providers.find((p) => p.id === provider)?.models ?? []).map((m) => (
                  <option key={m.id} value={m.id}>{m.label || m.id}</option>
                ))}
              </select>
            ) : (
              <input value={model} onChange={(e) => setModel(e.target.value)} disabled={busy} placeholder={provider === "lmstudio" ? "model id in LM Studio" : "model id"} style={{ fontSize: 12, padding: "2px 6px", borderRadius: 4, border: "1px solid var(--ac-border, #d8dad5)", flex: 1 }} />
            )
          )}
        </div>
      )}

      <div style={{ display: "flex", gap: 8, padding: 10, borderTop: "1px solid var(--ac-border, #d8dad5)" }}>
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={busy}
          placeholder={placeholder ?? 'e.g. "Add approval before Send Email"'}
          rows={2}
          style={{
            flex: 1,
            resize: "none",
            border: "1px solid var(--ac-border, #d8dad5)",
            borderRadius: 6,
            padding: "6px 8px",
            fontSize: 13,
            fontFamily: "inherit",
          }}
        />
        <button
          onClick={submit}
          disabled={busy || !draft.trim()}
          style={{
            padding: "0 16px",
            borderRadius: 6,
            border: "none",
            background: "var(--ac-tool-base, #1e6fd9)",
            color: "#fff",
            fontSize: 13,
            cursor: busy || !draft.trim() ? "default" : "pointer",
            opacity: busy || !draft.trim() ? 0.5 : 1,
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}
