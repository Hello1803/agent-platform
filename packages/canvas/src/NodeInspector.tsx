import { useEffect, useState } from "react";
import type { AgentNode, AgentSpecMutation, ToolBinding } from "@agent-platform/shared-types";
import { toolsForNode } from "./toolRegistry";
import { isModelCompatibleWithProvider } from "./specToFlow";
import type { AgentApiClient, AiProvidersResponse } from "./apiClient";
import { markdownToHtml } from "./markdown";

export interface NodeInspectorProps {
  node: AgentNode | null;
  tools?: ToolBinding[];
  onMutate: (mutations: AgentSpecMutation[]) => void;
  onClose?: () => void;
  /** Optional: when provided, ask_ai nodes get a Provider/Model section
   * backed by real server data (GET /ai/providers) instead of raw JSON
   * editing. Omitted entirely (not just hidden) when no client is
   * passed, so existing callers/tests that don't need this keep working
   * unchanged. */
  client?: AgentApiClient;
  /** Latest run variables — used by Preview nodes to render Markdown output. */
  runVariables?: Record<string, unknown> | null;
}

/** Node types whose config the Ask AI-style provider/model picker
 * applies to — both share the exact same config shape and executor
 * (backend/real_executors/ai.py's shared freetext executor), but the
 * inspector's dedicated dropdowns are scoped to `ask_ai` per the actual
 * ask ("Add provider and model selection to the Ask AI Inspector").
 * ai_agent nodes can still set config.provider/model by hand in the JSON
 * editor below — the backend validates/honors it identically either
 * way — they just don't get the dedicated dropdown UI. */
const PROVIDER_SELECTABLE_TYPES = new Set(["ask_ai"]);

/**
 * The "inspection window" — clicking a node on AgentCanvas (via its
 * onSelectNode prop) surfaces this. Shows every field that actually
 * matters for understanding what a node does (doc section 12's "why did
 * this execute?" starts with being able to SEE the node's config at all),
 * and is the direct, discoverable path to deleting a node — previously
 * the only way was knowing to select a node and press Backspace, which
 * AgentCanvas's default React Flow behavior already supports but with no
 * visible affordance telling anyone that.
 */
export function NodeInspector({ node, tools = [], onMutate, onClose, client, runVariables = null }: NodeInspectorProps) {
  const [label, setLabel] = useState("");
  const [configText, setConfigText] = useState("{}");
  const [configError, setConfigError] = useState<string | null>(null);
  const [providerStatus, setProviderStatus] = useState<AiProvidersResponse | null>(null);
  const [agentFiles, setAgentFiles] = useState<Array<{ path: string; size: number; suffix: string }>>([]);
  const [filesDir, setFilesDir] = useState<string>("");
  const [previewOpen, setPreviewOpen] = useState(false);
  const binding = node?.toolRefs?.map((ref) => tools.find((tool) => tool.id === ref)).find(Boolean);
  const compatibleTools = node ? toolsForNode(node.type) : [];
  const showProviderPicker = !!(node && client && PROVIDER_SELECTABLE_TYPES.has(node.type));

  useEffect(() => {
    if (node) {
      setLabel(node.label);
      setConfigText(JSON.stringify(node.config ?? {}, null, 2));
      setConfigError(null);
    }
  }, [node?.id]);

  useEffect(() => {
    if (!showProviderPicker || !client) return;
    let cancelled = false;
    client
      .getAiProviders()
      .then((result) => {
        if (!cancelled) setProviderStatus(result);
      })
      .catch(() => {
        /* provider picker fails soft — the raw JSON config editor below still works */
      });
    return () => {
      cancelled = true;
    };
  }, [showProviderPicker, client]);

  useEffect(() => {
    if (!client) return;
    let cancelled = false;
    const load = () =>
      client
        .listAgentFiles()
        .then((r) => {
          if (cancelled) return;
          setAgentFiles(r.files || []);
          setFilesDir(r.directory || "");
        })
        .catch(() => {
          /* soft fail — path fields still accept typing */
        });
    load();
    const t = window.setInterval(load, 8000);
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, [client, node?.id]);

  if (!node) {
    return (
      <div style={{ padding: 16, fontSize: 13, color: "var(--ac-ink-muted, #6b7280)", fontFamily: "var(--ac-font-body, system-ui)" }}>
        Select a node to inspect it.
      </div>
    );
  }

  const dirty = label !== node.label || configText !== JSON.stringify(node.config ?? {}, null, 2);

  const handleSave = () => {
    let parsedConfig: Record<string, unknown>;
    try {
      parsedConfig = JSON.parse(configText);
    } catch {
      setConfigError("Config must be valid JSON.");
      return;
    }
    setConfigError(null);
    const patch: Record<string, unknown> = {};
    if (label !== node.label) patch.label = label;
    if (configText !== JSON.stringify(node.config ?? {}, null, 2)) patch.config = parsedConfig;
    if (Object.keys(patch).length > 0) {
      onMutate([{ kind: "update_node", nodeId: node.id, patch }]);
    }
  };

  const handleDelete = () => {
    onMutate([{ kind: "remove_node", nodeId: node.id }]);
    onClose?.();
  };

  const configureBinding = (toolId: string) => {
    const definition = compatibleTools.find((tool) => tool.id === toolId);
    if (!definition) return;
    const patch = { toolId, displayName: definition.name, permissions: definition.permissions, mode: definition.execution };
    if (binding) onMutate([{ kind: "update_tool", toolId: binding.id, patch }]);
    else {
      const id = `tool_${node.id}`;
      onMutate([{ kind: "add_tool", tool: { id, ...patch } }, { kind: "update_node", nodeId: node.id, patch: { toolRefs: [id] } }]);
    }
  };

  // Reads the current draft config (configText, not node.config) so
  // provider/model changes compose correctly with in-progress, unsaved
  // JSON edits rather than clobbering them. Falls back to the last-known-
  // good config if the draft is mid-edit and momentarily invalid JSON.
  let draftConfig: Record<string, unknown>;
  try {
    draftConfig = JSON.parse(configText);
  } catch {
    draftConfig = node.config ?? {};
  }
  const currentProvider = typeof draftConfig.provider === "string" ? draftConfig.provider : "";
  const currentModel = typeof draftConfig.model === "string" ? draftConfig.model : "";

  const updateProviderModel = (nextProvider: string, nextModel: string) => {
    const next = { ...draftConfig };
    if (nextProvider) {
      next.provider = nextProvider;
      if (nextModel) next.model = nextModel;
      else delete next.model;
    } else {
      delete next.provider;
      delete next.model;
    }
    const serialized = JSON.stringify(next, null, 2);
    setConfigText(serialized);
    setConfigError(null);
    // Persist immediately — otherwise the dropdown looks like it works but
    // runs still use the server default until the user remembers Save.
    onMutate([{ kind: "update_node", nodeId: node.id, patch: { config: next } }]);
  };

  const handleProviderChange = (nextProvider: string) => {
    if (!nextProvider) {
      updateProviderModel("", "");
      return;
    }
    const providerModels = providerStatus?.providers.find((p) => p.id === nextProvider)?.models ?? [];
    // "Switching providers must automatically replace an incompatible
    // model" — never leave e.g. a leftover "claude-sonnet-5" selected
    // after switching the provider dropdown to "openai".
    const nextModel =
      currentModel && isModelCompatibleWithProvider(nextProvider, currentModel)
        ? currentModel
        : (providerModels[0]?.id ?? "");
    updateProviderModel(nextProvider, nextModel);
  };

  return (
    <div style={{ padding: 14, fontFamily: "var(--ac-font-body, system-ui)", display: "flex", flexDirection: "column", gap: 10, height: "100%", boxSizing: "border-box" }}>
      {/*
        Everything above the Save/Delete row lives in its own scrollable
        region, separate from the button row below. Previously the whole
        panel (including the buttons) was one flex column with no
        overflow handling — when the Config JSON was long enough that the
        textarea's `minHeight: 100` floor plus everything above it
        exceeded the fixed 320px panel height set by the parent
        (App.tsx), flexbox had nowhere to take that extra space from
        except squeezing the button row, which visually overlapped the
        textarea instead of scrolling below it. Giving this region its
        own `overflowY: auto` and the button row `flexShrink: 0` below
        means the buttons always keep their own reserved space and the
        long content scrolls independently, however long it gets.
      */}
      <div style={{ flex: 1, minHeight: 0, overflowY: "auto", display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.04em", color: "var(--ac-ink-muted, #6b7280)" }}>
            {node.type.replace(/_/g, " ")} · {node.category}
          </div>
          <code style={{ fontSize: 11, fontFamily: "var(--ac-font-mono, monospace)", color: "var(--ac-ink-muted, #6b7280)" }}>
            {node.id}
          </code>
        </div>
        {onClose && (
          <button onClick={onClose} style={{ border: "none", background: "transparent", cursor: "pointer", fontSize: 16, lineHeight: 1, color: "var(--ac-ink-muted, #6b7280)" }} aria-label="Close inspector">
            ×
          </button>
        )}
      </div>

      <label style={{ fontSize: 12, fontWeight: 600 }}>
        Label
        <input
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box" }}
        />
      </label>

      {(node.type === "ask_ai" || node.type === "ai_agent") && (
        <label style={{ fontSize: 12, fontWeight: 600 }}>
          Prompt / instructions
          <textarea
            value={typeof draftConfig.prompt === "string" ? draftConfig.prompt : ""}
            onChange={(e) => {
              const next = { ...draftConfig, prompt: e.target.value };
              setConfigText(JSON.stringify(next, null, 2));
              setConfigError(null);
            }}
            placeholder='e.g. "Summarize the spreadsheet. List top 3 vendors by amount."'
            rows={5}
            style={{ display: "block", width: "100%", marginTop: 4, padding: "6px 8px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box", fontFamily: "inherit", resize: "vertical", minHeight: 88 }}
          />
          <span style={{ display: "block", marginTop: 4, fontWeight: 400, color: "var(--ac-ink-muted, #6b7280)", fontSize: 11, lineHeight: 1.35 }}>
            Command the model follows. Upstream data (table_preview, file_contents, etc.) is attached automatically.
          </span>
        </label>
      )}

      {node.type === "ask_ai" && agentFiles.some((f) => [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"].includes(f.suffix)) && (
        <div style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 6 }}>
          <span style={{ fontWeight: 600 }}>Vision image (optional)</span>
          <select
            value={typeof draftConfig.image_path === "string" ? draftConfig.image_path : ""}
            onChange={(e) => {
              const next = { ...draftConfig };
              if (e.target.value) next.image_path = e.target.value;
              else delete next.image_path;
              setConfigText(JSON.stringify(next, null, 2));
              setConfigError(null);
            }}
            style={{ width: "100%", padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 12 }}
          >
            <option value="">None (use RAG images if any)</option>
            {agentFiles
              .filter((f) => [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"].includes(f.suffix))
              .map((f) => (
                <option key={f.path} value={f.path}>{f.path}</option>
              ))}
          </select>
        </div>
      )}

      {(node.type === "read_file" || node.type === "write_file") && (
        <div style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 6 }}>
          <span style={{ fontWeight: 600 }}>File path</span>
          {agentFiles.length > 0 && node.type === "read_file" && (
            <select
              value={typeof draftConfig.path === "string" && agentFiles.some((f) => f.path === draftConfig.path) ? String(draftConfig.path) : ""}
              onChange={(e) => {
                if (!e.target.value) return;
                const next = { ...draftConfig, path: e.target.value };
                setConfigText(JSON.stringify(next, null, 2));
                setConfigError(null);
              }}
              style={{ width: "100%", padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 12 }}
            >
              <option value="">Pick an uploaded file…</option>
              {agentFiles.map((f) => (
                <option key={f.path} value={f.path}>
                  {f.path} ({Math.max(1, Math.round(f.size / 1024))} KB)
                </option>
              ))}
            </select>
          )}
          <input
            value={typeof draftConfig.path === "string" ? draftConfig.path : ""}
            onChange={(e) => {
              const next = { ...draftConfig, path: e.target.value };
              setConfigText(JSON.stringify(next, null, 2));
              setConfigError(null);
            }}
            placeholder={node.type === "read_file" ? "or type path e.g. data.csv" : "e.g. output.txt"}
            style={{ display: "block", width: "100%", padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box", fontFamily: "var(--ac-font-mono, monospace)" }}
          />
          <span style={{ color: "var(--ac-ink-muted, #6b7280)", fontSize: 11 }}>
            {filesDir ? `Folder: ${filesDir}` : "Relative to agent_platform_files/"} — use header Upload, then pick here.
          </span>
        </div>
      )}

      {node.type === "write_file" && (
        <label style={{ fontSize: 12, fontWeight: 600 }}>
          Content variable
          <input
            value={typeof draftConfig.content_variable === "string" ? draftConfig.content_variable : "ai_response"}
            onChange={(e) => {
              const next = { ...draftConfig, content_variable: e.target.value };
              setConfigText(JSON.stringify(next, null, 2));
              setConfigError(null);
            }}
            placeholder="ai_response"
            style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box", fontFamily: "var(--ac-font-mono, monospace)" }}
          />
          <span style={{ display: "block", marginTop: 4, fontWeight: 400, color: "var(--ac-ink-muted, #6b7280)", fontSize: 11, lineHeight: 1.35 }}>
            Use <code>ai_response</code> after Ask AI so the written file is the model output, not the input file.
          </span>
        </label>
      )}

      {node.executionType && (
        <div style={{ fontSize: 12 }}>
          <span style={{ fontWeight: 600 }}>Execution: </span>
          <span
            style={{
              display: "inline-block",
              padding: "1px 8px",
              borderRadius: 10,
              fontSize: 11,
              background: node.executionType === "probabilistic" ? "var(--ac-ai-tint, #f1edff)" : "var(--ac-control-tint, #eef0f1)",
              color: node.executionType === "probabilistic" ? "var(--ac-ai-ink, #4a2fc7)" : "var(--ac-control-ink, #1d2124)",
            }}
          >
            {node.executionType}
            {node.confidenceThreshold != null && ` · min confidence ${Math.round(node.confidenceThreshold * 100)}%`}
          </span>
        </div>
      )}

      {node.toolRefs && node.toolRefs.length > 0 && (
        <div style={{ fontSize: 12 }}>
          <span style={{ fontWeight: 600 }}>Tools: </span>
          <code style={{ fontFamily: "var(--ac-font-mono, monospace)", fontSize: 11 }}>{node.toolRefs.join(", ")}</code>
        </div>
      )}

      {compatibleTools.length > 0 && node.type !== "read_file" && node.type !== "write_file" && node.type !== "code" && node.type !== "preview" && (
        <div style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 6 }}>
          <span style={{ fontWeight: 600 }}>Tool binding</span>
          <select value={binding?.toolId ?? ""} onChange={(e) => configureBinding(e.target.value)}>
            <option value="">⚠ Select a compatible tool</option>
            {compatibleTools.map((tool) => <option key={tool.id} value={tool.id}>{tool.name}</option>)}
          </select>
          {binding?.mode === "demo" && <span style={{ color: "#8a5b00" }}>Demo mode — no external operation is performed.</span>}
          {binding && compatibleTools.find((tool) => tool.id === binding.toolId)?.credentialRequired && (
            <label>Connection reference
              <input value={binding.credentialRef ?? ""} placeholder="e.g. gmail_primary" onChange={(e) => onMutate([{ kind: "update_tool", toolId: binding.id, patch: { credentialRef: e.target.value, mode: "connected" } }])} />
            </label>
          )}
        </div>
      )}

      {showProviderPicker && (
        <div style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 6 }}>
          <span style={{ fontWeight: 600 }}>AI provider</span>
          {!providerStatus ? (
            <span style={{ color: "var(--ac-ink-muted, #6b7280)" }}>Loading providers…</span>
          ) : (
            <>
              <select value={currentProvider} onChange={(e) => handleProviderChange(e.target.value)}>
                <option value="">Use server default ({providerStatus.current})</option>
                {providerStatus.providers.map((p) => (
                  <option key={p.id} value={p.id} disabled={p.status === "coming_soon" || !p.hasCredentials}>
                    {p.id.charAt(0).toUpperCase() + p.id.slice(1)}
                    {p.status === "coming_soon" ? " (coming soon)" : !p.hasCredentials ? " (no API key configured)" : ""}
                  </option>
                ))}
              </select>
              {currentProvider && (
                <select value={currentModel} onChange={(e) => updateProviderModel(currentProvider, e.target.value)}>
                  {(providerStatus.providers.find((p) => p.id === currentProvider)?.models ?? []).map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
                    </option>
                  ))}
                </select>
              )}
              {currentProvider && !providerStatus.providers.find((p) => p.id === currentProvider)?.hasCredentials && (
                <span style={{ color: "var(--ac-human-ink, #8a2e08)" }}>
                  No {currentProvider} API key is configured on the server — this node will fail clearly when run
                  (never silently fall back to another provider or a stub) until one is set.
                </span>
              )}
            </>
          )}
        </div>
      )}


      {node.type === "loop" && (
        <div style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 8 }}>
          <span style={{ fontWeight: 600 }}>For-each settings</span>
          <label>Items variable
            <input value={typeof draftConfig.items_variable === "string" ? draftConfig.items_variable : "ai_response"}
              onChange={(e) => { setConfigText(JSON.stringify({ ...draftConfig, items_variable: e.target.value }, null, 2)); setConfigError(null); }}
              style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box" }} />
          </label>
          <label>Items path (JSON key)
            <input value={typeof draftConfig.items_path === "string" ? draftConfig.items_path : "filtered_students"}
              onChange={(e) => { setConfigText(JSON.stringify({ ...draftConfig, items_path: e.target.value }, null, 2)); setConfigError(null); }}
              placeholder="filtered_students"
              style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box" }} />
          </label>
          <span style={{ color: "var(--ac-ink-muted, #6b7280)", fontSize: 11, lineHeight: 1.35 }}>
            Wire edges from this Loop: condition <code>_loop_continue == true</code> → body (Send Email), and <code>_loop_continue == false</code> → exit.
            Body should connect back to the Loop.
          </span>
        </div>
      )}

      {(node.type === "trigger_schedule") && (
        <div style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 8 }}>
          <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input
              type="checkbox"
              checked={draftConfig.enabled !== false}
              onChange={(e) => {
                setConfigText(JSON.stringify({ ...draftConfig, enabled: e.target.checked }, null, 2));
                setConfigError(null);
              }}
            />
            Enable automatic runs (server must stay up)
          </label>
          <label>Interval (minutes)
            <input
              type="number"
              min={1}
              value={typeof draftConfig.interval_minutes === "number" ? draftConfig.interval_minutes : 5}
              onChange={(e) => {
                setConfigText(JSON.stringify({ ...draftConfig, interval_minutes: Number(e.target.value) || 5 }, null, 2));
                setConfigError(null);
              }}
              style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box" }}
            />
          </label>
          <span style={{ color: "var(--ac-ink-muted, #6b7280)", fontSize: 11 }}>
            Backend polls about every 15s and runs this workflow when the interval elapses. Disable with env DISABLE_SCHEDULER=1.
          </span>
        </div>
      )}

      {node.type === "read_email" && (
        <div style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 8 }}>
          <label>Folder
            <input
              value={typeof draftConfig.folder === "string" ? draftConfig.folder : "INBOX"}
              onChange={(e) => { setConfigText(JSON.stringify({ ...draftConfig, folder: e.target.value }, null, 2)); setConfigError(null); }}
              style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box" }}
            />
          </label>
          <label>Query (UNSEEN / ALL / SEEN)
            <input
              value={typeof draftConfig.query === "string" ? draftConfig.query : "UNSEEN"}
              onChange={(e) => { setConfigText(JSON.stringify({ ...draftConfig, query: e.target.value }, null, 2)); setConfigError(null); }}
              style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box" }}
            />
          </label>
          <label>Limit
            <input
              type="number"
              min={1}
              value={typeof draftConfig.limit === "number" ? draftConfig.limit : 1}
              onChange={(e) => { setConfigText(JSON.stringify({ ...draftConfig, limit: Number(e.target.value) || 1 }, null, 2)); setConfigError(null); }}
              style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box" }}
            />
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input
              type="checkbox"
              checked={draftConfig.mark_as_read === true}
              onChange={(e) => {
                setConfigText(JSON.stringify({ ...draftConfig, mark_as_read: e.target.checked }, null, 2));
                setConfigError(null);
              }}
            />
            Mark as read after fetch (off by default — uses IMAP PEEK)
          </label>
        </div>
      )}

      {node.type === "preview" && (
        <div style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 8 }}>
          <label>Source variable
            <input
              value={typeof draftConfig.source === "string" ? draftConfig.source : "ai_response"}
              onChange={(e) => {
                setConfigText(JSON.stringify({ ...draftConfig, source: e.target.value }, null, 2));
                setConfigError(null);
              }}
              placeholder="ai_response"
              style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box" }}
            />
          </label>
          <span style={{ color: "var(--ac-ink-muted, #6b7280)", fontSize: 11 }}>
            Connect Ask AI (or other nodes) upstream, then Run. Markdown is rendered below.
          </span>
          {(() => {
            const srcKey = typeof draftConfig.source === "string" ? draftConfig.source : "ai_response";
            const fromRun =
              (runVariables && (runVariables.preview_content as string)) ||
              (runVariables && (runVariables.preview_markdown as string)) ||
              (runVariables && (runVariables[srcKey] as string)) ||
              (runVariables && (runVariables.ai_response as string)) ||
              "";
            const text = typeof fromRun === "string" ? fromRun : fromRun ? JSON.stringify(fromRun, null, 2) : "";
            if (!text) {
              return (
                <div style={{ padding: 10, borderRadius: 8, background: "#f9fafb", border: "1px solid #e5e7eb", color: "#6b7280" }}>
                  No preview yet — run the workflow so this node receives output.
                </div>
              );
            }
            return (
              <>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontWeight: 600 }}>Rendered output</span>
                  <button
                    type="button"
                    onClick={() => setPreviewOpen(true)}
                    style={{ padding: "4px 10px", borderRadius: 6, border: "1px solid #d8dad5", background: "#fff", fontSize: 11, cursor: "pointer" }}
                  >
                    Larger view
                  </button>
                </div>
                <div
                  style={{
                    padding: 12,
                    borderRadius: 8,
                    border: "1px solid #e5e7eb",
                    background: "#fff",
                    maxHeight: 280,
                    overflow: "auto",
                    fontSize: 13,
                    color: "#111827",
                  }}
                  dangerouslySetInnerHTML={{ __html: markdownToHtml(text) }}
                />
                {previewOpen && (
                  <div
                    style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.4)", zIndex: 80, display: "flex", alignItems: "center", justifyContent: "center" }}
                    onClick={() => setPreviewOpen(false)}
                  >
                    <div
                      style={{ background: "#fff", borderRadius: 12, width: "min(720px, 94vw)", maxHeight: "85vh", overflow: "auto", padding: 20, boxShadow: "0 16px 48px rgba(0,0,0,0.2)" }}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
                        <strong>{typeof draftConfig.title === "string" ? draftConfig.title : node.label}</strong>
                        <button type="button" onClick={() => setPreviewOpen(false)} style={{ border: "none", background: "transparent", cursor: "pointer", fontSize: 18 }}>×</button>
                      </div>
                      <div dangerouslySetInnerHTML={{ __html: markdownToHtml(text) }} style={{ fontSize: 14, lineHeight: 1.55 }} />
                    </div>
                  </div>
                )}
              </>
            );
          })()}
        </div>
      )}

      {node.type === "rag_index" && (
        <div style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 8 }}>
          <span style={{ fontWeight: 600 }}>Files to index</span>
          {agentFiles.length > 0 ? (
            <div style={{ maxHeight: 140, overflow: "auto", border: "1px solid var(--ac-border, #d8dad5)", borderRadius: 6, padding: 6 }}>
              {agentFiles.map((f) => {
                const selected = Array.isArray(draftConfig.paths) && (draftConfig.paths as string[]).includes(f.path);
                return (
                  <label key={f.path} style={{ display: "flex", gap: 8, alignItems: "center", padding: "3px 0", cursor: "pointer" }}>
                    <input
                      type="checkbox"
                      checked={!!selected}
                      onChange={() => {
                        const cur = Array.isArray(draftConfig.paths) ? [...(draftConfig.paths as string[])] : [];
                        const nextPaths = selected ? cur.filter((p) => p !== f.path) : [...cur, f.path];
                        setConfigText(JSON.stringify({ ...draftConfig, paths: nextPaths, index_name: draftConfig.index_name || "default" }, null, 2));
                        setConfigError(null);
                      }}
                    />
                    <span style={{ fontFamily: "var(--ac-font-mono, monospace)", fontSize: 11 }}>{f.path}</span>
                    <span style={{ color: "var(--ac-ink-muted, #6b7280)", fontSize: 10 }}>{Math.max(1, Math.round(f.size / 1024))} KB</span>
                  </label>
                );
              })}
            </div>
          ) : (
            <span style={{ color: "var(--ac-ink-muted, #6b7280)", fontSize: 11 }}>No files yet — use header Upload file.</span>
          )}
          <label>Or type paths (comma-separated)
            <input
              value={Array.isArray(draftConfig.paths) ? (draftConfig.paths as string[]).join(", ") : (typeof draftConfig.path === "string" ? draftConfig.path : "")}
              onChange={(e) => {
                const paths = e.target.value.split(",").map((s) => s.trim()).filter(Boolean);
                setConfigText(JSON.stringify({ ...draftConfig, paths, index_name: draftConfig.index_name || "default" }, null, 2));
                setConfigError(null);
              }}
              placeholder="handbook.pdf, photo.png"
              style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box" }}
            />
          </label>
          <label>Index name
            <input
              value={typeof draftConfig.index_name === "string" ? draftConfig.index_name : "default"}
              onChange={(e) => { setConfigText(JSON.stringify({ ...draftConfig, index_name: e.target.value }, null, 2)); setConfigError(null); }}
              style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box" }}
            />
          </label>
          <span style={{ color: "var(--ac-ink-muted, #6b7280)", fontSize: 11 }}>
            Upload files via the header Upload button first. Supports txt, md, csv, json, pdf, docx, xlsx.
          </span>
        </div>
      )}

      {node.type === "rag_query" && (
        <div style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 8 }}>
          <label>Query
            <textarea
              value={typeof draftConfig.query === "string" ? draftConfig.query : ""}
              onChange={(e) => { setConfigText(JSON.stringify({ ...draftConfig, query: e.target.value }, null, 2)); setConfigError(null); }}
              rows={3}
              placeholder="What is the refund policy?"
              style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box" }}
            />
          </label>
          <label>Index name
            <input
              value={typeof draftConfig.index_name === "string" ? draftConfig.index_name : "default"}
              onChange={(e) => { setConfigText(JSON.stringify({ ...draftConfig, index_name: e.target.value }, null, 2)); setConfigError(null); }}
              style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box" }}
            />
          </label>
          <label>Top K chunks
            <input
              type="number"
              value={typeof draftConfig.top_k === "number" ? draftConfig.top_k : 5}
              onChange={(e) => { setConfigText(JSON.stringify({ ...draftConfig, top_k: Number(e.target.value) || 5 }, null, 2)); setConfigError(null); }}
              style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box" }}
            />
          </label>
          <span style={{ color: "var(--ac-ink-muted, #6b7280)", fontSize: 11 }}>
            Output: <code>rag_context</code> / <code>rag_chunks</code> for Ask AI.
          </span>
        </div>
      )}

      {node.type === "api_call" && (
        <div style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 8 }}>
          <label>Method
            <select
              value={typeof draftConfig.method === "string" ? draftConfig.method : "GET"}
              onChange={(e) => { setConfigText(JSON.stringify({ ...draftConfig, method: e.target.value }, null, 2)); setConfigError(null); }}
              style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13 }}
            >
              {["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"].map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </label>
          <label>URL
            <input
              value={typeof draftConfig.url === "string" ? draftConfig.url : ""}
              onChange={(e) => { setConfigText(JSON.stringify({ ...draftConfig, url: e.target.value }, null, 2)); setConfigError(null); }}
              placeholder="https://api.example.com/items/{{id}}"
              style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box", fontFamily: "var(--ac-font-mono, monospace)" }}
            />
          </label>
          <label>Body (JSON or text, optional)
            <textarea
              value={typeof draftConfig.body === "string" ? draftConfig.body : (draftConfig.body != null ? JSON.stringify(draftConfig.body, null, 2) : "")}
              onChange={(e) => {
                let body: unknown = e.target.value;
                try { body = JSON.parse(e.target.value); } catch { /* keep string */ }
                setConfigText(JSON.stringify({ ...draftConfig, body }, null, 2));
                setConfigError(null);
              }}
              rows={4}
              placeholder='{"name": "{{Name}}"}'
              style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 12, boxSizing: "border-box", fontFamily: "var(--ac-font-mono, monospace)" }}
            />
          </label>
          <span style={{ color: "var(--ac-ink-muted, #6b7280)", fontSize: 11 }}>
            Use {"{{variable}}"} in URL/body. Response lands in <code>api_response</code> and <code>api_status</code>.
          </span>
        </div>
      )}

      {node.type === "code" && (
        <div style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 8 }}>
          <label>Python code
            <textarea
              value={typeof draftConfig.code === "string" ? draftConfig.code : ""}
              onChange={(e) => { setConfigText(JSON.stringify({ ...draftConfig, code: e.target.value, language: "python" }, null, 2)); setConfigError(null); }}
              rows={12}
              spellCheck={false}
              placeholder={'rows = input.get("table_preview") or []\nreturn {"row_count": len(rows)}'}
              style={{ display: "block", width: "100%", marginTop: 4, padding: "8px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 12, boxSizing: "border-box", fontFamily: "ui-monospace, monospace", lineHeight: 1.4 }}
            />
          </label>
          <span style={{ color: "var(--ac-ink-muted, #6b7280)", fontSize: 11, lineHeight: 1.35 }}>
            <code>input</code> is the workflow variable bag. <strong>return</strong> a dict to merge into state.
            No network/disk; allowed imports: json, math, re, datetime. Timeout 5s default.
          </span>
        </div>
      )}

      {node.type === "send_email" && (
        <div style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 8 }}>
          <label>To (recipient email)
            <input value={typeof draftConfig.to === "string" ? draftConfig.to : ""}
              onChange={(e) => { setConfigText(JSON.stringify({ ...draftConfig, to: e.target.value }, null, 2)); setConfigError(null); }}
              placeholder="recipient@example.com"
              style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box" }} />
          </label>
          <label>Subject
            <input value={typeof draftConfig.subject === "string" ? draftConfig.subject : ""}
              onChange={(e) => { setConfigText(JSON.stringify({ ...draftConfig, subject: e.target.value }, null, 2)); setConfigError(null); }}
              style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box" }} />
          </label>
          <label>Body template
            <textarea value={typeof draftConfig.body === "string" ? draftConfig.body : ""}
              onChange={(e) => { setConfigText(JSON.stringify({ ...draftConfig, body: e.target.value }, null, 2)); setConfigError(null); }}
              rows={4}
              placeholder="Leave empty to send the Ask AI / AI Agent output as the email body"
              style={{ display: "block", width: "100%", marginTop: 4, padding: "5px 7px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)", fontSize: 13, boxSizing: "border-box" }} />
          </label>
          <span style={{ color: "var(--ac-ink-muted, #6b7280)", fontSize: 11 }}>
            Use <code>{"{{current_item.Name}}"}</code>, <code>{"{{current_item.Email}}"}</code>, etc. Put a fixed address here for tests. Or leave empty if the CSV/AI draft contains an email (or use a Loop).
          </span>
        </div>
      )}

      <label style={{ fontSize: 12, fontWeight: 600, flex: 1, display: "flex", flexDirection: "column", minHeight: 100 }}>
        Config (JSON)
        <textarea
          value={configText}
          onChange={(e) => setConfigText(e.target.value)}
          style={{
            flex: 1,
            marginTop: 4,
            padding: "6px 8px",
            borderRadius: 5,
            border: `1px solid ${configError ? "var(--ac-human-base, #c2410c)" : "var(--ac-border, #d8dad5)"}`,
            fontSize: 12,
            fontFamily: "var(--ac-font-mono, monospace)",
            resize: "none",
            boxSizing: "border-box",
            minHeight: 100,
          }}
        />
        {configError && <span style={{ color: "var(--ac-human-ink, #8a2e08)", fontSize: 11, marginTop: 2 }}>{configError}</span>}
      </label>
      </div>

      <div style={{ display: "flex", gap: 8, flexShrink: 0 }}>
        <button
          onClick={handleSave}
          disabled={!dirty}
          style={{
            flex: 1,
            padding: "7px 0",
            borderRadius: 6,
            border: "none",
            background: dirty ? "var(--ac-tool-base, #1e6fd9)" : "var(--ac-border, #d8dad5)",
            color: "#fff",
            fontSize: 13,
            cursor: dirty ? "pointer" : "default",
          }}
        >
          Save changes
        </button>
        <button
          onClick={handleDelete}
          style={{
            padding: "7px 14px",
            borderRadius: 6,
            border: "1px solid var(--ac-human-base, #c2410c)",
            background: "transparent",
            color: "var(--ac-human-ink, #8a2e08)",
            fontSize: 13,
            cursor: "pointer",
          }}
        >
          Delete node
        </button>
      </div>
    </div>
  );
}
