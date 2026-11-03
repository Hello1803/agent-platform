import { useEffect, useMemo, useState } from "react";
import {
  AgentCanvas,
  AgentApiClient,
  ApiError,
  useAgentSession,
  useAgentRunStream,
  ChatPanel,
  TraceLog,
  ApprovalCard,
  NodePalette,
  NodeInspector,
  EdgeInspector,
  AiProviderSelector,
  WorkflowSelector,
  parseModelString,
  markdownToHtml,
} from "@agent-platform/canvas";
import demoSpecJson from "./data/demo-spec.json";
import type { AgentSpecification } from "@agent-platform/shared-types";

const DEMO_AGENT_ID = "agt_demo_invoice";

function getApiBaseUrl(): string {
  const params = new URLSearchParams(window.location.search);
  return params.get("api") ?? "http://localhost:8000";
}

/**
 * Distinguishes "the API was genuinely unreachable" (network failure —
 * fetch itself threw, e.g. connection refused, DNS failure, a CORS block)
 * from "the API responded, just with an error status" (ApiError — 401,
 * 403, 409, 422, 500, an Anthropic failure surfaced through /instructions,
 * etc.). Conflating these under one "Couldn't reach the API" message is
 * actively misleading for the second category, which needs a completely
 * different fix (check the error detail, not "is the server running").
 */
function describeBootstrapError(e: unknown): { message: string; kind: "network" | "api" } {
  if (e instanceof ApiError) {
    return { message: e.message, kind: "api" };
  }
  return { message: e instanceof Error ? e.message : String(e), kind: "network" };
}

type Mode = "build" | "test" | "use";

/**
 * The one page tying every previously-built, independently-tested piece
 * together: AgentCanvas + ChatPanel share ONE useAgentSession (so a chat
 * edit and a canvas drag can never show the user two different versions
 * of the agent — see useAgentSession.ts's doc comment), and a separate
 * useAgentRunStream drives Test/Use mode's live run view, including the
 * ApprovalCard for a paused human_approval node.
 *
 * Modes mirror doc section 14 (Build / Test / Use): Build shows the full
 * canvas + chat; Test adds a live trace overlay and run controls over the
 * SAME graph; Use hides the graph entirely — just a form and a result.
 */
export function App() {
  const client = useMemo(() => new AgentApiClient({ baseUrl: getApiBaseUrl() }), []);
  const [mode, setMode] = useState<Mode>("build");
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [showSmtpSetup, setShowSmtpSetup] = useState(false);
  const [showGmailOauthSetup, setShowGmailOauthSetup] = useState(false);
  const [gmailOauthForm, setGmailOauthForm] = useState({
    client_id: "",
    client_secret: "",
    redirect_uri: "http://localhost:8000/credentials/oauth/gmail/callback",
  });
  const [gmailOauthStatus, setGmailOauthStatus] = useState<string | null>(null);
  const [gmailCredId, setGmailCredId] = useState<string | null>(null);

  const [smtpForm, setSmtpForm] = useState({ email: "", password: "", host: "smtp.gmail.com", port: "587" });
  const [smtpStatus, setSmtpStatus] = useState<string | null>(null);
  const [smtpCredId, setSmtpCredId] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [agentFileList, setAgentFileList] = useState<Array<{ path: string; size: number }>>([]);
  const [showFilesPanel, setShowFilesPanel] = useState(false);



  // The bootstrap agent (created below if it doesn't exist) doubles as
  // BOTH the fixed demo workflow AND the Agent container every workflow
  // this app creates lives under — see WorkflowSelector.tsx's doc
  // comment. `selectedWorkflowId` is the one piece of UI state that
  // decides which workflow's real backend state useAgentSession/
  // useAgentRunStream load; switching it is the entire "switch
  // workflow" feature — those hooks already re-fetch on their own.
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string>(DEMO_AGENT_ID);
  const [bootstrapped, setBootstrapped] = useState(false);
  const [bootstrapError, setBootstrapError] = useState<string | null>(null);
  const [bootstrapErrorKind, setBootstrapErrorKind] = useState<"network" | "api">("network");

  // Self-bootstrapping for local dev: create the demo agent if it doesn't
  // exist yet. A real app would have a proper agent list/creation flow;
  // this just gets `npm run dev` to something on-screen with zero setup.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await client.getAgent(DEMO_AGENT_ID);
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) {
          try {
            const spec = { ...demoSpecJson, metadata: { ...demoSpecJson.metadata, id: DEMO_AGENT_ID } };
            await client.createAgent(spec as unknown as AgentSpecification);
          } catch (createErr) {
            if (!(createErr instanceof ApiError && createErr.status === 409)) {
              if (!cancelled) {
                const { message, kind } = describeBootstrapError(createErr);
                setBootstrapError(message);
                setBootstrapErrorKind(kind);
              }
              return;
            }
          }
        } else {
          if (!cancelled) {
            const { message, kind } = describeBootstrapError(e);
            setBootstrapError(message);
            setBootstrapErrorKind(kind);
          }
          return;
        }
      }
      if (!cancelled) setBootstrapped(true);
    })();
    return () => {
      cancelled = true;
    };
  }, [client]);

  const session = useAgentSession(client, selectedWorkflowId, bootstrapped);
  const runStream = useAgentRunStream(client, selectedWorkflowId);

  if (bootstrapError) {
    return (
      <Centered>
        {bootstrapErrorKind === "network" ? (
          <>
            <p>Couldn't reach the API at {getApiBaseUrl()}.</p>
            <p style={{ fontSize: 13, color: "#6b7280" }}>
              Start it with <code>PYTHONPATH=. uvicorn backend.api.app:app --reload</code>, or pass{" "}
              <code>?api=http://your-host:port</code> in the URL.
            </p>
          </>
        ) : (
          <>
            <p>The API at {getApiBaseUrl()} responded with an error while setting up the demo agent.</p>
            <p style={{ fontSize: 13, color: "#6b7280" }}>
              The server is reachable — this isn't a connectivity problem. Check the error detail below and the
              server's own logs.
            </p>
          </>
        )}
        <p style={{ fontSize: 12, color: "#9ca3af" }}>{bootstrapError}</p>
      </Centered>
    );
  }

  if (!bootstrapped || session.loading || !session.spec) {
    return <Centered>Loading…</Centered>;
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "10px 16px",
          borderBottom: "1px solid #d8dad5",
        }}
      >
        <div>
          <strong>{session.spec.metadata.name}</strong>
          <span style={{ marginLeft: 8, fontSize: 12, color: "#6b7280" }}>
            v{session.spec.metadata.version} · {session.spec.metadata.status}
          </span>
        </div>
        <WorkflowSelector
          client={client}
          agentId={session.spec.metadata.agentId ?? DEMO_AGENT_ID}
          selectedWorkflowId={selectedWorkflowId}
          onSelectWorkflow={(workflowId) => {
            setSelectedWorkflowId(workflowId);
            setSelectedNodeId(null); // node ids are workflow-specific — a stale selection would point at nothing (or the wrong thing) in the new workflow
            setSelectedEdgeId(null);
          }}
        />
        <AiProviderSelector client={client} spec={session.spec} onMutate={session.applyMutation} />
        <button
          onClick={() => { setSmtpStatus(null); setShowSmtpSetup(true); }}
          style={{ padding: "6px 12px", borderRadius: 6, border: "1px solid #d8dad5", background: "#fff", fontSize: 13, cursor: "pointer" }}
          title="Send mail using your email + app password (SMTP)"
        >
          Setup Email
        </button>
        <label
          style={{
            marginLeft: 8,
            padding: "6px 10px",
            borderRadius: 6,
            border: "1px solid var(--ac-border, #d8dad5)",
            background: "#fff",
            fontSize: 12,
            cursor: "pointer",
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          Upload file
          <input
            type="file"
            style={{ display: "none" }}
            onChange={async (e) => {
              const f = e.target.files?.[0];
              e.target.value = "";
              if (!f) return;
              setUploadStatus(`Uploading ${f.name}…`);
              try {
                const res = await client.uploadAgentFile(f);
                setUploadStatus(`Saved as ${res.path}`);
                try {
                  const listed = await client.listAgentFiles();
                  setAgentFileList((listed.files || []).map((x) => ({ path: x.path, size: x.size })));
                } catch { /* ignore */ }

              } catch (err) {
                setUploadStatus(err instanceof Error ? err.message : String(err));
              }
            }}
          />
        </label>
        {uploadStatus && (
          <span style={{ marginLeft: 8, fontSize: 11, color: "var(--ac-ink-muted, #6b7280)", maxWidth: 280 }} title={uploadStatus}>
            {uploadStatus}
          </span>
        )}
        <button
          type="button"
          onClick={async () => {
            setShowFilesPanel((v) => !v);
            try {
              const listed = await client.listAgentFiles();
              setAgentFileList((listed.files || []).map((x) => ({ path: x.path, size: x.size })));
            } catch (e) {
              setUploadStatus(e instanceof Error ? e.message : String(e));
            }
          }}
          style={{
            marginLeft: 8,
            padding: "6px 10px",
            borderRadius: 6,
            border: "1px solid var(--ac-border, #d8dad5)",
            background: showFilesPanel ? "#eef5ff" : "#fff",
            fontSize: 12,
            cursor: "pointer",
          }}
        >
          Files ({agentFileList.length || "…"})
        </button>


        <button
          onClick={async () => {
            setGmailOauthStatus(null);
            setShowGmailOauthSetup(true);
            try {
              const cfg = await client.getGmailOAuthConfig();
              setGmailOauthForm((prev) => ({
                client_id: cfg.client_id || prev.client_id,
                client_secret: prev.client_secret,
                redirect_uri: cfg.redirect_uri || prev.redirect_uri,
              }));
              if (cfg.configured) {
                setGmailOauthStatus("OAuth app is configured on the server. You can connect your Google account below.");
              }
            } catch {
              /* soft */
            }
          }}
          style={{ padding: "6px 12px", borderRadius: 6, border: "1px solid #d8dad5", background: "#fff", fontSize: 13, cursor: "pointer" }}
          title="Configure Google OAuth client + connect Gmail"
        >
          Connect Gmail
        </button>
        <nav style={{ display: "flex", gap: 4 }}>
          {(["build", "test", "use"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              style={{
                padding: "6px 14px",
                borderRadius: 6,
                border: "1px solid #d8dad5",
                background: mode === m ? "#1e6fd9" : "transparent",
                color: mode === m ? "#fff" : "#16181a",
                cursor: "pointer",
                textTransform: "capitalize",
                fontSize: 13,
              }}
            >
              {m}
            </button>
          ))}
        </nav>
      </header>
      {showFilesPanel && (
        <div style={{ padding: "10px 16px", borderBottom: "1px solid var(--ac-border, #d8dad5)", background: "#fafafa", fontSize: 12 }}>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>Agent files (click to copy path)</div>
          {agentFileList.length === 0 ? (
            <div style={{ color: "#6b7280" }}>No files yet. Use Upload file.</div>
          ) : (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {agentFileList.map((f) => (
                <button
                  key={f.path}
                  type="button"
                  title="Copy path"
                  onClick={() => {
                    void navigator.clipboard?.writeText(f.path);
                    setUploadStatus(`Copied: ${f.path}`);
                  }}
                  style={{
                    padding: "4px 10px",
                    borderRadius: 999,
                    border: "1px solid #d8dad5",
                    background: "#fff",
                    fontFamily: "ui-monospace, monospace",
                    fontSize: 11,
                    cursor: "pointer",
                  }}
                >
                  {f.path}
                  <span style={{ color: "#6b7280", marginLeft: 6 }}>{Math.max(1, Math.round(f.size / 1024))} KB</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}


      <main style={{ flex: 1, minHeight: 0 }}>
        {mode === "build" && <BuildMode spec={session.spec} session={session} selectedNodeId={selectedNodeId} onSelectNode={setSelectedNodeId} selectedEdgeId={selectedEdgeId} onSelectEdge={setSelectedEdgeId} client={client} runVariables={runStream.variables} />}
        {mode === "test" && (
          <TestMode
            spec={session.spec}
            runStream={runStream}
            client={client}
            onConfigureNode={(nodeId) => {
              setSelectedNodeId(nodeId);
              setMode("build");
            }}
          />
        )}
        {mode === "use" && <UseMode spec={session.spec} runStream={runStream} />}
      </main>

      {showSmtpSetup && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.35)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 50 }}>
          <div style={{ background: "#fff", borderRadius: 10, padding: 20, width: 400, maxWidth: "92vw", boxShadow: "0 8px 30px rgba(0,0,0,0.15)" }}>
            <h3 style={{ margin: "0 0 8px" }}>Setup Email (SMTP)</h3>
            <p style={{ fontSize: 13, color: "#6b7280", marginTop: 0 }}>
              Enter your email and an <strong>app password</strong> (for Gmail: Google Account → Security → App passwords).
              You can then send to anyone from Send Email nodes.
            </p>
            <label style={{ display: "block", fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
              Your email
              <input value={smtpForm.email} onChange={(e) => setSmtpForm({ ...smtpForm, email: e.target.value })}
                placeholder="you@gmail.com"
                style={{ display: "block", width: "100%", marginTop: 4, padding: 8, borderRadius: 6, border: "1px solid #d8dad5", boxSizing: "border-box" }} />
            </label>
            <label style={{ display: "block", fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
              App password
              <input type="password" value={smtpForm.password} onChange={(e) => setSmtpForm({ ...smtpForm, password: e.target.value })}
                placeholder="16-character app password"
                style={{ display: "block", width: "100%", marginTop: 4, padding: 8, borderRadius: 6, border: "1px solid #d8dad5", boxSizing: "border-box" }} />
            </label>
            <label style={{ display: "block", fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
              SMTP host
              <input value={smtpForm.host} onChange={(e) => setSmtpForm({ ...smtpForm, host: e.target.value })}
                style={{ display: "block", width: "100%", marginTop: 4, padding: 8, borderRadius: 6, border: "1px solid #d8dad5", boxSizing: "border-box" }} />
            </label>
            <label style={{ display: "block", fontSize: 12, fontWeight: 600, marginBottom: 12 }}>
              Port
              <input value={smtpForm.port} onChange={(e) => setSmtpForm({ ...smtpForm, port: e.target.value })}
                style={{ display: "block", width: "100%", marginTop: 4, padding: 8, borderRadius: 6, border: "1px solid #d8dad5", boxSizing: "border-box" }} />
            </label>
            {smtpCredId && (
              <div style={{ marginBottom: 12, padding: 10, borderRadius: 8, background: "#ecfdf5", border: "1px solid #a7f3d0" }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#065f46", marginBottom: 6 }}>Credential ID</div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <code style={{ flex: 1, fontSize: 13, wordBreak: "break-all", color: "#064e3b" }}>{smtpCredId}</code>
                  <button
                    type="button"
                    onClick={() => {
                      void navigator.clipboard?.writeText(smtpCredId);
                      setSmtpStatus((s) => (s && s.startsWith("Saved") ? s : "Copied credential id."));
                    }}
                    style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid #6ee7b7", background: "#fff", fontSize: 12, cursor: "pointer", whiteSpace: "nowrap" }}
                  >
                    Copy
                  </button>
                </div>
                <div style={{ fontSize: 11, color: "#047857", marginTop: 6 }}>
                  Paste this into Send Email / Read Email → Connection reference (tool mode = connected).
                </div>
              </div>
            )}
            {smtpStatus && <p style={{ fontSize: 13, color: smtpStatus.startsWith("Saved") || smtpStatus.startsWith("Copied") ? "#0f7b3a" : "#b91c1c", whiteSpace: "pre-wrap" as const }}>{smtpStatus}</p>}
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button onClick={() => setShowSmtpSetup(false)} style={{ padding: "8px 14px", borderRadius: 6, border: "1px solid #d8dad5", background: "#fff", cursor: "pointer" }}>Cancel</button>
              <button
                onClick={async () => {
                  try {
                    const result = await client.createSmtpCredential({
                      email: smtpForm.email,
                      password: smtpForm.password,
                      host: smtpForm.host,
                      port: Number(smtpForm.port) || 587,
                    });
                    setSmtpCredId(result.id);
                    setSmtpStatus(
                      "Saved. Credential id is shown above — use Copy, then paste into Send/Read Email connection reference. Saving again updates the same id."
                    );
                  } catch (e) {
                    setSmtpStatus(e instanceof Error ? e.message : String(e));
                  }
                }}
                style={{ padding: "8px 14px", borderRadius: 6, border: "none", background: "#1e6fd9", color: "#fff", cursor: "pointer" }}
              >
                {smtpCredId ? "Update" : "Save"}
              </button>
            </div>
          </div>
        </div>
      )}

      {showGmailOauthSetup && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.35)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 50 }} onClick={() => setShowGmailOauthSetup(false)}>
          <div style={{ background: "#fff", borderRadius: 12, padding: 20, width: "min(480px, 94vw)", maxHeight: "90vh", overflow: "auto", boxShadow: "0 12px 40px rgba(0,0,0,0.18)" }} onClick={(e) => e.stopPropagation()}>
            <h3 style={{ margin: "0 0 8px" }}>Connect Gmail (OAuth)</h3>
            <p style={{ fontSize: 12, color: "#4b5563", marginTop: 0, lineHeight: 1.45 }}>
              1) In Google Cloud Console create an OAuth <strong>Desktop</strong> or <strong>Web</strong> client.
              2) Add authorized redirect URI exactly as below.
              3) Paste Client ID + Secret here, Save, then Connect Google account.
            </p>
            <label style={{ display: "block", fontSize: 12, fontWeight: 600, marginBottom: 12 }}>
              Client ID
              <input
                value={gmailOauthForm.client_id}
                onChange={(e) => setGmailOauthForm({ ...gmailOauthForm, client_id: e.target.value })}
                placeholder="xxxx.apps.googleusercontent.com"
                style={{ display: "block", width: "100%", marginTop: 4, padding: 8, borderRadius: 6, border: "1px solid #d8dad5", boxSizing: "border-box" }}
              />
            </label>
            <label style={{ display: "block", fontSize: 12, fontWeight: 600, marginBottom: 12 }}>
              Client secret
              <input
                type="password"
                value={gmailOauthForm.client_secret}
                onChange={(e) => setGmailOauthForm({ ...gmailOauthForm, client_secret: e.target.value })}
                placeholder={gmailOauthForm.client_secret ? "" : "•••••••• (leave blank to keep existing)"}
                style={{ display: "block", width: "100%", marginTop: 4, padding: 8, borderRadius: 6, border: "1px solid #d8dad5", boxSizing: "border-box" }}
              />
            </label>
            <label style={{ display: "block", fontSize: 12, fontWeight: 600, marginBottom: 12 }}>
              Redirect URI
              <div style={{ display: "flex", gap: 8, marginTop: 4 }}>
                <input
                  value={gmailOauthForm.redirect_uri}
                  onChange={(e) => setGmailOauthForm({ ...gmailOauthForm, redirect_uri: e.target.value })}
                  style={{ flex: 1, padding: 8, borderRadius: 6, border: "1px solid #d8dad5", boxSizing: "border-box", fontSize: 12 }}
                />
                <button
                  type="button"
                  onClick={() => {
                    void navigator.clipboard?.writeText(gmailOauthForm.redirect_uri);
                    setGmailOauthStatus("Redirect URI copied.");
                  }}
                  style={{ padding: "8px 10px", borderRadius: 6, border: "1px solid #d8dad5", background: "#fff", fontSize: 12, cursor: "pointer" }}
                >
                  Copy
                </button>
              </div>
            </label>
            {gmailCredId && (
              <div style={{ marginBottom: 12, padding: 10, borderRadius: 8, background: "#ecfdf5", border: "1px solid #a7f3d0" }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#065f46", marginBottom: 6 }}>Gmail credential ID</div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <code style={{ flex: 1, fontSize: 13, wordBreak: "break-all" }}>{gmailCredId}</code>
                  <button type="button" onClick={() => void navigator.clipboard?.writeText(gmailCredId)} style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid #6ee7b7", background: "#fff", fontSize: 12, cursor: "pointer" }}>Copy</button>
                </div>
              </div>
            )}
            {gmailOauthStatus && (
              <p style={{ fontSize: 13, color: gmailOauthStatus.toLowerCase().includes("fail") || gmailOauthStatus.toLowerCase().includes("error") ? "#b91c1c" : "#0f7b3a", whiteSpace: "pre-wrap" }}>{gmailOauthStatus}</p>
            )}
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", flexWrap: "wrap" }}>
              <button onClick={() => setShowGmailOauthSetup(false)} style={{ padding: "8px 14px", borderRadius: 6, border: "1px solid #d8dad5", background: "#fff", cursor: "pointer" }}>Close</button>
              <button
                onClick={async () => {
                  try {
                    let secret = gmailOauthForm.client_secret;
                    if (!secret) {
                      const cfg = await client.getGmailOAuthConfig();
                      if (!cfg.client_secret_set) {
                        setGmailOauthStatus("Client secret is required the first time.");
                        return;
                      }
                      // Re-save with empty secret not allowed — user must paste again if rotating
                      setGmailOauthStatus("Enter client secret to save (required when updating).");
                      return;
                    }
                    await client.saveGmailOAuthConfig({
                      client_id: gmailOauthForm.client_id,
                      client_secret: secret,
                      redirect_uri: gmailOauthForm.redirect_uri,
                    });
                    setGmailOauthStatus("OAuth app saved on the server. Click Connect Google account next.");
                  } catch (e) {
                    setGmailOauthStatus(e instanceof Error ? e.message : String(e));
                  }
                }}
                style={{ padding: "8px 14px", borderRadius: 6, border: "1px solid #d8dad5", background: "#fff", cursor: "pointer" }}
              >
                Save OAuth app
              </button>
              <button
                onClick={async () => {
                  try {
                    if (gmailOauthForm.client_id && gmailOauthForm.client_secret) {
                      await client.saveGmailOAuthConfig({
                        client_id: gmailOauthForm.client_id,
                        client_secret: gmailOauthForm.client_secret,
                        redirect_uri: gmailOauthForm.redirect_uri,
                      });
                    }
                    const { authorize_url } = await client.startGmailOAuth("Gmail");
                    window.open(authorize_url, "_blank", "noopener,width=520,height=720");
                    setGmailOauthStatus(
                      "Google consent opened. After you approve, the callback page will confirm. Then use List credentials or the Gmail tool binding id on Send/Read Email."
                    );
                  } catch (e) {
                    setGmailOauthStatus(e instanceof Error ? e.message : String(e));
                  }
                }}
                style={{ padding: "8px 14px", borderRadius: 6, border: "none", background: "#1e6fd9", color: "#fff", cursor: "pointer" }}
              >
                Connect Google account
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function BuildMode({
  spec,
  session,
  selectedNodeId,
  onSelectNode,
  selectedEdgeId,
  onSelectEdge,
  client,
  runVariables,
}: {
  spec: AgentSpecification;
  session: ReturnType<typeof useAgentSession>;
  selectedNodeId: string | null;
  onSelectNode: (nodeId: string | null) => void;
  selectedEdgeId: string | null;
  onSelectEdge: (edgeId: string | null) => void;
  client: AgentApiClient;
  runVariables?: Record<string, unknown> | null;
}) {
  const selectedNode = selectedNodeId ? spec.nodes.find((n) => n.id === selectedNodeId) ?? null : null;
  const selectedEdge = selectedEdgeId ? spec.edges.find((e) => e.id === selectedEdgeId) ?? null : null;
  const edgeSourceType = selectedEdge ? spec.nodes.find((n) => n.id === selectedEdge.source)?.type ?? null : null;
  const [runtimeProvider, runtimeModel] = parseModelString(spec.runtimeConfig?.model);

  useEffect(() => {
    if (selectedNodeId && !selectedNode) onSelectNode(null);
  }, [selectedNodeId, selectedNode, onSelectNode]);
  useEffect(() => {
    if (selectedEdgeId && !selectedEdge) onSelectEdge(null);
  }, [selectedEdgeId, selectedEdge, onSelectEdge]);

  return (
    <div style={{ display: "flex", height: "100%" }}>
      <div style={{ width: 200, flexShrink: 0, borderRight: "1px solid #d8dad5" }}>
        <NodePalette />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <AgentCanvas
          spec={spec}
          onMutation={session.applyMutation}
          resyncNonce={session.resyncNonce}
          onSelectNode={onSelectNode}
          onSelectEdge={onSelectEdge}
        />
      </div>
      <div style={{ width: 340, flexShrink: 0, borderLeft: "1px solid #d8dad5", display: "flex", flexDirection: "column" }}>
        {selectedNode && (
          <div style={{ height: 360, flexShrink: 0, borderBottom: "1px solid #d8dad5", overflowY: "auto" }}>
            <NodeInspector node={selectedNode} tools={spec.tools} onMutate={session.applyMutation} onClose={() => onSelectNode(null)} client={client} runVariables={runVariables} />
          </div>
        )}
        {selectedEdge && !selectedNode && (
          <div style={{ height: 320, flexShrink: 0, borderBottom: "1px solid #d8dad5", overflowY: "auto" }}>
            <EdgeInspector edge={selectedEdge} sourceNodeType={edgeSourceType} onMutate={session.applyMutation} onClose={() => onSelectEdge(null)} />
          </div>
        )}
        <div style={{ flex: 1, minHeight: 0, padding: 12, boxSizing: "border-box" }}>
          <ChatPanel
            messages={session.messages}
            onSend={session.proposeInstruction}
            onApply={session.applyProposal}
            onCancel={session.cancelProposal}
            busy={session.chatBusy}
            client={client}
            defaultProvider={runtimeProvider}
            defaultModel={runtimeModel}
          />
        </div>
      </div>
    </div>
  );
}

function TestMode({
  spec,
  runStream,
  client,
  onConfigureNode,
}: {
  spec: AgentSpecification;
  runStream: ReturnType<typeof useAgentRunStream>;
  client: AgentApiClient;
  onConfigureNode: (nodeId: string) => void;
}) {
  const [previewOpen, setPreviewOpen] = useState(false);
  const previewText =
    typeof runStream.variables?.preview_content === "string"
      ? String(runStream.variables.preview_content)
      : typeof runStream.variables?.ai_response === "string"
        ? String(runStream.variables.ai_response)
        : "";
  return (
    <div style={{ display: "flex", height: "100%" }}>
      <div style={{ flex: 1, minWidth: 0, position: "relative" }}>
        <AgentCanvas
          spec={spec}
          onMutation={() => {}}
          readOnly
          trace={runStream.trace}
          activeNodeId={runStream.activeNodeId}
        />
      </div>
      <div style={{ width: 340, flexShrink: 0, padding: 12, boxSizing: "border-box", display: "flex", flexDirection: "column", gap: 12, overflowY: "auto" }}>
        <CompileStatusBanner spec={spec} client={client} onConfigureNode={onConfigureNode} />
        <RunControls runStream={runStream} />
        {runStream.status === "failed" && <RunErrorBanner error={runStream.error} />}
        {runStream.status === "paused" && <ApprovalGate runStream={runStream} />}
        <div style={{ flex: 1, minHeight: 0, overflowY: "auto", border: "1px solid #d8dad5", borderRadius: 8 }}>
          <TraceLog trace={runStream.trace} selectedNodeId={runStream.activeNodeId} />
        </div>
        {previewText && (
          <div style={{ border: "1px solid #d8dad5", borderRadius: 8, padding: 12, maxHeight: 220, overflow: "auto", background: "#fff" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div style={{ fontWeight: 600, fontSize: 12 }}>Preview</div>
              <button type="button" onClick={() => setPreviewOpen(true)} style={{ padding: "4px 10px", borderRadius: 6, border: "1px solid #d8dad5", background: "#fff", fontSize: 11, cursor: "pointer" }}>
                Larger view
              </button>
            </div>
            <div style={{ fontSize: 13, lineHeight: 1.5 }} dangerouslySetInnerHTML={{ __html: markdownToHtml(previewText) }} />
          </div>
        )}
        {previewOpen && previewText && (
          <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.4)", zIndex: 90, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setPreviewOpen(false)}>
            <div style={{ background: "#fff", borderRadius: 12, width: "min(720px, 94vw)", maxHeight: "85vh", overflow: "auto", padding: 20 }} onClick={(e) => e.stopPropagation()}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
                <strong>Preview</strong>
                <button type="button" onClick={() => setPreviewOpen(false)} style={{ border: "none", background: "transparent", cursor: "pointer", fontSize: 18 }}>×</button>
              </div>
              <div style={{ fontSize: 14, lineHeight: 1.55 }} dangerouslySetInnerHTML={{ __html: markdownToHtml(previewText) }} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Proactively runs a compile check whenever the spec's version changes,
 * so a broken tool binding (or any other compile-time issue) surfaces
 * BEFORE the user hits Run and gets a less specific runtime failure — the
 * direct fix for doc section 22/27's ask: turn "422 Spec does not
 * compile" into "Search Web is missing a required tool binding.
 * [Configure Search Web]".
 */
function CompileStatusBanner({
  spec,
  client,
  onConfigureNode,
}: {
  spec: AgentSpecification;
  client: AgentApiClient;
  onConfigureNode: (nodeId: string) => void;
}) {
  const [result, setResult] = useState<{ ok: boolean; error?: string | null; nodeId?: string | null } | null>(null);

  useEffect(() => {
    let cancelled = false;
    client.compileCheck(spec.metadata.id).then((r) => {
      if (!cancelled) setResult(r);
    }).catch(() => {
      if (!cancelled) setResult(null); // don't block Test mode on the check itself failing
    });
    return () => {
      cancelled = true;
    };
  }, [client, spec.metadata.id, spec.metadata.version]);

  if (!result || result.ok) return null;

  const node = result.nodeId ? spec.nodes.find((n) => n.id === result.nodeId) : undefined;

  return (
    <div
      style={{
        padding: "10px 12px",
        borderRadius: 8,
        background: "var(--ac-human-tint, #fbeee6)",
        border: "1px solid var(--ac-human-base, #c2410c)",
        color: "var(--ac-human-ink, #8a2e08)",
        fontSize: 13,
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 4 }}>This agent can't run yet.</div>
      <div style={{ marginBottom: node ? 8 : 0 }}>{result.error}</div>
      {node && (
        <button
          onClick={() => onConfigureNode(node.id)}
          style={{
            padding: "5px 12px",
            borderRadius: 6,
            border: "1px solid var(--ac-human-base, #c2410c)",
            background: "transparent",
            color: "var(--ac-human-ink, #8a2e08)",
            fontSize: 12,
            cursor: "pointer",
          }}
        >
          Configure {node.label}
        </button>
      )}
    </div>
  );
}

function UseMode({ spec, runStream }: { spec: AgentSpecification; runStream: ReturnType<typeof useAgentRunStream> }) {
  // "Use" mode hides the graph — goal, run, and a human-friendly result.
  const [showRaw, setShowRaw] = useState(false);
  const [previewOpen, setPreviewOpen] = useState(false);
  const vars = runStream.variables || {};
  const mainText =
    (typeof vars.preview_content === "string" && vars.preview_content) ||
    (typeof vars.ai_response === "string" && vars.ai_response) ||
    (typeof vars.draft_email === "string" && vars.draft_email) ||
    (typeof vars.summary === "string" && vars.summary) ||
    "";

  return (
    <Centered>
      <div style={{ maxWidth: 560, textAlign: "center", width: "100%" }}>
        <h2 style={{ marginBottom: 4 }}>{spec.metadata.name}</h2>
        <p style={{ color: "#6b7280", fontSize: 14 }}>{spec.goal}</p>
        <RunControls runStream={runStream} />
        {runStream.status === "failed" && <RunErrorBanner error={runStream.error} />}
        {runStream.status === "paused" && (
          <div style={{ marginTop: 16, display: "flex", justifyContent: "center" }}>
            <ApprovalGate runStream={runStream} />
          </div>
        )}
        {runStream.status === "completed" && (
          <div style={{ marginTop: 16, textAlign: "left" }}>
            {mainText ? (
              <>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <strong style={{ fontSize: 13 }}>Result</strong>
                  <div style={{ display: "flex", gap: 8 }}>
                    <button type="button" onClick={() => setPreviewOpen(true)} style={{ padding: "4px 10px", borderRadius: 6, border: "1px solid #d8dad5", background: "#fff", fontSize: 11, cursor: "pointer" }}>
                      Larger view
                    </button>
                    <button type="button" onClick={() => setShowRaw((v) => !v)} style={{ padding: "4px 10px", borderRadius: 6, border: "1px solid #d8dad5", background: "#fff", fontSize: 11, cursor: "pointer" }}>
                      {showRaw ? "Hide raw data" : "Show raw data"}
                    </button>
                  </div>
                </div>
                <div
                  style={{ background: "#f6f7f5", padding: 14, borderRadius: 8, fontSize: 14, lineHeight: 1.55 }}
                  dangerouslySetInnerHTML={{ __html: markdownToHtml(mainText) }}
                />
              </>
            ) : (
              <p style={{ color: "#6b7280", fontSize: 13 }}>Run finished — no summary text. Open raw data if needed.</p>
            )}
            {(showRaw || !mainText) && (
              <pre style={{ marginTop: 12, textAlign: "left", background: "#f6f7f5", padding: 12, borderRadius: 8, fontSize: 11, overflowX: "auto", maxHeight: 280 }}>
                {JSON.stringify(vars, null, 2)}
              </pre>
            )}
            {previewOpen && mainText && (
              <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.4)", zIndex: 90, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setPreviewOpen(false)}>
                <div style={{ background: "#fff", borderRadius: 12, width: "min(720px, 94vw)", maxHeight: "85vh", overflow: "auto", padding: 20 }} onClick={(e) => e.stopPropagation()}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
                    <strong>Result</strong>
                    <button type="button" onClick={() => setPreviewOpen(false)} style={{ border: "none", background: "transparent", cursor: "pointer", fontSize: 18 }}>×</button>
                  </div>
                  <div style={{ fontSize: 15, lineHeight: 1.55 }} dangerouslySetInnerHTML={{ __html: markdownToHtml(mainText) }} />
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </Centered>
  );
}

function RunErrorBanner({ error }: { error: string | null }) {
  return (
    <div
      style={{
        marginTop: 12,
        padding: "10px 14px",
        borderRadius: 8,
        background: "var(--ac-human-tint, #fbeee6)",
        color: "var(--ac-human-ink, #8a2e08)",
        border: "1px solid var(--ac-human-base, #c2410c)",
        fontSize: 13,
        textAlign: "left",
      }}
    >
      <strong>Run failed.</strong> {error ?? "No further details were provided."}
    </div>
  );
}

/**
 * Wraps ApprovalCard with its OWN in-flight tracking, rather than trying
 * to infer "is a resume request pending" from runStream.status. Within a
 * single render, status is a fixed literal value — TypeScript correctly
 * flags `status === "paused" && busy={status === "running"}` as
 * comparing two types that can't overlap in that scope, because they
 * genuinely can't within one render. What actually needs tracking is "did
 * the user click Approve/Reject and is that specific request still in
 * flight," which is local state independent of (and updated slightly
 * ahead of) the parent hook's status.
 */
function ApprovalGate({ runStream }: { runStream: ReturnType<typeof useAgentRunStream> }) {
  const [resuming, setResuming] = useState(false);
  if (!runStream.interrupt) return null;

  const handle = async (approved: boolean, reason?: string) => {
    setResuming(true);
    try {
      await runStream.resume({ approved, reason });
    } finally {
      setResuming(false);
    }
  };

  return (
    <ApprovalCard
      interrupt={runStream.interrupt}
      busy={resuming}
      onApprove={(reason) => void handle(true, reason)}
      onReject={(reason) => void handle(false, reason)}
    />
  );
}

function RunControls({ runStream }: { runStream: ReturnType<typeof useAgentRunStream> }) {
  const isBusy = runStream.status === "running";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, justifyContent: "center", marginTop: 12 }}>
      <button
        onClick={() => {
          runStream.reset();
          void runStream.start({});
        }}
        disabled={isBusy}
        style={{
          padding: "8px 20px",
          borderRadius: 6,
          border: "none",
          background: "#1e6fd9",
          color: "#fff",
          cursor: isBusy ? "default" : "pointer",
          opacity: isBusy ? 0.6 : 1,
        }}
      >
        {isBusy ? "Running…" : "Run Agent"}
      </button>
      <span style={{ fontSize: 12, color: "#6b7280", textTransform: "capitalize" }}>{runStream.status}</span>
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", flexDirection: "column", gap: 4 }}>
      {children}
    </div>
  );
}
