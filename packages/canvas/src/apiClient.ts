/**
 * Typed client for the backend API (backend/api/app.py). Every method
 * signature mirrors a route's actual request/response schema
 * (backend/api/schemas.py) — kept in sync by hand for now, same as the
 * other cross-language boundaries in this project (see the schema
 * cross-check pattern in backend/validation/check_examples.py).
 *
 * This is the piece that turns AgentCanvas's `onMutation` prop into a
 * real network call: see useAgentSession.ts for the wiring.
 */

import type { AgentSpecification, AgentSpecMutation, MutationResult } from "@agent-platform/shared-types";
import { streamSSE } from "./sseClient";

export interface ApiClientOptions {
  baseUrl: string;
  fetchImpl?: typeof fetch;
}

export interface InstructionResult {
  reasoning: string;
  attempts: number;
  result: MutationResult;
}

export interface WorkflowSummary {
  workflowId: string;
  agentId: string;
  name: string;
  description?: string | null;
  enabled: boolean;
  status: string;
  version: string;
  publishedVersion?: string | null;
  updatedAt: string;
}

export interface ChatProposal {
  reasoning: string;
  /** The raw mutation objects — pass back verbatim to applyChatProposal(). */
  mutations: Array<Record<string, unknown>>;
  ok: boolean;
  errors: Array<Record<string, unknown>>;
  attempts: number;
  baseVersion: string;
  provider: string;
  model?: string | null;
  /** A full preview of the spec as it WOULD look if applied — null when `ok` is false. */
  previewSpec: AgentSpecification | null;
}

export interface CompileCheckResult {
  ok: boolean;
  error?: string | null;
  /** The node the error is about, if it's about a specific node — lets the caller offer a "Configure X" action instead of just showing raw text. */
  nodeId?: string | null;
}

export interface RunResult {
  threadId: string;
  status: "completed" | "paused" | "failed";
  trace: Array<Record<string, unknown>>;
  variables: Record<string, unknown>;
  interrupt?: Record<string, unknown> | null;
  error?: string | null;
  /** The node whose executor raised, when status === "failed". Backed by
   * LangGraph's own checkpointed `state.next` (backend/api/app.py's
   * _partial_result_after_failure) — not a guess. `trace`/`variables`
   * above are ALSO preserved on failure now (every successful node's
   * output up to the failure), not reset to empty. */
  failedNodeId?: string | null;
}

export interface PublishResult {
  spec: AgentSpecification;
  publishedVersion: string;
}

export interface ToolRegistryEntry {
  id: string; name: string; description: string; category: string;
  node_types: string[]; permissions: string[]; credential_required: boolean;
  execution: "demo" | "connected";
}

export interface AiModelOption {
  id: string;
  label: string;
}

export interface AiProviderStatus {
  id: string;
  status: "available" | "coming_soon";
  hasCredentials: boolean;
  models: AiModelOption[];
}

export interface AiProvidersResponse {
  current: string;
  providers: AiProviderStatus[];
}

/** Thrown for any non-2xx response. `status` lets callers distinguish a
 * version conflict (409 — re-fetch and retry) from a validation failure
 * (422 — the mutation itself was wrong) from anything else. */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: unknown,
    message?: string
  ) {
    super(message ?? `API error ${status}: ${JSON.stringify(body)}`);
    this.name = "ApiError";
  }

  get isVersionConflict(): boolean {
    return this.status === 409;
  }

  get isValidationError(): boolean {
    return this.status === 422;
  }

  /** True for the specific "AI Builder unavailable, no API key configured" case (backend/api/app.py's get_anthropic_client dependency) — lets callers show a clean, actionable message instead of generic error text. */
  get isAiUnavailable(): boolean {
    return this.status === 503 && (this.body as any)?.reason === "no_api_key";
  }
}

export class AgentApiClient {
  private readonly baseUrl: string;
  private readonly fetchImpl: typeof fetch;

  constructor(options: ApiClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch.bind(globalThis);
  }

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
      method,
      headers: body !== undefined ? { "content-type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });

    const text = await response.text();
    const parsed = text ? JSON.parse(text) : undefined;

    if (!response.ok) {
      throw new ApiError(response.status, parsed?.detail ?? parsed);
    }
    return parsed as T;
  }

  createAgent(spec: AgentSpecification): Promise<AgentSpecification> {
    return this.request("POST", "/agents", { spec });
  }

  getAgent(agentId: string): Promise<AgentSpecification> {
    return this.request("GET", `/agents/${agentId}`);
  }

  listAgents(): Promise<AgentSpecification[]> {
    return this.request("GET", "/agents");
  }

  listVersions(agentId: string): Promise<AgentSpecification[]> {
    return this.request("GET", `/agents/${agentId}/versions`);
  }

  getPublishedAgent(agentId: string): Promise<AgentSpecification> {
    return this.request("GET", `/agents/${agentId}/published`);
  }

  getToolRegistry(): Promise<ToolRegistryEntry[]> {
    return this.request("GET", "/tools/registry");
  }

  /** Non-secret provider status only -- see backend/api/app.py's
   * /ai/providers route docstring for exactly what "non-secret" means
   * here (booleans, never key values). */
  getAiProviders(): Promise<AiProvidersResponse> {
    return this.request("GET", "/ai/providers");
  }

  createSmtpCredential(body: {
    email: string;
    password: string;
    host?: string;
    port?: number;
    name?: string;
    from_email?: string;
  }): Promise<{ id: string; provider: string; name: string; status: string }> {
    return this.request("POST", "/credentials/smtp", body);
  }

  startGmailOAuth(name = "Gmail"): Promise<{ authorize_url: string; state: string; redirect_uri: string }> {
    return this.request("GET", `/credentials/oauth/gmail/start?name=${encodeURIComponent(name)}`);
  }

  getGmailOAuthConfig(): Promise<{
    client_id: string;
    client_secret_set: boolean;
    redirect_uri: string;
    configured: boolean;
  }> {
    return this.request("GET", "/credentials/oauth/gmail/config");
  }

  saveGmailOAuthConfig(body: {
    client_id: string;
    client_secret: string;
    redirect_uri?: string;
  }): Promise<{ ok: boolean; client_id: string; redirect_uri: string; configured: boolean }> {
    return this.request("POST", "/credentials/oauth/gmail/config", body);
  }

  listCredentials(): Promise<Array<{ id: string; provider: string; name: string; status: string }>> {
    return this.request("GET", "/credentials");
  }

  applyMutations(
    agentId: string,
    mutations: AgentSpecMutation[],
    expectedVersion?: string
  ): Promise<MutationResult> {
    return this.request("POST", `/agents/${agentId}/mutations`, {
      mutations,
      expected_version: expectedVersion,
    });
  }

  sendInstruction(agentId: string, message: string): Promise<InstructionResult> {
    return this.request("POST", `/agents/${agentId}/instructions`, { message });
  }

  // ---- Multi-workflow ----------------------------------------------

  listWorkflows(agentId: string): Promise<WorkflowSummary[]> {
    return this.request<
      Array<{
        workflow_id: string;
        agent_id: string;
        name: string;
        description?: string | null;
        enabled: boolean;
        status: string;
        version: string;
        published_version?: string | null;
        updated_at: string;
      }>
    >("GET", `/agents/${agentId}/workflows`).then((rows) =>
      rows.map((r) => ({
        workflowId: r.workflow_id,
        agentId: r.agent_id,
        name: r.name,
        description: r.description,
        enabled: r.enabled,
        status: r.status,
        version: r.version,
        publishedVersion: r.published_version,
        updatedAt: r.updated_at,
      }))
    );
  }

  createWorkflow(agentId: string, name: string, description?: string): Promise<AgentSpecification> {
    return this.request("POST", `/agents/${agentId}/workflows`, { name, description });
  }

  getWorkflow(agentId: string, workflowId: string): Promise<AgentSpecification> {
    return this.request("GET", `/agents/${agentId}/workflows/${workflowId}`);
  }

  renameWorkflow(agentId: string, workflowId: string, name: string): Promise<AgentSpecification> {
    return this.request("POST", `/agents/${agentId}/workflows/${workflowId}/rename`, { name });
  }

  duplicateWorkflow(agentId: string, workflowId: string): Promise<AgentSpecification> {
    return this.request("POST", `/agents/${agentId}/workflows/${workflowId}/duplicate`);
  }

  deleteWorkflow(agentId: string, workflowId: string): Promise<void> {
    return this.request("DELETE", `/agents/${agentId}/workflows/${workflowId}`);
  }

  // ---- Conversational workflow builder: propose / preview / apply ---
  //
  // Mirrors backend/api/app.py's propose_chat_instruction /
  // apply_chat_proposal exactly: propose() never mutates anything on the
  // server (see the route's own docstring) — it's safe to call freely.
  // Only apply() commits, through the real mutation engine. Cancelling
  // is simply never calling apply() at all; there is deliberately no
  // "cancel" network call because there is nothing server-side to undo.

  proposeChatInstruction(
    agentId: string,
    workflowId: string,
    message: string,
    options?: { provider?: string; model?: string }
  ): Promise<ChatProposal> {
    return this.request<{
      reasoning: string;
      mutations: Array<Record<string, unknown>>;
      ok: boolean;
      errors: Array<Record<string, unknown>>;
      attempts: number;
      base_version: string;
      provider: string;
      model?: string | null;
      preview_spec: AgentSpecification | null;
    }>("POST", `/agents/${agentId}/workflows/${workflowId}/chat/propose`, {
      message,
      provider: options?.provider,
      model: options?.model,
    }).then((r) => ({
      reasoning: r.reasoning,
      mutations: r.mutations,
      ok: r.ok,
      errors: r.errors,
      attempts: r.attempts,
      baseVersion: r.base_version,
      provider: r.provider,
      model: r.model,
      previewSpec: r.preview_spec,
    }));
  }

  applyChatProposal(
    agentId: string,
    workflowId: string,
    mutations: Array<Record<string, unknown>>,
    expectedVersion: string,
    force = false
  ): Promise<AgentSpecification> {
    return this.request("POST", `/agents/${agentId}/workflows/${workflowId}/chat/apply`, {
      mutations,
      expected_version: expectedVersion,
      force,
    });
  }


  listAgentFiles(): Promise<{ directory: string; files: Array<{ path: string; size: number; suffix: string }> }> {
    return this.request("GET", "/files");
  }

  async uploadAgentFile(file: File, subdir = ""): Promise<{ ok: boolean; path: string; size: number; directory: string }> {
    const form = new FormData();
    form.append("file", file);
    const q = subdir ? `?subdir=${encodeURIComponent(subdir)}` : "";
    const res = await fetch(`${this.baseUrl}/files/upload${q}`, { method: "POST", body: form });
    if (!res.ok) {
      const text = await res.text();
      throw new ApiError(res.status, text || res.statusText);
    }
    return res.json();
  }

  compileCheck(agentId: string): Promise<CompileCheckResult> {
    return this.request<{ ok: boolean; error?: string | null; node_id?: string | null }>(
      "POST",
      `/agents/${agentId}/compile`
    ).then((r) => ({ ok: r.ok, error: r.error, nodeId: r.node_id }));
  }

  runAgent(agentId: string, seedVariables: Record<string, unknown> = {}): Promise<RunResult> {
    return this.request<{
      thread_id: string;
      status: RunResult["status"];
      trace: RunResult["trace"];
      variables: RunResult["variables"];
      interrupt?: RunResult["interrupt"];
      error?: string | null;
      failed_node_id?: string | null;
    }>("POST", `/agents/${agentId}/run`, { seed_variables: seedVariables }).then((r) => ({
      threadId: r.thread_id,
      status: r.status,
      trace: r.trace,
      variables: r.variables,
      interrupt: r.interrupt,
      error: r.error,
      failedNodeId: r.failed_node_id,
    }));
  }

  resumeAgent(agentId: string, threadId: string, resume: Record<string, unknown>): Promise<RunResult> {
    return this.request<{
      thread_id: string;
      status: RunResult["status"];
      trace: RunResult["trace"];
      variables: RunResult["variables"];
      interrupt?: RunResult["interrupt"];
      error?: string | null;
      failed_node_id?: string | null;
    }>("POST", `/agents/${agentId}/resume`, { thread_id: threadId, resume }).then((r) => ({
      threadId: r.thread_id,
      status: r.status,
      trace: r.trace,
      variables: r.variables,
      interrupt: r.interrupt,
      failedNodeId: r.failed_node_id,
      error: r.error,
    }));
  }

  publishAgent(agentId: string): Promise<PublishResult> {
    return this.request<{ spec: AgentSpecification; published_version: string }>(
      "POST",
      `/agents/${agentId}/publish`
    ).then((r) => ({ spec: r.spec, publishedVersion: r.published_version }));
  }

  /**
   * Streams a run over SSE, calling onTraceEvent as each node actually
   * executes, and resolving with the same RunResult /run returns
   * synchronously once the run finishes or pauses. See sseClient.ts for
   * why this uses fetch instead of EventSource (POST body support).
   */
  async runAgentStream(
    agentId: string,
    seedVariables: Record<string, unknown>,
    onTraceEvent: (event: Record<string, unknown>) => void,
    onNodeStarted?: (event: { node_id: string; node_label: string }) => void
  ): Promise<RunResult> {
    return this.streamRun(`/agents/${agentId}/run/stream`, { seed_variables: seedVariables }, onTraceEvent, onNodeStarted);
  }

  async resumeAgentStream(
    agentId: string,
    threadId: string,
    resume: Record<string, unknown>,
    onTraceEvent: (event: Record<string, unknown>) => void,
    onNodeStarted?: (event: { node_id: string; node_label: string }) => void
  ): Promise<RunResult> {
    return this.streamRun(
      `/agents/${agentId}/resume/stream`,
      { thread_id: threadId, resume },
      onTraceEvent,
      onNodeStarted
    );
  }

  private streamRun(
    path: string,
    body: unknown,
    onTraceEvent: (event: Record<string, unknown>) => void,
    onNodeStarted?: (event: { node_id: string; node_label: string }) => void
  ): Promise<RunResult> {
    return new Promise((resolve, reject) => {
      let result: RunResult | null = null;
      streamSSE({
        url: `${this.baseUrl}${path}`,
        body,
        fetchImpl: this.fetchImpl,
        onEvent: (event) => {
          if (event.event === "node_started") {
            onNodeStarted?.(event.data as { node_id: string; node_label: string });
          } else if (event.event === "trace") {
            onTraceEvent(event.data as Record<string, unknown>);
          } else if (event.event === "done") {
            const d = event.data as any;
            result = {
              threadId: d.thread_id,
              status: d.status,
              trace: d.trace,
              variables: d.variables,
              interrupt: d.interrupt,
              error: d.error,
              failedNodeId: d.failed_node_id,
            };
          }
        },
      })
        .then(() => {
          if (result) resolve(result);
          else reject(new Error("Stream ended without a 'done' event."));
        })
        .catch(reject);
    });
  }
}
