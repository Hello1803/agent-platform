/**
 * TypeScript types for the AgentSpecification IR.
 * Mirrors schema/agent-spec.schema.json 1:1 — this file and the JSON Schema
 * must be kept in sync. In CI, validate example specs against BOTH.
 *
 * These types are consumed by:
 *  - the visual node-based canvas (renders Node/Edge, writes canvasPosition)
 *  - the chat builder (reads/writes the whole spec as mutations)
 *  - the API client talking to the compiler/runtime
 */

export const AGENT_SPEC_VERSION = "1.0.0" as const;

export type AgentStatus = "draft" | "published" | "archived";

export interface AgentMetadata {
  id: string;
  name: string;
  description?: string;
  status: AgentStatus;
  version: string;
  publishedVersion?: string | null;
  createdAt: string; // ISO date-time
  updatedAt: string; // ISO date-time
  owner?: string;
  tags?: string[];
  /** The Agent container this workflow belongs to. Defaults to `id`
   * (this workflow's own id) when unset -- see backend/models/agent_spec.py's
   * AgentMetadata.effective_agent_id. */
  agentId?: string | null;
  /** Whether this workflow is active. */
  enabled?: boolean;
}

export type TriggerType =
  | "manual"
  | "schedule"
  | "webhook"
  | "email"
  | "file_change"
  | "external_event"
  | "api";

export interface Trigger {
  id: string;
  type: TriggerType;
  config?: Record<string, unknown>;
}

export type NodeType =
  // AI / Agent nodes
  | "ai_agent"
  | "ask_ai"
  | "classify"
  | "extract_information"
  | "summarize"
  | "make_decision"
  | "run_agent"
  // Tool nodes
  | "search_web"
  | "open_website"
  | "read_email"
  | "send_email"
  | "search_documents"
  | "read_file"
  | "write_file"
  | "database"
  | "api_call"
  | "code"
  | "preview"
  | "rag_index"
  | "rag_query"
  | "browser"
  // Control nodes
  | "if_else"
  | "switch"
  | "loop"
  | "parallel"
  | "merge"
  | "wait"
  | "retry"
  // Human nodes
  | "human_approval"
  | "ask_user"
  | "review"
  | "confirm"
  // Trigger nodes (as graph entry points)
  | "trigger_manual"
  | "trigger_schedule"
  | "trigger_email"
  | "trigger_webhook"
  | "trigger_file_change"
  | "trigger_external_event";

export type NodeCategory = "ai" | "tool" | "control" | "human" | "trigger";

export type DataType = "string" | "number" | "boolean" | "object" | "array" | "file" | "any";

export interface Port {
  id: string;
  name: string;
  dataType?: DataType;
  required?: boolean;
}

export interface CanvasPosition {
  x: number;
  y: number;
}

export interface SubgraphRef {
  agentId: string;
  versionPin?: string;
}

/** Distinguishes deterministic logic from LLM-driven decisions (see UX section 11). */
export type ExecutionType = "deterministic" | "probabilistic";

export interface AgentNode {
  id: string;
  type: NodeType;
  category: NodeCategory;
  label: string;
  executionType?: ExecutionType;
  /** Only meaningful when executionType === "probabilistic" */
  confidenceThreshold?: number;
  config?: Record<string, unknown>;
  inputs?: Port[];
  outputs?: Port[];
  toolRefs?: string[];
  subgraph?: SubgraphRef;
  canvasPosition?: CanvasPosition;
}

export type EdgeConditionKind = "deterministic" | "ai_decision";

export interface EdgeCondition {
  expression?: string;
  kind?: EdgeConditionKind;
}

export interface AgentEdge {
  id: string;
  source: string;
  sourceHandle?: string;
  target: string;
  targetHandle?: string;
  condition?: EdgeCondition;
  priority?: number;
  label?: string;
}

export interface ToolBinding {
  id: string;
  toolId: string;
  displayName?: string;
  credentialRef?: string;
  permissions: string[];
  rateLimit?: {
    maxCalls?: number;
    perSeconds?: number;
  };
  mode?: "demo" | "connected";
}

export type MemoryType = "none" | "conversation_buffer" | "vector_store" | "key_value" | "summary";
export type MemoryScope = "run" | "session" | "agent_global";

export interface MemoryConfig {
  enabled?: boolean;
  type?: MemoryType;
  scope?: MemoryScope;
  retentionDays?: number;
}

export interface PermissionSet {
  allow?: string[];
  /** Deny always overrides allow. */
  deny?: string[];
}

export type AutonomyLevel = "suggest" | "ask_permission" | "automatic";

export type ApprovalActionCategory =
  | "money"
  | "send_message"
  | "delete_data"
  | "account_change"
  | "external_publish"
  | "sensitive_info"
  | "other";

export interface ApprovalRule {
  id: string;
  appliesTo: {
    nodeId?: string;
    actionCategory?: ApprovalActionCategory;
  };
  autonomyLevel: AutonomyLevel;
  approverRole?: string;
}

export interface Policies {
  toolAllowlist?: string[];
  dataAccessBoundaries?: string[];
  promptInjectionDefenses?: boolean;
  auditLog?: boolean;
  maxRunDurationSeconds?: number;
  maxCostPerRun?: number;
}

export interface RetryPolicy {
  maxRetries?: number;
  backoffSeconds?: number;
}

export interface RuntimeConfig {
  model?: string;
  temperature?: number;
  maxTokens?: number;
  checkpointing?: boolean;
  retryPolicy?: RetryPolicy;
  timeoutSeconds?: number;
}

export interface AgentSpecification {
  specVersion: typeof AGENT_SPEC_VERSION;
  metadata: AgentMetadata;
  goal: string;
  instructions?: string;
  triggers?: Trigger[];
  nodes: AgentNode[];
  edges: AgentEdge[];
  tools?: ToolBinding[];
  memory?: MemoryConfig;
  permissions?: PermissionSet;
  humanApproval?: ApprovalRule[];
  policies?: Policies;
  runtimeConfig?: RuntimeConfig;
}

/**
 * A mutation is how BOTH the chat builder and the visual editor propose
 * changes to a spec. Neither is allowed to write the spec directly —
 * mutations go through validation before being merged (see Concurrent
 * Modifications / Agent Compiler in the architecture doc).
 */
export type AgentSpecMutation =
  | { kind: "add_node"; node: AgentNode }
  | { kind: "update_node"; nodeId: string; patch: Partial<AgentNode> }
  | { kind: "remove_node"; nodeId: string }
  | { kind: "add_edge"; edge: AgentEdge }
  | { kind: "update_edge"; edgeId: string; patch: Partial<AgentEdge> }
  | { kind: "remove_edge"; edgeId: string }
  | { kind: "update_metadata"; patch: Partial<AgentMetadata> }
  | { kind: "update_goal"; goal: string }
  | { kind: "update_instructions"; instructions: string }
  | { kind: "add_tool"; tool: ToolBinding }
  | { kind: "remove_tool"; toolId: string }
  | { kind: "update_tool"; toolId: string; patch: Record<string, unknown> }
  | { kind: "update_permissions"; patch: Partial<PermissionSet> }
  | { kind: "add_approval_rule"; rule: ApprovalRule }
  | { kind: "update_runtime_config"; patch: Partial<RuntimeConfig> };

export interface MutationResult {
  ok: boolean;
  spec?: AgentSpecification;
  errors?: ValidationError[];
}

export interface ValidationError {
  path: string;
  message: string;
  severity: "error" | "warning";
}
