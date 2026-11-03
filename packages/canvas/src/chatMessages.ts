/**
 * Pure chat-message construction, deliberately separated from React state
 * (useAgentSession.ts) so the "what does an instruction's outcome turn
 * into in the transcript" logic is unit-testable without rendering
 * anything — see __tests__/chatMessages.test.ts.
 */

import type { ChatProposal, InstructionResult } from "./apiClient";
import { ApiError } from "./apiClient";

export type ChatRole = "user" | "assistant" | "system";
export type ChatMessageStatus = "pending" | "ok" | "error";
/** Lifecycle of a proposal-carrying assistant message: "pending" shows
 * [Cancel]/[Apply], "applied"/"cancelled" show the resolved outcome
 * instead (buttons removed) — see ChatPanel.tsx. Undefined for messages
 * that never carried a proposal (a plain answer, an error, a legacy
 * /instructions response). */
export type ProposalStatus = "pending" | "applied" | "cancelled";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  timestamp: string;
  status: ChatMessageStatus;
  /** Whether this instruction actually changed the spec (false for a pure question that resolved to zero mutations). Undefined for user/system messages. */
  changedSpec?: boolean;
  /** Set when this message represents a proposal awaiting (or having
   * received) a Cancel/Apply decision — see proposeInstruction /
   * applyProposal / cancelProposal in useAgentSession.ts. */
  proposal?: ChatProposal;
  proposalStatus?: ProposalStatus;
}

let counter = 0;
function nextId(): string {
  counter += 1;
  return `msg_${Date.now()}_${counter}`;
}

export function buildUserMessage(content: string): ChatMessage {
  return { id: nextId(), role: "user", content, timestamp: new Date().toISOString(), status: "ok" };
}

/** The instruction succeeded (result.ok) — reasoning becomes the assistant's message, tagged with whether it actually changed the spec (vs. resolving to zero mutations, e.g. a pure question). */
export function buildAssistantMessage(outcome: InstructionResult): ChatMessage {
  return {
    id: nextId(),
    role: "assistant",
    content: outcome.reasoning,
    timestamp: new Date().toISOString(),
    status: "ok",
    changedSpec: outcome.result.spec != null,
  };
}

/**
 * A proposal came back from POST .../chat/propose — nothing has been
 * applied yet. Three distinct shapes, all rendered as one assistant
 * message:
 *   - zero mutations (a question/no-op): resolved immediately, no
 *     buttons — there's nothing to apply.
 *   - mutations, ok=true: a real proposal — proposalStatus "pending",
 *     ChatPanel renders [Cancel] [Apply Changes].
 *   - mutations, ok=false: the model's mutations didn't validate even
 *     after its own retry loop — shown with the validation errors, no
 *     buttons (nothing valid to apply).
 */
export function buildProposalMessage(proposal: ChatProposal): ChatMessage {
  const hasMutations = proposal.mutations.length > 0;
  const isPendingDecision = hasMutations && proposal.ok;
  let content = proposal.reasoning;
  if (hasMutations && !proposal.ok) {
    const firstError = proposal.errors[0]?.message as string | undefined;
    content += firstError ? `\n\nThat change didn't validate: ${firstError}` : "\n\nThat change didn't validate.";
  }
  return {
    id: nextId(),
    role: "assistant",
    content,
    timestamp: new Date().toISOString(),
    status: proposal.ok ? "ok" : "error",
    changedSpec: false, // nothing is applied yet, regardless of `ok`
    proposal: hasMutations ? proposal : undefined,
    proposalStatus: isPendingDecision ? "pending" : undefined,
  };
}

/**
 * The instruction failed — either the model refused (network/API-level
 * failure surfaced as ApiError from the /instructions route's 502), or
 * the mutations it proposed didn't validate (422) or hit a version
 * conflict (409). Distinguishes these in the message text since the
 * right next step differs: a conflict means "someone else edited this,
 * try again"; a validation failure means "rephrase or simplify."
 */
function extractValidationDetail(body: unknown): string | undefined {
  if (body == null) return undefined;
  if (typeof body === "string") return body;
  if (Array.isArray(body)) {
    const first = body[0] as any;
    return first?.message ?? (typeof first === "string" ? first : undefined);
  }
  if (typeof body === "object") {
    const obj = body as any;
    if (typeof obj.detail === "string") return obj.detail;
    if (Array.isArray(obj.detail)) {
      const first = obj.detail[0];
      return first?.message ?? (typeof first === "string" ? first : undefined);
    }
    if (Array.isArray(obj.errors) && obj.errors[0]?.message) return obj.errors[0].message;
    if (typeof obj.message === "string") return obj.message;
    if (typeof obj.error === "string") return obj.error;
  }
  return undefined;
}

export function buildErrorMessage(error: ApiError | Error): ChatMessage {
  let content: string;
  if (error instanceof ApiError && error.isAiUnavailable) {
    content = "AI Builder unavailable — no AI provider API key is configured on the server. Ask whoever's running this to set ANTHROPIC_API_KEY or OPENAI_API_KEY, or use the canvas and node inspector to edit directly for now.";
  } else if (error instanceof ApiError && error.isVersionConflict) {
    content = "The workflow was updated since this suggestion was made (canvas edit or another apply). The latest version is loaded — send the instruction again to get a fresh proposal.";
  } else if (error instanceof ApiError && error.isValidationError) {
    const detail = extractValidationDetail(error.body);
    const looksLikeMissingKey =
      typeof detail === "string" &&
      (/no AI provider is configured/i.test(detail) ||
        /API_KEY is not configured/i.test(detail) ||
        /is not configured on this server/i.test(detail));
    if (looksLikeMissingKey) {
      content =
        "AI Builder unavailable — no AI provider API key is configured on the server. Ask whoever's running this to set ANTHROPIC_API_KEY or OPENAI_API_KEY, or use the canvas and node inspector to edit directly for now.";
    } else {
      content = detail
        ? `That change didn't validate: ${detail}`
        : "That change didn't validate — try rephrasing or being more specific.";
    }
  } else if (error instanceof ApiError) {
    const detail = typeof error.body === "string" ? error.body : JSON.stringify(error.body);
    content = `Couldn't process that instruction${detail ? `: ${detail}` : "."}`;
  } else {
    content = error.message;
  }
  return { id: nextId(), role: "assistant", content, timestamp: new Date().toISOString(), status: "error" };
}

/** Human-readable one-liner for a single proposed mutation — used to
 * render a real, structured preview of what a proposal would do
 * (ChatPanel.tsx), rather than just showing the model's prose reasoning
 * and hoping it's accurate. Reads the raw mutation shape directly, the
 * same wire shape backend/models/agent_spec.py's AgentSpecMutation union
 * produces (see packages/shared-types/src/agent-spec.ts). */
export function summarizeMutation(mutation: Record<string, unknown>): string {
  const kind = mutation.kind as string;
  switch (kind) {
    case "add_node": {
      const node = mutation.node as any;
      return `+ Add node "${node?.label ?? node?.id}" (${node?.type})`;
    }
    case "update_node":
      return `~ Update node "${mutation.nodeId}"`;
    case "remove_node":
      return `− Remove node "${mutation.nodeId}"`;
    case "add_edge": {
      const edge = mutation.edge as any;
      return `+ Connect "${edge?.source}" → "${edge?.target}"`;
    }
    case "update_edge":
      return `~ Update connection "${mutation.edgeId}"`;
    case "remove_edge":
      return `− Disconnect "${mutation.edgeId}"`;
    case "update_metadata":
      return `~ Update agent details (${Object.keys((mutation.patch as object) ?? {}).join(", ")})`;
    case "update_goal":
      return "~ Update the agent's goal";
    case "update_instructions":
      return "~ Update the agent's instructions";
    case "add_tool": {
      const tool = mutation.tool as any;
      return `+ Add tool binding "${tool?.displayName ?? tool?.toolId}"`;
    }
    case "remove_tool":
      return `− Remove tool binding "${mutation.toolId}"`;
    case "update_tool":
      return `~ Update tool binding "${mutation.toolId}"`;
    case "update_permissions":
      return "~ Update permissions";
    case "add_approval_rule":
      return "+ Add a human-approval rule";
    case "update_runtime_config":
      return "~ Update runtime configuration";
    default:
      return `~ ${kind}`;
  }
}

/** Marks a previously-pending proposal message as applied or cancelled —
 * used by useAgentSession.ts's applyProposal/cancelProposal to update the
 * transcript in place rather than appending a new message, so the
 * [Cancel]/[Apply] buttons on THAT message disappear once resolved. */
export function withProposalResolution(message: ChatMessage, resolution: "applied" | "cancelled"): ChatMessage {
  return { ...message, proposalStatus: resolution };
}
