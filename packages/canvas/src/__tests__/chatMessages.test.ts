import { buildUserMessage, buildAssistantMessage, buildErrorMessage, buildProposalMessage, summarizeMutation, withProposalResolution } from "../chatMessages";
import { ApiError } from "../apiClient";
import type { InstructionResult, ChatProposal } from "../apiClient";
import type { AgentSpecification } from "@agent-platform/shared-types";

let passed = 0;
let failed = 0;

function assert(condition: boolean, message: string) {
  if (condition) passed++;
  else {
    failed++;
    console.error(`FAIL: ${message}`);
  }
}

// ---- buildUserMessage ----

{
  const msg = buildUserMessage("add an approval step");
  assert(msg.role === "user", "buildUserMessage: role is user");
  assert(msg.content === "add an approval step", "buildUserMessage: content preserved verbatim");
  assert(msg.status === "ok", "buildUserMessage: status is ok");
  assert(typeof msg.id === "string" && msg.id.length > 0, "buildUserMessage: has a non-empty id");
}

{
  const a = buildUserMessage("one");
  const b = buildUserMessage("two");
  assert(a.id !== b.id, "buildUserMessage: successive calls get distinct ids");
}

// ---- buildAssistantMessage ----

{
  const fakeSpec = { metadata: { id: "x" } } as unknown as AgentSpecification;
  const outcome: InstructionResult = {
    reasoning: "Added a translation step after the summary.",
    attempts: 1,
    result: { ok: true, spec: fakeSpec },
  };
  const msg = buildAssistantMessage(outcome);
  assert(msg.role === "assistant", "buildAssistantMessage: role is assistant");
  assert(msg.content === outcome.reasoning, "buildAssistantMessage: content is the reasoning text");
  assert(msg.status === "ok", "buildAssistantMessage: status is ok for a successful outcome");
  assert(msg.changedSpec === true, "buildAssistantMessage: changedSpec true when result.spec is present");
}

{
  // A pure question ("what does this agent do?") resolves to zero mutations — no spec in the result.
  const outcome: InstructionResult = {
    reasoning: "This agent searches the web then summarizes what it finds.",
    attempts: 1,
    result: { ok: true },
  };
  const msg = buildAssistantMessage(outcome);
  assert(msg.changedSpec === false, "buildAssistantMessage: changedSpec false when the instruction was just a question");
}

// ---- buildErrorMessage ----

{
  const conflictError = new ApiError(409, { errors: [{ path: "metadata.version", message: "stale" }] });
  const msg = buildErrorMessage(conflictError);
  assert(msg.status === "error", "buildErrorMessage: status is error");
  assert(msg.content.toLowerCase().includes("landed first"), "buildErrorMessage: version conflict gets a specific, actionable message");
}

{
  const validationError = new ApiError(422, { errors: [{ path: "nodes", message: "Node 'n_xyz' references unknown target." }] });
  const msg = buildErrorMessage(validationError);
  assert(msg.content.includes("Node 'n_xyz' references unknown target."), "buildErrorMessage: surfaces the actual validation detail when present");
}

{
  const validationErrorNoDetail = new ApiError(422, {});
  const msg = buildErrorMessage(validationErrorNoDetail);
  assert(msg.content.length > 0, "buildErrorMessage: falls back to a generic but non-empty message when no detail is present");
}

{
  const genericError = new ApiError(502, "model refused");
  const msg = buildErrorMessage(genericError);
  assert(msg.content.includes("502") === false, "buildErrorMessage: doesn't leak raw status codes into user-facing text");
}

{
  // The specific V9 fix: no ANTHROPIC_API_KEY configured should read as
  // "unavailable, here's what to do" — not a generic error or a raw
  // "Could not resolve authentication method" SDK exception string.
  const aiUnavailableError = new ApiError(503, { error: "AI Builder unavailable...", reason: "no_api_key" });
  assert(aiUnavailableError.isAiUnavailable, "ApiError: correctly identifies the no-API-key 503 shape");
  const msg = buildErrorMessage(aiUnavailableError);
  assert(msg.content.toLowerCase().includes("api key"), "buildErrorMessage: the no-API-key case gives specific, actionable guidance");
  assert(msg.content.includes("503") === false, "buildErrorMessage: no-API-key case also doesn't leak the raw status code");
}

{
  // A DIFFERENT 503 (e.g. a real outage) should NOT be misreported as the API-key case.
  const genericUnavailable = new ApiError(503, { error: "database connection lost" });
  assert(!genericUnavailable.isAiUnavailable, "ApiError: a 503 without reason=no_api_key is NOT misclassified as the API-key case");
}

{
  const plainError = new Error("network down");
  const msg = buildErrorMessage(plainError);
  assert(msg.content === "network down", "buildErrorMessage: handles a plain Error (not ApiError) too");
}

// ---- buildProposalMessage (Conversational Workflow Builder) ----

{
  const proposal: ChatProposal = {
    reasoning: "Added a human approval step before Send Email.",
    mutations: [
      { kind: "add_node", node: { id: "n_approval", type: "human_approval", label: "Approve send" } },
      { kind: "add_edge", edge: { id: "e1", source: "n_gen_email", target: "n_approval" } },
    ],
    ok: true,
    errors: [],
    attempts: 1,
    baseVersion: "1.0.0",
    provider: "anthropic",
    previewSpec: { metadata: { id: "x" } } as unknown as AgentSpecification,
  };
  const msg = buildProposalMessage(proposal);
  assert(msg.role === "assistant", "buildProposalMessage: role is assistant");
  assert(msg.content === proposal.reasoning, "buildProposalMessage: content is the reasoning");
  assert(msg.proposal === proposal, "buildProposalMessage: carries the full proposal for the UI to render");
  assert(msg.proposalStatus === "pending", "buildProposalMessage: a valid, mutating proposal starts pending (awaiting Cancel/Apply)");
  assert(msg.changedSpec === false, "buildProposalMessage: changedSpec is false -- NOTHING has been applied yet, only proposed");
}

{
  // A pure question -- zero mutations. Resolved immediately, no buttons.
  const proposal: ChatProposal = {
    reasoning: "This workflow searches the web then summarizes what it finds.",
    mutations: [],
    ok: true,
    errors: [],
    attempts: 1,
    baseVersion: "1.0.0",
    provider: "anthropic",
    previewSpec: null,
  };
  const msg = buildProposalMessage(proposal);
  assert(msg.proposal === undefined, "buildProposalMessage: a question (zero mutations) carries no proposal to act on");
  assert(msg.proposalStatus === undefined, "buildProposalMessage: a question has no pending decision");
}

{
  // Mutations that didn't validate even after the NL compiler's own retry loop.
  const proposal: ChatProposal = {
    reasoning: "Connecting to a node that doesn't exist.",
    mutations: [{ kind: "add_edge", edge: { id: "e_bad", source: "n_trigger", target: "n_missing" } }],
    ok: false,
    errors: [{ path: "<root>", message: "Edge 'e_bad' references unknown target node 'n_missing'." }],
    attempts: 2,
    baseVersion: "1.0.0",
    provider: "anthropic",
    previewSpec: null,
  };
  const msg = buildProposalMessage(proposal);
  assert(msg.status === "error", "buildProposalMessage: an invalid proposal is shown as an error");
  assert(msg.proposalStatus === undefined, "buildProposalMessage: an invalid proposal has nothing valid to Cancel/Apply, so no pending state");
  assert(msg.content.includes("references unknown target"), "buildProposalMessage: surfaces the actual validation error, not just the model's reasoning");
}

// ---- withProposalResolution ----

{
  const proposal: ChatProposal = { reasoning: "x", mutations: [{ kind: "update_goal", goal: "y" }], ok: true, errors: [], attempts: 1, baseVersion: "1.0.0", provider: "anthropic", previewSpec: null };
  const msg = buildProposalMessage(proposal);
  const applied = withProposalResolution(msg, "applied");
  assert(applied.proposalStatus === "applied", "withProposalResolution: marks the message as applied");
  assert(applied.id === msg.id, "withProposalResolution: preserves message identity (updates in place, doesn't create a new message)");

  const cancelled = withProposalResolution(msg, "cancelled");
  assert(cancelled.proposalStatus === "cancelled", "withProposalResolution: marks the message as cancelled");
  assert(msg.proposalStatus === "pending", "withProposalResolution: does not mutate the original message (pure function)");
}

// ---- summarizeMutation ----

{
  assert(
    summarizeMutation({ kind: "add_node", node: { id: "n1", label: "Approve send", type: "human_approval" } }).includes("Approve send"),
    "summarizeMutation: add_node names the node's label"
  );
  assert(
    summarizeMutation({ kind: "add_edge", edge: { source: "n1", target: "n2" } }).includes("n1") &&
      summarizeMutation({ kind: "add_edge", edge: { source: "n1", target: "n2" } }).includes("n2"),
    "summarizeMutation: add_edge names both endpoints"
  );
  assert(summarizeMutation({ kind: "remove_node", nodeId: "n1" }).includes("n1"), "summarizeMutation: remove_node names the node id");
  assert(summarizeMutation({ kind: "update_tool", toolId: "t1" }).includes("t1"), "summarizeMutation: update_tool names the tool id");
  assert(summarizeMutation({ kind: "some_future_kind" }).length > 0, "summarizeMutation: an unrecognized kind still produces a non-empty, non-crashing summary");
}

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
