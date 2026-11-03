/**
 * Wires AgentCanvas to the running API. This is the "last piece of glue"
 * called out at the end of the backend build: AgentCanvas only ever
 * proposes mutations via its onMutation prop — this hook is what actually
 * sends them to the server and feeds the authoritative result back in.
 *
 * ALSO owns the chat transcript and the propose/apply/cancel flow —
 * deliberately the SAME hook as canvas mutations, not a separate one.
 * Both write through to the exact same `spec` state, which is the whole
 * point: a chat-applied mutation and a canvas drag are just two
 * different sources of AgentSpecMutation, and if they held separate
 * copies of `spec` this hook would be reintroducing precisely the
 * split-brain problem the architecture doc's "single source of truth"
 * section exists to prevent. A chat PROPOSAL, by contrast, deliberately
 * does NOT touch `spec` at all until applyProposal() is called — see its
 * own doc comment below.
 *
 * Usage:
 *   const client = new AgentApiClient({ baseUrl: "http://localhost:8000" });
 *   const session = useAgentSession(client, agentId);
 *   if (session.spec) {
 *     <AgentCanvas spec={session.spec} onMutation={session.applyMutation}
 *                   resyncNonce={session.resyncNonce} />
 *     <ChatPanel messages={session.messages} onSend={session.proposeInstruction}
 *                onApply={session.applyProposal} onCancel={session.cancelProposal}
 *                busy={session.chatBusy} />
 *   }
 *
 * If the caller needs to do something ELSE before this agent is
 * guaranteed to exist (e.g. an app-level bootstrap step that creates a
 * demo agent on first load), pass `enabled: false` until that's done —
 * see the `enabled` param below. Without this, the hook's own initial
 * fetch races an independent bootstrap effect: both fire on mount, and if
 * bootstrap's create-agent call hasn't landed yet when this hook's GET
 * fires, that GET 404s and `spec` stays null forever, since nothing
 * retries it afterward.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { AgentSpecification, AgentSpecMutation } from "@agent-platform/shared-types";
import { AgentApiClient, ApiError, type ChatProposal } from "./apiClient";
import {
  type ChatMessage,
  buildUserMessage,
  buildErrorMessage,
  buildProposalMessage,
  withProposalResolution,
} from "./chatMessages";
import { parseModelString } from "./specToFlow";

export interface UseAgentSessionResult {
  spec: AgentSpecification | null;
  loading: boolean;
  /** Set on the most recent failed mutation; a version conflict or a validation error. Cleared on the next successful apply. */
  error: ApiError | null;
  /** Pass straight through to AgentCanvas's resyncNonce prop. */
  resyncNonce: number;
  /** Pass straight through to AgentCanvas's onMutation prop. Always a batch — see AgentCanvas.tsx's onMutation doc comment for why individual mutations are never submitted one at a time. */
  applyMutation: (mutations: AgentSpecMutation[]) => void;
  refetch: () => Promise<void>;
  /** The chat transcript — pass straight through to ChatPanel's messages prop. */
  messages: ChatMessage[];
  /** True while a chat request (propose or apply) is in flight — pass to ChatPanel's busy prop to disable the input. */
  chatBusy: boolean;
  /**
   * Sends a message to the AI builder. Does NOT mutate the spec —
   * appends a proposal message to the transcript (see
   * chatMessages.ts's buildProposalMessage) that ChatPanel renders with
   * [Cancel]/[Apply Changes] when it carries real mutations. Nothing
   * touches the workflow until applyProposal() is called for that
   * specific message.
   */
  proposeInstruction: (text: string, options?: { provider?: string; model?: string }) => Promise<void>;
  /** Commits a pending proposal through the real mutation engine (POST
   * .../chat/apply) and updates `spec` from the (re-validated) result. */
  applyProposal: (messageId: string) => Promise<void>;
  /** Discards a pending proposal — purely local, no network call, since
   * propose() never touched the server in the first place. */
  cancelProposal: (messageId: string) => void;
}

export function useAgentSession(client: AgentApiClient, agentId: string, enabled: boolean = true): UseAgentSessionResult {
  const [spec, setSpec] = useState<AgentSpecification | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);
  const [resyncNonce, setResyncNonce] = useState(0);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatBusy, setChatBusy] = useState(false);

  // Mutations can resolve out of order if the user fires several quickly
  // (e.g. two drags in a row, or a drag while an instruction is in
  // flight); track the newest request so a slow, stale response can't
  // clobber a newer one. Shared across applyMutation and proposeInstruction
  // since they write to the same `spec`.
  const requestSeq = useRef(0);

  const refetch = useCallback(async () => {
    setLoading(true);
    try {
      const latest = await client.getAgent(agentId);
      setSpec(latest);
      setError(null);
    } catch (e) {
      if (e instanceof ApiError) setError(e);
      else throw e;
    } finally {
      setLoading(false);
    }
  }, [client, agentId]);

  // Gated on `enabled` — see the doc comment above and App.tsx's bootstrap
  // effect for why: fetching before the caller's own setup (e.g. creating
  // a demo agent) has finished is a real race, not a hypothetical one.
  // `loading` starts true and stays true while disabled, so callers can
  // key their own "is this ready yet" check off `session.loading` the
  // same way regardless of whether they're using `enabled` or not.
  useEffect(() => {
    if (!enabled) return;
    void refetch();
    // Switching workflows (agentId changes) means re-fetching THAT
    // workflow's own real backend state, and starting its chat
    // transcript fresh — a proposal built against workflow A's node ids
    // would be nonsensical to show (or worse, apply) once the person has
    // switched to workflow B.
    setMessages([]);
  }, [refetch, enabled, agentId]);

  const applyMutation = useCallback(
    (mutations: AgentSpecMutation[]) => {
      if (!spec || mutations.length === 0) return;
      const seq = ++requestSeq.current;
      const expectedVersion = spec.metadata.version;

      client
        .applyMutations(agentId, mutations, expectedVersion)
        .then((result) => {
          if (seq !== requestSeq.current) return; // superseded by a newer mutation
          if (result.ok && result.spec) {
            setSpec(result.spec);
            setError(null);
          }
        })
        .catch(async (e: unknown) => {
          if (seq !== requestSeq.current) return;
          if (e instanceof ApiError) {
            setError(e);
            if (e.isVersionConflict) {
              // Someone else's edit (chat/NL compiler, or another canvas
              // session) landed first — pull the current authoritative
              // spec so the next attempt is based on reality.
              const latest = await client.getAgent(agentId);
              if (seq === requestSeq.current) setSpec(latest);
            }
            // Either way (conflict or plain validation failure), the
            // server made no change, so `spec` itself may be unchanged —
            // force AgentCanvas to discard its optimistic local edit by
            // bumping resyncNonce regardless.
            setResyncNonce((n) => n + 1);
          } else {
            throw e;
          }
        });
    },
    [client, agentId, spec]
  );

  /** The Agent container this workflow belongs to — read from the
   * loaded spec's own metadata (falls back to the workflow's own id,
   * matching backend/models/agent_spec.py's effective_agent_id) rather
   * than a separate prop, since the spec already carries this once
   * loaded and it's the one value the chat propose/apply routes
   * (/agents/{agentId}/workflows/{workflowId}/chat/...) actually need
   * beyond `agentId` (== workflowId) itself. */
  const containerAgentId = spec?.metadata.agentId ?? agentId;

  const proposeInstruction = useCallback(
    async (text: string, options?: { provider?: string; model?: string }) => {
      if (!text.trim() || chatBusy) return;
      setMessages((prev) => [...prev, buildUserMessage(text)]);
      setChatBusy(true);

      try {
        let provider = options?.provider;
        let model = options?.model;
        if (!provider && spec?.runtimeConfig?.model) {
          const [p, m] = parseModelString(spec.runtimeConfig.model);
          provider = p ?? undefined;
          model = model ?? m ?? undefined;
        }
        const proposal = await client.proposeChatInstruction(containerAgentId, agentId, text, {
          provider,
          model,
        });
        setMessages((prev) => [...prev, buildProposalMessage(proposal)]);
        // A proposal NEVER mutates the server — nothing to resync or
        // apply to `spec` here, unlike the old immediate-apply flow.
      } catch (e) {
        if (e instanceof ApiError) {
          setMessages((prev) => [...prev, buildErrorMessage(e)]);
        } else {
          const err = e instanceof Error ? e : new Error(String(e));
          setMessages((prev) => [...prev, buildErrorMessage(err)]);
          throw err;
        }
      } finally {
        setChatBusy(false);
      }
    },
    [client, agentId, containerAgentId, chatBusy, spec]
  );

  const applyProposal = useCallback(
    async (messageId: string, options?: { force?: boolean }) => {
      const message = messages.find((m) => m.id === messageId);
      if (!message?.proposal || message.proposalStatus !== "pending" || !spec) return;
      const seq = ++requestSeq.current;
      setChatBusy(true);
      const force = options?.force === true;
      try {
        let updated;
        try {
          updated = await client.applyChatProposal(
            containerAgentId,
            agentId,
            message.proposal.mutations,
            message.proposal.baseVersion,
            force
          );
        } catch (first) {
          if (!(first instanceof ApiError) || !first.isVersionConflict) throw first;
          // Offer overwrite: canvas moves / other edits bumped the version.
          const latest = await client.getAgent(agentId);
          if (seq === requestSeq.current) setSpec(latest);
          if (!force) {
            const ok =
              typeof window !== "undefined" &&
              window.confirm(
                "This workflow changed since the suggestion was made (for example you moved a node).\n\n" +
                  "Apply the chat changes on top of the CURRENT graph anyway?\n\n" +
                  "OK = overwrite / force apply\nCancel = keep the graph as it is"
              );
            if (!ok) {
              setMessages((prev) => [
                ...prev,
                {
                  id: `info-${Date.now()}`,
                  role: "assistant" as const,
                  content:
                    "Apply cancelled. The canvas was left as-is. Send the instruction again if you want a fresh proposal against the current layout.",
                  timestamp: new Date().toISOString(),
                  status: "ok" as const,
                },
              ]);
              return;
            }
          }
          updated = await client.applyChatProposal(
            containerAgentId,
            agentId,
            message.proposal.mutations,
            latest.metadata.version,
            true
          );
        }
        setMessages((prev) => prev.map((m) => (m.id === messageId ? withProposalResolution(m, "applied") : m)));
        if (seq === requestSeq.current) {
          setSpec(updated);
          setError(null);
          setResyncNonce((n) => n + 1);
        }
      } catch (e) {
        if (e instanceof ApiError) {
          setMessages((prev) => [...prev, buildErrorMessage(e)]);
          setError(e);
          if (e.isVersionConflict) {
            const latest = await client.getAgent(agentId);
            if (seq === requestSeq.current) {
              setSpec(latest);
              setResyncNonce((n) => n + 1);
            }
          }
        } else {
          throw e;
        }
      } finally {
        setChatBusy(false);
      }
    },
    [client, agentId, containerAgentId, messages, spec]
  );

  const cancelProposal = useCallback((messageId: string) => {
    // Purely local — propose() never touched the server, so there is
    // nothing to undo remotely. The workflow was never mutated.
    setMessages((prev) => prev.map((m) => (m.id === messageId ? withProposalResolution(m, "cancelled") : m)));
  }, []);

  return {
    spec,
    loading,
    error,
    resyncNonce,
    applyMutation,
    refetch,
    messages,
    chatBusy,
    proposeInstruction,
    applyProposal,
    cancelProposal,
  };
}
