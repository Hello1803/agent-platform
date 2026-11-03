/**
 * Drives a run over the SSE streaming endpoints, exposing exactly what
 * AgentCanvas's `trace`/`activeNodeId` props and a TraceLog need. Pairs
 * with useAgentSession.ts (which owns spec editing) — a real app uses
 * both: useAgentSession for the canvas's edit state, useAgentRunStream
 * for "Test"/"Use" mode's live run view (doc section 14's Build/Test/Use
 * modes).
 */

import { useCallback, useRef, useState } from "react";
import { AgentApiClient } from "./apiClient";
import type { TraceEventLike } from "./specToFlow";

export type RunStreamStatus = "idle" | "running" | "paused" | "completed" | "failed";

export interface UseAgentRunStreamResult {
  status: RunStreamStatus;
  /**
   * The trace so far — grows one event at a time as the SSE stream
   * delivers them (see the timing assertions in
   * backend/api/test_streaming.py, which verify events genuinely arrive
   * incrementally rather than all at once). Feed straight to
   * AgentCanvas's trace prop for nodes to light up live as they complete.
   */
  trace: TraceEventLike[];
  /**
   * The node currently executing, sourced from the backend's real
   * `node_started` SSE event (LangGraph's custom stream mode — see
   * compiler.py's _make_node_fn) which genuinely fires BEFORE that node's
   * executor runs, not a client-side guess. Cleared the moment a trace
   * event confirms that same node finished. A node that pauses the run
   * (human_approval) gets a node_started but never a matching trace event
   * — in that case activeNodeId correctly stays set to it until `reset()`,
   * which reads naturally as "still waiting on this node" while paused.
   */
  activeNodeId: string | null;
  variables: Record<string, unknown>;
  /** Present when status === "paused" — the human_approval (or ask_user/confirm/review) payload to show the approver. */
  interrupt: Record<string, unknown> | null;
  threadId: string | null;
  error: string | null;
  /** The node whose executor raised, when status === "failed" — sourced
   * from the backend's checkpointed state (see apiClient.ts's RunResult),
   * not a guess. `trace`/`variables` above are preserved up to the
   * failure point too, not reset — see backend/api/app.py's
   * _partial_result_after_failure. */
  failedNodeId: string | null;
  start: (seedVariables?: Record<string, unknown>) => Promise<void>;
  resume: (resumeValue: Record<string, unknown>) => Promise<void>;
  reset: () => void;
}

export function useAgentRunStream(client: AgentApiClient, agentId: string): UseAgentRunStreamResult {
  const [status, setStatus] = useState<RunStreamStatus>("idle");
  const [trace, setTrace] = useState<TraceEventLike[]>([]);
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null);
  const [variables, setVariables] = useState<Record<string, unknown>>({});
  const [interrupt, setInterrupt] = useState<Record<string, unknown> | null>(null);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [failedNodeId, setFailedNodeId] = useState<string | null>(null);

  // Guards against a stale stream's events landing after the user has
  // already started a new run (e.g. clicked "run" twice quickly).
  const generation = useRef(0);

  const reset = useCallback(() => {
    generation.current++;
    setStatus("idle");
    setTrace([]);
    setActiveNodeId(null);
    setVariables({});
    setInterrupt(null);
    setThreadId(null);
    setError(null);
    setFailedNodeId(null);
  }, []);

  const handleNodeStarted = useCallback((gen: number, event: { node_id: string; node_label: string }) => {
    if (gen !== generation.current) return;
    setActiveNodeId(event.node_id);
  }, []);

  const handleTraceEvent = useCallback((gen: number, event: Record<string, unknown>) => {
    if (gen !== generation.current) return;
    const typedEvent = event as unknown as TraceEventLike;
    setTrace((prev) => [...prev, typedEvent]);
    // This node just finished — clear activeNodeId only if it's still
    // pointing at THIS node (a fast subsequent node_started could already
    // have moved it on by the time this trace event is processed).
    setActiveNodeId((current) => (current === typedEvent.node_id ? null : current));
  }, []);

  const start = useCallback(
    async (seedVariables: Record<string, unknown> = {}) => {
      const gen = ++generation.current;
      setStatus("running");
      setTrace([]);
      setActiveNodeId(null);
      setInterrupt(null);
      setError(null);

      try {
        const result = await client.runAgentStream(
          agentId,
          seedVariables,
          (event) => handleTraceEvent(gen, event),
          (event) => handleNodeStarted(gen, event)
        );
        if (gen !== generation.current) return;
        setThreadId(result.threadId);
        setVariables(result.variables);
        if (result.status === "paused") {
          setStatus("paused");
          setInterrupt(result.interrupt ?? null);
        } else if (result.status === "failed") {
          setStatus("failed");
          setError(result.error ?? "Run failed.");
          setFailedNodeId(result.failedNodeId ?? null);
        } else {
          setStatus("completed");
          setActiveNodeId(null);
        }
      } catch (e) {
        if (gen !== generation.current) return;
        setStatus("failed");
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [client, agentId, handleTraceEvent, handleNodeStarted]
  );

  const resume = useCallback(
    async (resumeValue: Record<string, unknown>) => {
      if (!threadId) throw new Error("Cannot resume: no thread_id (has a run been started?).");
      const gen = ++generation.current;
      setStatus("running");
      setInterrupt(null);
      setError(null);

      try {
        const result = await client.resumeAgentStream(
          agentId,
          threadId,
          resumeValue,
          (event) => handleTraceEvent(gen, event),
          (event) => handleNodeStarted(gen, event)
        );
        if (gen !== generation.current) return;
        setVariables(result.variables);
        if (result.status === "paused") {
          setStatus("paused");
          setInterrupt(result.interrupt ?? null);
        } else if (result.status === "failed") {
          setStatus("failed");
          setError(result.error ?? "Run failed.");
          setFailedNodeId(result.failedNodeId ?? null);
        } else {
          setStatus("completed");
          setActiveNodeId(null);
        }
      } catch (e) {
        if (gen !== generation.current) return;
        setStatus("failed");
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [client, agentId, threadId, handleTraceEvent, handleNodeStarted]
  );

  return { status, trace, activeNodeId, variables, interrupt, threadId, error, failedNodeId, start, resume, reset };
}
