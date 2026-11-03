import { useEffect, useState } from "react";
import type { AgentApiClient, WorkflowSummary } from "./apiClient";

export interface WorkflowSelectorProps {
  client: AgentApiClient;
  agentId: string;
  selectedWorkflowId: string;
  onSelectWorkflow: (workflowId: string) => void;
  refreshKey?: number;
}

/**
 * Everything here is backed by the real backend workflow CRUD routes
 * (backend/api/app.py's create_workflow/list_workflows) -- this is
 * deliberately NOT a local list of labels. Selecting a workflow just
 * calls `onSelectWorkflow`, which the caller (App.tsx) uses to change
 * the `agentId` (== workflowId) param passed into useAgentSession --
 * that hook already re-fetches from the backend whenever that value
 * changes, so switching workflows here genuinely loads that workflow's
 * real backend state, not a cached or synthesized view of it.
 */
export function WorkflowSelector({ client, agentId, selectedWorkflowId, onSelectWorkflow, refreshKey }: WorkflowSelectorProps) {
  const [workflows, setWorkflows] = useState<WorkflowSummary[] | null>(null);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [error, setError] = useState<string | null>(null);

  const refresh = () => {
    client
      .listWorkflows(agentId)
      .then(setWorkflows)
      .catch(() => setError("Could not load workflows"));
  };

  useEffect(refresh, [client, agentId, refreshKey]);

  const handleCreate = async () => {
    const name = newName.trim();
    if (!name) return;
    try {
      const spec = await client.createWorkflow(agentId, name);
      setNewName("");
      setCreating(false);
      setError(null);
      refresh();
      onSelectWorkflow(spec.metadata.id);
    } catch {
      setError("Could not create workflow");
    }
  };

  if (error) return <span style={{ fontSize: 12, color: "var(--ac-human-ink, #8a2e08)" }}>{error}</span>;
  if (!workflows) return <span style={{ fontSize: 12, color: "var(--ac-ink-muted, #6b7280)" }}>Loading workflows…</span>;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13 }}>
      <span style={{ color: "var(--ac-ink-muted, #6b7280)" }}>Workflow</span>
      <select
        value={selectedWorkflowId}
        onChange={(e) => onSelectWorkflow(e.target.value)}
        style={{ fontSize: 13, padding: "4px 8px", borderRadius: 6, border: "1px solid var(--ac-border, #d8dad5)" }}
      >
        {workflows.map((w) => (
          <option key={w.workflowId} value={w.workflowId}>
            {w.name}
            {!w.enabled ? " (disabled)" : ""}
          </option>
        ))}
      </select>

      {creating ? (
        <>
          <input
            autoFocus
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void handleCreate();
              if (e.key === "Escape") {
                setCreating(false);
                setNewName("");
              }
            }}
            placeholder="Workflow name"
            style={{ fontSize: 13, padding: "4px 8px", borderRadius: 6, border: "1px solid var(--ac-border, #d8dad5)" }}
          />
          <button onClick={handleCreate} disabled={!newName.trim()} style={{ fontSize: 13, padding: "4px 10px", borderRadius: 6, border: "none", background: "var(--ac-tool-base, #1e6fd9)", color: "#fff", cursor: "pointer" }}>
            Create
          </button>
          <button onClick={() => { setCreating(false); setNewName(""); }} style={{ fontSize: 13, padding: "4px 10px", borderRadius: 6, border: "1px solid var(--ac-border, #d8dad5)", background: "transparent", cursor: "pointer" }}>
            Cancel
          </button>
        </>
      ) : (
        <button
          onClick={() => setCreating(true)}
          style={{ fontSize: 13, padding: "4px 10px", borderRadius: 6, border: "1px solid var(--ac-border, #d8dad5)", background: "transparent", cursor: "pointer" }}
        >
          + New Workflow
        </button>
      )}
    </div>
  );
}
