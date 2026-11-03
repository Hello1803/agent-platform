import type { NodeType } from "@agent-platform/shared-types";

/** UI projection of the backend registry. The backend remains authoritative. */
export interface ToolDefinition {
  id: string;
  name: string;
  nodeTypes: NodeType[];
  permissions: string[];
  credentialRequired: boolean;
  execution: "demo" | "connected";
}

export const TOOL_DEFINITIONS: ToolDefinition[] = [
  // execution: "connected" -- web.search has a real connected-mode
  // executor (backend/real_executors/search.py, via Anthropic's
  // server-side web search tool), gated by a binding's own `mode` just
  // like every other tool here. Was "demo" (and named "(demo)") when
  // the only implementation was a stub; kept in sync with
  // backend/tools/registry.py's identical fix.
  { id: "web.search", name: "Web Search", nodeTypes: ["search_web"], permissions: ["read"], credentialRequired: false, execution: "connected" },
  { id: "gmail.send", name: "Gmail (OAuth)", nodeTypes: ["send_email"], permissions: ["send"], credentialRequired: true, execution: "connected" },
  { id: "smtp.send", name: "Email (SMTP)", nodeTypes: ["send_email"], permissions: ["send"], credentialRequired: true, execution: "connected" },
  { id: "gmail.read", name: "Gmail (OAuth)", nodeTypes: ["read_email"], permissions: ["read"], credentialRequired: true, execution: "connected" },
  { id: "smtp.read", name: "Email (IMAP)", nodeTypes: ["read_email"], permissions: ["read"], credentialRequired: true, execution: "connected" },
  { id: "google_sheets.append_row", name: "Google Sheets", nodeTypes: ["database"], permissions: ["write"], credentialRequired: true, execution: "connected" },
  { id: "local.sqlite.append_row", name: "Local SQLite (demo)", nodeTypes: ["database"], permissions: ["write"], credentialRequired: false, execution: "demo" },
  // credentialRequired: false -- read_file/write_file do local, sandboxed
  // disk I/O and never actually fetch a secret to do it (see
  // backend/tools/registry.py's identical fix and its docstring for why).
  { id: "files.read", name: "File reader", nodeTypes: ["read_file"], permissions: ["read"], credentialRequired: false, execution: "connected" },
  { id: "files.write", name: "File writer", nodeTypes: ["write_file"], permissions: ["write"], credentialRequired: false, execution: "connected" },
];

export function toolsForNode(nodeType: NodeType): ToolDefinition[] {
  return TOOL_DEFINITIONS.filter((tool) => tool.nodeTypes.includes(nodeType));
}
