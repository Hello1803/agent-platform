import type { NodeCategory, NodeType } from "@agent-platform/shared-types";

export interface NodeCatalogEntry {
  type: NodeType;
  category: NodeCategory;
  label: string;
  description: string;
  icon: string;
  /** Sensible default config so a freshly-dropped node isn't immediately invalid — e.g. classify needs config.labels to compile at all. */
  defaultConfig?: Record<string, unknown>;
  defaultExecutionType?: "deterministic" | "probabilistic";
}

/**
 * Every NodeType the backend actually supports (backend/models/agent_spec.py's
 * NodeType enum), grouped and described for the palette — this is the "add
 * different node types" piece: previously the only way to get a node onto
 * the canvas was the NL compiler inferring one from a chat instruction;
 * there was no direct way to just drag e.g. a Summarize or Retry node on.
 */
export const NODE_CATALOG: NodeCatalogEntry[] = [
  // AI
  {
    type: "ai_agent", category: "ai", label: "AI Agent", icon: "✦",
    description: "Reason and act toward a goal",
    defaultExecutionType: "probabilistic", defaultConfig: { output_key: "ai_agent_output" },
  },
  {
    type: "ask_ai", category: "ai", label: "Ask AI", icon: "✦",
    description: "Single model call",
    defaultExecutionType: "probabilistic", defaultConfig: { output_key: "ai_response", prompt: "Analyze the current data and answer clearly." },
  },
  {
    type: "classify", category: "ai", label: "Classify", icon: "✦",
    description: "Label incoming content",
    defaultExecutionType: "probabilistic", defaultConfig: { labels: ["label_a", "label_b"] },
  },
  {
    type: "extract_information", category: "ai", label: "Extract", icon: "✦",
    description: "Pull structured fields",
    defaultExecutionType: "probabilistic", defaultConfig: { fields: ["field_1"] },
  },
  {
    type: "summarize", category: "ai", label: "Summarize", icon: "✦",
    description: "Condense gathered content",
    defaultExecutionType: "probabilistic",
  },
  {
    type: "make_decision", category: "ai", label: "AI Decision", icon: "✦",
    description: "Probabilistic branch",
    defaultExecutionType: "probabilistic",
  },
  {
    type: "run_agent", category: "ai", label: "Run Agent", icon: "✦",
    description: "Nested published agent",
    defaultExecutionType: "probabilistic",
  },

  // Tools
  { type: "search_web", category: "tool", label: "Search Web", icon: "⚙", description: "Query the public web" },
  { type: "open_website", category: "tool", label: "Open Website", icon: "⚙", description: "Fetch a URL" },
  { type: "read_email", category: "tool", label: "Read Email", icon: "⚙", description: "Read from an inbox" },
  { type: "send_email", category: "tool", label: "Send Email", icon: "⚙", description: "Send via Gmail/SMTP — body comes from Ask AI when left empty", defaultConfig: { subject: "Message from agent", body: "", email_field: "Email" } },
  { type: "search_documents", category: "tool", label: "Search Documents", icon: "⚙", description: "Query indexed documents" },
  { type: "read_file", category: "tool", label: "Read File", icon: "⚙", description: "Read CSV/XLSX/text from the agent files directory", defaultConfig: { path: "data.csv" } },
  { type: "write_file", category: "tool", label: "Write File", icon: "⚙", description: "Write CSV/XLSX/text into the agent files directory", defaultConfig: { path: "output.txt", content_variable: "ai_response" } },
  { type: "database", category: "tool", label: "Database", icon: "⚙", description: "Read or write records" },
  { type: "api_call", category: "tool", label: "HTTP Request", icon: "⚙", description: "Call any HTTP API (GET/POST/…)", defaultConfig: { method: "GET", url: "https://httpbin.org/get", headers: {}, body: null } },
  { type: "rag_index", category: "tool", label: "Index Documents", icon: "⚙", description: "Chunk & index files for RAG (txt/pdf/docx/csv/…)", defaultConfig: { paths: ["notes.txt"], index_name: "default", chunk_size: 800 } },
  { type: "rag_query", category: "tool", label: "Query Knowledge", icon: "⚙", description: "Retrieve relevant chunks for a question", defaultConfig: { query: "", index_name: "default", top_k: 5 } },
  { type: "preview", category: "tool", label: "Preview", icon: "👁", description: "Show upstream text as rendered Markdown (no file write)", defaultConfig: { source: "ai_response" } },
  { type: "code", category: "tool", label: "Code", icon: "⚙", description: "Run a short Python snippet on workflow variables", defaultConfig: { language: "python", code: "# input = variables from earlier nodes\n# return a dict to merge into state\nrows = input.get(\"table_preview\") or input.get(\"rows\") or []\nreturn {\"row_count\": len(rows), \"sample\": rows[:5]}" } },
  { type: "browser", category: "tool", label: "Browser", icon: "⚙", description: "Automate a browser" },
  // Control
  { type: "if_else", category: "control", label: "If / Else", icon: "⌥", description: "Branch on a condition" },
  { type: "switch", category: "control", label: "Switch", icon: "⌥", description: "Branch on multiple cases" },
  { type: "loop", category: "control", label: "Loop", icon: "⌥", description: "For-each over a list (e.g. filtered students)", defaultConfig: { items_variable: "ai_response", items_path: "filtered_students", item_variable: "current_item", email_field: "Email" } },
  { type: "parallel", category: "control", label: "Parallel", icon: "⌥", description: "Run branches concurrently" },
  { type: "merge", category: "control", label: "Merge", icon: "⌥", description: "Join branches back together" },
  { type: "wait", category: "control", label: "Wait", icon: "⌥", description: "Pause for a duration" },
  { type: "retry", category: "control", label: "Retry", icon: "⌥", description: "Retry on failure" },

  // Human
  { type: "human_approval", category: "human", label: "Human Approval", icon: "✋", description: "Pause for approve/reject" },
  { type: "ask_user", category: "human", label: "Ask User", icon: "✋", description: "Request free-text input" },
  { type: "review", category: "human", label: "Review", icon: "✋", description: "Request open-ended review" },
  { type: "confirm", category: "human", label: "Confirm", icon: "✋", description: "Request yes/no confirmation" },

  // Triggers
  { type: "trigger_manual", category: "trigger", label: "Manual Trigger", icon: "⚡", description: "Start on demand" },
  { type: "trigger_schedule", category: "trigger", label: "Schedule", icon: "⚡", description: "Start on a time interval (server-side)", defaultConfig: { enabled: true, interval_minutes: 5 } },
  { type: "trigger_email", category: "trigger", label: "New Email", icon: "⚡", description: "Start on incoming email" },
  { type: "trigger_webhook", category: "trigger", label: "Webhook", icon: "⚡", description: "Start on an HTTP call" },
  { type: "trigger_file_change", category: "trigger", label: "File Change", icon: "⚡", description: "Start on a file change" },
  { type: "trigger_external_event", category: "trigger", label: "External Event", icon: "⚡", description: "Start on an external event" },
];

export const NODE_CATEGORIES: NodeCategory[] = ["trigger", "ai", "tool", "control", "human"];

/**
 * Mirrors backend/compiler/compiler.py's REQUIRED_CAPABILITY exactly —
 * kept in sync by hand (same cross-language pattern as the schema/TS
 * types/Pydantic trio). This is the fix for a real bug: dropping a tool
 * node from the palette with no tool binding produces a spec that's
 * structurally valid (Pydantic accepts it) but fails at COMPILE time with
 * an opaque error the user only discovers when they try to run it. See
 * catalogEntryToAddNodeMutation in specToFlow.ts, which uses this to
 * auto-provision a starter tool binding atomically with the node itself.
 */
export const REQUIRED_CAPABILITY: Partial<Record<NodeType, "read" | "write" | "send">> = {
  search_web: "read",
  read_email: "read",
  send_email: "send",
  database: "write",
};

export function catalogEntriesByCategory(category: NodeCategory): NodeCatalogEntry[] {
  return NODE_CATALOG.filter((e) => e.category === category);
}
