import { NODE_CATALOG, NODE_CATEGORIES, catalogEntriesByCategory, type NodeCatalogEntry } from "./nodeCatalog";

export const DRAG_DATA_TYPE = "application/agent-node-type";

const CATEGORY_TITLES: Record<string, string> = {
  trigger: "Triggers",
  ai: "AI",
  tool: "Tools",
  control: "Control",
  human: "Human",
};

const CATEGORY_COLORS: Record<string, string> = {
  trigger: "var(--ac-trigger-base, #1b8a5a)",
  ai: "var(--ac-ai-base, #6d4aff)",
  tool: "var(--ac-tool-base, #1e6fd9)",
  control: "var(--ac-control-base, #33393f)",
  human: "var(--ac-human-base, #c2410c)",
};

export interface NodePaletteProps {
  /** Restrict to specific node types, e.g. to hide types a plan/tier doesn't support. Omit to show the full catalog. */
  filter?: (entry: NodeCatalogEntry) => boolean;
}

/**
 * A draggable node library — drag an entry onto AgentCanvas to add it
 * (AgentCanvas's onDrop handler reads DRAG_DATA_TYPE and turns it into an
 * add_node mutation with a real canvas position, computed via
 * useReactFlow's screenToFlowPosition so it lands exactly where dropped).
 * Previously the only way to get a new node onto the canvas was inferring
 * one from a chat instruction — this is the direct, discoverable path.
 */
export function NodePalette({ filter }: NodePaletteProps) {
  const entries = filter ? NODE_CATALOG.filter(filter) : NODE_CATALOG;

  return (
    <div style={{ fontFamily: "var(--ac-font-body, system-ui)", overflowY: "auto", height: "100%" }}>
      {NODE_CATEGORIES.map((category) => {
        const categoryEntries = entries.filter((e) => e.category === category);
        if (categoryEntries.length === 0) return null;
        return (
          <div key={category} style={{ marginBottom: 4 }}>
            <div
              style={{
                fontSize: 11,
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.04em",
                color: "var(--ac-ink-muted, #6b7280)",
                padding: "10px 12px 4px",
              }}
            >
              {CATEGORY_TITLES[category] ?? category}
            </div>
            {categoryEntries.map((entry) => (
              <div
                key={entry.type}
                draggable
                onDragStart={(e) => {
                  e.dataTransfer.setData(DRAG_DATA_TYPE, entry.type);
                  e.dataTransfer.effectAllowed = "move";
                }}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  padding: "6px 12px",
                  cursor: "grab",
                  borderLeft: `3px solid ${CATEGORY_COLORS[category]}`,
                  margin: "2px 6px",
                  borderRadius: 4,
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "var(--ac-canvas-bg, #f6f7f5)")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                title={`Drag onto the canvas to add a ${entry.label} node`}
              >
                <span aria-hidden="true" style={{ color: CATEGORY_COLORS[category], flexShrink: 0 }}>
                  {entry.icon}
                </span>
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 500 }}>{entry.label}</div>
                  <div style={{ fontSize: 11, color: "var(--ac-ink-muted, #6b7280)" }}>{entry.description}</div>
                </div>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}

// Re-exported so consumers building a custom palette (e.g. filtered to
// just one category) can still reach the raw catalog data.
export { NODE_CATALOG, catalogEntriesByCategory };
